<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3Next GDN MTP Target-Verify Path

This note summarizes the PyTorch backend path that should guide the AutoDeploy
FLA cached gated-delta fix for Qwen3.5 MTP.

## Why This Matters

The AutoDeploy CUDA graph failure is currently in the target Qwen3.5 FLA
gated-delta path. The failing AutoDeploy op uses
`BatchInfo.get_absorbed_info()`, which folds extend requests into prefill
requests:

- `tensorrt_llm/_torch/auto_deploy/custom_ops/attention_interface.py`
- `BatchInfo.get_absorbed_info()`

That is a pre-MTP abstraction. For MTP target verification, the verification
region is not normal prefill and not a single-token decode. It is a short
per-request sequence containing the target token plus drafted tokens:

```text
tokens_per_verify_sequence = max_total_draft_tokens + 1
```

Qwen3Next already handles this explicitly in the PyTorch backend GDN mixer.
That path should be the reference for adapting the AutoDeploy FLA cached
gated-delta op.

## Entry Point

Qwen3Next linear-attention layers use `Qwen3NextGatedDeltaNet`:

- `tensorrt_llm/_torch/models/modeling_qwen3_next.py`
- `Qwen3NextLinearDecoderLayer.__init__()`
- `tensorrt_llm/_torch/modules/mamba/gdn_mixer.py`
- `Qwen3NextGatedDeltaNet`

In `Qwen3NextGatedDeltaNet.forward()`, the batch is split by metadata:

```python
num_prefills = attn_metadata.num_contexts
num_decodes = attn_metadata.seq_lens.shape[0] - num_prefills
num_prefill_tokens = attn_metadata.num_ctx_tokens
num_decode_tokens = attn_metadata.num_tokens - num_prefill_tokens
batch_split_size = [num_prefills, num_decodes]
state_indices = mamba_metadata.state_indices[: num_prefills + num_decodes]
state_indices_p, state_indices_d = torch.split(state_indices, batch_split_size)
```

In MTP target verification, the "decode" side is the verification region. It is
not restricted to one token per sequence. Its token count is expected to be:

```python
num_decode_tokens == num_decodes * (spec_metadata.max_total_draft_tokens + 1)
```

The target-verify path is enabled when:

```python
is_target_verify = (
    num_decodes > 0
    and spec_metadata is not None
    and attn_metadata.kv_cache_manager.is_speculative()
    and layer_cache is not None
)
```

When this is true, the layer also obtains speculative intermediate state
buffers:

```python
intermediate_conv_states = layer_cache.intermediate_conv_window
intermediate_ssm_states = layer_cache.intermediate_ssm
```

## Projection

The layer projects hidden states into:

```python
mixed_qkv, z, b, a
```

using `in_proj_qkvz` and `in_proj_ba`. The recurrent gated-delta part consumes
`q`, `k`, `v`, `a`, and `b`. The `z` branch is applied after the core recurrent
output.

## Mixed Prefill Plus Target Verify

If the batch contains prefill/context requests, `forward()` dispatches to
`forward_extend()`.

`forward_extend()` first splits prefill tokens from verification tokens:

```python
mixed_qkv_p, mixed_qkv_d = torch.split(
    mixed_qkv, [num_prefill_tokens, num_decode_tokens], dim=0
)
a_p, a_d = torch.split(a, [num_prefill_tokens, num_decode_tokens], dim=0)
b_p, b_d = torch.split(b, [num_prefill_tokens, num_decode_tokens], dim=0)
```

### Prefill Side

The prefill side is handled as normal context processing:

1. Run `causal_conv1d_fn(...)` over the prefill tokens.
2. Run `chunk_gated_delta_rule(...)` over the prefill recurrent sequence.
3. Write the final prefill recurrent state back to the real `ssm_states` slots.

This is allowed because prefill/context tokens are accepted by definition.

### Target-Verify Side

The target-verify side is treated differently.

First, the verification token region is reshaped:

```python
draft_token_num = spec_metadata.max_total_draft_tokens + 1
mixed_qkv_d = mixed_qkv_d.reshape(num_decodes, draft_token_num, -1).transpose(1, 2)
```

Then the convolution recurrence uses `causal_conv1d_update_triton(...)` with
intermediate-state writes:

```python
intermediate_state_indices = torch.arange(
    num_decodes, dtype=torch.int32, device=state_indices_d.device
)

mixed_qkv_d = causal_conv1d_update_triton(
    mixed_qkv_d,
    conv_states,
    self.conv1d.weight,
    self.conv1d.bias,
    activation=self.activation,
    conv_state_indices=state_indices_d,
    intermediate_conv_window=intermediate_conv_states,
    intermediate_state_indices=intermediate_state_indices,
)
```

This computes the conv outputs for all candidate verify steps but does not
blindly commit the last candidate conv state as the request state. Each
candidate step's conv state is saved in `intermediate_conv_states`.

After conv processing, the target-verify tensors are reshaped into recurrent
GDN form:

```python
query_d = query[:, num_prefill_tokens:, :, :].reshape(
    num_decodes, draft_token_num, num_k_heads, head_k_dim
)
key_d = key[:, num_prefill_tokens:, :, :].reshape(
    num_decodes, draft_token_num, num_k_heads, head_k_dim
)
value_d = value[:, num_prefill_tokens:, :, :].reshape(
    num_decodes, draft_token_num, num_v_heads, head_v_dim
)
a_d = a[num_prefill_tokens:].reshape(num_decodes, draft_token_num, -1)
b_d = b[num_prefill_tokens:].reshape(num_decodes, draft_token_num, -1)
```

Then the recurrent gated-delta verify path calls:

```python
fused_recurrent_gated_delta_rule_update(
    q=query_d,
    k=key_d,
    v=value_d,
    g=g_d,
    beta=beta_d,
    initial_state_source=ssm_states[state_indices_d],
    initial_state_indices=torch.arange(num_decodes, dtype=torch.int32, device=...),
    use_qk_l2norm_in_kernel=True,
    disable_state_update=True,
    intermediate_states_buffer=intermediate_ssm_states,
    cache_steps=draft_token_num,
)
```

The important fields are:

- `initial_state_source=ssm_states[state_indices_d]`: start from the real state
  before target verification.
- `disable_state_update=True`: do not mutate the real recurrent state to the
  end of the full target-plus-draft sequence.
- `intermediate_states_buffer=intermediate_ssm_states`: save every candidate
  recurrent state.
- `cache_steps=draft_token_num`: store one intermediate state per target-verify
  step.

This is the recurrent-state equivalent of avoiding KV/SSM over-commit during
speculative verification.

## Pure Target Verify

If there are no prefill/context tokens in the batch, `forward()` dispatches to
`forward_decode()`. Under `is_target_verify`, the same core pattern is used:

1. Assert that `mixed_qkv`, `a`, and `b` contain
   `num_decodes * (max_total_draft_tokens + 1)` rows.
2. Reshape to `[num_decodes, draft_token_num, ...]`.
3. Run `causal_conv1d_update_triton(...)` with `intermediate_conv_window`.
4. Run `fused_recurrent_gated_delta_rule_update(...)` with
   `disable_state_update=True` and `intermediate_states_buffer`.

## Acceptance Commit

The target-verify forward pass only computes candidate outputs and saves
candidate recurrent states. It does not decide which candidate state is real.

After sampling determines how many draft tokens were accepted,
`MambaCacheManager.update_mamba_states()` commits the accepted state:

```python
num_accepted_draft_tokens = (
    num_accepted_tokens[num_contexts:num_contexts + num_gens] - 1
)

accepted_ssm = intermediate_ssm_states[:, src_state_indices, num_accepted_draft_tokens]
all_ssm_states[:, state_indices_d, :] = accepted_ssm

accepted_conv = intermediate_conv_states[:, src_state_indices, num_accepted_draft_tokens]
all_conv_states[:, state_indices_d, :] = accepted_conv
```

So the state lifecycle is:

```text
real state before verify
  -> run target over target token plus draft tokens
  -> save every candidate conv/GDN state into intermediate buffers
  -> sampler accepts k tokens
  -> cache manager commits intermediate_state[k]
```

This avoids needing to "rewind" the real recurrent state after the verify pass.
The real state is not advanced past acceptance in the first place.

## Implications for AutoDeploy FLA Cached GDN

The current AutoDeploy FLA cached GDN path is still two-way:

```text
prefill-or-absorbed-extend + decode
```

For MTP it likely needs the same three-way structure as `triton_backend_mamba`:

```text
prefill + extend/target-verify + decode
```

Concretely:

1. Use `BatchInfo.get_num_sequences()` and `BatchInfo.get_num_tokens()` instead
   of `get_absorbed_info()` in the MTP-capable path.
2. Keep normal prefill behavior for true prefill tokens.
3. Add an explicit extend/target-verify branch for
   `num_extend > 0` / speculative verification tokens.
4. In that branch, reshape the extend tokens as
   `[num_extend, tokens_per_extend, ...]`.
5. Start from the real `delta_cache` state for the corresponding request slots.
6. Run a recurrent gated-delta update that can store each candidate final state
   into an intermediate spec cache.
7. Do not update `delta_cache` to the end of the whole verify sequence during
   target verification.
8. Let the post-sampling cache manager commit the accepted intermediate
   recurrent state.

This also explains why eager `torch-simple` can appear to work while CUDA graph
capture fails. Eager execution can tolerate Python-side host reads that CUDA
graph capture forbids, and nonzero acceptance only proves that the speculative
loop ran. It does not prove the recurrent state was committed with the accepted
candidate semantics.

## Fast Failure Context

The CUDA graph failure stack for the current AutoDeploy path is:

```text
fla_cached_gated_delta_rule
  -> chunk_gated_delta_rule
  -> chunk_local_cumsum
  -> prepare_chunk_indices
  -> .tolist()
```

The `.tolist()` comes from FLA varlen/chunk metadata preparation. The immediate
capture failure may be fixed by avoiding that path for fixed rectangular MTP
verify sequences, but the semantic fix still needs accepted-state handling as
described above.
