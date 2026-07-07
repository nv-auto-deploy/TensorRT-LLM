# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Equivalence tests for the AutoDeploy greedy-decode sampler fast path.

The fast path must hand off exactly the same greedy token IDs and replay-bound
state (the ``store.new_tokens`` device buffer consumed by the overlap
scheduler's next-input gather) as the generic TorchSampler over many
same-process decode handoffs, and must fall back to the generic path for any
batch it does not cover.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.auto_deploy.shim.ad_executor import instantiate_sampler
from tensorrt_llm._torch.auto_deploy.shim.ad_sampler import (
    ADFastGreedySampleState,
    ADGreedyDecodeTorchSampler,
)
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestState, convert_wordlist
from tensorrt_llm._torch.pyexecutor.sampler import TorchSampler, _request_strategy
from tensorrt_llm._torch.pyexecutor.sampling_utils import GREEDY
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.bindings.executor import SamplingConfig as ExecutorSamplingConfig
from tensorrt_llm.llmapi.llm_args import SamplerType

VOCAB_SIZE = 128
MAX_SEQ_LEN = 256
NUM_HANDOFFS = 24


def _make_sampler_args(max_num_sequences: int = 2) -> TorchSampler.Args:
    return TorchSampler.Args(
        max_seq_len=MAX_SEQ_LEN,
        max_draft_len=0,
        max_total_draft_tokens=0,
        max_num_sequences=max_num_sequences,
        max_beam_width=1,
    )


def _make_request(
    seq_slot: int,
    prompt: list[int],
    max_new_tokens: int,
    end_id: int,
    stop_words_list=None,
) -> LlmRequest:
    req = LlmRequest(
        request_id=seq_slot,
        seq_slot=seq_slot,
        input_tokens=list(prompt),
        max_new_tokens=max_new_tokens,
        end_id=end_id,
        stop_words_list=convert_wordlist(stop_words_list) if stop_words_list else None,
        sampling_config=SamplingConfig(),
        is_streaming=False,
    )
    return req


def _setup_generation_request(sampler: TorchSampler, req: LlmRequest) -> ScheduledRequests:
    """Register the request with the sampler store (as the real context step would)
    and return a generation-only batch containing it."""
    ctx_batch = ScheduledRequests()
    ctx_batch.append_context_request(req)
    sampler.setup_sampler_step(ctx_batch)
    req.state = LlmRequestState.GENERATION_IN_PROGRESS
    gen_batch = ScheduledRequests()
    gen_batch.generation_requests = [req]
    return gen_batch


def _step_logits(step: int, num_rows: int = 1, forced_token: int = None) -> torch.Tensor:
    """Deterministic per-step logits whose argmax is (7 * step + 3) % VOCAB_SIZE."""
    g = torch.Generator(device="cuda").manual_seed(1234 + step)
    logits = torch.rand((num_rows, VOCAB_SIZE), generator=g, device="cuda", dtype=torch.float32)
    token = forced_token if forced_token is not None else (7 * step + 3) % VOCAB_SIZE
    logits[:, token] += 10.0
    return logits


@pytest.mark.parametrize("sampler_type", [SamplerType.auto, SamplerType.TorchSampler])
@pytest.mark.parametrize("pp_size", [1, 2])
def test_sampler_factory_gates_fast_path_to_pp1(sampler_type, pp_size):
    config = SimpleNamespace(
        speculative_config=None,
        sampler_type=sampler_type,
        max_seq_len=MAX_SEQ_LEN,
        max_beam_width=1,
        disable_overlap_scheduler=False,
    )
    mapping = SimpleNamespace(pp_size=pp_size)

    sampler = instantiate_sampler(config, max_num_sequences=2, dist_mapping=mapping, engine=None)

    if pp_size == 1:
        assert isinstance(sampler, ADGreedyDecodeTorchSampler)
    else:
        assert type(sampler) is TorchSampler


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("finish_mode", ["end_id", "length"])
def test_fast_greedy_matches_generic_over_handoffs(finish_mode):
    end_id = 99
    max_new_tokens = NUM_HANDOFFS + 1 if finish_mode == "length" else 2 * NUM_HANDOFFS
    prompt = [11, 12, 13]

    generic = TorchSampler(_make_sampler_args())
    fast = ADGreedyDecodeTorchSampler(_make_sampler_args())

    req_g = _make_request(0, prompt, max_new_tokens, end_id)
    req_f = _make_request(0, prompt, max_new_tokens, end_id)
    assert _request_strategy(req_f, vocab_size=2**31) == GREEDY

    batch_g = _setup_generation_request(generic, req_g)
    batch_f = _setup_generation_request(fast, req_f)

    engaged_steps = 0
    for step in range(NUM_HANDOFFS + 2):
        if req_g.state == LlmRequestState.GENERATION_COMPLETE:
            break
        forced = end_id if (finish_mode == "end_id" and step == NUM_HANDOFFS) else None
        logits = _step_logits(step, forced_token=forced)
        outputs_g = {"logits": logits.clone()}
        outputs_f = {"logits": logits.clone()}

        state_g = generic.sample_async(batch_g, outputs_g, [0])
        state_f = fast.sample_async(batch_f, outputs_f, [0])
        assert isinstance(state_f, ADFastGreedySampleState), "fast path did not engage"
        engaged_steps += 1

        generic.update_requests(state_g)
        fast.update_requests(state_f)

        # Exact greedy token IDs and request-visible state per handoff.
        assert req_f.get_tokens(0) == req_g.get_tokens(0), f"token mismatch at step {step}"
        assert req_f.state == req_g.state, f"state mismatch at step {step}"
        assert req_f.is_finished == req_g.is_finished
        assert req_f.is_finished_due_to_length == req_g.is_finished_due_to_length
        assert req_f.py_decoding_iter == req_g.py_decoding_iter

        # Replay-bound state: the device buffer the overlap scheduler gathers the
        # next captured-graph input from must match bit-exactly.
        torch.cuda.synchronize()
        assert torch.equal(fast.store.new_tokens, generic.store.new_tokens), (
            f"store.new_tokens mismatch at step {step}"
        )

    assert engaged_steps >= NUM_HANDOFFS, f"only {engaged_steps} fast handoffs ran"
    assert req_g.state == LlmRequestState.GENERATION_COMPLETE
    assert req_f.state == LlmRequestState.GENERATION_COMPLETE
    if finish_mode == "length":
        assert req_f.is_finished_due_to_length
    else:
        assert not req_f.is_finished_due_to_length


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fast_greedy_multi_request_batch():
    """Two concurrent greedy requests share one batch; tokens land in their slots."""
    fast = ADGreedyDecodeTorchSampler(_make_sampler_args())
    generic = TorchSampler(_make_sampler_args())

    reqs_f = [_make_request(s, [5 + s], 32, end_id=999) for s in (0, 1)]
    reqs_g = [_make_request(s, [5 + s], 32, end_id=999) for s in (0, 1)]
    for sampler, reqs in ((fast, reqs_f), (generic, reqs_g)):
        ctx = ScheduledRequests()
        for r in reqs:
            ctx.append_context_request(r)
        sampler.setup_sampler_step(ctx)
        for r in reqs:
            r.state = LlmRequestState.GENERATION_IN_PROGRESS

    for step in range(20):
        g = torch.Generator(device="cuda").manual_seed(99 + step)
        logits = torch.rand((2, VOCAB_SIZE), generator=g, device="cuda", dtype=torch.float32)
        logits[0, (3 * step + 1) % VOCAB_SIZE] += 10.0
        logits[1, (5 * step + 2) % VOCAB_SIZE] += 10.0

        for sampler, reqs, expect_fast in ((fast, reqs_f, True), (generic, reqs_g, False)):
            batch = ScheduledRequests()
            batch.generation_requests = list(reqs)
            state = sampler.sample_async(batch, {"logits": logits.clone()}, [0])
            assert isinstance(state, ADFastGreedySampleState) == expect_fast
            sampler.update_requests(state)

        for rf, rg in zip(reqs_f, reqs_g):
            assert rf.get_tokens(0) == rg.get_tokens(0), f"step {step}"
        torch.cuda.synchronize()
        assert torch.equal(fast.store.new_tokens, generic.store.new_tokens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fast_greedy_fallback_cases():
    """Batches outside the fast-path contract must route to the generic sampler."""
    fast = ADGreedyDecodeTorchSampler(_make_sampler_args())

    # Stop-words request: generic path (device stop-word machinery) must run.
    req_sw = _make_request(0, [1, 2], 32, end_id=999, stop_words_list=[[42]])
    batch_sw = _setup_generation_request(fast, req_sw)
    state = fast.sample_async(batch_sw, {"logits": _step_logits(0)}, [0])
    assert not isinstance(state, ADFastGreedySampleState)
    fast.update_requests(state)

    # Context batch: falls back (and performs the generic store setup + sampling).
    fast2 = ADGreedyDecodeTorchSampler(_make_sampler_args())
    req_ctx = _make_request(1, [1, 2, 3], 32, end_id=999)
    ctx_batch = ScheduledRequests()
    ctx_batch.append_context_request(req_ctx)
    state = fast2.sample_async(ctx_batch, {"logits": _step_logits(1, num_rows=1)}, [0, 1])
    assert not isinstance(state, ADFastGreedySampleState)

    # Non-greedy request: falls back.
    fast3 = ADGreedyDecodeTorchSampler(_make_sampler_args())
    exec_config = ExecutorSamplingConfig(beam_width=1)
    exec_config.top_k = 8
    exec_config.temperature = 0.7
    req_tp = LlmRequest(
        request_id=0,
        seq_slot=0,
        input_tokens=[4, 5],
        max_new_tokens=8,
        end_id=999,
        sampling_config=SamplingConfig(exec_config),
        is_streaming=False,
    )
    assert _request_strategy(req_tp, vocab_size=2**31) != GREEDY
    batch_tp = _setup_generation_request(fast3, req_tp)
    state = fast3.sample_async(batch_tp, {"logits": _step_logits(2)}, [0])
    assert not isinstance(state, ADFastGreedySampleState)

    # CUDA-graph padding: logits may carry trailing padded rows; the fast path
    # must ignore them and still engage.
    fast4 = ADGreedyDecodeTorchSampler(_make_sampler_args())
    req_pad = _make_request(0, [7], 32, end_id=999)
    batch_pad = _setup_generation_request(fast4, req_pad)
    logits = _step_logits(3, num_rows=2)  # row 1 is a padded row
    state = fast4.sample_async(batch_pad, {"logits": logits}, [0])
    assert isinstance(state, ADFastGreedySampleState)
    fast4.update_requests(state)
    assert req_pad.get_tokens(0)[-1] == (7 * 3 + 3) % VOCAB_SIZE


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_outstanding_fast_state_owns_pinned_snapshot():
    """A second sample cannot overwrite an unconsumed fast-state host snapshot."""
    sampler = ADGreedyDecodeTorchSampler(_make_sampler_args())
    req = _make_request(0, [9], 32, end_id=999)
    batch = _setup_generation_request(sampler, req)

    token_a = 17
    token_b = 29
    token_c = 41
    state_a = sampler.sample_async(batch, {"logits": _step_logits(0, forced_token=token_a)}, [0])
    assert isinstance(state_a, ADFastGreedySampleState)

    # The persistent pinned mirror still belongs to state A. State B must use
    # generic per-state storage rather than overwrite that mirror.
    state_b = sampler.sample_async(batch, {"logits": _step_logits(1, forced_token=token_b)}, [0])
    assert not isinstance(state_b, ADFastGreedySampleState)

    sampler.update_requests(state_a)
    assert req.get_tokens(0)[-1] == token_a
    sampler.update_requests(state_b)
    assert req.get_tokens(0)[-1] == token_b

    state_c = sampler.sample_async(batch, {"logits": _step_logits(2, forced_token=token_c)}, [0])
    assert isinstance(state_c, ADFastGreedySampleState)
    sampler.update_requests(state_c)
    assert req.get_tokens(0)[-1] == token_c
