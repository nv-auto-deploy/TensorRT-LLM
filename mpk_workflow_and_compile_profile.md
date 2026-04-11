# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Gemma MPK Workflow And Compile Profile

This note records the current AutoDeploy -> MPK integration workflow as it
exists today, and the standalone compile timings that were measured before
making assumptions about the compile bottleneck.

## 1. Current AutoDeploy -> MPK Workflow

### 1.1 Real current path in the codebase

Today the compile-stage hook is:

- [lower_to_mpk.py](tensorrt_llm/_torch/auto_deploy/transform/library/lower_to_mpk.py)

The flow is:

```text
AutoDeploy compile-stage GraphModule
    -> GemmaMpkTranslator.build_plan(model)
    -> translation plan stored in autodeploy_meta["mpk_translation_plan"]
    -> if dry_run_only=false:
         wrap root graph with GemmaMpkRuntimeWrapper
         autodeploy_meta["mpk_runtime_mode"] = "mirage_runtime"
```

The wrapper is:

- [runtime_wrapper.py](tensorrt_llm/_torch/auto_deploy/mpk/runtime_wrapper.py)

Its contract is:

```text
wrapped FX graph
    -> call_module("gemma_mpk_runtime", ...)
    -> GemmaMpkRuntimeWrapper.forward(...)
    -> mpk_callable(*args, **kwargs)
```

Once `mpk_runtime_mode == "mirage_runtime"`, the normal compile transform is
skipped:

- [compile_model.py](tensorrt_llm/_torch/auto_deploy/transform/library/compile_model.py)

```text
if autodeploy_meta["mpk_runtime_mode"] == "mirage_runtime":
    skip compile_model
```

So the intended runtime shape is:

```text
FX graph
  -> lower_to_mpk
  -> gemma_mpk_runtime(...)
  -> direct MPK/Mirage execution
```

not:

```text
FX graph
  -> lower_to_mpk
  -> compile_model / cudagraph
  -> runtime
```

### 1.2 Important reality check

The wrapper path exists, but the full-model live Mirage callable is still not
implemented.

The current function is:

- [mirage_bridge.py](tensorrt_llm/_torch/auto_deploy/mpk/mirage_bridge.py)
  `build_gemma_mirage_runtime_callable(...)`

Today it still raises `NotImplementedError` for the full Gemma path and reports
the remaining gap/partial-step counts.

So the real status is:

- graph analysis and wrapping: implemented
- compile-model bypass for Mirage mode: implemented
- live full-model Mirage callable: not implemented yet

### 1.3 What is already live and real today

Even though the full-model callable is not finished, several live Mirage pieces
are real and tested:

- live attention-side sublayer/block correctness helpers
- live split-dense Gemma-style MoE block correctness helper
- real Mirage `PersistentKernel` task registration and execution for those block
  pieces

Those live helpers are in:

- [mirage_bridge.py](tensorrt_llm/_torch/auto_deploy/mpk/mirage_bridge.py)

and are tested in:

- [test_gemma4_mpk_layer.py](tests/unittest/auto_deploy/singlegpu/transformations/library/test_gemma4_mpk_layer.py)

## 2. Compile Dataflow Inside Mirage

### 2.1 Generic Mirage `KNGraph` compile

Generic Mirage graphs like:

```text
gelu -> mul
```

go through:

- [kernel.py](../common/mirage/python/mirage/kernel.py)

The path is:

```text
graph.compile(inputs=[...], target_cc=90)
    -> validate input dims/strides
    -> generate_cuda_program(...)
    -> write temporary test.cu
    -> nvcc compile to temporary .so
    -> import generated launcher
```

Important observation:

- there is no persistent on-disk compile cache in this normal `compile()` path
- caching is only on the live Python graph object via `_is_compiled`

That means:

```text
new Mirage graph object
    -> pays full transpile + nvcc cost again
```

### 2.2 MPK `PersistentKernel` compile

MPK `PersistentKernel` compile goes through:

- [persistent_kernel.py](../common/mirage/python/mirage/mpk/persistent_kernel.py)

The path is roughly:

```text
PersistentKernel.compile()
    -> kn_graph.generate_task_graph(...)
    -> write generated runtime CUDA
    -> nvcc compile
    -> import generated launcher
```

Like the generic path, this is also compile-once-per-object unless we add our
own higher-level reuse.

## 3. Standalone Compile Timing Measurements

These timings were measured from standalone Python calls, not inferred.

Environment:

- target CC: `90`
- same local Mirage install under:
  `/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python`

### 3.1 Generic Mirage activation graph: shape `(1, 8, 64)`

Graph:

```text
out = mul(gelu(a), b)
```

Measured by timing:

- `generate_cuda_program(...)`
- `subprocess.check_call(nvcc ...)`
- total `graph.compile(...)`

Result:

| Case | Codegen | `nvcc` | Other Python/import | Total |
|---|---:|---:|---:|---:|
| generic `gelu+mul`, `(1, 8, 64)` | `0.00897 s` | `11.94233 s` | `0.01302 s` | `11.96432 s` |

### 3.2 Generic Mirage activation graph: shape `(1, 64)`

Same graph:

```text
out = mul(gelu(a), b)
```

Result:

| Case | Codegen | `nvcc` | Other Python/import | Total |
|---|---:|---:|---:|---:|
| generic `gelu+mul`, `(1, 64)` | `0.00891 s` | `11.78761 s` | `0.01127 s` | `11.80779 s` |

### 3.3 Small MPK `PersistentKernel` compile

Case:

- one `linear_with_residual_layer`
- input/output shape `(1, 512)`

Result:

| Case | `nvcc` | Other Python/import | Total |
|---|---:|---:|---:|
| PK `linear_with_residual`, `(1, 512)` | `22.27655 s` | `0.03328 s` | `22.30983 s` |

## 4. What These Numbers Actually Mean

### 4.1 The bottleneck is real, and it is mostly `nvcc`

For the standalone generic activation case:

- Mirage code generation itself is tiny: about `0.009 s`
- Python/import overhead is tiny: about `0.01 s`
- almost all compile time is `nvcc`: about `11.8 - 11.9 s`

So the statement:

```text
"the expensive generic Mirage activation compile path"
```

is supported by direct evidence, and more precisely it means:

```text
fresh generic activation graph
    -> cheap Mirage codegen
    -> expensive nvcc compile
```

### 4.2 The issue is not just the 2D FFN shape

The 2D `(1, 64)` activation graph and the 3D `(1, 8, 64)` activation graph
compile in almost the same total time.

## 5. Stage-By-Stage Synthetic Full-Layer Profile

The bridge now exposes a standalone profiler for the synthetic full live-layer
path:

- [mirage_bridge.py](tensorrt_llm/_torch/auto_deploy/mpk/mirage_bridge.py)
  `profile_mirage_gemma_full_layer_split_dense_compile_stages(...)`

The measured command was:

```bash
bash -ic 'f1 && PYTHONWARNINGS=ignore \
  PYTHONPATH=$PWD:/lustre/fs1/portfolios/coreai/projects/coreai_comparch_autodeploy/users/bmarimuthu/common/mirage/python \
  python3 - <<\"PY\" 2>/tmp/mpk_full_layer_profile.err
from pprint import pprint
from tensorrt_llm._torch.auto_deploy.mpk.mirage_bridge import (
    profile_mirage_gemma_full_layer_split_dense_compile_stages,
)

results = profile_mirage_gemma_full_layer_split_dense_compile_stages(
    seed=0,
    verbose=True,
)
print("FINAL_PROFILE")
pprint(results)
PY'
```

### 5.1 Completed stages before the back-half stall

These timings were emitted directly by the helper before the run stopped making
forward progress:

| Stage | Compile | Launch | Notes |
|---|---:|---:|---|
| `attn_pk` | `26.390 s` | `0.062 s` | attention front half + paged attention + output proj/residual |
| `ffn_phase1` | `26.876 s` | `0.004 s` | RMSNorm + gate/up projection |
| `ffn_activation` | `11.908 s` | `0.003 s` | generic Mirage `gelu+mul`; codegen `0.009 s`, `nvcc` `11.898 s` |
| `ffn_phase2` | `26.963 s` | not yet printed when sampled | down projection compile completed |

### 5.2 What the stage profile already tells us

1. The front-half compile budget is already about:

```text
26.390 + 26.876 + 11.908 + 26.963 ~= 92.137 s
```

2. The generic activation stage still follows the same pattern as the standalone
   microbenchmark:
   - tiny codegen
   - compile time almost entirely in `nvcc`

3. The larger `PersistentKernel` stages are materially more expensive than the
   tiny single-task PK microbenchmark:
   - synthetic full-layer PK stages are about `26-27 s`
   - tiny standalone PK `linear_with_residual` was about `22.3 s`

### 5.3 Important blocker: the back-half delay is not purely compile time

The stage-profile run stopped emitting progress after `ffn_phase2_compile_s`.
A `gdb` stack sample of the still-running Python process showed the main thread
blocked in:

```text
torch.cuda.synchronize()
  -> cudaDeviceSynchronize()
```

That means the observed long tail after the front-half compiles is not just
"Mirage compile is slow". At least one later back-half step is stalling in live
GPU execution or launch synchronization.

So the current evidence-backed picture is:

```text
front-half compile cost: measured
back-half total delay: not explained by compile alone
back-half runtime/launch stall: confirmed by stack sample
```

## 6. Practical Implication

The next optimization/debugging step should not be "assume activation compile is
the whole problem". The data now says:

- yes, fresh PK and generic activation compiles are expensive
- but the synthetic full-layer slowdown also includes a later CUDA-side stall

So the right next split is:

1. keep the measured compile timings as the baseline cost model
2. isolate which back-half live stage is hanging in `cudaDeviceSynchronize`
3. only then decide whether to optimize by:
   - caching compiles
   - reducing the number of PK boundaries
   - adding a dedicated MPK task for Gemma GELU-mul

That suggests the full-layer slowdown is not explained purely by the FFN helper
shape. More likely causes are:

- repeated fresh compiles of multiple helper graphs
- one or more later full-layer stages triggering additional compiles
- cumulative wall-clock from several `~12s` generic compiles and `~22s` PK
  compiles

### 4.3 Why the full-layer helper can still feel "stuck"

The first synthetic full-layer helper currently tries to compose:

- attention PK compile
- FFN phase-1 PK compile
- FFN activation generic compile
- FFN phase-2 PK compile
- router PK compile
- expert PK compile
- MoE activation generic compile
- phase-3 PK compile

If those are all fresh compiles, the expected wall-clock can easily reach
minutes even before runtime execution.

That does not by itself prove a bug, but it explains why the full-layer helper
can look hung while still being CPU-active.

## 5. Can We Call MPK Compilation Standalone And Profile It?

Yes.

That is exactly what was done above.

Two practical standalone profiling modes are already possible:

1. Generic Mirage graph compile profiling

```text
construct mirage.new_kernel_graph()
    -> graph.compile(inputs=[...], target_cc=90)
    -> monkeypatch timing around:
         generate_cuda_program(...)
         subprocess.check_call(...)
```

2. MPK `PersistentKernel` compile profiling

```text
construct tiny PersistentKernel
    -> register a small task
    -> time compile_persistent_kernel_with_patches(pk)
    -> monkeypatch timing around subprocess.check_call(...)
```

So yes: we can call the compilation standalone, and we already have hard timing
data for the current activation-style case and for one small PK case.

## 6. Most Useful Next Profiling Step

The next most useful profiling step is not another isolated activation graph.
It is stage-by-stage timing of the first synthetic full-layer helper.

That would answer:

- which subcompile dominates in the full-layer composition
- whether the long wall-clock is just the sum of expected compile costs
- whether one specific helper stage is pathological

The most relevant stages to time are:

- attention PK compile
- FFN phase-1 PK compile
- FFN activation generic compile
- FFN phase-2 PK compile
- router PK compile
- expert PK compile
- MoE activation generic compile
- phase-3 PK compile

That should be done before changing architecture based on guesswork.
