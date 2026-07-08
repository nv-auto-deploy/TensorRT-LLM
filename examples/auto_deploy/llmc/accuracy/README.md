# Accuracy suite (llmc standalone package)

MMLU + GSM8K accuracy tests for the standalone `llmc` package (AutoDeploy).

## Source of truth / how this ships

This directory is the **source** of the accuracy suite. It is copied into the
generated standalone `llm-compiler` repository by
`examples/auto_deploy/llmc/create_standalone_package.py` (which the
TensorRT-LLM → llm-compiler sync runs), landing at `tests/accuracy/` in that
repo. On copy, `tensorrt_llm._torch.auto_deploy` imports are rewritten to `llmc`
(so `from tensorrt_llm._torch.auto_deploy.llm import LLM` becomes
`from llmc.llm import LLM`).

Edit here in TensorRT-LLM — do **not** edit the copy in llm-compiler (it is
regenerated and overwritten on every sync). The suite is intentionally kept out
of TensorRT-LLM's own pytest collection (it lives under `examples/`); TRT-LLM's
in-tree AutoDeploy accuracy tests are separate
(`tests/integration/defs/accuracy/test_llm_api_autodeploy.py`).

The GitLab/JET CI that *runs* this suite in llm-compiler (`.gitlab-ci.yml`,
`workloads/accuracy.yaml`, `docker/Dockerfile.accuracy`) is llm-compiler-owned
and not part of this sync.

## Layout

| Path | What |
|---|---|
| `accuracy_core.py` | Dependency-free harness: reference loading, statistics (no scipy/trtllm), `registry_yaml_extra`, and spec-dec acceptance-rate helpers. CPU-importable. |
| `integration/test_llm_api_llmc.py` | The accuracy tests. Drive the AutoDeploy `LLM` on the TRT-LLM runtime; `tensorrt_llm` is imported lazily (via `importorskip`) so the module still collects on CPU. |
| `unit/test_accuracy_core.py` | Pure-CPU unit tests for the harness logic. |
| `references/{gsm8k,mmlu}.yaml` | Expected accuracy per model (keyed by HF id, with quant variants). |
| `conftest.py` | Puts this dir on `sys.path` so both subtrees can `import accuracy_core`; registers the `large`/`blackwell` markers. |

## Running (in the standalone package)

```bash
# Unit tests (CPU, no tensorrt_llm):
pytest tests/accuracy/unit -q

# Integration tests (need the TRT-LLM runtime; one class per model):
pytest tests/accuracy/integration/test_llm_api_llmc.py::TestLlama31_8B
```

In llm-compiler CI the integration tests are fanned out into one JET job per
class by `scripts/generate_accuracy_products.py`.

## Large models and platform routing

Models that don't fit a single 8×80GB node (DeepSeek-R1, Qwen3.5-397B, GLM-4.7,
gpt-oss-120b) are tagged `@pytest.mark.large`, and NVFP4/MXFP4 classes
`@pytest.mark.blackwell`. **All classes run** (no opt-in gate for now; only
`@pytest.mark.skip` classes are dropped). `generate_accuracy_products.py` routes
each product to a cluster: `@large`/`@blackwell` → `LLMC_ACCURACY_PLATFORM_BIG`,
else `LLMC_ACCURACY_PLATFORM`; `LLMC_ACCURACY_GPUS_PER_NODE` sets
`nodes = ceil(world_size / N)`. `TestDeepSeekR1` runs full-weight; gpt-oss / Qwen /
GLM are `xfail(strict=False)` (truncated smoke configs / no full reference yet).

## Model registry

`accuracy_core.registry_yaml_extra()` resolves each model's build config from the
shared model registry (`runners/trtllm/model_registry/models.yaml` in the
standalone package, mirrored from `examples/auto_deploy/model_registry/`). A model
that is commented-out (disabled) in that registry resolves to `[]`. A test for a
disabled model can either be **waived** (`@pytest.mark.skip`) or **load its config
directly** via `accuracy_core.registry_config_paths(...)` — the config *files* live
in `configs/` regardless of the registry entry, so the test needn't depend on (or
re-enable) `models.yaml`. `TestNemotronNanoV3` / `TestNemotronSuper120B` (disabled
upstream: device-side assert / OOM) and the MTP test do the latter.

## MTP (speculative decoding)

`TestNemotronSuperV3MTP` runs Nemotron-Super-V3 A12B FP8 at world_size 4 with MTP
multi-token-prediction spec-dec. MTP is a test-specific spec-dec config, not a
registry model variant, so the test loads `super_v3_mtp.yaml` **directly** via
`accuracy_core.registry_config_paths("super_v3_mtp.yaml")` (→
`speculative_config: {decoding_type: MTP, ...}`) rather than through the registry
overlay — matching upstream `test_mtp`, which also passes that yaml with
`world_size=4`. It scores GSM8K and asserts a spec-dec acceptance-rate floor
(≥0.50) via `accuracy_core.check_acceptance_rate` (reads `llm.get_stats()`
`specDecodingStats`; needs `enable_iter_perf_stats=True`).
