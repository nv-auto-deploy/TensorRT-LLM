# Environment recovery (container / lease reset)

After a container or SLURM lease reset, the Python environment is frequently wiped and
`import tensorrt_llm` fails. This is the recovery sequence that worked, plus the CUDA-13
toolchain pitfalls that the flashinfer / trtllm-gen FMHA JIT paths hit on B200 (sm_100).

All `pip install` use `--break-system-packages` because the system Python is PEP-668
externally-managed.

## 0. First: check whether you even need to recover

```bash
python3 -c "import tensorrt_llm; from tensorrt_llm._torch.auto_deploy import LLM; print('AD OK')"
which sweep trtllm-serve
pip show tensorrt_llm | grep -iE "Editable|Location"   # confirm the LIVE checkout
```
If that prints `AD OK` and `sweep` resolves, skip the rest. **Always confirm the editable
location** — `import tensorrt_llm` follows the `__editable__*.pth` finder, which can map the
`tensorrt_llm` package to a *sibling* checkout (e.g. `TensorRT-LLM1`), not the tree you think
you're testing.

## 1. Reinstall the editable package + core deps

```bash
cd <TensorRT-LLM checkout>
pip install --break-system-packages wheel_stub setuptools          # build backend needs wheel_stub
pip install --break-system-packages -e . --no-build-isolation --no-deps   # editable, libs from working tree
pip install --break-system-packages -r requirements.txt            # pulls torch, transformers, etc.
```

If `-r requirements.txt` resolves a conflicting torch (cu12 vs cu13), let it settle on the
cu13 build that `requirements.txt` pins (`torch>=2.10,<=2.11`, `nvidia-nccl-cu13`, etc.). Then:

```bash
pip install --break-system-packages mpi4py blake3 "xgrammar==0.1.32" "transformers==5.5.4" "wheel>=0.46.2"
pip install --break-system-packages "tensorrt~=10.15.1"            # match requirements pin
```

## 2. CUDA-13 runtime libs + linker symlinks

The C++ extension expects `libcublasLt.so.13` etc. The pip cu13 bundle lives at
`~/.local/lib/python3.12/site-packages/nvidia/cu13/lib`. Export it and add the symlinks the
JIT linker expects:

```bash
CUDIR=/home/egeva/.local/lib/python3.12/site-packages/nvidia/cu13
export CUDA_HOME=$CUDIR
export LD_LIBRARY_PATH="$CUDIR/lib:$LD_LIBRARY_PATH"
export PATH="$CUDIR/bin:$PATH"
ln -sfn "$CUDIR/lib" "$CUDIR/lib64"                       # flashinfer links -L .../lib64
ln -sf "$CUDIR/lib/libcudart.so.13" "$CUDIR/lib/libcudart.so"   # unversioned -lcudart
mkdir -p "$CUDIR/lib/stubs"
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$CUDIR/lib/libcuda.so"        # driver stub
ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so.1 "$CUDIR/lib/stubs/libcuda.so"
```

## 3. CUDA toolkit (nvcc) version MUST match CUDART

The flashinfer mamba + trtllm-gen FMHA kernels JIT-compile at server startup. Three distinct
failures, all from a toolkit-version mismatch:

- `#error "CUDA compiler and CUDA toolkit headers are incompatible"` — CCCL compat check.
  Workaround: `export FLASHINFER_EXTRA_CUDAFLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK`
  (flashinfer reads this env; see `flashinfer/jit/cpp_ext.py`).
- `ptxas ... fatal : Unsupported .version 9.3; current version is '9.0'` — nvcc emits newer
  PTX than the bundled ptxas accepts. Fix by **pinning all toolkit components to one version**:
  ```bash
  pip install --break-system-packages \
    "nvidia-cuda-nvcc==13.0.88" "nvidia-cuda-crt==13.0.88" "nvidia-nvvm==13.0.88"
  ```
  (Match `CUDART_VERSION` — check `grep "define CUDA_VERSION " $CUDIR/include/cuda.h`; 13000 → 13.0.x.)
  Then clear the stale JIT cache: `rm -rf ~/.cache/flashinfer/*/100a/cached_ops/selective_state_update_*`.

## 4. FMHA NVRTC `cuda.h` / `cuda/std/type_traits` not found

The prebuilt trtllm-gen FMHA lib JIT-compiles via NVRTC and resolves `#include <cuda.h>` and
`#include <cuda/std/type_traits>` through a symlink baked into the build dir:
`cpp/build/tensorrt_llm/kernels/trtllmGenKernels/fmha/cuda`. After a rebuild on a different
host this symlink points at a dead `/usr/local/cuda/include`. NVRTC needs **both** `cuda.h`
(flat) **and** `cuda/std/*` (which lives under `cccl/cuda/` in a real CUDA tree). Recreate it
as a directory mirror:

```bash
SLINK=<checkout>/cpp/build/tensorrt_llm/kernels/trtllmGenKernels/fmha/cuda
CUDA_INCL=/usr/local/cuda-13.1/targets/x86_64-linux/include   # a real CUDA toolkit include dir
rm -rf "$SLINK"; mkdir -p "$SLINK"
for e in "$CUDA_INCL"/*; do ln -sf "$e" "$SLINK/$(basename "$e")"; done
rm -f "$SLINK/cuda"; ln -sfn "$CUDA_INCL/cccl/cuda" "$SLINK/cuda"   # so cuda/std/type_traits resolves
# verify: ls "$SLINK/cuda.h" "$SLINK/cuda/std/type_traits"
```

## 5. Make `auto-dev install` work (bare `python`)

`auto-dev` invokes a bare `python` (not `python3`). Shim it:
```bash
mkdir -p ~/.local/bin && ln -sf "$(which python3)" ~/.local/bin/python
```
Note `auto-dev install -s` does NOT pass `--break-system-packages`; if its editable-install
step fails on PEP-668, run the editable install manually (step 1).

## 6. Verify

```bash
python3 -c "import tensorrt_llm; from tensorrt_llm._torch.auto_deploy import LLM; print('AD OK')"
```
Then a 1-point smoke sweep at c=1 should land ~402 tps/u (TP2) / ~405 (TP4); if it's wildly
lower, a perf transform isn't applying (check the server log / use the `ad-conf-check` skill).
