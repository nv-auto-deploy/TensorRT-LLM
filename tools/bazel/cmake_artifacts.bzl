# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local-only Bazel wrappers for TensorRT-LLM CMake native artifacts."""

_NON_HERMETIC_CMAKE_EXECUTION_REQUIREMENTS = {
    "local": "1",
    "no-cache": "1",
    "no-remote": "1",
    "no-remote-cache": "1",
    "no-remote-exec": "1",
    "no-sandbox": "1",
}


def _trtllm_cmake_artifacts_impl(ctx):
    libtensorrt_llm = ctx.actions.declare_file("%s/libtensorrt_llm.so" % ctx.label.name)
    nvinfer_plugin = ctx.actions.declare_file("%s/libnvinfer_plugin_tensorrt_llm.so" % ctx.label.name)
    bindings = ctx.actions.declare_file("%s/%s" % (ctx.label.name, ctx.attr.bindings_output_name))
    executor_worker = ctx.actions.declare_file("%s/executorWorker" % ctx.label.name)
    decoder_attention_0 = ctx.actions.declare_file("%s/libdecoder_attention_0.so" % ctx.label.name)
    decoder_attention_1 = ctx.actions.declare_file("%s/libdecoder_attention_1.so" % ctx.label.name)
    pg_utils = ctx.actions.declare_file("%s/libpg_utils.so" % ctx.label.name)

    args = [
        libtensorrt_llm.path,
        nvinfer_plugin.path,
        bindings.path,
        executor_worker.path,
        decoder_attention_0.path,
        decoder_attention_1.path,
        pg_utils.path,
        ctx.attr.cuda_architectures,
        ctx.attr.tensorrt_root,
        ctx.attr.python,
        ctx.attr.conan,
        str(ctx.attr.jobs),
    ]

    ctx.actions.run_shell(
        inputs = ctx.files.srcs,
        outputs = [
            libtensorrt_llm,
            nvinfer_plugin,
            bindings,
            executor_worker,
            decoder_attention_0,
            decoder_attention_1,
            pg_utils,
        ],
        arguments = args,
        command = """
set -euo pipefail

libtensorrt_llm_out="$1"
nvinfer_plugin_out="$2"
bindings_out="$3"
executor_worker_out="$4"
decoder_attention_0_out="$5"
decoder_attention_1_out="$6"
pg_utils_out="$7"
default_cuda_architectures="$8"
default_tensorrt_root="$9"
default_python="${10}"
default_conan="${11}"
default_jobs="${12}"

mkdir -p "$(dirname "$libtensorrt_llm_out")"
mkdir -p "$(dirname "$nvinfer_plugin_out")"
mkdir -p "$(dirname "$bindings_out")"
mkdir -p "$(dirname "$executor_worker_out")"
mkdir -p "$(dirname "$decoder_attention_0_out")"
mkdir -p "$(dirname "$decoder_attention_1_out")"
mkdir -p "$(dirname "$pg_utils_out")"
rm -f \\
    "$libtensorrt_llm_out" \\
    "$nvinfer_plugin_out" \\
    "$bindings_out" \\
    "$executor_worker_out" \\
    "$decoder_attention_0_out" \\
    "$decoder_attention_1_out" \\
    "$pg_utils_out"

cpp_cmake_real="$(readlink -f cpp/CMakeLists.txt)"
workspace_root="$(cd "$(dirname "$(dirname "$cpp_cmake_real")")" && pwd -P)"

select_tool() {
    local env_value="$1"
    local default_relative="$2"
    local fallback="$3"
    local candidate
    local resolved

    if [[ -n "$env_value" ]]; then
        candidate="$env_value"
    elif [[ -x "$workspace_root/$default_relative" ]]; then
        candidate="$workspace_root/$default_relative"
    elif [[ -x "$default_relative" ]]; then
        candidate="$default_relative"
    else
        candidate="$fallback"
    fi

    if [[ "$candidate" == */* && "$candidate" != /* && -x "$workspace_root/$candidate" ]]; then
        candidate="$workspace_root/$candidate"
    fi

    if [[ "$candidate" == */* ]]; then
        if [[ ! -x "$candidate" ]]; then
            echo "Required executable is missing or not executable: $candidate" >&2
            return 1
        fi
        printf '%s\\n' "$candidate"
        return 0
    fi

    if ! resolved="$(command -v "$candidate")"; then
        echo "Required executable was not found on PATH: $candidate" >&2
        return 1
    fi
    printf '%s\\n' "$resolved"
}

python_bin="$(select_tool "${TRTLLM_BAZEL_PYTHON:-}" "$default_python" python3)"
conan_bin="$(select_tool "${TRTLLM_BAZEL_CONAN:-}" "$default_conan" conan)"
cuda_architectures="${TRTLLM_BAZEL_CMAKE_CUDA_ARCHITECTURES:-$default_cuda_architectures}"
tensorrt_root="${TensorRT_ROOT:-${TENSORRT_ROOT:-$default_tensorrt_root}}"
jobs="${TRTLLM_BAZEL_CMAKE_JOBS:-$default_jobs}"
mpi_c_compiler="${TRTLLM_BAZEL_MPI_C_COMPILER:-}"
mpi_cxx_compiler="${TRTLLM_BAZEL_MPI_CXX_COMPILER:-}"
mpi_home="${TRTLLM_BAZEL_MPI_HOME:-${MPI_HOME:-}}"

infer_mpi_home() {
    local compiler="$1"

    if [[ -z "$compiler" || "$compiler" != */bin/* ]]; then
        return 0
    fi

    (cd "$(dirname "$compiler")/.." && pwd -P)
}

prepend_env_path() {
    local env_name="$1"
    local entry="$2"
    local current_value="${!env_name:-}"

    if [[ ! -d "$entry" ]]; then
        return 0
    fi

    if [[ -n "$current_value" ]]; then
        export "$env_name=$entry:$current_value"
    else
        export "$env_name=$entry"
    fi
}

extract_include_flags() {
    local compiler="$1"
    local compile_flags
    local include_flags=()
    local next_is_include="0"
    local token

    if [[ -z "$compiler" ]]; then
        return 0
    fi

    if ! compile_flags="$("$compiler" --showme:compile 2>/dev/null)"; then
        return 0
    fi

    for token in $compile_flags; do
        if [[ "$next_is_include" == "1" ]]; then
            include_flags+=("-I$token")
            next_is_include="0"
        elif [[ "$token" == "-I" ]]; then
            next_is_include="1"
        elif [[ "$token" == -I* ]]; then
            include_flags+=("$token")
        fi
    done

    printf '%s\\n' "${include_flags[*]:-}"
}

if [[ "$jobs" == "0" ]]; then
    if command -v nproc >/dev/null; then
        jobs="$(nproc)"
    else
        jobs="1"
    fi
fi

if [[ -n "$mpi_home" ]]; then
    if [[ ! -d "$mpi_home" ]]; then
        echo "MPI home directory does not exist: $mpi_home" >&2
        exit 1
    fi
    mpi_home="$(cd "$mpi_home" && pwd -P)"
    prepend_env_path PATH "$mpi_home/bin"
fi

if [[ -z "$mpi_c_compiler" ]] && command -v mpicc >/dev/null; then
    mpi_c_compiler="$(command -v mpicc)"
fi
if [[ -z "$mpi_cxx_compiler" ]] && command -v mpicxx >/dev/null; then
    mpi_cxx_compiler="$(command -v mpicxx)"
fi

if [[ -z "$mpi_home" ]]; then
    mpi_home="$(infer_mpi_home "$mpi_c_compiler")"
    if [[ -z "$mpi_home" ]]; then
        mpi_home="$(infer_mpi_home "$mpi_cxx_compiler")"
    fi
    if [[ -n "$mpi_home" ]]; then
        prepend_env_path PATH "$mpi_home/bin"
    fi
fi

if [[ -n "$mpi_home" ]]; then
    export MPI_HOME="$mpi_home"
    if [[ -z "${OPAL_PREFIX:-}" && -d "$mpi_home/share/openmpi" ]]; then
        export OPAL_PREFIX="$mpi_home"
    fi
    prepend_env_path LD_LIBRARY_PATH "$mpi_home/lib64"
    prepend_env_path LD_LIBRARY_PATH "$mpi_home/lib"
    prepend_env_path PKG_CONFIG_PATH "$mpi_home/lib64/pkgconfig"
    prepend_env_path PKG_CONFIG_PATH "$mpi_home/lib/pkgconfig"
    prepend_env_path CMAKE_PREFIX_PATH "$mpi_home"
fi

cmake_cuda_flags="${TRTLLM_BAZEL_CMAKE_CUDA_FLAGS:-}"
mpi_cuda_include_flags="${TRTLLM_BAZEL_MPI_CUDA_INCLUDE_FLAGS:-}"
if [[ -z "$mpi_cuda_include_flags" ]]; then
    mpi_cuda_include_flags="$(extract_include_flags "$mpi_c_compiler")"
fi
if [[ -z "$mpi_cuda_include_flags" ]]; then
    mpi_cuda_include_flags="$(extract_include_flags "$mpi_cxx_compiler")"
fi
if [[ -z "$mpi_cuda_include_flags" && -n "$mpi_home" && -f "$mpi_home/include/mpi.h" ]]; then
    mpi_cuda_include_flags="-I$mpi_home/include"
fi
if [[ -n "$mpi_cuda_include_flags" ]]; then
    if [[ -n "$cmake_cuda_flags" ]]; then
        cmake_cuda_flags="$cmake_cuda_flags $mpi_cuda_include_flags"
    else
        cmake_cuda_flags="$mpi_cuda_include_flags"
    fi
fi

mpi_cmake_args=()
if [[ -n "$cmake_cuda_flags" ]]; then
    mpi_cmake_args+=("-DCMAKE_CUDA_FLAGS=$cmake_cuda_flags")
fi
if [[ -n "$mpi_c_compiler" ]]; then
    mpi_cmake_args+=("-DCMAKE_C_COMPILER=$mpi_c_compiler")
    mpi_cmake_args+=("-DMPI_C_COMPILER=$mpi_c_compiler")
fi
if [[ -n "$mpi_cxx_compiler" ]]; then
    mpi_cmake_args+=("-DCMAKE_CXX_COMPILER=$mpi_cxx_compiler")
    mpi_cmake_args+=("-DMPI_CXX_COMPILER=$mpi_cxx_compiler")
fi

work_root="$(mktemp -d "${TMPDIR:-/tmp}/trtllm-bazel-cmake.XXXXXX")"
cleanup() {
    local status="$?"
    if [[ "$status" -eq 0 ]]; then
        rm -rf "$work_root"
    else
        echo "Preserving failed TensorRT-LLM CMake build directory: $work_root" >&2
    fi
    exit "$status"
}
trap cleanup EXIT

source_root="$work_root/src"
cmake_build_dir="$work_root/cmake-build"
conan_output_dir="$cmake_build_dir/conan"
mkdir -p "$source_root" "$cmake_build_dir"

tar -C "$workspace_root" \\
    --exclude='cpp/build' \\
    --exclude='cpp/build_*' \\
    --exclude='cpp/cmake-build-*' \\
    --exclude='cpp/bazel-*' \\
    --exclude='cpp/.cache' \\
    --exclude='cpp/__pycache__' \\
    -cf - cpp | tar -C "$source_root" -xf -

mkdir -p "$source_root/tensorrt_llm"
cp "$workspace_root/tensorrt_llm/version.py" "$source_root/tensorrt_llm/version.py"
ln -s "$workspace_root/3rdparty" "$source_root/3rdparty"

export CONAN_HOME="$work_root/conan-home"
export PYTHONDONTWRITEBYTECODE=1

echo "Using Python executable: $python_bin" >&2
echo "Using Conan executable: $conan_bin" >&2
echo "Using CONAN_HOME: $CONAN_HOME" >&2
echo "Using CMAKE_CUDA_ARCHITECTURES: $cuda_architectures" >&2
echo "Using TensorRT_ROOT: $tensorrt_root" >&2
echo "Using extra CMAKE_CUDA_FLAGS: ${cmake_cuda_flags:-<CMake default>}" >&2
echo "Using MPI_HOME: ${MPI_HOME:-<unset>}" >&2
echo "Using OPAL_PREFIX: ${OPAL_PREFIX:-<unset>}" >&2
echo "Using CMAKE_C_COMPILER: ${mpi_c_compiler:-<CMake default>}" >&2
echo "Using CMAKE_CXX_COMPILER: ${mpi_cxx_compiler:-<CMake default>}" >&2
echo "Using MPI_C_COMPILER: ${mpi_c_compiler:-<CMake default>}" >&2
echo "Using MPI_CXX_COMPILER: ${mpi_cxx_compiler:-<CMake default>}" >&2

"$conan_bin" profile detect -f

set +e
"$conan_bin" install \\
    --build=missing \\
    --no-remote \\
    --output-folder="$conan_output_dir" \\
    -s "build_type=Release" \\
    "$source_root/cpp"
conan_status="$?"
set -e
if [[ "$conan_status" -ne 0 ]]; then
    echo "Conan install failed while running with --no-remote." >&2
    echo "This target intentionally does not fetch remote packages from Bazel actions." >&2
    echo "Populate the local Conan cache for the TensorRT-LLM CMake dependencies, then rebuild." >&2
    exit "$conan_status"
fi

cmake \\
    -G Ninja \\
    -S "$source_root/cpp" \\
    -B "$cmake_build_dir" \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DBUILD_PYT=ON \\
    -DBUILD_TESTS=OFF \\
    -DBUILD_BENCHMARKS=OFF \\
    -DBUILD_DEEP_EP=OFF \\
    -DBUILD_DEEP_GEMM=OFF \\
    -DBUILD_FLASH_MLA=OFF \\
    -DBUILD_MICRO_BENCHMARKS=OFF \\
    -DNVTX_DISABLE=ON \\
    '-DBUILD_WHEEL_TARGETS=tensorrt_llm;nvinfer_plugin_tensorrt_llm;bindings;executorWorker' \\
    "-DPython_EXECUTABLE=$python_bin" \\
    "-DPython3_EXECUTABLE=$python_bin" \\
    "-DCMAKE_CUDA_ARCHITECTURES=$cuda_architectures" \\
    "-DCMAKE_TOOLCHAIN_FILE=$conan_output_dir/conan_toolchain.cmake" \\
    "-DTensorRT_ROOT=$tensorrt_root" \\
    "${mpi_cmake_args[@]}"

cmake \\
    --build "$cmake_build_dir" \\
    --config Release \\
    --parallel "$jobs" \\
    --target build_wheel_targets \\
    -- \\
    -d keepdepfile

copy_exact_output() {
    local src="$1"
    local dst="$2"
    local description="$3"

    if [[ ! -f "$src" ]]; then
        echo "Expected CMake output for $description was not produced: $src" >&2
        find "$cmake_build_dir/tensorrt_llm" -maxdepth 4 -type f -name '*.so' -print >&2 || true
        return 1
    fi

    cp -f "$src" "$dst"
}

copy_exact_output \\
    "$cmake_build_dir/tensorrt_llm/libtensorrt_llm.so" \\
    "$libtensorrt_llm_out" \\
    "libtensorrt_llm"
copy_exact_output \\
    "$cmake_build_dir/tensorrt_llm/plugins/libnvinfer_plugin_tensorrt_llm.so" \\
    "$nvinfer_plugin_out" \\
    "nvinfer_plugin_tensorrt_llm"
copy_exact_output \\
    "$cmake_build_dir/tensorrt_llm/executor_worker/executorWorker" \\
    "$executor_worker_out" \\
    "executorWorker"
copy_exact_output \\
    "$cmake_build_dir/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/libdecoder_attention_0.so" \\
    "$decoder_attention_0_out" \\
    "libdecoder_attention_0"
copy_exact_output \\
    "$cmake_build_dir/tensorrt_llm/kernels/decoderMaskedMultiheadAttention/libdecoder_attention_1.so" \\
    "$decoder_attention_1_out" \\
    "libdecoder_attention_1"
copy_exact_output \\
    "$cmake_build_dir/tensorrt_llm/runtime/utils/libpg_utils.so" \\
    "$pg_utils_out" \\
    "libpg_utils"
chmod 755 "$executor_worker_out"

shopt -s nullglob
bindings_matches=("$cmake_build_dir"/tensorrt_llm/nanobind/bindings*.so)
shopt -u nullglob
if [[ "${#bindings_matches[@]}" -ne 1 ]]; then
    echo "Expected exactly one CMake bindings output under $cmake_build_dir/tensorrt_llm/nanobind." >&2
    printf 'Found bindings candidates: %s\\n' "${bindings_matches[@]:-<none>}" >&2
    find "$cmake_build_dir/tensorrt_llm/nanobind" -maxdepth 2 -type f -name '*.so' -print >&2 || true
    exit 1
fi
cp -f "${bindings_matches[0]}" "$bindings_out"
""",
        execution_requirements = _NON_HERMETIC_CMAKE_EXECUTION_REQUIREMENTS,
        mnemonic = "TensorRTLLMCMakeArtifacts",
        progress_message = "Building TensorRT-LLM CMake native artifacts",
        use_default_shell_env = True,
    )

    return [
        DefaultInfo(files = depset([
            libtensorrt_llm,
            nvinfer_plugin,
            bindings,
            executor_worker,
            decoder_attention_0,
            decoder_attention_1,
            pg_utils,
        ])),
        OutputGroupInfo(
            decoder_attention_0 = depset([decoder_attention_0]),
            decoder_attention_1 = depset([decoder_attention_1]),
            libtensorrt_llm = depset([libtensorrt_llm]),
            nvinfer_plugin_tensorrt_llm = depset([nvinfer_plugin]),
            pg_utils = depset([pg_utils]),
            tensorrt_llm_bindings = depset([bindings]),
            executor_worker = depset([executor_worker]),
            trtllm_runtime_libs = depset([
                libtensorrt_llm,
                nvinfer_plugin,
                decoder_attention_0,
                decoder_attention_1,
                pg_utils,
            ]),
        ),
    ]


trtllm_cmake_artifacts = rule(
    implementation = _trtllm_cmake_artifacts_impl,
    attrs = {
        "bindings_output_name": attr.string(
            default = "bindings.cpython-312-x86_64-linux-gnu.so",
            doc = "Stable declared filename for the copied Python 3.12 nanobind extension.",
        ),
        "conan": attr.string(
            default = ".venv-3.12/bin/conan",
            doc = "Default Conan executable, used when TRTLLM_BAZEL_CONAN is unset and the path exists.",
        ),
        "cuda_architectures": attr.string(
            default = "90-real",
            doc = "Default CMAKE_CUDA_ARCHITECTURES; override with TRTLLM_BAZEL_CMAKE_CUDA_ARCHITECTURES.",
        ),
        "jobs": attr.int(
            default = 0,
            doc = "Parallel CMake build jobs. Zero means nproc; override with TRTLLM_BAZEL_CMAKE_JOBS.",
        ),
        "python": attr.string(
            default = ".venv-3.12/bin/python3",
            doc = "Default Python executable, used when TRTLLM_BAZEL_PYTHON is unset and the path exists.",
        ),
        "srcs": attr.label_list(
            allow_files = True,
            doc = "Source files that should invalidate the local CMake artifact action.",
        ),
        "tensorrt_root": attr.string(
            default = "/usr/local/tensorrt",
            doc = "Default TensorRT_ROOT; TensorRT_ROOT or TENSORRT_ROOT from the shell environment wins.",
        ),
    },
    doc = "Builds TensorRT-LLM CMake native artifacts once and exposes them through output groups.",
)
