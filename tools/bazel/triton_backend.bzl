# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local-only Bazel wrappers for Triton TensorRT-LLM backend artifacts."""

_NON_HERMETIC_CMAKE_EXECUTION_REQUIREMENTS = {
    "local": "1",
    "no-cache": "1",
    "no-remote": "1",
    "no-remote-cache": "1",
    "no-remote-exec": "1",
    "no-sandbox": "1",
}


def _triton_tensorrt_llm_backend_impl(ctx):
    backend_lib = ctx.actions.declare_file("%s/libtriton_tensorrtllm.so" % ctx.label.name)
    executor_worker = ctx.actions.declare_file("%s/trtllmExecutorWorker" % ctx.label.name)
    libtensorrt_llm = ctx.actions.declare_file("%s/libtensorrt_llm.so" % ctx.label.name)
    nvinfer_plugin_tensorrt_llm = ctx.actions.declare_file(
        "%s/libnvinfer_plugin_tensorrt_llm.so" % ctx.label.name,
    )
    decoder_attention_0 = ctx.actions.declare_file("%s/libdecoder_attention_0.so" % ctx.label.name)
    decoder_attention_1 = ctx.actions.declare_file("%s/libdecoder_attention_1.so" % ctx.label.name)
    pg_utils = ctx.actions.declare_file("%s/libpg_utils.so" % ctx.label.name)

    args = [
        backend_lib.path,
        executor_worker.path,
        libtensorrt_llm.path,
        nvinfer_plugin_tensorrt_llm.path,
        decoder_attention_0.path,
        decoder_attention_1.path,
        pg_utils.path,
        ctx.file.libtensorrt_llm.path,
        ctx.file.nvinfer_plugin_tensorrt_llm.path,
        ctx.file.executor_worker.path,
        ctx.file.decoder_attention_0.path,
        ctx.file.decoder_attention_1.path,
        ctx.file.pg_utils.path,
        ctx.attr.triton_repo_tag,
        ctx.attr.tensorrt_root,
        str(ctx.attr.jobs),
        ctx.attr.cmake,
        ctx.attr.ninja,
    ]

    inputs = depset(
        direct = [
            ctx.file.libtensorrt_llm,
            ctx.file.nvinfer_plugin_tensorrt_llm,
            ctx.file.executor_worker,
            ctx.file.decoder_attention_0,
            ctx.file.decoder_attention_1,
            ctx.file.pg_utils,
        ],
        transitive = [
            depset(ctx.files.srcs),
            depset(ctx.files.trtllm_srcs),
        ],
    )

    ctx.actions.run_shell(
        inputs = inputs,
        outputs = [
            backend_lib,
            executor_worker,
            libtensorrt_llm,
            nvinfer_plugin_tensorrt_llm,
            decoder_attention_0,
            decoder_attention_1,
            pg_utils,
        ],
        arguments = args,
        command = """
set -euo pipefail

backend_lib_out="$1"
executor_worker_out="$2"
libtensorrt_llm_out="$3"
nvinfer_plugin_out="$4"
decoder_attention_0_out="$5"
decoder_attention_1_out="$6"
pg_utils_out="$7"
libtensorrt_llm_in="$(readlink -f "$8")"
nvinfer_plugin_in="$(readlink -f "$9")"
executor_worker_in="$(readlink -f "${10}")"
decoder_attention_0_in="$(readlink -f "${11}")"
decoder_attention_1_in="$(readlink -f "${12}")"
pg_utils_in="$(readlink -f "${13}")"
default_triton_repo_tag="${14}"
default_tensorrt_root="${15}"
default_jobs="${16}"
default_cmake="${17}"
default_ninja="${18}"

mkdir -p "$(dirname "$backend_lib_out")"
mkdir -p "$(dirname "$executor_worker_out")"
mkdir -p "$(dirname "$libtensorrt_llm_out")"
mkdir -p "$(dirname "$nvinfer_plugin_out")"
mkdir -p "$(dirname "$decoder_attention_0_out")"
mkdir -p "$(dirname "$decoder_attention_1_out")"
mkdir -p "$(dirname "$pg_utils_out")"
rm -f \\
    "$backend_lib_out" \\
    "$executor_worker_out" \\
    "$libtensorrt_llm_out" \\
    "$nvinfer_plugin_out" \\
    "$decoder_attention_0_out" \\
    "$decoder_attention_1_out" \\
    "$pg_utils_out"

triton_cmake_real="$(readlink -f triton_backend/inflight_batcher_llm/CMakeLists.txt)"
workspace_root="$(cd "$(dirname "$triton_cmake_real")/../.." && pwd -P)"

select_tool() {
    local env_value="$1"
    local default_tool="$2"
    local fallback="$3"
    local candidate
    local resolved

    if [[ -n "$env_value" ]]; then
        candidate="$env_value"
    elif [[ -n "$default_tool" ]]; then
        candidate="$default_tool"
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

resolve_triton_repo_tag() {
    local attr_value="$1"
    local dockerfile="$workspace_root/docker/Dockerfile.multi"
    local triton_base_tag

    if [[ -n "${TRTLLM_BAZEL_TRITON_REPO_TAG:-}" ]]; then
        printf '%s\\n' "$TRTLLM_BAZEL_TRITON_REPO_TAG"
        return 0
    fi

    if [[ -n "$attr_value" && "$attr_value" != "auto" ]]; then
        printf '%s\\n' "$attr_value"
        return 0
    fi

    triton_base_tag="$(sed -n 's/^ARG TRITON_BASE_TAG=//p' "$dockerfile" 2>/dev/null | tail -n 1)"
    if [[ -n "$triton_base_tag" ]]; then
        printf 'r%s\\n' "${triton_base_tag%%-py3*}"
    else
        printf 'main\\n'
    fi
}

cmake_bin="$(select_tool "${TRTLLM_BAZEL_CMAKE:-}" "$default_cmake" cmake)"
ninja_bin="$(select_tool "${TRTLLM_BAZEL_NINJA:-}" "$default_ninja" ninja)"
tensorrt_root="${TensorRT_ROOT:-${TENSORRT_ROOT:-$default_tensorrt_root}}"
jobs="${TRTLLM_BAZEL_TRITON_CMAKE_JOBS:-${TRTLLM_BAZEL_CMAKE_JOBS:-$default_jobs}}"
triton_repo_tag="$(resolve_triton_repo_tag "$default_triton_repo_tag")"
triton_common_repo_tag="${TRTLLM_BAZEL_TRITON_COMMON_REPO_TAG:-$triton_repo_tag}"
triton_core_repo_tag="${TRTLLM_BAZEL_TRITON_CORE_REPO_TAG:-$triton_repo_tag}"
triton_backend_repo_tag="${TRTLLM_BAZEL_TRITON_BACKEND_REPO_TAG:-$triton_repo_tag}"
mpi_c_compiler="${TRTLLM_BAZEL_MPI_C_COMPILER:-}"
mpi_cxx_compiler="${TRTLLM_BAZEL_MPI_CXX_COMPILER:-}"
mpi_home="${TRTLLM_BAZEL_MPI_HOME:-${MPI_HOME:-}}"

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

cmake_cuda_flags="${TRTLLM_BAZEL_TRITON_CMAKE_CUDA_FLAGS:-${TRTLLM_BAZEL_CMAKE_CUDA_FLAGS:-}}"
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

work_root="$(mktemp -d "${TMPDIR:-/tmp}/trtllm-bazel-triton.XXXXXX")"
cleanup() {
    local status="$?"
    if [[ "$status" -eq 0 ]]; then
        rm -rf "$work_root"
    else
        echo "Preserving failed Triton TensorRT-LLM backend build directory: $work_root" >&2
    fi
    exit "$status"
}
trap cleanup EXIT

source_root="$work_root/src"
cmake_build_dir="$work_root/cmake-build"
mkdir -p "$source_root" "$cmake_build_dir"

tar -C "$workspace_root" \\
    --exclude='triton_backend/inflight_batcher_llm/build' \\
    --exclude='triton_backend/inflight_batcher_llm/build_*' \\
    --exclude='triton_backend/inflight_batcher_llm/cmake-build-*' \\
    --exclude='triton_backend/inflight_batcher_llm/bazel-*' \\
    --exclude='triton_backend/inflight_batcher_llm/__pycache__' \\
    -cf - triton_backend/inflight_batcher_llm | tar -C "$source_root" -xf -

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

trtllm_build_dir="$source_root/cpp/build/tensorrt_llm"
trtllm_decoder_attention_dir="$trtllm_build_dir/kernels/decoderMaskedMultiheadAttention"
trtllm_runtime_utils_dir="$trtllm_build_dir/runtime/utils"
mkdir -p \\
    "$trtllm_build_dir/plugins" \\
    "$trtllm_build_dir/executor_worker" \\
    "$trtllm_decoder_attention_dir" \\
    "$trtllm_runtime_utils_dir"
cp -f "$libtensorrt_llm_in" "$trtllm_build_dir/libtensorrt_llm.so"
cp -f "$nvinfer_plugin_in" "$trtllm_build_dir/plugins/libnvinfer_plugin_tensorrt_llm.so"
install -m 755 "$executor_worker_in" "$trtllm_build_dir/executor_worker/executorWorker"
cp -f "$decoder_attention_0_in" "$trtllm_decoder_attention_dir/libdecoder_attention_0.so"
cp -f "$decoder_attention_1_in" "$trtllm_decoder_attention_dir/libdecoder_attention_1.so"
cp -f "$pg_utils_in" "$trtllm_runtime_utils_dir/libpg_utils.so"

echo "Using CMake executable: $cmake_bin" >&2
echo "Using Ninja executable: $ninja_bin" >&2
echo "Using TRTLLM_DIR: $source_root" >&2
echo "Using TensorRT_ROOT: $tensorrt_root" >&2
echo "Using Triton common repo tag: $triton_common_repo_tag" >&2
echo "Using Triton core repo tag: $triton_core_repo_tag" >&2
echo "Using Triton backend repo tag: $triton_backend_repo_tag" >&2
echo "Using CMake build jobs: $jobs" >&2
echo "Using extra CMAKE_CUDA_FLAGS: ${cmake_cuda_flags:-<CMake default>}" >&2
echo "Using MPI_HOME: ${MPI_HOME:-<unset>}" >&2
echo "Using OPAL_PREFIX: ${OPAL_PREFIX:-<unset>}" >&2
echo "Using CMAKE_C_COMPILER: ${mpi_c_compiler:-<CMake default>}" >&2
echo "Using CMAKE_CXX_COMPILER: ${mpi_cxx_compiler:-<CMake default>}" >&2

"$cmake_bin" \\
    -G Ninja \\
    -S "$source_root/triton_backend/inflight_batcher_llm" \\
    -B "$cmake_build_dir" \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \\
    '-DCMAKE_INSTALL_RPATH=$ORIGIN' \\
    "-DCMAKE_MAKE_PROGRAM=$ninja_bin" \\
    "-DCMAKE_INSTALL_PREFIX=$cmake_build_dir/install" \\
    "-DTRTLLM_DIR=$source_root" \\
    -DUSE_CXX11_ABI=ON \\
    -DBUILD_TESTS=OFF \\
    "-DTRITON_COMMON_REPO_TAG=$triton_common_repo_tag" \\
    "-DTRITON_CORE_REPO_TAG=$triton_core_repo_tag" \\
    "-DTRITON_BACKEND_REPO_TAG=$triton_backend_repo_tag" \\
    "-DTensorRT_ROOT=$tensorrt_root" \\
    "${mpi_cmake_args[@]}"

"$cmake_bin" \\
    --build "$cmake_build_dir" \\
    --config Release \\
    --parallel "$jobs" \\
    --target triton-tensorrt-llm-backend \\
    -- \\
    -d keepdepfile

"$cmake_bin" \\
    --build "$cmake_build_dir" \\
    --config Release \\
    --parallel "$jobs" \\
    --target install \\
    -- \\
    -d keepdepfile

copy_exact_output() {
    local src="$1"
    local dst="$2"
    local description="$3"

    if [[ ! -f "$src" ]]; then
        echo "Expected Triton backend CMake output for $description was not produced: $src" >&2
        find "$cmake_build_dir" -maxdepth 4 -type f -name 'libtriton_tensorrtllm.so' -o -name 'trtllmExecutorWorker' -print >&2 || true
        return 1
    fi

    cp -f "$src" "$dst"
}

copy_exact_output \\
    "$cmake_build_dir/libtriton_tensorrtllm.so" \\
    "$backend_lib_out" \\
    "libtriton_tensorrtllm"
copy_exact_output \\
    "$cmake_build_dir/trtllmExecutorWorker" \\
    "$executor_worker_out" \\
    "trtllmExecutorWorker"
cp -pf "$libtensorrt_llm_in" "$libtensorrt_llm_out"
cp -pf "$nvinfer_plugin_in" "$nvinfer_plugin_out"
cp -pf "$decoder_attention_0_in" "$decoder_attention_0_out"
cp -pf "$decoder_attention_1_in" "$decoder_attention_1_out"
cp -pf "$pg_utils_in" "$pg_utils_out"
chmod 755 "$executor_worker_out"
""",
        execution_requirements = _NON_HERMETIC_CMAKE_EXECUTION_REQUIREMENTS,
        mnemonic = "TritonTensorRTLLMBackend",
        progress_message = "Building Triton TensorRT-LLM backend artifacts",
        use_default_shell_env = True,
    )

    return [
        DefaultInfo(files = depset([
            backend_lib,
            executor_worker,
            libtensorrt_llm,
            nvinfer_plugin_tensorrt_llm,
            decoder_attention_0,
            decoder_attention_1,
            pg_utils,
        ])),
        OutputGroupInfo(
            decoder_attention_0 = depset([decoder_attention_0]),
            decoder_attention_1 = depset([decoder_attention_1]),
            libtriton_tensorrtllm = depset([backend_lib]),
            libtensorrt_llm = depset([libtensorrt_llm]),
            nvinfer_plugin_tensorrt_llm = depset([nvinfer_plugin_tensorrt_llm]),
            pg_utils = depset([pg_utils]),
            trtllm_runtime_libs = depset([
                libtensorrt_llm,
                nvinfer_plugin_tensorrt_llm,
                decoder_attention_0,
                decoder_attention_1,
                pg_utils,
            ]),
            trtllm_executor_worker = depset([executor_worker]),
        ),
    ]


triton_tensorrt_llm_backend = rule(
    implementation = _triton_tensorrt_llm_backend_impl,
    attrs = {
        "cmake": attr.string(
            default = "cmake",
            doc = "Default CMake executable; override with TRTLLM_BAZEL_CMAKE.",
        ),
        "decoder_attention_0": attr.label(
            allow_single_file = True,
            mandatory = True,
            doc = "TensorRT-LLM decoder attention runtime shared library to stage beside libtensorrt_llm.",
        ),
        "decoder_attention_1": attr.label(
            allow_single_file = True,
            mandatory = True,
            doc = "TensorRT-LLM decoder attention runtime shared library to stage beside libtensorrt_llm.",
        ),
        "executor_worker": attr.label(
            allow_single_file = True,
            mandatory = True,
            doc = "TensorRT-LLM executorWorker executable to stage for Triton CMake.",
        ),
        "jobs": attr.int(
            default = 0,
            doc = "Parallel CMake build jobs. Zero means nproc; override with TRTLLM_BAZEL_TRITON_CMAKE_JOBS.",
        ),
        "libtensorrt_llm": attr.label(
            allow_single_file = True,
            mandatory = True,
            doc = "TensorRT-LLM shared library to stage for Triton CMake.",
        ),
        "ninja": attr.string(
            default = "ninja",
            doc = "Default Ninja executable; override with TRTLLM_BAZEL_NINJA.",
        ),
        "nvinfer_plugin_tensorrt_llm": attr.label(
            allow_single_file = True,
            mandatory = True,
            doc = "TensorRT-LLM TensorRT plugin shared library to stage for Triton CMake.",
        ),
        "pg_utils": attr.label(
            allow_single_file = True,
            mandatory = True,
            doc = "TensorRT-LLM process group utility runtime shared library to stage beside libtensorrt_llm.",
        ),
        "srcs": attr.label_list(
            allow_files = True,
            doc = "Triton backend source files that should invalidate the local CMake action.",
        ),
        "tensorrt_root": attr.string(
            default = "/usr/local/tensorrt",
            doc = "Default TensorRT_ROOT; TensorRT_ROOT or TENSORRT_ROOT from the shell environment wins.",
        ),
        "triton_repo_tag": attr.string(
            default = "auto",
            doc = "Triton repo tag for common/core/backend. 'auto' mirrors jenkins/scripts/get_triton_tag.sh.",
        ),
        "trtllm_srcs": attr.label_list(
            allow_files = True,
            doc = "TensorRT-LLM C++ source files that should invalidate the local CMake action.",
        ),
    },
    doc = "Builds the Triton TensorRT-LLM backend through its local CMake/Ninja project.",
)
