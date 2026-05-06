# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local-only Bazel wrappers for TensorRT-LLM wheel packaging."""

_NON_HERMETIC_WHEEL_EXECUTION_REQUIREMENTS = {
    "local": "1",
    "no-cache": "1",
    "no-remote": "1",
    "no-remote-cache": "1",
    "no-remote-exec": "1",
    "no-sandbox": "1",
}


def _trtllm_wheel_impl(ctx):
    wheel_dir = ctx.actions.declare_directory(ctx.label.name)
    script = ctx.file.script
    source_files = depset(ctx.files.srcs + [script])
    source_manifest = ctx.actions.declare_file("%s.sources" % ctx.label.name)
    ctx.actions.write(
        output = source_manifest,
        content = "\n".join([
            "%s\t%s" % (src.path, src.short_path)
            for src in source_files.to_list()
        ]) + "\n",
    )

    args = [
        source_manifest.path,
        script.path,
        script.short_path,
        wheel_dir.path,
        ctx.attr.python,
        ctx.attr.cuda_architectures,
    ] + list(ctx.attr.args)

    ctx.actions.run_shell(
        inputs = depset([source_manifest], transitive = [source_files]),
        outputs = [wheel_dir],
        arguments = args,
        command = """
set -euo pipefail

source_manifest="$1"
script_input_path="$2"
script_relpath="$3"
dist_dir="$4"
default_python="$5"
default_cuda_architectures="$6"
shift 6

case "$dist_dir" in
    /*) ;;
    *) dist_dir="$PWD/$dist_dir" ;;
esac

script_real="$(readlink -f "$script_input_path")"
workspace_root="$(cd "$(dirname "$(dirname "$script_real")")" && pwd -P)"

select_python() {
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
            echo "Required Python executable is missing or not executable: $candidate" >&2
            return 1
        fi
        printf '%s\\n' "$candidate"
        return 0
    fi

    if ! resolved="$(command -v "$candidate")"; then
        echo "Required Python executable was not found on PATH: $candidate" >&2
        return 1
    fi
    printf '%s\\n' "$resolved"
}

python_bin="$(select_python "${TRTLLM_BAZEL_PYTHON:-}" "$default_python" python3)"
echo "Using Python executable: $python_bin" >&2

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

mpi_c_compiler="${TRTLLM_BAZEL_MPI_C_COMPILER:-}"
mpi_cxx_compiler="${TRTLLM_BAZEL_MPI_CXX_COMPILER:-}"
mpi_home="${TRTLLM_BAZEL_MPI_HOME:-${MPI_HOME:-}}"

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

wheel_extra_cmake_args=()
wheel_extra_args=()
user_controls_benchmarks="0"
user_controls_cuda_architectures="0"
for user_arg in "$@"; do
    if [[ "$user_arg" == "--benchmarks" || "$user_arg" == *"BUILD_BENCHMARKS="* ]]; then
        user_controls_benchmarks="1"
    fi
    if [[ "$user_arg" == "--cuda_architectures" || "$user_arg" == "--cuda_architectures="* \
        || "$user_arg" == "-a" || "$user_arg" == -a* ]]; then
        user_controls_cuda_architectures="1"
    fi
done
if [[ "$user_controls_benchmarks" != "1" ]]; then
    wheel_extra_cmake_args+=(--extra-cmake-vars "BUILD_BENCHMARKS=OFF")
fi
if [[ "$user_controls_cuda_architectures" != "1" ]]; then
    cuda_architectures="${TRTLLM_BAZEL_WHEEL_CUDA_ARCHITECTURES:-}"
    if [[ -z "$cuda_architectures" ]]; then
        cuda_architectures="${TRTLLM_BAZEL_CMAKE_CUDA_ARCHITECTURES:-}"
    fi
    if [[ -z "$cuda_architectures" ]]; then
        cuda_architectures="$default_cuda_architectures"
    fi
    if [[ -n "$cuda_architectures" ]]; then
        wheel_extra_args+=(--cuda_architectures "$cuda_architectures")
    fi
else
    cuda_architectures="<user-provided>"
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
if [[ -n "$cmake_cuda_flags" ]]; then
    wheel_extra_cmake_args+=(--extra-cmake-vars "CMAKE_CUDA_FLAGS=$cmake_cuda_flags")
fi
if [[ -n "$mpi_c_compiler" ]]; then
    wheel_extra_cmake_args+=(--extra-cmake-vars "CMAKE_C_COMPILER=$mpi_c_compiler")
    wheel_extra_cmake_args+=(--extra-cmake-vars "MPI_C_COMPILER=$mpi_c_compiler")
fi
if [[ -n "$mpi_cxx_compiler" ]]; then
    wheel_extra_cmake_args+=(--extra-cmake-vars "CMAKE_CXX_COMPILER=$mpi_cxx_compiler")
    wheel_extra_cmake_args+=(--extra-cmake-vars "MPI_CXX_COMPILER=$mpi_cxx_compiler")
fi

echo "Using MPI_HOME: ${MPI_HOME:-<unset>}" >&2
echo "Using OPAL_PREFIX: ${OPAL_PREFIX:-<unset>}" >&2
echo "Using CMAKE_C_COMPILER: ${mpi_c_compiler:-<CMake default>}" >&2
echo "Using CMAKE_CXX_COMPILER: ${mpi_cxx_compiler:-<CMake default>}" >&2
echo "Using extra CMAKE_CUDA_FLAGS: ${cmake_cuda_flags:-<CMake default>}" >&2
echo "Using wheel CUDA architectures: ${cuda_architectures:-<build_wheel.py default>}" >&2

rm -rf "$dist_dir"
mkdir -p "$dist_dir"

tmp_base="${TMPDIR:-/tmp}"
mkdir -p "$tmp_base"
work_dir="$(mktemp -d "$tmp_base/trtllm-bazel-wheel.XXXXXX")"
cleanup_work_dir() {
    local status="$?"

    trap - EXIT
    if [[ "$status" -eq 0 ]]; then
        rm -rf "$work_dir"
    else
        echo "Preserving TensorRT-LLM wheel work directory after failure: $work_dir" >&2
    fi
    exit "$status"
}
trap cleanup_work_dir EXIT

source_dir="$work_dir/source"
build_dir="$work_dir/build"
mkdir -p "$source_dir" "$build_dir"
export TRTLLM_DG_CACHE_DIR="$work_dir/dg_cache"
mkdir -p "$TRTLLM_DG_CACHE_DIR"

"$python_bin" - "$source_manifest" "$source_dir" <<'PY'
import os
import shutil
import stat
import sys

manifest_path = sys.argv[1]
source_dir = sys.argv[2]

with open(manifest_path, encoding="utf-8") as manifest:
    for raw_line in manifest:
        line = raw_line.rstrip("\\n")
        if not line:
            continue
        exec_path, relpath = line.split("\\t", 1)
        if os.path.isabs(relpath) or relpath == ".." or relpath.startswith("../"):
            raise RuntimeError(f"Refusing to stage source outside root: {relpath}")

        dst = os.path.join(source_dir, relpath)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(exec_path):
            shutil.copytree(exec_path, dst, symlinks=False, dirs_exist_ok=True)
        else:
            shutil.copy2(exec_path, dst, follow_symlinks=True)
            mode = os.stat(dst).st_mode
            os.chmod(dst, mode | stat.S_IWUSR)
PY

script="$source_dir/$script_relpath"
"$python_bin" "$script" --build_dir "$build_dir" --dist_dir "$dist_dir" \
    "${wheel_extra_args[@]}" "${wheel_extra_cmake_args[@]}" "$@"
""",
        execution_requirements = _NON_HERMETIC_WHEEL_EXECUTION_REQUIREMENTS,
        mnemonic = "TensorRTLLMWheel",
        progress_message = "Building TensorRT-LLM wheel into %{output}",
        use_default_shell_env = True,
    )

    return [DefaultInfo(files = depset([wheel_dir]))]


trtllm_wheel = rule(
    implementation = _trtllm_wheel_impl,
    attrs = {
        "args": attr.string_list(
            doc = "Extra arguments forwarded to scripts/build_wheel.py after --dist_dir.",
        ),
        "script": attr.label(
            allow_single_file = True,
            mandatory = True,
            doc = "Wheel build script to execute.",
        ),
        "python": attr.string(
            default = ".venv-3.12/bin/python3",
            doc = (
                "Repo-relative Python executable used to run the wheel action "
                + "when TRTLLM_BAZEL_PYTHON is unset."
            ),
        ),
        "cuda_architectures": attr.string(
            default = "90-real",
            doc = (
                "Default CUDA architectures forwarded to the wheel builder; "
                + "override with TRTLLM_BAZEL_WHEEL_CUDA_ARCHITECTURES or "
                + "TRTLLM_BAZEL_CMAKE_CUDA_ARCHITECTURES."
            ),
        ),
        "srcs": attr.label_list(
            allow_files = True,
            doc = (
                "Source files staged into a writable checkout-shaped tree "
                + "before running the wheel script."
            ),
        ),
    },
    doc = (
        "Stages declared TensorRT-LLM sources, runs the wheel builder, and "
        + "exposes its dist directory."
    ),
)
