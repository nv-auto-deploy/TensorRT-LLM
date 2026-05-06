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
        script.short_path,
        wheel_dir.path,
    ] + list(ctx.attr.args)

    ctx.actions.run_shell(
        inputs = depset([source_manifest], transitive = [source_files]),
        outputs = [wheel_dir],
        arguments = args,
        command = """
set -euo pipefail

source_manifest="$1"
script_relpath="$2"
dist_dir="$3"
shift 3

case "$dist_dir" in
    /*) ;;
    *) dist_dir="$PWD/$dist_dir" ;;
esac

rm -rf "$dist_dir"
mkdir -p "$dist_dir"

tmp_base="${TMPDIR:-/tmp}"
mkdir -p "$tmp_base"
work_dir="$(mktemp -d "$tmp_base/trtllm-bazel-wheel.XXXXXX")"
trap 'rm -rf "$work_dir"' EXIT

source_dir="$work_dir/source"
build_dir="$work_dir/build"
mkdir -p "$source_dir" "$build_dir"
export TRTLLM_DG_CACHE_DIR="$work_dir/dg_cache"
mkdir -p "$TRTLLM_DG_CACHE_DIR"

python3 - "$source_manifest" "$source_dir" <<'PY'
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
python3 "$script" --build_dir "$build_dir" --dist_dir "$dist_dir" "$@"
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
