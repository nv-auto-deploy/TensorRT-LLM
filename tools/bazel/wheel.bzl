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
    args = [script.path, wheel_dir.path] + list(ctx.attr.args)

    ctx.actions.run_shell(
        inputs = [script],
        outputs = [wheel_dir],
        arguments = args,
        command = """
set -euo pipefail

script="$1"
dist_dir="$2"
shift 2

rm -rf "$dist_dir"
mkdir -p "$dist_dir"
python3 "$script" --dist_dir "$dist_dir" "$@"
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
    },
    doc = "Runs the existing TensorRT-LLM wheel builder and exposes its dist directory.",
)
