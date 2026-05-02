# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bazel helpers for TensorRT-LLM CI pytest selector targets."""

_PYTEST_SELECTOR_RUNNER = "//tools/bazel:pytest_selector_runner"
_PYTEST_SELECTOR_TEST_WRAPPER = "//ci/bazel:pytest_selector_test.sh"
_PYTEST_ENV_INHERIT = [
    "CUDA_VISIBLE_DEVICES",
    "LLM_MODELS_ROOT",
    "NVIDIA_VISIBLE_DEVICES",
    "TRTLLM_BAZEL_REPO_ROOT",
]


def _tags_with_manual(tags, manual):
    normalized_tags = list(tags or [])
    if manual and "manual" not in normalized_tags:
        normalized_tags.append("manual")
    return normalized_tags


def _runner_args(selector, pytest_args):
    args = [
        "$(location %s)" % _PYTEST_SELECTOR_RUNNER,
        "--selector",
        selector,
    ]
    for pytest_arg in pytest_args or []:
        args.extend(["--pytest-arg", pytest_arg])
    return args


def trtllm_pytest_case(
        name,
        selector,
        tags = None,
        target_compatible_with = None,
        size = "large",
        timeout = "long",
        pytest_args = None,
        manual = True):
    """Defines a Bazel test target for one TensorRT-LLM pytest selector."""

    if not selector:
        fail("trtllm_pytest_case requires a non-empty selector")

    native.sh_test(
        name = name,
        srcs = [_PYTEST_SELECTOR_TEST_WRAPPER],
        args = _runner_args(selector, pytest_args),
        data = [_PYTEST_SELECTOR_RUNNER],
        env_inherit = _PYTEST_ENV_INHERIT,
        size = size,
        tags = _tags_with_manual(tags, manual),
        target_compatible_with = list(target_compatible_with or []),
        timeout = timeout,
    )
