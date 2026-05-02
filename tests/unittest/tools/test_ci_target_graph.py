# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LOCAL_SCRIPTS_DIR = REPO_ROOT / "scripts"
repo_root_text = str(REPO_ROOT)
if repo_root_text in sys.path:
    sys.path.remove(repo_root_text)
sys.path.insert(0, repo_root_text)

scripts_module = types.ModuleType("scripts")
scripts_module.__path__ = [str(LOCAL_SCRIPTS_DIR)]
sys.modules["scripts"] = scripts_module

from scripts.ci_target_graph.generate import build_manifest  # noqa: E402
from scripts.ci_target_graph.generate_bazel_autodeploy import (  # noqa: E402
    bazel_cases_from_manifest,
    build_autodeploy_h100_build,
)
from scripts.ci_target_graph.selector_parser import parse_pytest_selector  # noqa: E402
from scripts.ci_target_graph.validate import (  # noqa: E402
    summarize_missing_jenkins_stage_targets,
    targets_without_jenkins_stage,
)

FIXTURE_REPO_ROOT = Path(__file__).resolve().parent / "fixtures" / "ci_target_graph"

H100_SELECTOR = (
    "accuracy/test_llm_api_autodeploy.py::TestLlama3_1_8B::test_auto_dtype[trtllm-False-1]"
)
DGX_H100_SELECTOR = (
    "accuracy/test_llm_api_autodeploy.py::TestLlama3_1_8B::test_auto_dtype[trtllm-False-4]"
)
AUTO_TRIGGER_OTHERS_SELECTOR = (
    "accuracy/test_llm_api_autodeploy.py::TestLlama3_1_8B::test_attention_dp[1]"
)
MULTI_PATH_SELECTOR = (
    'unittest/_torch/test_model_config.py unittest/llmapi/test_llm_args.py -k "config"'
)
MULTI_PATHS = [
    "unittest/_torch/test_model_config.py",
    "unittest/llmapi/test_llm_args.py",
]
H100_AUTODEPLOY_PRE_SHARD_STAGES = [
    "H100_PCIe-AutoDeploy-1",
    "H100_PCIe-AutoDeploy-DeepSeek-1",
    "H100_PCIe-AutoDeploy-GptOss-1",
    "H100_PCIe-AutoDeploy-Others-1",
]
EXPECTED_TARGET_IDS = [
    (
        "//ci_target_graph/l0_dgx_h100:"
        "l0_dgx_h100__e000__t0000__accuracy_test_llm_api_autodeploy_py_"
        "testllama3_1_8b_test_auto_ff2c09f896"
    ),
    (
        "//ci_target_graph/l0_h100:"
        "l0_h100__e000__t0000__accuracy_test_llm_api_autodeploy_py_"
        "testllama3_1_8b_test_auto_3a05522434"
    ),
]


def _assert_source(
    source: dict[str, Any],
    *,
    yaml_file: str,
    entry_index: int,
    test_index: int,
    jenkins_stages: list[str],
) -> None:
    expected = {
        "yaml": f"tests/integration/test_lists/test-db/{yaml_file}",
        "context": yaml_file.removesuffix(".yml"),
        "entry_index": entry_index,
        "test_index": test_index,
        "jenkins_stages": jenkins_stages,
    }
    if "jenkins_stage_scope" in source:
        assert source["jenkins_stage_scope"] == "pre_shard_candidates"
        expected["jenkins_stage_scope"] = "pre_shard_candidates"
    assert source == expected


def _parse_pytest_selector(raw: str) -> dict[str, Any]:
    return parse_pytest_selector(raw).to_dict()


def _build_manifest() -> dict[str, Any]:
    return build_manifest(FIXTURE_REPO_ROOT)


def _manifest_targets(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    return manifest["targets"]


def _target_by_selector(targets: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {target["selector"]["raw"]: target for target in targets}


def _assert_selector(parsed: dict[str, Any], raw: str, path: str) -> None:
    assert parsed == {
        "raw": raw,
        "path": path,
        "paths": [path],
        "pytest_args": [],
        "timeout_minutes": None,
        "isolation": False,
    }


def _assert_autodeploy_target(
    target: dict[str, Any],
    *,
    raw: str,
    stage: str,
    yaml_file: str,
    gpu_count: int,
    jenkins_stages: list[str] | None = None,
) -> None:
    assert target["kind"] == "pytest_selector"

    selector = target["selector"]
    _assert_selector(selector, raw, raw)

    _assert_source(
        target["source"],
        yaml_file=yaml_file,
        entry_index=0,
        test_index=0,
        jenkins_stages=jenkins_stages or [stage],
    )

    assert target["constraints"] == {
        "stage": "pre_merge",
        "backend": "autodeploy",
        "orchestrator": "mpi",
        "auto_trigger": None,
        "gpu_wildcards": ["*h100*"],
        "system_gpu_count": {"gte": gpu_count, "lte": gpu_count},
    }

    tags = set(target["tags"])
    assert {
        "stage:pre_merge",
        "backend:autodeploy",
        "orchestrator:mpi",
        "gpu:*h100*",
        f"system_gpu_count:gte{gpu_count}_lte{gpu_count}",
    }.issubset(tags)
    if gpu_count > 1:
        assert "multi_gpu" in tags
    else:
        assert "multi_gpu" not in tags

    assert target["component_hints"] == ["tests/integration/defs/accuracy"]


def test_parse_plain_pytest_selector() -> None:
    parsed = _parse_pytest_selector(H100_SELECTOR)

    _assert_selector(parsed, H100_SELECTOR, H100_SELECTOR)


@pytest.mark.parametrize(
    ("raw", "expected_args"),
    [
        (
            'unittest/auto_deploy/singlegpu/models -k "llama and not slow"',
            ["-k", "llama and not slow"],
        ),
        (
            'unittest/auto_deploy/singlegpu -m "gpu and not flaky"',
            ["-m", "gpu and not flaky"],
        ),
        (
            "unittest/auto_deploy/standalone --tb=short --disable-warnings --unknown-option=value",
            ["--tb=short", "--disable-warnings", "--unknown-option=value"],
        ),
    ],
)
def test_parse_selector_preserves_pytest_args(raw: str, expected_args: list[str]) -> None:
    parsed = _parse_pytest_selector(raw)

    assert parsed["raw"] == raw
    assert parsed["path"] == raw.split()[0]
    assert parsed["paths"] == [raw.split()[0]]
    assert parsed["pytest_args"] == expected_args
    assert parsed["timeout_minutes"] is None
    assert parsed["isolation"] is False


def test_parse_selector_extracts_execution_policy() -> None:
    raw = (
        "accuracy/test_llm_api_autodeploy.py::TestLlama3_1_8B::"
        "test_auto_dtype[trtllm-False-1] ISOLATION TIMEOUT (90)"
    )
    parsed = _parse_pytest_selector(raw)

    assert parsed == {
        "raw": raw,
        "path": H100_SELECTOR,
        "paths": [H100_SELECTOR],
        "pytest_args": [],
        "timeout_minutes": 90,
        "isolation": True,
    }


def test_parse_selector_records_multiple_pytest_paths() -> None:
    parsed = _parse_pytest_selector(MULTI_PATH_SELECTOR)

    assert parsed["raw"] == MULTI_PATH_SELECTOR
    assert parsed["path"] == MULTI_PATHS[0]
    assert parsed["paths"] == MULTI_PATHS
    assert parsed["pytest_args"] == ["-k", "config"]
    assert parsed["timeout_minutes"] is None
    assert parsed["isolation"] is False


def test_build_manifest_from_minimal_autodeploy_fixtures() -> None:
    first_manifest = _build_manifest()
    second_manifest = _build_manifest()

    assert first_manifest["schema_version"] == 1
    first_target_ids = [target["target_id"] for target in _manifest_targets(first_manifest)]
    assert first_target_ids == sorted(first_target_ids)
    assert set(EXPECTED_TARGET_IDS).issubset(first_target_ids)
    assert first_target_ids == [
        target["target_id"] for target in _manifest_targets(second_manifest)
    ]

    targets = _target_by_selector(_manifest_targets(first_manifest))
    assert {H100_SELECTOR, DGX_H100_SELECTOR}.issubset(targets)

    _assert_autodeploy_target(
        targets[H100_SELECTOR],
        raw=H100_SELECTOR,
        stage="H100_PCIe-AutoDeploy-1",
        yaml_file="l0_h100.yml",
        gpu_count=1,
        jenkins_stages=H100_AUTODEPLOY_PRE_SHARD_STAGES,
    )
    _assert_autodeploy_target(
        targets[DGX_H100_SELECTOR],
        raw=DGX_H100_SELECTOR,
        stage="DGX_H100-4_GPUs-AutoDeploy-1",
        yaml_file="l0_dgx_h100.yml",
        gpu_count=4,
    )


def test_auto_trigger_others_excludes_named_model_family_stages() -> None:
    manifest = _build_manifest()
    target = _target_by_selector(_manifest_targets(manifest))[AUTO_TRIGGER_OTHERS_SELECTOR]

    assert target["constraints"]["auto_trigger"] == "others"
    assert "auto_trigger:others" in target["tags"]
    stages = set(target["source"]["jenkins_stages"])
    assert "H100_PCIe-AutoDeploy-Others-1" in stages
    assert "H100_PCIe-AutoDeploy-DeepSeek-1" not in stages
    assert "H100_PCIe-AutoDeploy-GptOss-1" not in stages


def test_manifest_component_hints_include_all_multi_path_prefixes() -> None:
    manifest = _build_manifest()
    target = _target_by_selector(_manifest_targets(manifest))[MULTI_PATH_SELECTOR]

    assert target["selector"]["path"] == MULTI_PATHS[0]
    assert target["selector"]["paths"] == MULTI_PATHS
    assert target["selector"]["pytest_args"] == ["-k", "config"]
    assert set(target["component_hints"]) >= {
        "tests/unittest/_torch",
        "tests/unittest/llmapi",
    }


def test_build_manifest_accepts_string_repo_root() -> None:
    manifest = build_manifest(str(FIXTURE_REPO_ROOT))

    assert manifest["schema_version"] == 1
    assert len(manifest["targets"]) == 4


def test_generate_autodeploy_h100_bazel_cases_from_manifest() -> None:
    manifest = _build_manifest()
    targets = _target_by_selector(_manifest_targets(manifest))
    cases = {case.name: case for case in bazel_cases_from_manifest(manifest)}

    h100_case = cases[targets[H100_SELECTOR]["target_id"].split(":", 1)[1]]
    assert h100_case.selector == H100_SELECTOR
    assert h100_case.pytest_args == []
    assert {
        "backend:autodeploy",
        "gpu:*h100*",
        "gpu:h100",
        "gpu_count:1",
        "requires:cuda",
        "requires:model_cache",
    }.issubset(set(h100_case.tags))
    assert h100_case.target_compatible_with == [
        "//platforms/gpu:h100",
        "//platforms/gpu_count:one",
    ]
    assert h100_case.timeout == "long"
    assert h100_case.deps == [
        "//tensorrt_llm/_torch/auto_deploy:runtime",
        "//tests/integration/defs/accuracy:accuracy_tests",
    ]

    multi_path_case = cases[targets[MULTI_PATH_SELECTOR]["target_id"].split(":", 1)[1]]
    assert multi_path_case.selector == MULTI_PATHS[0]
    assert multi_path_case.pytest_args == [MULTI_PATHS[1], "-k", "config"]
    assert multi_path_case.deps == ["//tensorrt_llm/_torch/auto_deploy:runtime"]


def test_render_autodeploy_h100_build_file_marks_generated_content() -> None:
    rendered = build_autodeploy_h100_build(_build_manifest())

    assert "Generated by scripts/ci_target_graph/generate_bazel_autodeploy.py" in rendered
    assert "trtllm_pytest_case(" in rendered
    assert "autodeploy_h100_generated" in rendered
    assert "//tensorrt_llm/_torch/auto_deploy:runtime" in rendered
    assert "//tests/integration/defs/accuracy:accuracy_tests" in rendered


def test_validate_reports_targets_without_jenkins_stage_candidates() -> None:
    manifest = {
        "targets": [
            {
                "source": {
                    "yaml": "tests/integration/test_lists/test-db/l0_example.yml",
                    "jenkins_stages": [],
                },
                "constraints": {
                    "stage": "pre_merge",
                    "backend": "autodeploy",
                },
            },
            {
                "source": {
                    "yaml": "tests/integration/test_lists/test-db/l0_example.yml",
                    "jenkins_stages": ["Example-1"],
                },
                "constraints": {
                    "stage": "pre_merge",
                    "backend": "autodeploy",
                },
            },
        ],
    }

    missing_targets = targets_without_jenkins_stage(manifest)

    assert len(missing_targets) == 1
    assert summarize_missing_jenkins_stage_targets(missing_targets) == [
        (
            1,
            "tests/integration/test_lists/test-db/l0_example.yml",
            "pre_merge",
            "autodeploy",
        ),
    ]


def test_cli_writes_manifest_json(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(REPO_ROOT)]
    )

    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.ci_target_graph.generate",
            "--repo-root",
            str(FIXTURE_REPO_ROOT),
            "--output",
            str(output),
        ],
        check=True,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    manifest = json.loads(output.read_text())
    assert manifest["schema_version"] == 1
    target_ids = [target["target_id"] for target in manifest["targets"]]
    assert target_ids == sorted(target_ids)
    assert set(EXPECTED_TARGET_IDS).issubset(target_ids)
