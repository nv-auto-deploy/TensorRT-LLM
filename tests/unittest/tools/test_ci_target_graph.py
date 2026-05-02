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
from scripts.ci_target_graph.select_impacted import (  # noqa: E402
    BazelQueryError,
    ChangedFile,
    broad_fallback_reason_for_path,
    normalize_changed_path,
    owner_labels_for_path,
    parse_name_status_z,
    select_impacted,
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


class _FakeQueryClient:
    def __init__(
        self,
        *,
        query_results: list[str] | None = None,
        fallback_query_results: list[str] | None = None,
        cquery_results: list[str] | None = None,
        incompatible_query_results: list[str] | None = None,
        fail_impacted_query: bool = False,
        fail_fallback_query: bool = False,
        fail_cquery: bool = False,
    ) -> None:
        self.query_results = query_results or []
        self.fallback_query_results = fallback_query_results
        self.cquery_results = cquery_results or []
        self.incompatible_query_results = incompatible_query_results or []
        self.fail_impacted_query = fail_impacted_query
        self.fail_fallback_query = fail_fallback_query
        self.fail_cquery = fail_cquery
        self.queries: list[str] = []
        self.cqueries: list[tuple[str, str]] = []

    def query(self, expression: str) -> list[str]:
        self.queries.append(expression)
        if self.fail_impacted_query and "rdeps(" in expression:
            raise BazelQueryError("fake impacted query failure")
        if "target_compatible_with" in expression:
            return self.incompatible_query_results
        if self.fail_fallback_query and "tests(" in expression:
            raise BazelQueryError("fake broad fallback query failure")
        if "tests(" in expression and self.fallback_query_results is not None:
            return self.fallback_query_results
        return self.query_results

    def cquery(self, expression: str, platform: str) -> list[str]:
        self.cqueries.append((expression, platform))
        if self.fail_cquery:
            raise BazelQueryError("fake platform query failure")
        return self.cquery_results


def _changed_file(path: str) -> ChangedFile:
    return ChangedFile(
        status="M",
        paths=(path,),
        raw_paths=(path,),
        source="test",
    )


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


def test_select_impacted_normalizes_repo_relative_paths() -> None:
    assert normalize_changed_path(r"scripts\ci_target_graph/./select_impacted.py") == (
        "scripts/ci_target_graph/select_impacted.py",
        None,
    )

    normalized, reason = normalize_changed_path("/absolute/path.py")
    assert normalized is None
    assert reason == "absolute path"

    normalized, reason = normalize_changed_path("scripts/../secret.py")
    assert normalized is None
    assert reason == "path escapes the repository"


def test_select_impacted_parses_nul_name_status_with_rename_evidence() -> None:
    parsed = parse_name_status_z(
        b"R100\0old/path.py\0new/path.py\0A\0scripts/ci_target_graph/generate.py\0",
        source="fixture",
    )

    assert parsed.fallback_reasons == ()
    assert parsed.changed_files[0].status == "R100"
    assert parsed.changed_files[0].raw_paths == ("old/path.py", "new/path.py")
    assert parsed.changed_files[0].paths == ("old/path.py", "new/path.py")
    assert parsed.changed_files[1].paths == ("scripts/ci_target_graph/generate.py",)


@pytest.mark.parametrize(
    "path",
    [
        "ci/bazel/defs.bzl",
        "platforms/BUILD.bazel",
        ".bazelrc",
        ".bazelversion",
        "MODULE.bazel",
        "MODULE.bazel.lock",
        "requirements_bazel_lock.txt",
        "requirements.txt",
        "jenkins/L0_Test.groovy",
        ".github/workflows/build.yml",
        "tests/integration/test_lists/test-db/l0_h100.yml",
        "tests/integration/test_lists/waives.txt",
    ],
)
def test_select_impacted_broad_fallback_policy_paths(path: str) -> None:
    assert broad_fallback_reason_for_path(path) is not None


def test_select_impacted_modeled_owner_mapping() -> None:
    assert owner_labels_for_path("tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py") == (
        "//tensorrt_llm/_torch/auto_deploy:runtime",
    )
    assert owner_labels_for_path("tests/integration/defs/accuracy/test_llm_api.py") == (
        "//tests/integration/defs/accuracy:accuracy_tests",
    )
    assert owner_labels_for_path("scripts/ci_target_graph/select_impacted.py") == (
        "//scripts/ci_target_graph:ci_target_graph_lib",
    )
    assert owner_labels_for_path("tests/integration/defs/accuracy/nested/test_llm_api.py") == ()
    assert owner_labels_for_path("docs/source/index.rst") == ()


def test_select_impacted_query_failure_uses_broad_fallback() -> None:
    query_client = _FakeQueryClient(
        query_results=["//ci/bazel:impacted_test"],
        fallback_query_results=["//ci/bazel:fallback_test"],
        cquery_results=["//ci/bazel:fallback_test"],
        fail_impacted_query=True,
    )

    result = select_impacted(
        repo_root=REPO_ROOT,
        base="upstream/main",
        platform="//platforms:h100_4gpu",
        changed_files=[_changed_file("tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py")],
        smoke_targets=["//ci/bazel:smoke_test"],
        query_client=query_client,
    )

    assert result.fallback_used
    assert any("impacted Bazel query failed" in reason for reason in result.fallback_reasons)
    assert result.candidate_targets == ("//ci/bazel:fallback_test",)
    assert result.selected_targets == ("//ci/bazel:fallback_test", "//ci/bazel:smoke_test")
    assert any("rdeps(" in expression for expression in query_client.queries)
    assert any("tests(" in expression for expression in query_client.queries)


def test_select_impacted_empty_impacted_query_uses_broad_fallback() -> None:
    query_client = _FakeQueryClient(
        query_results=[],
        fallback_query_results=["//ci/bazel:fallback_test"],
    )

    result = select_impacted(
        repo_root=REPO_ROOT,
        base="upstream/main",
        changed_files=[_changed_file("tensorrt_llm/_torch/auto_deploy/shim/ad_executor.py")],
        smoke_targets=["//ci/bazel:smoke_test"],
        query_client=query_client,
    )

    assert result.selection_available
    assert result.fallback_reasons == (
        "impacted Bazel query returned no targets for modeled owners; using broad fallback",
    )
    assert result.candidate_targets == ("//ci/bazel:fallback_test",)
    assert result.selected_targets == ("//ci/bazel:fallback_test", "//ci/bazel:smoke_test")
    assert any("rdeps(" in expression for expression in query_client.queries)
    assert any("tests(" in expression for expression in query_client.queries)


def test_select_impacted_broad_fallback_query_failure_marks_unavailable() -> None:
    query_client = _FakeQueryClient(fail_fallback_query=True)

    result = select_impacted(
        repo_root=REPO_ROOT,
        base="upstream/main",
        changed_files=[_changed_file("requirements.txt")],
        smoke_targets=["//ci/bazel:smoke_test"],
        query_client=query_client,
    )

    assert not result.selection_available
    assert result.candidate_targets == ()
    assert result.selected_targets == ()
    assert result.smoke_targets == ("//ci/bazel:smoke_test",)
    assert any("broad fallback Bazel query failed" in warning for warning in result.warnings)
    assert result.to_json_dict()["selection_available"] is False


def test_select_impacted_platform_fallback_failure_marks_unavailable() -> None:
    query_client = _FakeQueryClient(
        query_results=["//ci/bazel:impacted_test"],
        fail_cquery=True,
        fail_fallback_query=True,
    )

    result = select_impacted(
        repo_root=REPO_ROOT,
        base="upstream/main",
        platform="//platforms:h100_4gpu",
        changed_files=[_changed_file("tensorrt_llm/_torch/auto_deploy/runtime.py")],
        smoke_targets=["//ci/bazel:smoke_test"],
        query_client=query_client,
    )

    assert not result.selection_available
    assert result.candidate_targets == ("//ci/bazel:impacted_test",)
    assert result.selected_targets == ()
    assert any("platform cquery failed" in warning for warning in result.warnings)
    assert any("broad fallback Bazel query failed" in warning for warning in result.warnings)


def test_select_impacted_result_json_is_stable_and_sorted() -> None:
    query_client = _FakeQueryClient(
        query_results=["//ci/bazel:z_test", "//ci/bazel:a_test", "//ci/bazel:a_test"],
    )

    result = select_impacted(
        repo_root=REPO_ROOT,
        base="upstream/main",
        head="feature",
        changed_files=[_changed_file("scripts/ci_target_graph/generate.py")],
        smoke_targets=["//ci/bazel:smoke_test"],
        query_client=query_client,
    )
    payload = result.to_json_dict()

    assert payload["schema_version"] == 1
    assert payload["base"] == "upstream/main"
    assert payload["head"] == "feature"
    assert payload["fallback"] == {"reasons": [], "used": False}
    assert payload["owner_labels"] == ["//scripts/ci_target_graph:ci_target_graph_lib"]
    assert payload["candidate_targets"] == ["//ci/bazel:a_test", "//ci/bazel:z_test"]
    assert payload["selection_available"] is True
    assert payload["selected_targets"] == [
        "//ci/bazel:a_test",
        "//ci/bazel:smoke_test",
        "//ci/bazel:z_test",
    ]
    assert json.loads(result.to_json()) == payload


def test_select_impacted_platform_and_manual_filters_use_fake_query_client() -> None:
    query_client = _FakeQueryClient(
        query_results=["//ci/bazel:manual_gpu_test", "//ci/bazel:wrong_platform_test"],
        cquery_results=["//ci/bazel:manual_gpu_test (abcdef)"],
    )

    result = select_impacted(
        repo_root=REPO_ROOT,
        base="upstream/main",
        platform="//platforms:h100_4gpu",
        changed_files=[_changed_file("tensorrt_llm/_torch/auto_deploy/runtime.py")],
        manual_policy="only",
        include_tags=["backend:autodeploy"],
        exclude_tags=["slow"],
        query_client=query_client,
    )

    assert result.selected_targets == ("//ci/bazel:manual_gpu_test",)
    assert query_client.queries
    query_expression = query_client.queries[0]
    assert 'attr("tags", "manual"' in query_expression
    assert 'attr("tags", "backend:autodeploy"' in query_expression
    assert 'except attr("tags", "slow"' in query_expression
    assert query_client.cqueries == [
        (
            "set(//ci/bazel:manual_gpu_test //ci/bazel:wrong_platform_test)",
            "//platforms:h100_4gpu",
        )
    ]
