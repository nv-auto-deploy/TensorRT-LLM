# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Parse pytest selector strings from TensorRT-LLM CI test lists."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from typing import Any

_TIMEOUT_TOKEN_RE = re.compile(r"^TIMEOUT\s*\(?\s*(?P<minutes>\d+)\s*\)?$")
_TIMEOUT_VALUE_RE = re.compile(r"^\(?\s*(?P<minutes>\d+)\s*\)?$")

_PYTEST_OPTIONS_WITH_VALUES = {
    "-c",
    "-k",
    "-m",
    "-o",
    "-p",
    "--basetemp",
    "--cache-clear",
    "--color",
    "--confcutdir",
    "--cov",
    "--cov-config",
    "--cov-report",
    "--csv",
    "--durations",
    "--junit-xml",
    "--maxfail",
    "--output-dir",
    "--rootdir",
    "--tb",
    "--test-list",
    "--test-prefix",
    "--timeout",
    "--timeout-method",
    "--waives-file",
}


@dataclass(frozen=True)
class ParsedSelector:
    """Structured view of a raw pytest selector from a test-db YAML entry."""

    raw: str
    path: str | None
    paths: list[str]
    pytest_args: list[str]
    timeout_minutes: int | None
    isolation: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable selector shape used by the manifest."""
        return {
            "raw": self.raw,
            "path": self.path,
            "paths": self.paths,
            "pytest_args": self.pytest_args,
            "timeout_minutes": self.timeout_minutes,
            "isolation": self.isolation,
        }


def parse_pytest_selector(raw: str) -> ParsedSelector:
    """Parse a raw test list selector without rejecting unfamiliar syntax.

    The test-db convention stores timeout markers as minutes, for example
    ``TIMEOUT (90)``. Jenkins and the integration test parser multiply that
    value by 60 before applying pytest-timeout, so this parser keeps the
    manifest field in minutes.
    """
    text = str(raw).strip()
    tokens = _split_selector(text)
    paths: list[str] = []
    pytest_args: list[str] = []
    timeout_minutes: int | None = None
    isolation = False
    option_waiting_for_value = False

    index = 0
    while index < len(tokens):
        token = tokens[index]

        cleaned_token, token_had_isolation = _remove_isolation_marker(token)
        if token_had_isolation:
            isolation = True
        if not cleaned_token:
            index += 1
            continue
        token = cleaned_token

        parsed_timeout, consumed = _parse_timeout(tokens, index, token)
        if consumed:
            if parsed_timeout is not None:
                timeout_minutes = parsed_timeout
            index += consumed
            continue

        if option_waiting_for_value or token.startswith("-"):
            pytest_args.append(token)
        else:
            paths.append(token)

        if option_waiting_for_value:
            option_waiting_for_value = False
        elif _option_expects_value(token):
            option_waiting_for_value = True

        index += 1

    path = paths[0] if paths else None
    return ParsedSelector(
        raw=text,
        path=path,
        paths=paths,
        pytest_args=pytest_args,
        timeout_minutes=timeout_minutes,
        isolation=isolation,
    )


def _split_selector(text: str) -> list[str]:
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


def _remove_isolation_marker(token: str) -> tuple[str, bool]:
    parts = token.split(",")
    if "ISOLATION" not in parts:
        return token, False

    cleaned_parts = [part for part in parts if part != "ISOLATION"]
    return ",".join(cleaned_parts).strip(), True


def _parse_timeout(tokens: list[str], index: int, token: str) -> tuple[int | None, int]:
    token_match = _TIMEOUT_TOKEN_RE.match(token)
    if token_match:
        return int(token_match.group("minutes")), 1

    if token != "TIMEOUT":
        return None, 0

    if index + 1 >= len(tokens):
        return None, 1

    value_match = _TIMEOUT_VALUE_RE.match(tokens[index + 1])
    if value_match:
        return int(value_match.group("minutes")), 2

    return None, 1


def _option_expects_value(token: str) -> bool:
    if "=" in token:
        return False
    return token in _PYTEST_OPTIONS_WITH_VALUES
