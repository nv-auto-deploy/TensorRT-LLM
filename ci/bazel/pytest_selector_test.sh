#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: pytest_selector_test.sh <pytest-selector-runner> [runner args...]" >&2
    exit 2
fi

runner="$1"
shift

exec "${runner}" "$@"
