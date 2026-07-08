# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Conftest for the accuracy suite (tests/accuracy/).

Shared harness (``accuracy_core``) + ``references/`` for both the unit tests
(``unit/``, pure CPU logic) and the integration tests (``integration/``, which
run on the TRT-LLM runtime via ``llmc.llm.LLM``). This inserts the accuracy dir
on ``sys.path`` so both subtrees can ``import accuracy_core``.

The standalone purity guard (root ``tests/conftest.py``) forbids importing
tensorrt_llm during a broad ``tests/`` run and exempts only
``tests/accuracy/integration`` -- those tests deliberately drive the TRT-LLM
runtime, so scope the run to that path when tensorrt_llm is present.
"""

import os
import sys

# Make ``accuracy_core`` importable as a top-level module from the test files.
sys.path.insert(0, os.path.dirname(__file__))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "large: model needs more than one 8x80GB node (bigger-memory GPU or "
        "multi-node); routed to the big/Blackwell cluster.",
    )
    config.addinivalue_line(
        "markers",
        "blackwell: model needs a Blackwell GPU (NVFP4/MXFP4); the products "
        "generator routes it to the big/Blackwell cluster.",
    )
