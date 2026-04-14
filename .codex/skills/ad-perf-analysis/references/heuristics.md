<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Heuristics

- Prefer one change per iteration.
- Prefer reusing existing artifacts before recollecting them.
- If the graph dump is absent, continue analysis but mark semantic confidence lower.
- If host waits dominate GPU gaps, collect a deep host trace before recommending deeper GPU work.
