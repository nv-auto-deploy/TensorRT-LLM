<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# MiniMax M2 Notes

MiniMax M2 is the first validation target for this skill.

Useful diagnostics:

- decode-only windows when possible
- one-layer sanity runs with `TLLM_OVERRIDE_LAYER_NUM=1`
- compare AutoDeploy serving config against nearby AD registry configs
- pay attention to MoE routing, dispatch/combine, and host-side waits before attention
