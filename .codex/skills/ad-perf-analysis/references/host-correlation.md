<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Host Correlation

Primary joins come from `correlationId` across:

- `CUPTI_ACTIVITY_KIND_RUNTIME`
- `CUPTI_ACTIVITY_KIND_KERNEL`
- `CUPTI_ACTIVITY_KIND_MEMCPY`
- `CUPTI_ACTIVITY_KIND_MEMSET`
- `CUPTI_ACTIVITY_KIND_SYNCHRONIZATION`

If `CUDA_CALLCHAINS` exists, also join:

- `CUPTI_ACTIVITY_KIND_RUNTIME.callchainId`
- `CUDA_CALLCHAINS.id`

This enables:

- host launch attribution
- memcpy attribution
- host wait classification
- optional stack-attributed hot paths
