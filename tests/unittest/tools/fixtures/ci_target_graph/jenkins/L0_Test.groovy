// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

def configureTestStages() {
    x86TestConfigs = [
        "H100_PCIe-AutoDeploy-1": ["h100-cr", "l0_h100", 1, 1],
        "H100_PCIe-AutoDeploy-Others-1": ["h100-cr", "l0_h100", 1, 1],
        "H100_PCIe-AutoDeploy-DeepSeek-1": ["h100-cr", "l0_h100", 1, 1],
        "H100_PCIe-AutoDeploy-GptOss-1": ["h100-cr", "l0_h100", 1, 1],
    ]

    x86SlurmTestConfigs = [
        "DGX_H100-4_GPUs-AutoDeploy-1": ["auto:dgx-h100-x4", "l0_dgx_h100", 1, 1, 4],
    ]
}
