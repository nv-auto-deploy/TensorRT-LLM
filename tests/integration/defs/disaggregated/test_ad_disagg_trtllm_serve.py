# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from pathlib import Path

import pytest
from defs.common import get_free_port_in_ci as get_free_port
from defs.conftest import llm_models_root
from disagg_test_utils import (
    run_ctx_worker,
    run_disagg_server,
    run_gen_worker,
    terminate,
    wait_for_disagg_server_ready,
)
from openai import OpenAI

pytest_plugins = ["disagg_test_utils"]

SERVER_START_TIMEOUT_S = 600
TINYLLAMA_MODEL_DIR = "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"
AUTODEPLOY_BACKEND = "_autodeploy"


def tinyllama_model_path():
    return str(Path(llm_models_root()) / TINYLLAMA_MODEL_DIR)


def worker_cuda_devices(num_workers):
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        devices = [device.strip() for device in visible_devices.split(",") if device.strip()]
        if len(devices) < num_workers:
            pytest.skip(
                f"AutoDeploy trtllm-serve disagg smoke requires {num_workers} "
                f"visible GPUs, got {len(devices)}"
            )
        return devices[:num_workers]

    return [str(device) for device in range(num_workers)]


def autodeploy_worker_config(disable_overlap_scheduler=False):
    config = {
        "backend": AUTODEPLOY_BACKEND,
        "max_batch_size": 1,
        "cuda_graph_config": {"batch_sizes": [1]},
        "cache_transceiver_config": {"backend": "DEFAULT"},
    }
    if disable_overlap_scheduler:
        config["disable_overlap_scheduler"] = True

    return config


def proxy_config(port, ctx_port, gen_port):
    return {
        "hostname": "localhost",
        "port": port,
        "backend": AUTODEPLOY_BACKEND,
        "context_servers": {
            "num_instances": 1,
            "urls": [f"localhost:{ctx_port}"],
        },
        "generation_servers": {
            "num_instances": 1,
            "urls": [f"localhost:{gen_port}"],
        },
    }


@pytest.mark.skip_less_device_memory(30000)
@pytest.mark.skip_less_device(2)
@pytest.mark.timeout(900)
@pytest.mark.asyncio(loop_scope="module")
async def test_openai_completion(disagg_port, work_dir):
    """Smoke test AutoDeploy disagg through trtllm-serve and the OpenAI API.

    The lower-level tests in ``test_ad_disagg.py`` drive AutoDeploy workers
    directly and inspect context/generation handoff metadata. This test instead
    verifies the trtllm-serve deployment shape: context worker, generation
    worker, disaggregated proxy, and an OpenAI-compatible completion request.
    """
    model = tinyllama_model_path()
    ctx_device, gen_device = worker_cuda_devices(2)
    ctx_port = get_free_port()
    gen_port = get_free_port()
    ctx_worker = None
    gen_worker = None
    disagg_server = None

    try:
        ctx_worker = run_ctx_worker(
            model,
            autodeploy_worker_config(disable_overlap_scheduler=True),
            work_dir,
            port=ctx_port,
            device=ctx_device,
        )
        gen_worker = run_gen_worker(
            model,
            autodeploy_worker_config(),
            work_dir,
            port=gen_port,
            device=gen_device,
        )
        disagg_server = run_disagg_server(
            proxy_config(disagg_port, ctx_port, gen_port), work_dir, disagg_port
        )
        await wait_for_disagg_server_ready(disagg_port, timeout=SERVER_START_TIMEOUT_S)

        client = OpenAI(
            api_key="tensorrt_llm",
            base_url=f"http://localhost:{disagg_port}/v1",
        )
        response = client.completions.create(
            model=model,
            prompt="What is the capital of Germany?",
            max_tokens=10,
            temperature=0,
            extra_body={"ignore_eos": True},
        )
    finally:
        terminate(ctx_worker, gen_worker, disagg_server)

    assert response.choices
    assert response.choices[0].text
