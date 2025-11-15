# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from collections import defaultdict
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import torch
from strenum import StrEnum
from torch._prims_common import DeviceLikeType

from tensorrt_llm._torch.attention_backend.interface import AttentionRuntimeFeatures
from tensorrt_llm._torch.pyexecutor._util import _create_kv_cache_manager, get_kv_cache_manager_cls
from tensorrt_llm._torch.pyexecutor.config_utils import is_mla
from tensorrt_llm._torch.pyexecutor.guided_decoder import GuidedDecoder
from tensorrt_llm._torch.pyexecutor.llm_request import get_draft_token_length
from tensorrt_llm._torch.pyexecutor.py_executor_creator import get_guided_decoding_config
from tensorrt_llm._torch.pyexecutor.seq_slot_manager import SeqSlotManager
from tensorrt_llm._torch.speculative import (
    SpecMetadata,
    _get_spec_drafter,
    _get_spec_resource_manager,
    get_spec_metadata,
)
from tensorrt_llm._torch.speculative.mtp import SampleStateTensorsMTP
from tensorrt_llm._utils import nvtx_range
from tensorrt_llm.llmapi.llm_args import (
    ContextChunkingPolicy,
    DecodingBaseConfig,
    LoadFormat,
    TorchLlmArgs,
)
from tensorrt_llm.llmapi.tokenizer import TokenizerBase

from ...._utils import mpi_rank, mpi_world_size
from ....bindings.internal.batch_manager import CacheType
from ....mapping import Mapping
from ...distributed import MPIDist
from ...pyexecutor.model_engine import ModelEngine, PyTorchModelEngine
from ...pyexecutor.py_executor import PyExecutor
from ...pyexecutor.resource_manager import KVCacheManager, ResourceManager, ResourceManagerType
from ...pyexecutor.sampler import SampleStateTensors, TorchSampler
from ...pyexecutor.scheduler import (
    BindCapacityScheduler,
    BindMicroBatchScheduler,
    ScheduledRequests,
    SimpleScheduler,
)
from ..custom_ops.attention_interface import SequenceInfo
from ..distributed import common as dist
from ..llm_args import LlmArgs
from ..transform.optimizer import InferenceOptimizer
from ..utils.logger import ad_logger
from .interface import CachedSequenceInterface, GetInferenceModel


class _CacheManagerWithFakePool(KVCacheManager):
    """We use the default KVCacheManager but with a fake pool by setting head_dim=0.

    The actual cache pools are managed by auto_deploy layerwise cache pools.
    """

    def __init__(
        self,
        kv_cache_config,
        num_blocks: int,
        tokens_per_block: int,
        max_seq_len: int,
        max_batch_size: int,
    ):
        self.num_blocks = num_blocks
        super().__init__(
            kv_cache_config=kv_cache_config,
            kv_cache_type=CacheType.SELF,
            num_layers=1,
            num_kv_heads=1,
            head_dim=0,
            tokens_per_block=tokens_per_block,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=Mapping(),
        )

    def calculate_max_num_blocks(
        self, kv_cache_config, head_dim, tokens_per_block, mapping, dtype, kv_factor
    ) -> Tuple[int, int]:
        """Calculate the maximum number of blocks needed for the cache."""
        # TODO: this is VERY hacky... Ideally, we want to compute the number of blocks
        # just like in the original implementation. However, let's wait for the layer-wise attention
        # implementation before over-optimizing the function here
        ad_logger.info("Using fake cache manager with head_dim=0 and num pages:", self.num_blocks)
        return self.num_blocks, 0


def construct_draft_llm_args(
    ad_config: LlmArgs,
    draft_spec_config: DecodingBaseConfig,
    enable_chunked_context: bool,
) -> TorchLlmArgs:
    """Construct a TorchLlmArgs for the draft model from AutoDeploy config.

    Args:
        ad_config: The AutoDeploy LLM configuration
        draft_spec_config: The speculative decoding config for the draft model
        enable_chunked_context: Whether chunked prefill is enabled

    Returns:
        A TorchLlmArgs instance suitable for creating a PyTorchModelEngine
    """
    # Extract common fields as a dict
    common_fields = {
        "model": ad_config.model,
        "tokenizer": ad_config.tokenizer,
        "max_batch_size": ad_config.max_batch_size,
        "max_seq_len": ad_config.max_seq_len,
        "max_beam_width": ad_config.max_beam_width,
        "max_num_tokens": ad_config.max_num_tokens,
        "max_input_len": ad_config.max_input_len,
        "kv_cache_config": ad_config.kv_cache_config,
        "enable_chunked_prefill": enable_chunked_context,
        "attn_backend": ad_config.attn_backend,
        "disable_overlap_scheduler": ad_config.disable_overlap_scheduler,
        "speculative_config": draft_spec_config,
        "checkpoint_loader": getattr(ad_config, "draft_checkpoint_loader", None),
    }

    # Add other fields that may exist in ad_config
    optional_fields = [
        "dtype",
        "trust_remote_code",
        "sparse_attention_config",
        "lora_config",
        "scheduler_config",
        "garbage_collection_gen0_threshold",
        "skip_tokenizer_init",
        "tokenizer_mode",
        "revision",
        "tokenizer_revision",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "context_parallel_size",
        "gpus_per_node",
        "enable_lora",
        "guided_decoding_backend",
        "peft_cache_config",
        "cache_transceiver_config",
        "decoding_config",
    ]

    for field in optional_fields:
        if hasattr(ad_config, field):
            value = getattr(ad_config, field)
            if value is not None:  # Only add if not None
                common_fields[field] = value

    draft_llm_args = TorchLlmArgs(**common_fields)

    # Handle load_format separately
    if draft_spec_config.load_format == "dummy":
        draft_llm_args.load_format = LoadFormat.DUMMY

    draft_llm_args.tensor_parallel_size = ad_config.world_size

    return draft_llm_args


# Used for compatibility with interfaces that expect max_num_tokens to be an int.
def get_max_num_tokens(num_tokens_limit: Optional[int], max_seq_len: int, batch_size: int) -> int:
    if num_tokens_limit is not None:
        return num_tokens_limit
    return max_seq_len * batch_size


def create_draft_kv_cache_manager_maybe(
    draft_model_engine: Optional[PyTorchModelEngine],
    ad_config: LlmArgs,
    dist_mapping: Mapping,
) -> Optional[KVCacheManager]:
    if draft_model_engine is None or not draft_model_engine.model.model_config.is_generation:
        return None

    # Get the appropriate KV cache manager class
    kv_cache_manager_cls = get_kv_cache_manager_cls(draft_model_engine.model.model_config)

    return _create_kv_cache_manager(
        model_engine=draft_model_engine,
        kv_cache_manager_cls=kv_cache_manager_cls,
        mapping=dist_mapping,
        kv_cache_config=ad_config.kv_cache_config,
        tokens_per_block=ad_config.attn_page_size,
        max_seq_len=ad_config.max_seq_len,
        max_batch_size=ad_config.max_batch_size,
        spec_config=ad_config.speculative_config,
        sparse_attn_config=ad_config.sparse_attention_config,
        max_num_tokens=ad_config.max_num_tokens,
        max_beam_width=ad_config.max_beam_width,
        is_draft=True,
        kv_connector_manager=None,  # KV connector not supported with draft models in AutoDeploy
        estimating_kv_cache=False,
    )


def create_spec_resource_manager(
    engine: "ADEngine",
    draft_model_engine: Optional[PyTorchModelEngine] = None,
):
    """
    Create a speculative resource manager for the given ADEngine and optional draft model.

    This is a convenience helper that extracts the necessary configuration from the engine
    objects to create the spec resource manager, rather than requiring all parameters to be
    passed explicitly.

    Args:
        engine: The ADEngine (target model engine) that stores ad_config
        draft_model_engine: Optional draft model engine for speculative decoding

    Returns:
        A spec resource manager or None if speculative config is not set
    """
    max_num_tokens = get_max_num_tokens(
        engine.llm_args.max_num_tokens, engine.llm_args.max_seq_len, engine.ad_config.max_batch_size
    )

    return _get_spec_resource_manager(
        spec_config=engine.spec_config,
        model_config=engine.model_config,
        max_num_requests=engine.get_max_num_sequences(),
        max_seq_len=engine.llm_args.max_seq_len,
        max_num_tokens=max_num_tokens,
        draft_model_engine=draft_model_engine,
    )


def get_spec_drafter(
    model_engine: "ADEngine",
    draft_model_engine,
    sampler,
    spec_resource_manager,
    guided_decoder: Optional[GuidedDecoder] = None,
):
    """
    Create a speculative drafter for the given ADEngine.

    This is a convenience helper that extracts spec_config and max_num_requests
    from the ADEngine and delegates to _get_spec_drafter, similar to the
    get_spec_drafter function in tensorrt_llm._torch.speculative.utils but
    adapted for ADEngine.

    Args:
        model_engine: The ADEngine (target model engine)
        draft_model_engine: Draft model engine for speculative decoding
        sampler: The sampler instance
        spec_resource_manager: Speculative resource manager
        guided_decoder: Optional guided decoder for structured generation

    Returns:
        A speculative drafter or None if speculative config is not set
    """
    spec_config = model_engine.spec_config
    if spec_config is None:
        return None

    max_num_requests = model_engine.get_max_num_sequences()

    return _get_spec_drafter(
        spec_config=spec_config,
        max_num_requests=max_num_requests,
        draft_model_engine=draft_model_engine,
        sampler=sampler,
        spec_resource_manager=spec_resource_manager,
        guided_decoder=guided_decoder,
    )


class ADEngine(ModelEngine):
    """The AutoDeploy Engine (ADEngine) is the main engine interface to execute AutoDeploy models.

    It follows the ``ModelEngine`` abstractions and is responsible for building the ad-optimized
    model, converting TRT-LLM scheduled requests into ad-native (pytorch-native) inputs, running
    the model, and returning correctly formatted logits.
    """

    @property
    def _device(self) -> DeviceLikeType:
        return self.cache_seq_interface.device

    @property
    def run_with_spec_decode(self):
        return self.spec_config is not None

    @property
    def model_config(self):
        return self.ad_config.create_factory()._get_model_config()

    @classmethod
    def build_from_config(cls, ad_config: LlmArgs):
        """Build the ADEngine using the LlmArgs that gets passed through from the LLM."""

        print(f"[TRACE] Building ADEngine from config: {ad_config}")

        max_batch_size = ad_config.max_batch_size
        max_seq_len = ad_config.max_seq_len
        attn_page_size = ad_config.attn_page_size
        max_num_tokens = ad_config.max_num_tokens
        max_beam_width = ad_config.max_beam_width

        # update device to contain the current default device if it's in cuda
        device = torch.device(ad_config.device)
        if device.type == "cuda" and device.index is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        device = str(device)

        factory = ad_config.create_factory()

        # initialize seq info object
        seq_info = SequenceInfo(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            page_size=attn_page_size,
            max_num_tokens=max_num_tokens,
            vocab_size_padded=factory.vocab_size_padded,
        )

        # TODO (lucaslie): consider how we move args around InferenceOptimizer.__init__,
        # ADEngine.__init__, and ADEngine.build_from_config. Seems a bit unnatural atm.

        # construct inference optimizer
        build_and_optimize = InferenceOptimizer(factory=factory, config=ad_config.transforms)

        # construct engine
        return cls(
            build_and_optimize,
            seq_info,
            device,
            ad_config,
            max_beam_width,
        )

    @torch.inference_mode()
    def __init__(
        self,
        get_inference_model: GetInferenceModel,
        seq_info: SequenceInfo,
        device: DeviceLikeType,
        ad_config: LlmArgs,
        max_beam_width: int = 1,
    ) -> None:
        """Initialize the engine with model and sequence information."""
        # NOTE (lucaslie): create a fake Namespace to satisfy PyExecutor requirements...
        # This is not correctly declared in the base ModelEngine class though...
        self.llm_args = SimpleNamespace()
        self.llm_args.print_iter_log = False
        self.llm_args.enable_iter_perf_stats = False
        self.llm_args.enable_iter_req_stats = False
        self.llm_args.stream_interval = 1
        self.llm_args.attention_dp_config = None
        self.llm_args.batch_wait_timeout_ms = 0
        self.llm_args.batch_wait_timeout_iters = 0
        self.llm_args.batch_wait_max_tokens_ratio = 0.0
        self.llm_args.max_num_tokens = seq_info.max_num_tokens
        self.llm_args.max_seq_len = seq_info.max_seq_len
        self.iter_counter = 0

        # NOTE (lucaslie): not a declared base member in the base class; required by PyExecutor...
        self.max_beam_width = max_beam_width
        self.enable_attention_dp = False

        self.ad_config = ad_config
        self.spec_config = ad_config.speculative_config
        print(f"[TRACE] Spec config in ADEngine.__init__: {self.spec_config}")

        # construct cache sequence interface
        self.cache_seq_interface = CachedSequenceInterface(
            sequence_info=seq_info,
            device=device,
        )

        # build model
        self.model = get_inference_model(self.cache_seq_interface)

        # start fresh with fixed seed
        torch.manual_seed(42)

    @nvtx_range("ad_prepare_inputs")
    def _prepare_inputs(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: ResourceManager,
        new_tensors_device: Optional[SampleStateTensors] = None,
        spec_metadata: Optional[SpecMetadata] = None,
    ) -> List[bool]:
        """Prepare inputs for AD Model from scheduled requests."""
        # cache manager

        print(
            f"[TRACE] Preparing inputs for AD Model from scheduled requests: {scheduled_requests}"
        )
        print(f"[TRACE] Context requests length: {len(scheduled_requests.context_requests)}")
        print(f"[TRACE] Generation requests length: {len(scheduled_requests.generation_requests)}")

        assert (
            self.spec_config is None or self.spec_config.spec_dec_mode.support_overlap_scheduler()
        ), (
            f"{self.spec_config.spec_dec_mode.decoding_type} must support the overlap scheduler currently"
        )

        new_tokens, new_tokens_lens, next_draft_tokens = None, None, None
        if new_tensors_device is not None:
            print(f"[TRACE] new_tensors_device type: {type(new_tensors_device)}")
            new_tokens = new_tensors_device.new_tokens
            # print(f"new_tokens: {new_tokens}")
            # print(f"new_tokens.shape: {new_tokens.shape}")

            # Note(govind): The following if statement should always be true if we are using
            # the overlap scheduler with speculative decoding.
            if isinstance(new_tensors_device, SampleStateTensorsMTP):
                assert self.ad_config.speculative_config is not None
                new_tokens_lens = new_tensors_device.new_tokens_lens  # [batch]
                next_draft_tokens = new_tensors_device.next_draft_tokens  # [batch, draft_len]
                print(f"new_tokens_lens: {new_tokens_lens}")
                new_tokens_lens = len(new_tokens_lens)
                print(f"new_tokens_lens (length): {new_tokens_lens}")
                print(f"new_tokens: {new_tokens}")
                print(f"new_tokens.shape: {new_tokens.shape}")
                print(f"next_draft_tokens: {next_draft_tokens}")
                print(f"next_draft_tokens.shape: {next_draft_tokens.shape}")

        # TODO(govind): Manage guided decoding + speculative decoding together.

        kv_cache_manager = resource_manager.get_resource_manager(
            ResourceManagerType.KV_CACHE_MANAGER
        )

        print(f"[TRACE] KV cache manager: {kv_cache_manager}")
        # requests in order of context, generate
        context_requests = scheduled_requests.context_requests
        gen_requests = scheduled_requests.generation_requests
        extend_requests = []
        first_draft_requests = []
        generation_requests = []

        # separate requests into extend requests (contains draft tokens) and other
        # generation requests (requests that do not have draft tokens)
        for request in gen_requests:
            if get_draft_token_length(request) > 0 or next_draft_tokens is not None:
                extend_requests.append(request)
            else:
                generation_requests.append(request)

            assert not request.py_is_first_draft, (
                "first draft requests should not be here, ADEngine only supports target models in spec dec"
            )

        print(f"[TRACE] Extend requests: {extend_requests}")
        print(f"[TRACE] First draft requests: {first_draft_requests}")
        print(
            f"[TRACE] Generation requests (not extend or first draft requests): {generation_requests}"
        )

        if new_tokens is not None:
            print(f"[TRACE] New tokens shape: {new_tokens.shape}")

        # info to be extracted
        input_ids: List[List[int]] = []
        input_pos: List[int] = []
        last_logit_only: List[bool] = []
        page_assignments: List[List[int]] = []
        slot_idx: List[int] = []
        all_gather_indices: List[List[int]] = []
        extra_args: Dict[str, List[torch.Tensor]] = defaultdict(list)

        dummy_token = -1

        # look at context requests first
        for request in context_requests:
            # store input ids and pos of first token in sequence
            # NOTE: begin_compute > 0 indicates block reuse
            # NOTE: end_compute will be used in the future for chunked prefill
            all_prompt_tokens = request.get_tokens(0)
            begin_compute = request.context_current_position
            end_compute = begin_compute + request.context_chunk_size
            prompt_tokens = all_prompt_tokens[begin_compute:end_compute]

            input_ids.append(prompt_tokens)
            input_pos.append(begin_compute)

            request.py_batch_idx = request.seq_slot
            last_logit_only.append(True)

            # get cache indices and truncate the number of blocks according to end_compute
            cache_indices = kv_cache_manager.get_cache_indices(request)
            num_active_blocks = kv_cache_manager.get_num_kv_blocks(end_compute)
            page_assignments.append(cache_indices[:num_active_blocks])

            # store seq slot idx
            slot_idx.append(request.seq_slot)

            # store extra arguments
            if request.py_multimodal_data is not None:
                for k, v in request.py_multimodal_data.items():
                    extra_args[k].append(v)

        # look at generate requests next
        # TODO: we should also handle extend requests (for speculative decoding) here

        for request in extend_requests:
            if new_tokens is None or request.is_dummy or request.py_batch_idx is None:
                print("Processing possibly non-dummy request")
                print(
                    f"new_tokens: {new_tokens}, request.is_dummy: {request.is_dummy},\
                        request.py_batch_idx: {request.py_batch_idx},\
                        request.max_beam_num_tokens: {request.max_beam_num_tokens}"
                )
                input_ids.append(
                    [request.get_token(0, request.get_num_tokens(0) - 1)]
                    + [token for token in request.py_draft_tokens]
                )
                num_tokens_seen = request.max_beam_num_tokens - 1
                input_pos.append(num_tokens_seen)
            else:
                print(f"length of request.py_draft_tokens: {len(request.py_draft_tokens)}")
                print(f"py_batch_idx: {request.py_batch_idx}")

                gather_indices_to_extend = [
                    x * new_tokens_lens + request.py_batch_idx
                    for x in range(len(request.py_draft_tokens) + 1)
                ]

                print(f"gather_indices_to_extend: {gather_indices_to_extend}")
                all_gather_indices.append(gather_indices_to_extend)
                dummy_draft_tokens = [dummy_token for _ in range(len(request.py_draft_tokens))]
                print(dummy_draft_tokens)
                input_ids.append([dummy_token] + dummy_draft_tokens)

                num_tokens_seen = request.max_beam_num_tokens - 1
                input_pos.append(num_tokens_seen)

            request.py_batch_idx = request.seq_slot

            slot_idx.append(request.seq_slot)

            last_logit_only.append(False)

            # get cache indices
            cache_indices = kv_cache_manager.get_cache_indices(request)
            page_assignments.append(cache_indices)

        for request in generation_requests:
            print("About to append the following to input_pos:")
            print("Request.max_beam_num_tokens: ", request.max_beam_num_tokens)
            print("Request.py_draft_tokens: ", request.py_draft_tokens)

            # new_tokens are provided when the overlap scheduler is enabled.
            if new_tokens is None or request.is_dummy or request.py_batch_idx is None:
                input_ids.append([request.get_token(0, request.get_num_tokens(0) - 1)])
                input_pos.append(request.max_beam_num_tokens - 1)

            else:
                input_ids.append([dummy_token])
                input_pos.append(request.max_beam_num_tokens)
                all_gather_indices.append([request.py_batch_idx])

            print(f"[TRACE] request.seq_slot: {request.seq_slot}")
            print(f"[TRACE] request.py_batch_idx: {request.py_batch_idx}")

            request.py_batch_idx = request.seq_slot

            # store seq slot idx
            # TODO: double-check if this is correct for the overlap scheduler
            slot_idx.append(request.seq_slot)

            # return all logits
            last_logit_only.append(False)

            # get cache indices
            cache_indices = kv_cache_manager.get_cache_indices(request)
            page_assignments.append(cache_indices)

        print("Finished for loop over generation requests.")
        # update the sequence info object now
        self.cache_seq_interface.info.nest_sequences(
            input_ids,
            input_pos=input_pos,
            page_assignments=page_assignments,
            slot_idx=slot_idx,
            **extra_args,
        )

        flat_gather_indices = [
            gather_idx
            for seq_gather_indices in all_gather_indices
            for gather_idx in seq_gather_indices
        ]

        print(f"[TRACE] input_ids: {input_ids}")
        print(f"[TRACE] input_pos: {input_pos}")
        print(f"[TRACE] page_assignments: {page_assignments}")
        print(f"[TRACE] flat_gather_indices: {flat_gather_indices}")
        print(f"new_tokens (unflattened): {new_tokens}")

        print(
            f"[TRACE] sequence info (before rescatter): {self.cache_seq_interface.info.named_args}"
        )
        # scatter the new tokens into the input_ids tensor if provided
        if new_tokens is not None:
            self.cache_seq_interface.info.rescatter_input_ids(
                ungathered_input_ids=new_tokens.flatten(),  # ensure it's flattened
                gather_idx=flat_gather_indices,
                scatter_ref=dummy_token,
            )

        print(
            f"[TRACE] sequence info (after rescatter): {self.cache_seq_interface.info.named_args}"
        )

        return last_logit_only

    @nvtx_range("ad_compute_logits")
    def _compute_logits(self) -> List[torch.Tensor]:
        # run the model
        print("[TRACE] In _compute_logits(). Running model with named args")
        print(f"[TRACE] Named args: {self.cache_seq_interface.info.named_args}")
        logits: torch.Tensor = self.model(**self.cache_seq_interface.named_args)[0]

        # return a list of tensors
        return self.cache_seq_interface.info.unnest_sequences(logits)

    def get_max_num_sequences(self) -> int:
        """Maximum number of sequences supported by the engine."""
        return self.cache_seq_interface.info.max_batch_size

    @torch.inference_mode()
    def forward(
        self,
        scheduled_requests: ScheduledRequests,
        resource_manager: ResourceManager,
        new_tensors_device: Optional[torch.Tensor] = None,
        gather_context_logits: bool = False,
        cache_indirection_buffer: Optional[torch.Tensor] = None,
    ):
        """Run forward from scheduled requests; main entrypoint that gets called by the executor."""

        spec_metadata = None
        if self.run_with_spec_decode:
            spec_resource_manager = resource_manager.get_resource_manager(
                ResourceManagerType.SPEC_RESOURCE_MANAGER
            )

            batch_size = scheduled_requests.batch_size
            max_num_tokens = get_max_num_tokens(
                self.llm_args.max_num_tokens, self.llm_args.max_seq_len, batch_size
            )

            target_model_config, _ = self.ad_config.create_factory()._get_model_config()

            spec_metadata = get_spec_metadata(
                spec_config=self.spec_config,
                model_config=target_model_config,
                max_num_requests=batch_size,
                max_num_tokens=max_num_tokens,
                spec_resource_manager=spec_resource_manager,
                is_draft_model=False,
            )

        # convert requests and store in sequence info object

        last_logit_only = self._prepare_inputs(
            scheduled_requests, resource_manager, new_tensors_device, spec_metadata
        )

        print("[TRACE] In forward(). Finished preparing inputs. Now computing logits.")
        # compute all logits
        logits = self._compute_logits()

        print(f"[TRACE] logits: {logits}")
        print(f"length of logits: {len(logits)}")

        print(
            "[TRACE] In forward(). Finished computing logits. Now gathering and concatenating logits."
        )

        print(f"[TRACE] last_logit_only: {last_logit_only}")
        print("About to gather and concatenate logits.")
        # gather+cat logits
        logits_flat = torch.cat(
            [ls_one_seq[-last_only:] for ls_one_seq, last_only in zip(logits, last_logit_only)],
            dim=0,
        )

        print(
            f"[TRACE] In forward(). Finished gathering and concatenating logits. Flattened logits: {logits_flat}"
        )
        return {"logits": logits_flat}


def create_draft_model_engine_maybe(
    ad_config: LlmArgs,
    engine,
    dist_mapping: Mapping,
    mpi_dist: MPIDist,
    max_draft_len: int,
    max_total_draft_tokens: int,
) -> Optional[PyTorchModelEngine]:
    """Create a draft model engine for speculative decoding.

    Args:
        ad_config: The AutoDeploy LLM configuration
        engine: The target model engine (ADEngine)
        dist_mapping: The distributed mapping configuration
        mpi_dist: The MPI distribution object
        max_draft_len: Maximum draft length for speculative decoding
        max_total_draft_tokens: Maximum total draft tokens

    Returns:
        PyTorchModelEngine configured as a draft model, or None if not needed
    """
    spec_config = ad_config.speculative_config
    if spec_config is None or not spec_config.spec_dec_mode.has_draft_model():
        return None

    has_spec_drafter = spec_config.spec_dec_mode.has_spec_drafter()
    print(f"[TRACE] Has spec drafter: {has_spec_drafter}")

    draft_spec_config = copy.copy(spec_config)

    use_chain_drafter = (
        ad_config.guided_decoding_backend is None
        and draft_spec_config._allow_chain_drafter
        and draft_spec_config._allow_greedy_draft_tokens
        and ad_config.attn_backend == "TRTLLM"
    )

    print(f"[TRACE] Use chain drafter: {use_chain_drafter}")
    print("[TRACE] ad_config.guided_decoding_backend: ", ad_config.guided_decoding_backend)
    print(
        "[TRACE] draft_spec_config._allow_chain_drafter: ",
        draft_spec_config._allow_chain_drafter,
    )
    print(
        "[TRACE] draft_spec_config._allow_greedy_draft_tokens: ",
        draft_spec_config._allow_greedy_draft_tokens,
    )
    print("[TRACE] ad_config.attn_backend: ", ad_config.attn_backend)
    if use_chain_drafter:

        def drafting_loop_wrapper(model):
            from tensorrt_llm._torch.speculative.drafting_loops import ChainDrafter

            return ChainDrafter(max_draft_len, max_total_draft_tokens, model)
    else:
        drafting_loop_wrapper = None

    enable_chunked_context = ad_config.enable_chunked_prefill
    if ad_config.attn_backend == "FLASHINFER_STAR_ATTENTION":
        enable_chunked_context = False

    kv_cache_config = ad_config.kv_cache_config

    # chunk_unit_size may be changed to 64 when using flash mla
    attn_runtime_features = AttentionRuntimeFeatures(
        chunked_prefill=enable_chunked_context,
        cache_reuse=kv_cache_config.enable_block_reuse,
        has_speculative_draft_tokens=has_spec_drafter,
        chunk_size=engine.llm_args.max_num_tokens,
    )

    # Construct TorchLlmArgs for the draft model
    draft_llm_args = construct_draft_llm_args(
        ad_config=ad_config,
        draft_spec_config=draft_spec_config,
        enable_chunked_context=enable_chunked_context,
    )

    print(
        f"[AUTODEPLOY] Creating draft_model_engine with model_path: {draft_spec_config.speculative_model_dir}"
    )
    print(f"[TRACE] Dist: {mpi_dist}")

    print(
        f"[TRACE] Calling draft_model_engine constructor in create_autodeploy_executor "
        f"with the following arguments:\n"
        f"[TRACE] model_path: {draft_spec_config.speculative_model_dir}\n"
        f"[TRACE] llm_args: {draft_llm_args}\n"
        f"[TRACE] mapping: {dist_mapping}\n"
        f"[TRACE] attn_runtime_features: {attn_runtime_features}\n"
        f"[TRACE] dist: {mpi_dist}\n"
        f"[TRACE] spec_config: {draft_spec_config}\n"
        f"[TRACE] is_draft_model: True"
    )
    draft_model_engine = PyTorchModelEngine(
        model_path=draft_spec_config.speculative_model_dir,
        llm_args=draft_llm_args,
        mapping=dist_mapping,
        attn_runtime_features=attn_runtime_features,
        dist=mpi_dist,
        spec_config=draft_spec_config,
        is_draft_model=True,
        drafting_loop_wrapper=drafting_loop_wrapper,
    )

    # For DeepseekV3 MTP, we need to set the num_hidden_layers to 1 for the draft model
    if draft_spec_config.spec_dec_mode.is_mtp_eagle():
        draft_model_engine.model.model_config.pretrained_config.num_hidden_layers = 1

    draft_model_engine.kv_cache_manager_key = ResourceManagerType.DRAFT_KV_CACHE_MANAGER

    # The goal here is to share the embed_tokens and lm_head weights between the draft model and the target model.
    # This is done by referencing the submodules from the target model in the draft model.
    # This code expects that the submodules in the target model have a forward() function,
    # which is not the case when the target model is an ADEngine.
    # We need to ensure compatibility with this case.

    assert ad_config.speculative_config.spec_dec_mode.is_draft_target(), (
        "Currently, the code is only supported for draft target mode."
    )

    # If the model is an MLA model, we need to set
    # draft_model_engine.attn_runtime_features.chunked_prefill to False
    target_model_config = engine.model_config
    print(f"[TRACE] Engine model pretrained config: {target_model_config}")
    print(f"[TRACE] Is MLA: {is_mla(target_model_config)}")

    if is_mla(target_model_config):
        draft_model_engine.attn_runtime_features.chunked_prefill = False

    return draft_model_engine


def create_autodeploy_executor(ad_config: LlmArgs, tokenizer: Optional[TokenizerBase] = None):
    """Create an AutoDeploy executor from the given configuration and tokenizer.
    The tokenizer is required for guided decoding.

    This is the entrypoint API to the _autodeploy backend.
    """
    # initialize process groups
    world_size = mpi_world_size()
    rank = mpi_rank()
    print(f"[TRACE] World size: {world_size}")
    print(f"[TRACE] Rank: {rank}")
    dist_mapping = Mapping(rank=rank, world_size=world_size, tp_size=world_size)
    print(f"[TRACE] Dist mapping: {dist_mapping}")
    mpi_dist = MPIDist(dist_mapping)
    print(f"[TRACE] MPI dist: {mpi_dist}")
    ad_logger.set_rank(rank)
    torch.cuda.set_device(rank)
    port = mpi_dist.broadcast(dist.get_free_port())  # use MPI broadcast to pick a free port
    dist.initialize_or_skip(rank, world_size, port)

    # some config
    assert ad_config.max_beam_width <= 1, "_autodeploy + beam_search is not supported"

    max_num_sequences = ad_config.max_batch_size * dist_mapping.pp_size
    # some derivative properties
    max_draft_len = (
        0 if ad_config.speculative_config is None else ad_config.speculative_config.max_draft_len
    )
    max_total_draft_tokens = (
        0
        if ad_config.speculative_config is None
        else ad_config.speculative_config.max_total_draft_tokens
    )

    print(f"[TRACE] Max draft len: {max_draft_len}")
    print(f"[TRACE] Max total draft tokens: {max_total_draft_tokens}")

    # initialize model engine
    engine = ADEngine.build_from_config(ad_config=ad_config)

    draft_model_engine = create_draft_model_engine_maybe(
        ad_config=ad_config,
        engine=engine,
        dist_mapping=dist_mapping,
        mpi_dist=mpi_dist,
        max_draft_len=max_draft_len,
        max_total_draft_tokens=max_total_draft_tokens,
    )
    print("[TRACE] AD Config: ", ad_config)

    # check kvcache config for partial block reuse
    # TODO: copy_on_partial_reuse is not supported yet, see
    # https://github.com/NVIDIA/TensorRT-LLM/issues/7142 for more details.
    enable_block_reuse = ad_config.kv_cache_config.enable_block_reuse
    enable_partial_reuse = ad_config.kv_cache_config.enable_partial_reuse
    copy_on_partial_reuse = ad_config.kv_cache_config.copy_on_partial_reuse
    if enable_block_reuse and enable_partial_reuse and copy_on_partial_reuse:
        raise RuntimeError(
            f"partial block reuse with {copy_on_partial_reuse=} set to True is NOT supported"
            " in AutoDeploy. Please set it to False via the kv_cache_config.copy_on_partial_reuse "
            "field in tensorrt_llm._torch.auto_deploy.llm_args.LlmArgs."
        )

    # TODO: detect whether SSM layer is present in the model and raise an error or disable block
    # reuse with a warning --> see https://github.com/NVIDIA/TensorRT-LLM/issues/7142. For now, we
    # just emit a general warning.
    if enable_block_reuse:
        ad_logger.warning(
            f"{enable_block_reuse=} is enabled. Note that this is not supported for SSM layers and"
            " may lead to incorrect results if the model contains SSM layers."
        )

    # resource managers
    print(
        f"[TRACE] engine.cache_seq_interface.info.num_pages: {engine.cache_seq_interface.info.num_pages}"
    )
    print(f"[TRACE] ad_config.attn_page_size: {ad_config.attn_page_size}")
    print(f"[TRACE] ad_config.max_seq_len: {ad_config.max_seq_len}")
    print(f"[TRACE] ad_config.max_batch_size: {ad_config.max_batch_size}")
    kv_cache_manager = _CacheManagerWithFakePool(
        ad_config.kv_cache_config,
        num_blocks=engine.cache_seq_interface.info.num_pages,
        tokens_per_block=ad_config.attn_page_size,
        max_seq_len=ad_config.max_seq_len,
        max_batch_size=ad_config.max_batch_size,
    )
    seq_slot_manager = SeqSlotManager(max_num_sequences=max_num_sequences)

    draft_kv_cache_manager = create_draft_kv_cache_manager_maybe(
        draft_model_engine, ad_config, dist_mapping
    )

    spec_resource_manager = create_spec_resource_manager(
        engine=engine,
        draft_model_engine=draft_model_engine,
    )

    resource_manager = ResourceManager(
        {
            ResourceManagerType.KV_CACHE_MANAGER: kv_cache_manager,
            ResourceManagerType.DRAFT_KV_CACHE_MANAGER: draft_kv_cache_manager,
            ResourceManagerType.SEQ_SLOT_MANAGER: seq_slot_manager,
            ResourceManagerType.SPEC_RESOURCE_MANAGER: spec_resource_manager,
        }
    )

    resource_manager.resource_managers.move_to_end(ResourceManagerType.KV_CACHE_MANAGER, last=True)

    # TODO: consider passing through scheduler_config arguments here. Not doing this for now since
    # it requires correctly setting up the C++ pybind scheduler config from the LLMArgs and then
    # processing the arguments here...

    # Chunked prefill
    if ad_config.enable_chunked_prefill:
        chunk_unit_size = ad_config.attn_page_size
        chunking_policy = ContextChunkingPolicy.FIRST_COME_FIRST_SERVED
        ctx_chunk_config: Tuple[StrEnum, int] = (chunking_policy, chunk_unit_size)
    else:
        ctx_chunk_config = None

    # scheduling
    capacitor_scheduler = BindCapacityScheduler(
        max_num_requests=ad_config.max_batch_size,
        kv_cache_manager=kv_cache_manager.impl,
        peft_cache_manager=None,
    )
    mb_scheduler = BindMicroBatchScheduler(
        max_batch_size=ad_config.max_batch_size,
        max_num_tokens=engine.llm_args.max_num_tokens,
        ctx_chunk_config=ctx_chunk_config,
    )
    scheduler = SimpleScheduler(capacitor_scheduler, mb_scheduler)

    # search sampler with speculative decoding
    sampler_args = TorchSampler.Args(
        max_seq_len=ad_config.max_seq_len,
        max_draft_len=max_draft_len,
        max_total_draft_tokens=max_total_draft_tokens,
        max_num_sequences=max_num_sequences,
        max_beam_width=ad_config.max_beam_width,
    )
    sampler = TorchSampler(sampler_args)

    # Guided (structured) decoding.
    guided_decoder = None
    if (
        (guided_decoding_backend := ad_config.guided_decoding_backend) is not None
    ) and dist_mapping.is_last_pp_rank():
        vocab_size_padded = engine.cache_seq_interface.info.vocab_size_padded
        if vocab_size_padded is None:
            raise RuntimeError(
                "Could not determine the vocabulary size. Required for guided decoding."
            )
        guided_decoding_config = get_guided_decoding_config(
            guided_decoding_backend=guided_decoding_backend, tokenizer=tokenizer
        )
        guided_decoder = GuidedDecoder(
            guided_decoding_config=guided_decoding_config,
            max_num_sequences=ad_config.max_batch_size,
            vocab_size_padded=vocab_size_padded,
        )

    drafter = get_spec_drafter(
        model_engine=engine,
        draft_model_engine=draft_model_engine,
        sampler=sampler,
        spec_resource_manager=spec_resource_manager,
    )

    print(f"[TRACE] Drafter in AutoDeploy executor: {drafter}")

    # creating the executor object
    py_executor = PyExecutor(
        resource_manager,
        scheduler,
        model_engine=engine,
        sampler=sampler,
        dist=mpi_dist,
        max_num_sequences=max_num_sequences,
        disable_overlap_scheduler=ad_config.disable_overlap_scheduler,
        max_input_len=ad_config.max_input_len,
        max_batch_size=ad_config.max_batch_size,
        max_draft_len=max_draft_len,
        max_total_draft_tokens=max_total_draft_tokens,
        max_beam_width=ad_config.max_beam_width,
        guided_decoder=guided_decoder,
        drafter=drafter,
    )
    return py_executor
