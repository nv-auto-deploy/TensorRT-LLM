# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Device-resident greedy decode fast path for the AutoDeploy TorchSampler.

Steady-state single-beam greedy decode is host-cadence-bound at small batch: the
generic ``TorchSampler.sample_async`` spends hundreds of microseconds per token on
per-step pinned-tensor staging, step-indexer arithmetic, and the device
finish-reasons kernel swarm, all to select one argmax token per request. On
multi-rank TP deployments every rank re-runs that host tail per token, which
delays the next captured-graph replay launch and turns into spin-wait inside the
first allreduce of the next decoder body.

``ADGreedyDecodeTorchSampler`` adds a tightly gated fast path for batches where
every scheduled request is a plain greedy generation request (no beam search, no
draft tokens, no logprobs, no stop words, no min-length / embedding-bias / d2t).
The fast path keeps token selection device-resident (one argmax + one scatter
into the persistent ``store.new_tokens`` buffer consumed by the overlap
scheduler's next-input gather), issues a single async D2H copy into a persistent
pinned mirror, and defers stop-criteria evaluation to the host in
``update_requests`` via the pre-existing ``_handle_stop_criteria`` helper, which
implements the same END_ID > LENGTH > STOP_WORDS precedence as the device
finish-reasons writer. Any batch that does not qualify falls back to the generic
TorchSampler implementation unchanged.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch

from tensorrt_llm._torch.pyexecutor.llm_request import (
    LlmRequest,
    LlmRequestState,
    get_draft_token_length,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManager
from tensorrt_llm._torch.pyexecutor.sampler import (
    DEFAULT_BEAM_IDX,
    SampleStateTensors,
    SampleStateTensorsHostTorch,
    SampleStateTorch,
    TorchSampler,
    _request_strategy,
    add_token,
)
from tensorrt_llm._torch.pyexecutor.sampling_utils import GREEDY
from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests
from tensorrt_llm._utils import prefer_pinned

__all__ = ["ADFastGreedySampleState", "ADGreedyDecodeTorchSampler"]


@dataclass(kw_only=True)
class ADFastGreedySampleState(SampleStateTorch):
    """Marker state emitted by the fast greedy path; routed in ``update_requests``."""


class ADGreedyDecodeTorchSampler(TorchSampler):
    """TorchSampler with a low-overhead path for steady greedy decode batches.

    Falls back to the generic TorchSampler for any request/batch the fast path
    does not cover, so generic sampling, streaming, cancellation, and
    multi-request behavior are preserved.
    """

    # Strategy probe: greediness does not depend on vocab size (same probe value
    # as TorchSampler._can_use_fast_greedy_path).
    _STRATEGY_PROBE_VOCAB_SIZE = 2**31

    def __init__(self, args: TorchSampler.Args):
        super().__init__(args)
        # Cache keyed on the batch's seq-slot tuple; steady decode reuses it and
        # performs zero per-step host tensor creation for the scatter indices.
        self._fast_slots_key: Optional[tuple] = None
        self._fast_slots_cuda: Optional[torch.Tensor] = None
        # Persistent pinned mirror of store.new_tokens (one async D2H per step).
        self._fast_new_tokens_host: Optional[torch.Tensor] = None
        # The pinned mirror belongs to one returned fast state until that state
        # is consumed. Fall back to the generic path rather than let a second
        # sample overwrite an outstanding snapshot.
        self._fast_state_pending = False

    def _fast_greedy_batch(
        self, scheduled_requests: ScheduledRequests, model_outputs: dict[str, Any]
    ) -> Optional[list[LlmRequest]]:
        """Return the generation requests iff the whole batch qualifies, else None."""
        if self._fast_state_pending or self.max_beam_width > 1 or self.async_worker_enabled():
            return None
        # d2t token translation belongs to the speculative one-model path.
        if "d2t" in model_outputs:
            return None
        # Context requests need generic logits selection and store setup.
        if scheduled_requests.context_requests:
            return None
        requests = scheduled_requests.generation_requests
        if not requests:
            return None
        logits = model_outputs["logits"]
        # One logits row per request, in scheduled order; CUDA-graph padding rows
        # (appended after the real requests) may trail and are ignored.
        if logits.dim() != 2 or logits.shape[0] < len(requests):
            return None
        for req in requests:
            if (
                req.is_dummy
                or req.py_is_draft
                or req.py_seq_slot is None
                or get_draft_token_length(req) > 0
                or req.py_return_log_probs
                or req.py_return_generation_logits
                or req.py_stop_words_list
                or req.py_min_length
                or req._py_embedding_bias_1d is not None
                or _request_strategy(req, vocab_size=self._STRATEGY_PROBE_VOCAB_SIZE) != GREEDY
            ):
                return None
        return requests

    @torch.inference_mode()
    def sample_async(
        self,
        scheduled_requests: ScheduledRequests,
        model_outputs: dict[str, Any],
        num_context_logits_prefix_sum: list[int],
        resource_manager: Optional[ResourceManager] = None,
    ) -> SampleStateTorch:
        requests = self._fast_greedy_batch(scheduled_requests, model_outputs)
        if requests is None:
            return super().sample_async(
                scheduled_requests,
                model_outputs,
                num_context_logits_prefix_sum,
                resource_manager,
            )

        # Keep new-request store bookkeeping identical to the generic path. For a
        # steady generation-only batch this early-returns without staging work.
        self.setup_sampler_step(scheduled_requests)

        new_tokens = self.store.new_tokens
        slots = tuple(req.py_seq_slot for req in requests)
        if slots != self._fast_slots_key:
            self._fast_slots_cuda = torch.tensor(slots, dtype=torch.int64, device=new_tokens.device)
            self._fast_slots_key = slots

        logits = model_outputs["logits"]
        next_tokens = torch.argmax(logits[: len(requests)], dim=-1)
        # Same buffer the generic path fills: the overlap scheduler gathers the
        # next step's input ids from it, so the device-side handoff is unchanged.
        new_tokens[0, :, 0].scatter_(0, self._fast_slots_cuda, next_tokens.to(new_tokens.dtype))

        if self._fast_new_tokens_host is None:
            self._fast_new_tokens_host = torch.empty_like(
                new_tokens, device="cpu", pin_memory=prefer_pinned()
            )
        self._fast_new_tokens_host.copy_(new_tokens, non_blocking=True)

        sampler_event = self._record_sampler_event()
        state = ADFastGreedySampleState(
            requests=requests,
            device=SampleStateTensors(new_tokens=new_tokens),
            host=SampleStateTensorsHostTorch(
                new_tokens=self._fast_new_tokens_host,
                finish_reasons=None,
                first_finish_reasons=None,
                logprobs_state=None,
            ),
            sampler_event=sampler_event,
        )
        self._fast_state_pending = True
        return state

    def update_requests(
        self,
        state: SampleStateTorch,
        resource_manager: Optional[ResourceManager] = None,
    ) -> None:
        if not isinstance(state, ADFastGreedySampleState):
            super().update_requests(state, resource_manager)
            return

        if state.sampler_event:
            state.sampler_event.synchronize()
        assert state.host is not None
        new_tokens_list = state.host.new_tokens.tolist()
        for req in state.requests:
            if req.state == LlmRequestState.GENERATION_COMPLETE:
                continue
            new_token = add_token(req, new_tokens_list, beam_idx=DEFAULT_BEAM_IDX)
            # Same END_ID > LENGTH > STOP_WORDS precedence as the device
            # finish-reasons writer; stop words are excluded by eligibility.
            self._handle_stop_criteria(
                req, new_token, max_seq_len=self.max_seq_len, beam_idx=DEFAULT_BEAM_IDX
            )
            req.py_num_accepted_draft_tokens = 0
            req.py_rewind_len = 0
            req.py_decoding_iter += 1
        self._fast_state_pending = False
