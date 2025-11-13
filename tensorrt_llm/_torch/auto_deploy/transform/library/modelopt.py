from __future__ import annotations

from typing import Tuple

import torch
from torch.fx import GraphModule

from tensorrt_llm._torch.auto_deploy.utils.pattern_matcher import (
    ADPatternMatcherPass,
    register_ad_pattern,
)

from ..interface import BaseTransform, TransformInfo, TransformRegistry


def _fake_quant_mo_linear_pattern_fp8(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    amax_in: torch.Tensor,
    amax_w: torch.Tensor,
) -> torch.Tensor:
    a_in = torch.ops.aten.detach.default(amax_in)
    a_w = torch.ops.aten.detach.default(amax_w)
    q_in = torch.ops.tensorrt.quantize_op.default(x, a_in, 8, 4, False, False)
    q_w = torch.ops.tensorrt.quantize_op.default(w, a_w, 8, 4, False, False)
    out = torch.ops.auto_deploy.torch_linear_simple.default(q_in, q_w, b)
    return out


def _fake_quant_mo_linear_repl_fp8(
    x: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    amax_in: torch.Tensor,
    amax_w: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.auto_deploy.torch_fake_quant_mo_linear.default(
        x, w, b, amax_in, amax_w, 8, 4, False, False
    )


@TransformRegistry.register("match_fake_quant_mo_linear")
class MatchFakeQuantMOLinear(BaseTransform):
    """
    Replace:
        detach(amax_in)  -> trt.quantize_op(x,  amax_in, nb, gs, u, nr)
        detach(amax_w)   -> trt.quantize_op(w,  amax_w,  nb, gs, u, nr)
        -> auto_deploy.torch_linear_simple(qx, qw, b)

    with:
        auto_deploy.torch_fake_quant_mo_linear(
            x, w, b, amax_in, amax_w, nb, gs, u, nr
        )
    """

    def _apply(
        self,
        gm: GraphModule,
        cm,
        factory,
        shared_config,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph
        patterns = ADPatternMatcherPass()
        In = 3072
        Out = 18432
        dummy_args = [
            torch.randn(1, In, device="meta", dtype=torch.bfloat16),  # x
            torch.randn(Out, In, device="meta", dtype=torch.bfloat16),  # w
            torch.randn(Out, device="meta", dtype=torch.bfloat16),  # b
            torch.randn((), device="meta", dtype=torch.bfloat16),  # amax_in (scalar tensor)
            torch.randn((), device="meta", dtype=torch.bfloat16),  # amax_w  (scalar tensor)
        ]

        register_ad_pattern(
            search_fn=_fake_quant_mo_linear_pattern_fp8,
            replace_fn=_fake_quant_mo_linear_repl_fp8,
            patterns=patterns,
            dummy_args=dummy_args,
            op_ignore_types={},
            scalar_workaround={},
        )

        num_matches = patterns.apply(graph)

        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        return gm, info
