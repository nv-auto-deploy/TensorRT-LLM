from functools import partial
from typing import Literal, Tuple

import torch
import torch.nn as nn
from pydantic import Field
from torch.fx import GraphModule, Node

from ...utils.module import get_submodule_of_param
from ...utils.node_utils import is_op
from ...utils.pattern_matcher import ADPatternMatcherPass, register_ad_pattern
from ..interface import BaseTransform, TransformConfig, TransformInfo, TransformRegistry


def _moe_dense_mlp_pattern(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_w: torch.Tensor,
    gate_up_b: torch.Tensor,
    down_w: torch.Tensor,
    down_b: torch.Tensor,
    alpha: float = 1.0,
    limit: float = 10.0,
    minus_limit: float = -10.0,
) -> torch.Tensor:
    batch_size = hidden_states.shape[0]
    hidden_size = hidden_states.shape[2]
    hidden_states = hidden_states.reshape(-1, hidden_size)  # (num_tokens, hidden_size)
    num_experts = routing_weights.shape[1]

    hidden_states = hidden_states.repeat(num_experts, 1)
    hidden_states = hidden_states.view(num_experts, -1, hidden_size)
    gate_up = torch.bmm(hidden_states, gate_up_w) + gate_up_b.unsqueeze(-2)
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=minus_limit, max=limit)
    glu = gate * torch.sigmoid(gate * alpha)
    next_states = torch.bmm(((up + 1) * glu), down_w)
    next_states = next_states + down_b.unsqueeze(-2)
    next_states = next_states.view(num_experts, batch_size, -1, hidden_size)
    next_states = (
        next_states * routing_weights.transpose(0, 1).view(num_experts, batch_size, -1)[..., None]
    )
    next_states = next_states.sum(dim=0)  # [B, S, H]
    return next_states


def _moe_dense_mlp_repl(
    hidden_states: torch.Tensor,
    routing_weights: torch.Tensor,
    gate_up_w: torch.Tensor,
    gate_up_b: torch.Tensor,
    down_w: torch.Tensor,
    down_b: torch.Tensor,
    alpha: float,
    limit: float,
    minus_limit: float,
) -> torch.Tensor:
    return torch.ops.auto_deploy.torch_moe_dense_mlp(
        hidden_states, routing_weights, gate_up_w, gate_up_b, down_w, down_b, alpha, limit
    )


@TransformRegistry.register("match_dense_moe_pattern")
class MatchMOEDenseMLP(BaseTransform):
    def _apply(
        self,
        gm: GraphModule,
        cm,
        factory,
        shared_config,
    ) -> Tuple[GraphModule, TransformInfo]:
        graph = gm.graph
        patterns = ADPatternMatcherPass()

        B, S, H = 2, 4, 8  # batch, seq, hidden
        E, In = 3, 16  # experts, intermediate (I); gate_up has 2I
        T = B * S

        dummy_args = [
            torch.randn(B, S, H, device="meta", dtype=torch.float16),  # hidden_states
            torch.randn(T, E, device="meta", dtype=torch.float16),  # routing_weights
            torch.randn(E, H, 2 * In, device="meta", dtype=torch.float16),  # gate_up_w  [E,H,2I]
            torch.randn(E, 2 * In, device="meta", dtype=torch.float16),  # gate_up_b  [E,2I]
            torch.randn(E, In, H, device="meta", dtype=torch.float16),  # down_w     [E,I,H]
            torch.randn(E, H, device="meta", dtype=torch.float16),  # down_b     [E,H]
            1.07,
            10.1,
            -10.1,
        ]

        op_ignore_types = {
            torch.ops.aten.view.default: (int,),
            torch.ops.aten.reshape.default: (int,),
            torch.ops.auto_deploy.view.default: (int,),
            torch.ops.aten.repeat.default: (int,),
            torch.ops.aten.slice.Tensor: (int,),
            torch.ops.aten.unsqueeze.default: (int,),
            torch.ops.aten.transpose.int: (int,),
        }

        scalar_workaround = {"alpha": 1.07, "limit": 10.1, "minus_limit": -10.1}

        register_ad_pattern(
            search_fn=_moe_dense_mlp_pattern,
            replace_fn=_moe_dense_mlp_repl,
            patterns=patterns,
            dummy_args=dummy_args,
            op_ignore_types=op_ignore_types,
            scalar_workaround=scalar_workaround,
        )

        num_matches = patterns.apply(graph)
        info = TransformInfo(
            skipped=False,
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        return gm, info


def _get_alpha_limit_from_dense(node: Node) -> Tuple[float, float]:
    # torch_moe_dense_mlp(hidden, routing, gu_w, gu_b, dn_w, dn_b, alpha, limit)
    # alpha/limit may be in args or kwargs
    alpha = node.kwargs.get("alpha", None)
    limit = node.kwargs.get("limit", None)
    if alpha is None:
        alpha = float(node.args[6]) if len(node.args) >= 7 else 1.0
    if limit is None:
        limit = float(node.args[7]) if len(node.args) >= 8 else 10.0
    return float(alpha), float(limit)


def _get_topk_from_router(node: Node) -> int:
    # torch_moe_router(hidden, weight, bias, top_k=2)
    if "top_k" in node.kwargs:
        return int(node.kwargs["top_k"])
    return int(node.args[3]) if len(node.args) >= 4 else 2


def _get_packed_mxfp4_expert_layout(qcfg: dict):
    checkpoint_layout = qcfg.get("checkpoint_layout")
    if checkpoint_layout is None:
        return None
    return getattr(checkpoint_layout, "packed_mxfp4_experts", None)


class MXFP4MLPConfig(TransformConfig):
    mxfp4_backend: Literal["auto", "torch", "triton"] = Field(
        default="auto",
        description=(
            "Backend for packed MXFP4 MoE lowering. 'auto' selects the torch reference path "
            "for checkpoint-layout packed experts and Triton for legacy quant_method=mxfp4."
        ),
    )


def _get_mxfp4_moe_op(backend: Literal["torch", "triton"]):
    if backend == "torch":
        return torch.ops.auto_deploy.torch_mxfp4_moe
    return torch.ops.auto_deploy.triton_mxfp4_moe


def _layout_layer_from_names(checkpoint_layout: object, source_names: dict[str, str]):
    for source_name in source_names.values():
        layer = checkpoint_layout.layer_from_runtime_name(source_name)
        if layer is not None:
            return layer
    return None


def _load_mxfp4_expert_layout_hook(
    state_dict,
    prefix,
    *args,
    checkpoint_layout: object,
    target_names: dict[str, str],
    layer: int,
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
) -> None:
    prefix = prefix or ""
    checkpoint_layout.load_runtime_buffers(
        state_dict,
        prefix,
        layer=layer,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        target_gate_up_blocks=target_names["gate_up_blocks"],
        target_gate_up_scales=target_names["gate_up_scales"],
        target_down_blocks=target_names["down_blocks"],
        target_down_scales=target_names["down_scales"],
        num_experts=num_experts,
    )


def _get_mxfp4_expert_dims(
    gm: GraphModule,
    gate_up_w_name: str,
    gate_up_b_name: str,
    down_b_name: str,
) -> Tuple[int, int, int]:
    gu_b = gm.get_parameter(gate_up_b_name)  # [E, 2I]
    gu_w = gm.get_parameter(gate_up_w_name)  # [E, H, 2I] or [E, 2I, H]
    dn_b = gm.get_parameter(down_b_name)  # [E, H]

    E = int(gu_b.shape[0])
    I2 = int(gu_b.shape[1])
    In = I2 // 2

    assert gu_w.dim() == 3, "gate_up_w must be rank-3"
    if gu_w.shape[1] == I2:
        H = int(gu_w.shape[2])
    else:
        H = int(dn_b.shape[1])
    return E, H, In


def _register_mxfp4_expert_params(
    gm: GraphModule,
    gate_up_w_name: str,
    gate_up_b_name: str,
    down_w_name: str,
    down_b_name: str,
) -> Tuple[str, str, str, str]:
    """Create (if missing) the four MXFP4 params under the experts module and return their full names.

    Returns:
      (gu_blocks_name, gu_scales_name, dn_blocks_name, dn_scales_name)
    """
    E, H, In = _get_mxfp4_expert_dims(gm, gate_up_w_name, gate_up_b_name, down_b_name)

    # Compute block dims (assume divisible; zero-init anyway)
    H_blk = max(1, H // 32)
    I_blk = max(1, In // 32)

    experts_mod, experts_path, _ = get_submodule_of_param(gm, gate_up_w_name)

    # New param names under experts module
    gu_blocks_name = "gate_up_proj_blocks"
    gu_scales_name = "gate_up_proj_scales"
    dn_blocks_name = "down_proj_blocks"
    dn_scales_name = "down_proj_scales"

    # Zero-init tensors (uint8 for blocks/scales)
    gu_blocks = torch.zeros((E, 2 * In, H_blk, 16), dtype=torch.uint8)
    gu_scales = torch.zeros((E, 2 * In, H_blk), dtype=torch.uint8)
    dn_blocks = torch.zeros((E, H, I_blk, 16), dtype=torch.uint8)
    dn_scales = torch.zeros((E, H, I_blk), dtype=torch.uint8)

    experts_mod.register_parameter(gu_blocks_name, nn.Parameter(gu_blocks, requires_grad=False))
    experts_mod.register_parameter(gu_scales_name, nn.Parameter(gu_scales, requires_grad=False))
    experts_mod.register_parameter(dn_blocks_name, nn.Parameter(dn_blocks, requires_grad=False))
    experts_mod.register_parameter(dn_scales_name, nn.Parameter(dn_scales, requires_grad=False))

    # Full GM attribute paths for new params
    prefix = (experts_path + ".") if experts_path else ""
    return (
        prefix + gu_blocks_name,
        prefix + gu_scales_name,
        prefix + dn_blocks_name,
        prefix + dn_scales_name,
    )


@TransformRegistry.register("quantize_mxfp4_moe")
class InsertMXFP4MLP(BaseTransform):
    """
    Replace (torch_moe_router -> torch_moe_dense_mlp) with a single MXFP4 MoE op,
    and register MXFP4 expert params (blocks + scales) on the experts module.
    """

    algo_name: str = "mxfp4"
    config: MXFP4MLPConfig

    @classmethod
    def get_config_class(cls):
        return MXFP4MLPConfig

    def _resolve_backend(
        self,
        qcfg: dict,
        checkpoint_layout: object | None,
    ) -> Literal["torch", "triton"]:
        backend = self.config.mxfp4_backend
        if backend != "auto":
            return backend
        if checkpoint_layout is not None or qcfg.get("expert_quant_method") == self.algo_name:
            return "torch"
        return "triton"

    def _apply(
        self,
        gm: GraphModule,
        cm,
        factory,
        shared_config,
    ) -> Tuple[GraphModule, TransformInfo]:
        qcfg = factory.get_quant_config()
        if not qcfg:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )
        checkpoint_layout = _get_packed_mxfp4_expert_layout(qcfg)
        if qcfg.get("quant_method", "") != self.algo_name and checkpoint_layout is None:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )
        num_matches = 0
        mxfp4_backend = self._resolve_backend(qcfg, checkpoint_layout)
        mxfp4_op = _get_mxfp4_moe_op(mxfp4_backend)

        for n in list(gm.graph.nodes):
            if not is_op(n, torch.ops.auto_deploy.torch_moe_dense_mlp):
                continue

            # Expect: torch_moe_dense_mlp(hidden, routing, gu_w, gu_b, dn_w, dn_b, alpha, limit)
            if len(n.args) < 6:
                continue

            hidden_node = n.args[0]
            routing_node = n.args[1]
            gate_up_w_node = n.args[2]
            gate_up_b_node = n.args[3]
            down_w_node = n.args[4]
            down_b_node = n.args[5]

            if not isinstance(routing_node, Node) or not is_op(
                routing_node, torch.ops.auto_deploy.torch_moe_router
            ):
                continue

            # Router params: weight, bias, top_k
            router_weight_node = routing_node.args[1]
            router_bias_node = routing_node.args[2]
            top_k = _get_topk_from_router(routing_node)

            # Resolve parameter names so we can find the experts module
            if gate_up_w_node.op != "get_attr" or gate_up_b_node.op != "get_attr":
                continue
            if down_w_node.op != "get_attr" or down_b_node.op != "get_attr":
                continue

            gu_w_name = gate_up_w_node.target
            gu_b_name = gate_up_b_node.target
            dn_w_name = down_w_node.target
            dn_b_name = down_b_node.target
            source_names = {
                "gate_up_weight": gu_w_name,
                "gate_up_bias": gu_b_name,
                "down_weight": dn_w_name,
                "down_bias": dn_b_name,
            }
            layer = None
            if checkpoint_layout is not None:
                layer = _layout_layer_from_names(checkpoint_layout, source_names)
                if layer is None:
                    continue
            num_experts, hidden_size, intermediate_size = _get_mxfp4_expert_dims(
                gm, gu_w_name, gu_b_name, dn_b_name
            )

            # Register MXFP4 params on experts
            gu_blocks_name, gu_scales_name, dn_blocks_name, dn_scales_name = (
                _register_mxfp4_expert_params(gm, gu_w_name, gu_b_name, dn_w_name, dn_b_name)
            )
            if checkpoint_layout is not None:
                gm._register_load_state_dict_pre_hook(
                    partial(
                        _load_mxfp4_expert_layout_hook,
                        checkpoint_layout=checkpoint_layout,
                        target_names={
                            "gate_up_blocks": gu_blocks_name,
                            "gate_up_scales": gu_scales_name,
                            "down_blocks": dn_blocks_name,
                            "down_scales": dn_scales_name,
                        },
                        layer=layer,
                        num_experts=num_experts,
                        hidden_size=hidden_size,
                        intermediate_size=intermediate_size,
                    )
                )

            # Alpha/limit (from dense call)
            alpha, limit = _get_alpha_limit_from_dense(n)

            # Insert the new get_attr nodes for MXFP4 params
            with gm.graph.inserting_before(n):
                gu_blocks_attr = gm.graph.create_node("get_attr", gu_blocks_name)
                gu_scales_attr = gm.graph.create_node("get_attr", gu_scales_name)
                dn_blocks_attr = gm.graph.create_node("get_attr", dn_blocks_name)
                dn_scales_attr = gm.graph.create_node("get_attr", dn_scales_name)

            n.target = mxfp4_op.default
            n.kwargs = {}

            # torch_mxfp4_moe/triton_mxfp4_moe(
            #   hidden_states,
            #   router_weight, router_bias, top_k,
            #   gate_up_blocks, gate_up_bias, gate_up_scales, alpha, limit,
            #   down_blocks, down_bias, down_scales)
            new_args = (
                hidden_node,
                router_weight_node,
                router_bias_node,
                top_k,
                gu_blocks_attr,
                gate_up_b_node,
                gu_scales_attr,
                float(alpha),
                float(limit),
                dn_blocks_attr,
                down_b_node,
                dn_scales_attr,
            )
            n.args = new_args

            # Remove the now-unneeded router node if nobody else uses it
            if len(routing_node.users) == 0:
                gm.graph.erase_node(routing_node)

            num_matches += 1

        info = TransformInfo(
            skipped=(num_matches == 0),
            num_matches=num_matches,
            is_clean=num_matches == 0,
            has_valid_shapes=num_matches == 0,
        )
        return gm, info
