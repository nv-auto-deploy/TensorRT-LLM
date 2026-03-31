# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import math
from functools import partial
from typing import Callable, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.fx import GraphModule, Node

from tensorrt_llm._torch.utils import ActivationType

from ...models.factory import ModelFactory
from ...shim.interface import CachedSequenceInterface
from ...utils.node_utils import is_op
from ...utils.quantization_utils import (
    is_mixed_precision_config,
    mixed_precision_has_algo,
    should_skip_mixed_precision_quantization,
    should_skip_quantization,
)
from ..interface import SharedConfig, TransformInfo, TransformRegistry
from .quantization import (
    FP8LinearQuantizationFromConfig,
    NVFP4LinearQuantizationFromConfig,
    Quantization,
)


def _quantize_moe_node(
    gm: GraphModule,
    node: Node,
    quant_impl: Quantization,
    quantized_op: Callable[..., Node],
):
    """
    Replace a torch.ops.auto_deploy.torch_moe node with its quantized version,
    quantizing each expert weight list and registering scales + hooks.
    Automatically handles different scale configurations per quantization type.
    """
    w1_names, w2_names, w3_names = _extract_moe_weight_param_lists(node)

    # Capture original weight arg nodes (positions 3-5) before the transform.
    # For the pre-pattern-match (per-expert stacked) case these are aten.stack nodes
    # whose get_attr inputs reference per-expert params we will delete.  Erase them
    # after the main replacement so that graph lint passes.
    orig_weight_arg_nodes: List[Node] = [
        n
        for n in (node.args[3], node.args[4], node.args[5])
        if isinstance(n, Node)
        and n.op == "call_function"
        and n.target is torch.ops.aten.stack.default
    ]

    scale_keys = quant_impl.scale_names()

    def quantize_experts(
        weight_names: List[str],
    ) -> Tuple[Node, List[List[Node]]]:
        """Quantize expert weights and return (stacked_weight_node, per-expert scale nodes).

        Handles two cases:
        - Single stacked parameter (post-pattern-match): one entry in ``weight_names``,
          the param is already shape ``(E, I, H)``.  Quantize in-place, create a
          ``get_attr`` node, and return it together with per-expert scale nodes.
        - Per-expert parameters (pre-pattern-match, from ``aten.stack`` inputs):
          multiple entries in ``weight_names``.  Quantize each expert, stack them
          into a new stacked parameter (per-expert params are removed), and return
          a single ``get_attr`` node for the stacked param.
        """
        scale_nodes_group: List[List[Node]] = []

        if len(weight_names) == 1:
            # Already-stacked case: quantize the single (E, I, H) param in-place
            name = weight_names[0]
            orig_weight = gm.get_parameter(name)
            new_weight = quant_impl.quantize_weight(orig_weight)

            modname, _, attrname = name.rpartition(".")
            submod = gm.get_submodule(modname)
            setattr(submod, attrname, nn.Parameter(new_weight, requires_grad=False))

            for scale_name, scale_val in quant_impl.default_scales(orig_weight.shape).items():
                submod.register_buffer(scale_name, scale_val)

            gm._register_load_state_dict_pre_hook(partial(quant_impl.load_hook, weight_name=name))

            with gm.graph.inserting_before(node):
                weight_node = gm.graph.get_attr(name)
                scales = [gm.graph.get_attr(modname + "." + s) for s in scale_keys]
                scale_nodes_group.append(scales)

            return weight_node, scale_nodes_group

        # Per-expert case: quantize each expert, then register a stacked param.
        # No per-expert weight get_attr nodes are created — only scale nodes.
        quant_weights: List[torch.Tensor] = []
        for name in weight_names:
            orig_weight = gm.get_parameter(name)
            new_weight = quant_impl.quantize_weight(orig_weight)

            modname, _, attrname = name.rpartition(".")
            submod = gm.get_submodule(modname)
            setattr(submod, attrname, nn.Parameter(new_weight, requires_grad=False))
            quant_weights.append(gm.get_parameter(name))

            for scale_name, scale_val in quant_impl.default_scales(orig_weight.shape).items():
                submod.register_buffer(scale_name, scale_val)

            gm._register_load_state_dict_pre_hook(partial(quant_impl.load_hook, weight_name=name))

            with gm.graph.inserting_before(node):
                scales = [gm.graph.get_attr(modname + "." + s) for s in scale_keys]
                scale_nodes_group.append(scales)

        # Derive canonical name: "experts.0.w1", "experts.1.w1" → "experts.w1_stacked_quant"
        parts = weight_names[0].split(".")
        idx_pos = next((i for i, p in enumerate(parts) if p.isdigit()), len(parts) - 1)
        parent_path = ".".join(parts[:idx_pos])
        attr_suffix = "_".join(parts[idx_pos + 1 :])
        stacked_name = f"{parent_path}.{attr_suffix}_stacked_quant"

        stacked = torch.stack(quant_weights, dim=0)

        # Register the stacked param, then delete per-expert weight params
        stacked_parts = stacked_name.split(".")
        current: nn.Module = gm
        for part in stacked_parts[:-1]:
            if not hasattr(current, part):
                current.add_module(part, nn.Module())
            current = getattr(current, part)
        current.register_parameter(stacked_parts[-1], nn.Parameter(stacked, requires_grad=False))

        for name in weight_names:
            name_parts = name.split(".")
            parent_mod: nn.Module = gm
            for part in name_parts[:-1]:
                parent_mod = getattr(parent_mod, part)
            parent_mod._parameters.pop(name_parts[-1], None)

        with gm.graph.inserting_before(node):
            stacked_node = gm.graph.get_attr(stacked_name)
            stacked_node.meta["val"] = gm.get_parameter(stacked_name)

        return stacked_node, scale_nodes_group

    # Quantize all three expert weights
    w1_weight_node, w1_scales = quantize_experts(w1_names)
    w2_weight_node, w2_scales = quantize_experts(w2_names)
    if w3_names:
        w3_weight_node, w3_scales = quantize_experts(w3_names)
    else:
        w3_weight_node, w3_scales = None, []

    # Collect scale tensors per scale type across w1, w2, w3
    def collect_scales(index: int) -> Tuple[List[Node], List[Node], List[Node]]:
        return (
            [s[index] for s in w1_scales],
            [s[index] for s in w2_scales],
            [s[index] for s in w3_scales],
        )

    # Prepare args: w1/w2/w3 are single stacked Tensor nodes; scales remain as lists.
    args = [
        node.args[0],  # x
        node.args[1],  # selected_experts
        node.args[2],  # routing_weights
        w1_weight_node,
        w2_weight_node,
        w3_weight_node,
    ]

    for idx in range(len(scale_keys)):
        s1, s2, s3 = collect_scales(idx)
        args.extend([s1, s2, s3])

    # Extract is_gated_mlp and act_fn from the original node
    # These can be in args[6:] or in kwargs
    is_gated_mlp = True  # default
    act_fn = ActivationType.Silu  # default

    if len(node.args) > 6:
        is_gated_mlp = node.args[6]
    elif "is_gated_mlp" in node.kwargs:
        is_gated_mlp = node.kwargs["is_gated_mlp"]

    if len(node.args) > 7:
        act_fn = node.args[7]
    elif "act_fn" in node.kwargs:
        act_fn = node.kwargs["act_fn"]

    # Prepare kwargs for the quantized op
    kwargs = {
        "is_gated_mlp": is_gated_mlp,
        "act_fn": act_fn,
    }

    # Replace the current node with the quantized version
    with gm.graph.inserting_after(node):
        new_node = gm.graph.call_function(
            quantized_op,
            args=tuple(args),
            kwargs=kwargs,
        )
        node.replace_all_uses_with(new_node)
        gm.graph.erase_node(node)

    # Erase the now-dead aten.stack weight nodes and their per-expert get_attr inputs.
    # This is only needed when the original weight args were aten.stack nodes (pre-pattern-match).
    for stack_node in orig_weight_arg_nodes:
        if stack_node.users:
            continue  # still used, skip
        per_expert_inputs = list(stack_node.args[0]) if stack_node.args else []
        gm.graph.erase_node(stack_node)
        for inp_node in per_expert_inputs:
            if isinstance(inp_node, Node) and not inp_node.users:
                gm.graph.erase_node(inp_node)


# TODO(Fridah-nv): robust handling similar to `extract_param_names_from_lin_node` or expand it
def _extract_moe_weight_param_lists(moe_node: Node) -> Tuple[List[str], List[str], List[str]]:
    """
    Given a torch.ops.moe.torch_moe node in gm.graph, extract the parameter names for
    w1_weight, w2_weight, and w3_weight.

    Handles both formats:
    - Stacked: single get_attr Node per weight (post-PR2 format, returns single-element list).
    - Per-expert: list of get_attr Nodes (legacy format, returns multi-element list).

    Returns:
      (w1_names, w2_names, w3_names), each a list of strings like 'layer.expert_0.w1.weight'
    """
    # args layout: (x, selected_experts, routing_weights, w1_weight, w2_weight, w3_weight)
    if len(moe_node.args) < 6:
        raise RuntimeError(
            f"Expected moe_node.args to have at least 6 entries, got {len(moe_node.args)}"
        )
    w1_arg, w2_arg, w3_arg = moe_node.args[3], moe_node.args[4], moe_node.args[5]

    def _unwrap(arg) -> List[str]:
        if arg is None:
            return []
        if isinstance(arg, (list, tuple)):
            # Legacy per-expert format
            names: List[str] = []
            for elt in arg:
                if not isinstance(elt, Node) or elt.op != "get_attr":
                    raise RuntimeError(
                        f"Expected each list element to be a get_attr Node, got {elt}"
                    )
                names.append(elt.target)
            return names
        if isinstance(arg, Node) and arg.op == "get_attr":
            # Stacked format: single get_attr node (post-pattern-match)
            return [arg.target]
        if (
            isinstance(arg, Node)
            and arg.op == "call_function"
            and arg.target is torch.ops.aten.stack.default
        ):
            # Pre-pattern-match: aten.stack of per-expert get_attr nodes
            stack_inputs = arg.args[0] if arg.args else []
            names = []
            for elt in stack_inputs:
                if not isinstance(elt, Node) or elt.op != "get_attr":
                    raise RuntimeError(f"Expected get_attr Node in aten.stack inputs, got {elt!r}")
                names.append(elt.target)
            return names
        raise TypeError(
            f"Expected a get_attr Node or list/tuple of get_attr Nodes, got {type(arg)}"
        )

    return _unwrap(w1_arg), _unwrap(w2_arg), _unwrap(w3_arg)


@TransformRegistry.register("quantize_fp8_moe")
class QuantizeFP8MOE(FP8LinearQuantizationFromConfig):
    """
    Traverse gm, find every torch.ops.auto_deploy.torch_moe, and replace it with the
    quantized version using the quant_algo from quant_config.
    """

    def target_op(self):
        return torch.ops.auto_deploy.torch_quant_fp8_moe

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        qcfg = factory.get_quant_config()
        if not qcfg:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        is_mixed = is_mixed_precision_config(qcfg)
        if is_mixed:
            if not mixed_precision_has_algo(qcfg, self.algo_name):
                return gm, TransformInfo(
                    skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
                )
            quantized_layers = qcfg.get("quantized_layers", {})
        elif qcfg.get("quant_algo", "").upper() != self.algo_name:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        excluded_patterns = qcfg.get("exclude_modules", [])
        count = 0

        for node in list(gm.graph.nodes):
            if not is_op(node, torch.ops.auto_deploy.torch_moe):
                continue

            w1_names, w2_names, w3_names = _extract_moe_weight_param_lists(node)
            all_weight_names = w1_names + w2_names + w3_names

            if any(should_skip_quantization(n, excluded_patterns) for n in all_weight_names):
                continue

            if is_mixed and any(
                should_skip_mixed_precision_quantization(n, self.algo_name, quantized_layers)
                for n in all_weight_names
            ):
                continue

            _quantize_moe_node(gm, node, self, self.target_op())
            count += 1

        info = TransformInfo(
            skipped=(count == 0),
            num_matches=count,
            is_clean=(count == 0),
            has_valid_shapes=True,
        )
        return gm, info


@TransformRegistry.register("quantize_nvfp4_moe")
class QuantizeNVFP4MOE(NVFP4LinearQuantizationFromConfig):
    """
    Traverse gm, find every torch.ops.auto_deploy.torch_moe, and replace it with the
    quantized version using the quant_algo from quant_config.
    """

    def target_op(self):
        return torch.ops.auto_deploy.torch_quant_nvfp4_moe

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        qcfg = factory.get_quant_config()
        if not qcfg:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        is_mixed = is_mixed_precision_config(qcfg)
        if is_mixed:
            if not mixed_precision_has_algo(qcfg, self.algo_name):
                return gm, TransformInfo(
                    skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
                )
            quantized_layers = qcfg.get("quantized_layers", {})
        elif qcfg.get("quant_algo", "").upper() != self.algo_name:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        excluded_patterns = qcfg.get("exclude_modules", [])
        count = 0

        for node in list(gm.graph.nodes):
            if not is_op(node, torch.ops.auto_deploy.torch_moe):
                continue

            w1_names, w2_names, w3_names = _extract_moe_weight_param_lists(node)
            all_weight_names = w1_names + w2_names + w3_names

            if any(should_skip_quantization(n, excluded_patterns) for n in all_weight_names):
                continue

            if is_mixed and any(
                should_skip_mixed_precision_quantization(n, self.algo_name, quantized_layers)
                for n in all_weight_names
            ):
                continue

            _quantize_moe_node(gm, node, self, self.target_op())
            count += 1

        info = TransformInfo(
            skipped=(count == 0),
            num_matches=count,
            is_clean=(count == 0),
            has_valid_shapes=True,
        )
        return gm, info


@TransformRegistry.register("quantize_finegrained_fp8_moe")
class QuantizeFineGrainedFP8MOE(Quantization):
    """
    Traverse gm, find every torch.ops.auto_deploy.torch_moe, and replace it with the
    FineGrainedFP8 quantized version.

    This transform handles FineGrained FP8 quantization config format:
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["gate", "lm_head"]
        }
    """

    algo_name = "fp8"

    def target_op(self):
        return torch.ops.auto_deploy.torch_quant_finegrained_fp8_moe

    def quantize_weight(self, w: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(w, dtype=torch.float8_e4m3fn, device=w.device)

    def scale_names(self) -> List[str]:
        return ["weight_scale_inv"]

    def default_scales(self, original_weight_shape: Tuple) -> Dict[str, torch.Tensor]:
        # Default block size is 128x128 for FineGrained FP8.
        # Supports both 2D (N, K) and 3D (E, N, K) stacked shapes.
        *batch_dims, N, K = original_weight_shape
        block_n, block_k = 128, 128
        scale_shape = (*batch_dims, math.ceil(N / block_n), math.ceil(K / block_k))
        return {"weight_scale_inv": torch.ones(scale_shape, dtype=torch.bfloat16)}

    def build_custom_args_for_linear(self, scales: Dict[str, "Node"]) -> Tuple:
        return ([scales["weight_scale_inv"]],)

    def load_hook(self, state_dict, prefix, *args, weight_name: str):
        """Load hook to handle HF FineGrainedFP8 checkpoint format."""
        if weight_name not in state_dict:
            return

        weight = state_dict[weight_name]
        if weight.dtype == torch.float8_e4m3fn:
            scale_inv_name = weight_name + "_scale_inv"
            if scale_inv_name in state_dict:
                mod_prefix = weight_name.rsplit(".", 1)[0]
                state_dict[mod_prefix + ".weight_scale_inv"] = state_dict[scale_inv_name]

    def _apply(
        self,
        gm: GraphModule,
        cm: CachedSequenceInterface,
        factory: ModelFactory,
        shared_config: SharedConfig,
    ) -> Tuple[GraphModule, TransformInfo]:
        # Gate by quant_method in quant_config (HF style)
        qcfg = factory.get_quant_config()
        if not qcfg:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        quant_method = str(qcfg.get("quant_method", "")).lower()
        if quant_method != self.algo_name:
            return gm, TransformInfo(
                skipped=True, num_matches=0, is_clean=True, has_valid_shapes=True
            )

        excluded_patterns = qcfg.get("modules_to_not_convert", [])
        count = 0

        for node in list(gm.graph.nodes):
            if not is_op(node, torch.ops.auto_deploy.torch_moe):
                continue

            # Check experts are allowed (no excludes)
            w1_names, w2_names, w3_names = _extract_moe_weight_param_lists(node)
            if any(
                should_skip_quantization(n, excluded_patterns)
                for n in (w1_names + w2_names + w3_names)
            ):
                continue

            _quantize_moe_node(gm, node, self, self.target_op())
            count += 1

        info = TransformInfo(
            skipped=(count == 0),
            num_matches=count,
            is_clean=(count == 0),
            has_valid_shapes=True,
        )
        return gm, info
