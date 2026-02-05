#!/usr/bin/env python3
"""Module-Level Graph Comparison: AD vs HF Debugging Tool.

This module compares AutoDeploy (AD) outputs with HuggingFace (HF) model outputs
at module boundaries to identify which module first diverges.

Compares at:
    - embed_tokens: Embedding layer output
    - self_attn: Attention layer output
    - block_sparse_moe: MoE layer output
    - norm: Final normalization output
    - lm_head: Output projection

Usage at breakpoint in optimizer.py:
    from tensorrt_llm._torch.auto_deploy.utils.graph_debug_compare import run_comparison
    run_comparison(mod, cm, self.factory)
"""

import argparse
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

# Import custom models to trigger their registration with AutoConfig
# This ensures the custom config classes are known when loading models
# flake8: noqa: F401
from ..models.custom import Glm4MoeLiteForCausalLM
from ..models.hf import AutoModelForCausalLMFactory
from .debug_interpreter import run_interpreter_with_captures
from .graph_debug_utils import compare_tensors, load_debug_artifacts
from .logger import extract_graph_metadata

# ============================================================================
# Scatter Plot Configuration
# ============================================================================
SCATTER_MAX_POINTS = 50000  # Max points to plot (subsample if more)


def _load_hf_model_with_custom_registry(
    model_path: str,
    num_layers: int,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    """Load an HF model, using AutoDeploy's custom model registry if applicable.

    AutoDeploy registers custom model implementations (e.g., Glm4MoeLiteForCausalLM)
    via AutoModelForCausalLMFactory.register_custom_model_cls(). The standard
    transformers.AutoModelForCausalLM doesn't know about these registrations.

    This function checks if the model's config type has a custom implementation
    registered and uses it instead of the default AutoModelForCausalLM.

    Args:
        model_path: HuggingFace model path or local directory
        num_layers: Number of hidden layers to use (for debugging with fewer layers)
        torch_dtype: Torch dtype for model weights

    Returns:
        Loaded and configured HF model
    """
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.num_hidden_layers = num_layers
    config.use_cache = False

    # Check if there's a custom model registered for this config type
    config_cls_name = type(config).__name__
    custom_model_cls = AutoModelForCausalLMFactory._custom_model_mapping.get(config_cls_name, None)

    if custom_model_cls is not None:
        print(f"  Using custom model from AD registry: {custom_model_cls.__name__}")
        # Use the custom model class's from_pretrained
        hf_model = custom_model_cls.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    else:
        print(f"  Using standard AutoModelForCausalLM (config: {config_cls_name})")
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

    hf_model.eval()
    return hf_model


def plot_tensor_scatter(
    hf_tensor: torch.Tensor,
    ad_tensor: torch.Tensor,
    out_png: str,
    title: str,
    max_points: int = SCATTER_MAX_POINTS,
    seed: int = 0,
) -> tuple:
    """Flatten two tensors, subsample, and produce an HF vs AD scatter plot.

    Args:
        hf_tensor: HuggingFace reference tensor
        ad_tensor: AutoDeploy tensor to compare
        out_png: Output path for PNG file
        title: Plot title
        max_points: Maximum number of points to plot (subsamples if more)
        seed: Random seed for subsampling

    Returns:
        Tuple of (correlation, num_points_plotted)
    """
    hf_flat = hf_tensor.detach().float().cpu().reshape(-1)
    ad_flat = ad_tensor.detach().float().cpu().reshape(-1)

    min_len = min(hf_flat.numel(), ad_flat.numel())
    if min_len == 0:
        print(f"      Warning: Scatter skipped (empty tensors): {title}")
        return float("nan"), 0

    hf_flat = hf_flat[:min_len]
    ad_flat = ad_flat[:min_len]
    finite_mask = torch.isfinite(hf_flat) & torch.isfinite(ad_flat)
    hf_flat = hf_flat[finite_mask]
    ad_flat = ad_flat[finite_mask]

    if hf_flat.numel() == 0:
        print(f"      Warning: Scatter skipped (no finite points): {title}")
        return float("nan"), 0

    hf_np = hf_flat.numpy()
    ad_np = ad_flat.numpy()

    n_points = hf_np.size
    if n_points > max_points:
        rng = np.random.default_rng(seed)
        indices = rng.choice(n_points, size=max_points, replace=False)
        hf_np = hf_np[indices]
        ad_np = ad_np[indices]
        n_points = hf_np.size

    corr = float(np.corrcoef(hf_np, ad_np)[0, 1]) if hf_np.size >= 2 else float("nan")

    png_dir = os.path.dirname(out_png)
    if png_dir:
        os.makedirs(png_dir, exist_ok=True)

    lo = float(min(hf_np.min(), ad_np.min()))
    hi = float(max(hf_np.max(), ad_np.max()))
    if lo == hi:
        lo -= 1.0
        hi += 1.0

    plt.figure(figsize=(6, 6), dpi=150)
    plt.scatter(hf_np, ad_np, s=1, alpha=0.25, linewidths=0)
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="black", alpha=0.8)
    plt.xlabel("HF (reference)")
    plt.ylabel("AutoDeploy")
    plt.title(f"{title}\nPearson r = {corr:.5f}")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    print(f"      Scatter saved: {out_png} (corr={corr:.4f}, points={n_points})")
    return corr, n_points


def _find_ad_weight(
    ad_state: Dict[str, torch.Tensor],
    hf_module_path: str,
    hf_key: str,
) -> Tuple[Optional[torch.Tensor], Optional[str], Optional[str]]:
    """Find matching AD weight with multiple strategies.

    Handles:
    1. _orig_mod. prefix with dot notation (new format)
    2. Fused MoE weights (experts stacked into single tensors)
    3. Underscore format (legacy fallback)

    Returns:
        (weight_tensor, matched_key, slice_info) or (None, None, None) if not found
    """
    # Strategy 1: _orig_mod. prefix with dot notation
    orig_mod_key = f"_orig_mod.{hf_module_path}.{hf_key}"
    if orig_mod_key in ad_state:
        return ad_state[orig_mod_key], orig_mod_key, None

    # Strategy 2: Check for fused MoE weights
    expert_match = re.match(r"experts\.(\d+)\.(w[123])\.weight", hf_key)
    if expert_match:
        expert_idx = int(expert_match.group(1))
        weight_type = expert_match.group(2)

        # Find fused weight key
        for ad_key in ad_state.keys():
            if weight_type in ("w1", "w3") and "fused_moe_w3_w1_stacked" in ad_key:
                fused = ad_state[ad_key]
                intermediate_size = fused.shape[1] // 2
                if weight_type == "w3":
                    extracted = fused[expert_idx, :intermediate_size, :]
                    return extracted, ad_key, f"[{expert_idx}, :{intermediate_size}, :]"
                else:  # w1
                    extracted = fused[expert_idx, intermediate_size:, :]
                    return extracted, ad_key, f"[{expert_idx}, {intermediate_size}:, :]"
            elif weight_type == "w2" and "fused_moe_w2_stacked" in ad_key:
                extracted = ad_state[ad_key][expert_idx]
                return extracted, ad_key, f"[{expert_idx}]"

    # Strategy 3: Original underscore format (fallback)
    ad_prefix = hf_module_path.replace(".", "_") + "_"
    ad_key_candidate = ad_prefix + hf_key.replace(".", "_")
    if ad_key_candidate in ad_state:
        return ad_state[ad_key_candidate], ad_key_candidate, None

    # Strategy 4: Suffix/contains match (flexible matching)
    for ad_k, ad_v in ad_state.items():
        if ad_key_candidate in ad_k or ad_k.endswith(ad_key_candidate):
            return ad_v, ad_k, None
        # Also try matching just the param name with prefix
        param_suffix = hf_key.replace(".", "_")
        if ad_k.endswith("_" + param_suffix) and ad_prefix.rstrip("_") in ad_k:
            return ad_v, ad_k, None

    return None, None, None


# ============================================================================
# Configuration
# ============================================================================
# Tolerances for bfloat16 comparison - tightened to catch 1-bit errors
# bfloat16 has ~7 bits of mantissa. 0.007812 is 2^-7.
# We want to flag diffs > 1e-4 even for values > 1.0.
ATOL = 1e-3
RTOL = 1e-3  # 0.01% relative tolerance


# ============================================================================
# Stage 1: Coarse Module Boundary Comparison
# ============================================================================


def _discover_hf_modules(hf_model) -> List[str]:
    """Discover key modules from the HF model structure for GLM-4.7-Flash.

    GLM-4.7-Flash structure (from modeling_glm4_moe_lite.py):
    - model.embed_tokens: Embedding layer (nn.Embedding)
    - model.layers.0.self_attn: Glm4MoeLiteAttention (MLA)
    - model.layers.0.mlp: Glm4MoeLiteMLP (dense MLP)
    - model.layers.1.self_attn: Glm4MoeLiteAttention (MLA)
    - model.layers.1.mlp: Glm4MoeLiteMoE (MoE layer)
    - model.norm: Glm4MoeLiteRMSNorm (final layer norm)
    - lm_head: Output projection (nn.Linear)

    Returns:
        List of module paths to hook
    """
    modules_to_hook = []

    # Find the inner model (accessed via .model attribute for CausalLM models)
    inner_model = getattr(hf_model, "model", hf_model)

    # 1. embed_tokens (Glm4MoeLiteModel.embed_tokens)
    if hasattr(inner_model, "embed_tokens"):
        modules_to_hook.append("model.embed_tokens")

    # 2. Layer 0 and Layer 1 attention and FFN
    # GLM-4.7-Flash: Layer 0 has dense MLP (Glm4MoeLiteMLP), Layer 1+ has MoE (Glm4MoeLiteMoE)
    layers = getattr(inner_model, "layers", None)
    if layers is not None:
        # Layer 0 (dense MLP)
        if len(layers) > 0:
            layer0 = layers[0]
            if hasattr(layer0, "self_attn"):
                modules_to_hook.append("model.layers.0.self_attn")
            if hasattr(layer0, "mlp"):
                modules_to_hook.append("model.layers.0.mlp")

        # Layer 1 (MoE for GLM-4.7-Flash)
        if len(layers) > 1:
            layer1 = layers[1]
            if hasattr(layer1, "self_attn"):
                modules_to_hook.append("model.layers.1.self_attn")
            if hasattr(layer1, "mlp"):
                modules_to_hook.append("model.layers.1.mlp")

    # 3. Final norm (Glm4MoeLiteRMSNorm)
    if hasattr(inner_model, "norm"):
        modules_to_hook.append("model.norm")

    # 4. lm_head (on the outer model - Glm4MoeLiteForCausalLM)
    if hasattr(hf_model, "lm_head"):
        modules_to_hook.append("lm_head")

    print(f"[_discover_hf_modules] Discovered modules: {modules_to_hook}")
    return modules_to_hook


def _build_module_mapping(hf_modules: List[str]) -> Dict[str, str]:
    """Build a mapping from HF module paths to simplified AD keys.

    For GLM-4.7-Flash, we need layer-specific keys:
    - model.layers.0.self_attn -> layer0_self_attn
    - model.layers.0.mlp -> layer0_mlp
    - model.layers.1.self_attn -> layer1_self_attn
    - model.layers.1.mlp -> layer1_mlp

    Args:
        hf_modules: List of HF module paths

    Returns:
        Dict mapping HF paths to simplified keys
    """
    mapping = {}
    for hf_path in hf_modules:
        parts = hf_path.split(".")
        if len(parts) == 1:
            # Simple case: "lm_head" -> "lm_head"
            mapping[hf_path] = hf_path
        elif parts[-1] in ("embed_tokens", "embeddings"):
            mapping[hf_path] = "embed_tokens"
        elif parts[-1] in ("self_attn", "attention", "attn"):
            # Check if this is a layer-specific module: model.layers.N.self_attn
            if len(parts) >= 4 and parts[1] == "layers":
                layer_idx = parts[2]
                mapping[hf_path] = f"layer{layer_idx}_self_attn"
            else:
                mapping[hf_path] = "self_attn"
        elif parts[-1] in ("block_sparse_moe", "moe", "mlp", "feed_forward", "ffn"):
            # Check if this is a layer-specific module: model.layers.N.mlp
            if len(parts) >= 4 and parts[1] == "layers":
                layer_idx = parts[2]
                mapping[hf_path] = f"layer{layer_idx}_mlp"
            else:
                mapping[hf_path] = "ffn"
        elif parts[-1] in ("norm", "final_layernorm", "ln_f"):
            mapping[hf_path] = "norm"
        else:
            # Fallback: use the last part
            mapping[hf_path] = parts[-1]

    print(f"[_build_module_mapping] Module mapping: {mapping}")
    return mapping


class HFHookCapture:
    """Context manager to capture HF model inputs and outputs at module boundaries."""

    def __init__(self, model, module_names: List[str]):
        """Initialize hook capture.

        Args:
            model: HuggingFace model
            module_names: List of module names to hook (e.g., ["embed_tokens", "self_attn"])
        """
        self.model = model
        self.module_names = module_names
        self.captured: Dict[str, Any] = {}  # outputs
        self.captured_inputs: Dict[str, Any] = {}  # inputs
        self.handles = []

    def __enter__(self):
        for name in self.module_names:
            module = self._find_module(name)
            if module is not None:
                # Register pre-hook to capture inputs
                pre_handle = module.register_forward_pre_hook(self._make_pre_hook(name))
                self.handles.append(pre_handle)
                # Register forward hook to capture outputs
                handle = module.register_forward_hook(self._make_hook(name))
                self.handles.append(handle)
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _find_module(self, name: str):
        """Find a module by name in the model."""
        # Try direct access first
        try:
            module = self.model
            for part in name.split("."):
                module = getattr(module, part)
            return module
        except AttributeError:
            pass

        # Search in named_modules
        for full_name, mod in self.model.named_modules():
            if full_name.endswith(name) or name in full_name:
                return mod

        print(f"Warning: Could not find module '{name}'")
        return None

    def _clone_value(self, v: Any) -> Any:
        """Clone a tensor value, preserving structure for tuples/lists."""
        if isinstance(v, torch.Tensor):
            return v.detach().clone()
        elif isinstance(v, tuple):
            return tuple(self._clone_value(x) for x in v)
        elif isinstance(v, list):
            return [self._clone_value(x) for x in v]
        return v

    def _make_pre_hook(self, name: str):
        """Create a pre-hook to capture inputs."""

        def pre_hook(module, args):
            # Capture all tensor args
            cloned_args = tuple(self._clone_value(a) for a in args)
            self.captured_inputs[name] = cloned_args

        return pre_hook

    def _make_hook(self, name: str):
        """Create a forward hook to capture outputs."""

        def hook(module, input, output):
            # Handle different output types
            self.captured[name] = self._clone_value(output)

        return hook


def stage1_coarse_comparison(
    hf_model,
    ad_gm,
    ad_metadata: Dict[str, Any],
    input_ids: torch.Tensor,
    ad_named_args: Optional[Dict[str, Any]] = None,
    device: str = "cuda",
    output_dir: Optional[str] = None,
) -> Tuple[bool, Optional[str], Dict[str, Dict[str, Any]]]:
    """Run Stage 1: Coarse module boundary comparison.

    Args:
        hf_model: HuggingFace model
        ad_gm: AD GraphModule (final)
        ad_metadata: AD metadata dict
        input_ids: Input token IDs
        ad_named_args: Optional dict of named args to feed AD GraphModule
        device: Device to run on

    Returns:
        Tuple of:
        - all_passed: True if all comparisons pass
        - first_failing_module: Name of first failing module (or None)
        - results: Dict of comparison results per module
    """

    def _tensor_stats(t: torch.Tensor) -> Dict[str, Any]:
        if t is None:
            return {"status": "none"}
        if not isinstance(t, torch.Tensor):
            return {"status": "non_tensor", "type": str(type(t))}
        if t.numel() == 0:
            return {"shape": tuple(t.shape), "dtype": str(t.dtype), "status": "empty"}
        tf = t.float()
        return {
            "shape": tuple(t.shape),
            "dtype": str(t.dtype),
            "min": tf.min().item(),
            "max": tf.max().item(),
            "has_nan": torch.isnan(tf).any().item(),
            "has_inf": torch.isinf(tf).any().item(),
        }

    def _hf_tensor_leaves(v: Any) -> List[torch.Tensor]:
        """Deterministically extract tensor leaves from HF outputs (tensor / tuple / list)."""
        if isinstance(v, torch.Tensor):
            return [v]
        if isinstance(v, (tuple, list)):
            return [t for t in v if isinstance(t, torch.Tensor)]
        return []

    print("\n" + "=" * 80)
    print("STAGE 1: Coarse Module Boundary Comparison")
    print("=" * 80)

    # Dynamically discover modules from the HF model
    hf_modules = _discover_hf_modules(hf_model)

    # Move models to device
    hf_model = hf_model.to(device)
    ad_gm = ad_gm.to(device)
    input_ids = input_ids.to(device)

    # Run HF model with hooks
    print("\nRunning HF model with hooks...")
    with torch.inference_mode():
        with HFHookCapture(hf_model, hf_modules) as hf_capture:
            _ = hf_model(input_ids)
            hf_captured = hf_capture.captured
            hf_captured_inputs = hf_capture.captured_inputs

    print(f"HF captured modules (outputs): {list(hf_captured.keys())}")
    print(f"HF captured modules (inputs): {list(hf_captured_inputs.keys())}")

    # Build module mapping dynamically from discovered modules
    module_mapping = _build_module_mapping(hf_modules)

    # Find AD output nodes by pattern matching on node names
    # GLM-4.7-Flash node naming patterns from dumped graphs:
    #
    # Export stage (001_export_export_to_gm.txt):
    #   - model_embed_tokens_embedding
    #   - model_layers_0_self_attn_o_proj_torch_linear_simple_3
    #   - model_layers_0_mlp_down_proj_torch_linear_simple_6
    #   - model_layers_1_self_attn_o_proj_torch_linear_simple_10
    #   - model_layers_1_mlp_add_13 (MoE residual add after shared experts)
    #   - model_norm_mul_20
    #   - lm_head_torch_linear_simple_15
    #
    # Final stage (054_compile_multi_stream_moe.txt):
    #   - model_embed_tokens_embedding
    #   - torch_quant_nvfp4_linear_default_7 (layer 0 o_proj quantized)
    #   - torch_quant_nvfp4_linear_default_4 (layer 0 mlp quantized)
    #   - torch_quant_nvfp4_linear_default_3 (layer 1 o_proj quantized)
    #   - model_layers_1_mlp_add_13 or model_layers_1_mlp_to_21 (MoE output)
    #   - flashinfer_rms_norm_8 (final norm fused)
    #   - lm_head_torch_linear_simple_15
    ad_node_patterns = {
        "embed_tokens": r"^model_embed_tokens_embedding$",
        # Layer 0: dense MLP (Glm4MoeLiteMLP)
        "layer0_self_attn": (
            r"^(model_layers_0_self_attn_o_proj_torch_linear_simple_\d+"
            r"|torch_quant_nvfp4_linear_default_7)$"
        ),
        "layer0_mlp": (
            r"^(model_layers_0_mlp_down_proj_torch_linear_simple_\d+"
            r"|torch_quant_nvfp4_linear_default_4"
            r"|model_layers_0_add_5)$"
        ),
        # Layer 1: MoE (Glm4MoeLiteMoE)
        "layer1_self_attn": (
            r"^(model_layers_1_self_attn_o_proj_torch_linear_simple_\d+"
            r"|torch_quant_nvfp4_linear_default_3)$"
        ),
        "layer1_mlp": r"^(model_layers_1_mlp_add_\d+|model_layers_1_mlp_to_21)$",
        # Final outputs
        "norm": r"^(model_norm_mul_\d+|flashinfer_rms_norm_8)$",
        "lm_head": r"^lm_head_torch_linear_simple_\d+$",
    }

    # Find matching nodes in the AD graph
    all_node_names = [n.name for n in ad_gm.graph.nodes]
    ad_output_nodes: Dict[str, Optional[str]] = {}

    print("\nMatching AD nodes by pattern:")
    for key, pattern in ad_node_patterns.items():
        matches = [n for n in all_node_names if re.match(pattern, n)]
        if matches:
            # Take the last match (usually the final output of that module)
            ad_output_nodes[key] = matches[-1]
            print(f"  [{key}] -> {matches[-1]}")
        else:
            ad_output_nodes[key] = None
            print(f"  [{key}] -> NO MATCH for pattern: {pattern}")

    # Build list of nodes to capture
    ad_capture_nodes = set(n for n in ad_output_nodes.values() if n is not None)

    # Build inputs for the AD graph in placeholder order
    placeholders = [n.target for n in ad_gm.graph.nodes if n.op == "placeholder"]
    ad_inputs = {}
    missing_inputs = []
    ad_source = ad_named_args or {}
    for name in placeholders:
        if name in ad_source:
            val = ad_source[name]
            ad_inputs[name] = val.to(device) if isinstance(val, torch.Tensor) else val
        elif name == "input_ids":
            ad_inputs[name] = input_ids
        else:
            missing_inputs.append(name)

    if missing_inputs:
        print(f"WARNING: Missing inputs for placeholders: {missing_inputs}")

    with torch.inference_mode():
        ad_captured = run_interpreter_with_captures(
            ad_gm,
            ad_capture_nodes,
            **ad_inputs,
        )

    print(f"AD captured nodes: {list(ad_captured.keys())}")

    # Compare at each boundary (inputs and outputs)
    results = {}
    all_passed = True
    first_failing_module = None

    def _compare_module_weights(
        hf_model,
        ad_gm,
        hf_module_path: str,
        device: str,
    ) -> Dict[str, Any]:
        """Compare weights between HF module and AD using direct key matching.

        Both state_dicts use the same dot-notation keys, so just compare directly.
        """
        hf_state = hf_model.state_dict()
        ad_state = ad_gm.state_dict()

        # DEBUG: Print state dict keys for MoE modules
        if "mlp" in hf_module_path and "layers.1" in hf_module_path:
            print("\n  [DEBUG] State Dict Keys for MoE layer:")
            print("  [DEBUG] HF state dict keys (mlp/experts):")
            hf_moe_keys = sorted([k for k in hf_state.keys() if "layers.1.mlp" in k])
            for k in hf_moe_keys:
                shape = tuple(hf_state[k].shape) if hasattr(hf_state[k], "shape") else "N/A"
                print(f"    HF: {k} -> shape={shape}")

            print("  [DEBUG] AD state dict keys (mlp/experts):")
            ad_moe_keys = sorted(
                [k for k in ad_state.keys() if "layers.1.mlp" in k or "layers_1_mlp" in k]
            )
            for k in ad_moe_keys:
                shape = tuple(ad_state[k].shape) if hasattr(ad_state[k], "shape") else "N/A"
                print(f"    AD: {k} -> shape={shape}")

            # Also check if there are any mismatches in key sets
            hf_set = set(hf_moe_keys)
            ad_set = set(ad_moe_keys)
            only_in_hf = hf_set - ad_set
            only_in_ad = ad_set - hf_set
            if only_in_hf:
                print(f"  [DEBUG] Keys ONLY in HF: {sorted(only_in_hf)}")
            if only_in_ad:
                print(f"  [DEBUG] Keys ONLY in AD: {sorted(only_in_ad)}")

            # DEBUG: Check what the graph actually references (get_attr nodes)
            print("  [DEBUG] AD graph 'get_attr' nodes for MoE weights:")
            for node in ad_gm.graph.nodes:
                if node.op == "get_attr" and (
                    "layers_1_mlp" in node.target or "layers.1.mlp" in node.target
                ):
                    # Get the actual tensor from the graph
                    try:
                        attr_val = ad_gm
                        for part in node.target.split("."):
                            attr_val = getattr(attr_val, part)
                        shape = tuple(attr_val.shape) if hasattr(attr_val, "shape") else "N/A"
                        # Check if this tensor is same object as state_dict
                        state_key = node.target
                        in_state = state_key in ad_state
                        same_obj = in_state and (
                            ad_state[state_key].data_ptr() == attr_val.data_ptr()
                        )
                        print(
                            f"    get_attr: {node.target} -> shape={shape},"
                            f" in_state_dict={in_state},"
                            f"same_obj={same_obj}"
                        )
                    except Exception as e:
                        print(f"    get_attr: {node.target} -> ERROR: {e}")

        # Filter to keys matching this module path
        prefix = hf_module_path + "."
        hf_keys = [k for k in hf_state.keys() if k.startswith(prefix)]

        if not hf_keys:
            return {"status": "skip", "reason": f"No HF weights for '{hf_module_path}'"}

        weight_comparisons = []
        for hf_key in hf_keys:
            hf_weight = hf_state[hf_key]

            if hf_key not in ad_state:
                weight_comparisons.append(
                    {
                        "hf_key": hf_key,
                        "match": False,
                        "reason": "not in AD",
                    }
                )
                continue

            ad_weight = ad_state[hf_key]
            hf_w = hf_weight.to(device)
            ad_w = ad_weight.to(device)

            if hf_w.shape != ad_w.shape:
                weight_comparisons.append(
                    {
                        "hf_key": hf_key,
                        "match": False,
                        "reason": f"shape: HF={tuple(hf_w.shape)} AD={tuple(ad_w.shape)}",
                    }
                )
                continue

            diff = (hf_w.float() - ad_w.float()).abs()
            max_diff = diff.max().item()
            match = max_diff < 1e-6

            # DEBUG: Print first few values for mismatching expert weights
            if (
                not match
                and "experts" in hf_key
                and ("gate_up_proj" in hf_key or "down_proj" in hf_key)
            ):
                hf_flat = hf_w.flatten()[:10].tolist()
                ad_flat = ad_w.flatten()[:10].tolist()
                print(f"\n  [DEBUG] Weight value comparison for {hf_key}:")
                print(f"    HF first 10 values: {[f'{v:.4f}' for v in hf_flat]}")
                print(f"    AD first 10 values: {[f'{v:.4f}' for v in ad_flat]}")
                # Check if AD values look like random initialization (typically small random values)
                ad_mean = ad_w.float().mean().item()
                ad_std = ad_w.float().std().item()
                hf_mean = hf_w.float().mean().item()
                hf_std = hf_w.float().std().item()
                print(f"    HF stats: mean={hf_mean:.6f}, std={hf_std:.6f}")
                print(f"    AD stats: mean={ad_mean:.6f}, std={ad_std:.6f}")

            weight_comparisons.append(
                {
                    "hf_key": hf_key,
                    "match": match,
                    "max_diff": max_diff,
                }
            )

        all_match = all(c.get("match", False) for c in weight_comparisons)
        return {
            "match": all_match,
            "comparisons": weight_comparisons,
            "num_weights": len(weight_comparisons),
            "num_matched": sum(1 for c in weight_comparisons if c.get("match", False)),
        }

    print("\n--- Comparison Results (using node name patterns) ---")
    for hf_name, ad_key in module_mapping.items():
        module_results: Dict[str, Any] = {}
        module_passed = True
        print(f"\n[{ad_key}] Comparing HF '{hf_name}' vs AD node")

        # 1. Compare weights FIRST (always)
        print("  [weights] Comparing weights...")
        weight_result = _compare_module_weights(hf_model, ad_gm, hf_name, device)
        module_results["weights"] = weight_result

        if weight_result.get("status") == "skip":
            print(f"  [weights] SKIP: {weight_result.get('reason')}")
        elif weight_result.get("status") == "error":
            print(f"  [weights] ERROR: {weight_result.get('reason')}")
        else:
            num_weights = weight_result.get("num_weights", 0)
            num_matched = weight_result.get("num_matched", 0)
            if weight_result.get("match"):
                print(f"  [weights] PASS: All {num_weights} weights match")
            else:
                print(f"  [weights] FAIL: {num_matched}/{num_weights} weights match")
                # Print which weights don't match
                for comp in weight_result.get("comparisons", []):
                    if not comp.get("match", False):
                        reason = comp.get("reason", f"max_diff={comp.get('max_diff', '?')}")
                        print(f"    - MISMATCH: {comp['hf_key']} ({reason})")
                module_passed = False

        # 2. Compare activations
        hf_tensor = hf_captured.get(hf_name)
        ad_node_name = ad_output_nodes.get(ad_key)

        if hf_tensor is None:
            print(f"  [outputs] SKIP: HF module '{hf_name}' not captured")
            module_results["outputs"] = {"status": "skip", "reason": "HF not captured"}
            results[ad_key] = module_results
            continue

        # Get first tensor from HF output (handles tuples)
        hf_tensors = _hf_tensor_leaves(hf_tensor)
        if not hf_tensors:
            print(f"  [outputs] SKIP: HF output has no tensors (type={type(hf_tensor)})")
            module_results["outputs"] = {"status": "skip", "reason": "no tensor leaves"}
            results[ad_key] = module_results
            continue

        hf_out = hf_tensors[0]  # Use first tensor for comparison
        print(f"  [HF] shape={hf_out.shape}, dtype={hf_out.dtype}")

        if ad_node_name is None:
            print(f"  [AD] No matching node found for '{ad_key}' pattern")
            module_results["outputs"] = {"status": "no_match", "reason": "pattern not found"}
            module_passed = False
        elif ad_node_name not in ad_captured:
            print(f"  [AD] Node '{ad_node_name}' not captured")
            module_results["outputs"] = {"status": "not_captured", "node": ad_node_name}
            module_passed = False
        else:
            ad_out = ad_captured[ad_node_name]
            print(f"  [AD] node='{ad_node_name}', shape={ad_out.shape}, dtype={ad_out.dtype}")

            # Compare tensors
            comparison = compare_tensors(hf_out, ad_out, atol=ATOL, rtol=RTOL)
            module_results["outputs"] = comparison | {"node": ad_node_name}

            if comparison["match"]:
                print(f"  [outputs] PASS (max_diff={comparison['max_diff']:.6f})")
            else:
                print(
                    f"  [outputs] FAIL (max_diff={comparison['max_diff']:.6f}, "
                    f"mean_diff={comparison['mean_diff']:.6f})"
                )
                module_passed = False

                # Save scatter plot on failure
                if output_dir:
                    scatter_png = os.path.join(output_dir, f"{ad_key}_output_scatter.png")
                    plot_tensor_scatter(hf_out, ad_out, scatter_png, f"{ad_key}: HF vs AD")

        results[ad_key] = module_results
        if not module_passed:
            all_passed = False
            if first_failing_module is None:
                first_failing_module = ad_key
                # Summarize the issue
                weights_ok = weight_result.get("match", False) or weight_result.get("status") in (
                    "skip",
                    "error",
                )
                outputs_ok = module_results.get("outputs", {}).get("match", False)
                if weights_ok and not outputs_ok:
                    print(f"  -> First divergence at {ad_key}: weights OK, activations DIFFER")
                elif not weights_ok:
                    print(f"  -> First divergence at {ad_key}: weights MISMATCH")

    return all_passed, first_failing_module, results


# ============================================================================
# Inline Comparison Entry Point (for use at breakpoint)
# ============================================================================


def run_comparison(mod, cm, factory, device: str = "cuda", output_dir: Optional[str] = None):
    """Run HF vs AD comparison inline with live GraphModule.

    Usage at breakpoint in optimizer.py:
        from tensorrt_llm._torch.auto_deploy.utils.graph_debug_compare import run_comparison
        run_comparison(mod, cm, self.factory, output_dir="debug_scatter_plots")

    Args:
        mod: The transformed AD GraphModule
        cm: CachedSequenceInterface with named_args
        factory: ModelFactory with checkpoint_path and config
        device: Device to run on
        output_dir: Optional directory to save scatter plots for each block comparison
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Scatter plots will be saved to: {output_dir}")
    # Get model path and num_layers from factory
    model_path = getattr(factory, "checkpoint_path", None) or getattr(factory, "model", None)
    num_layers = (
        factory.model_kwargs.get("num_hidden_layers", 1) if hasattr(factory, "model_kwargs") else 1
    )

    if model_path is None:
        print("ERROR: Could not get model path from factory")
        return

    print("=" * 80)
    print("INLINE COMPARISON: AD vs HF")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Num layers: {num_layers}")

    # Extract metadata from live model
    print("\nExtracting metadata from AD model...")
    ad_metadata = extract_graph_metadata(mod)

    # Get inputs from cm
    input_ids = cm.named_args.get("input_ids")
    if input_ids is None:
        print("ERROR: No input_ids in cm.named_args")
        return

    # Load HF model using AD's custom model registry
    print(f"\nLoading HuggingFace model: {model_path}")
    hf_model = _load_hf_model_with_custom_registry(model_path, num_layers)

    # Run comparison
    all_passed, failing_module, results = stage1_coarse_comparison(
        hf_model,
        mod,
        ad_metadata,
        input_ids,
        ad_named_args=cm.named_args,
        device=device,
        output_dir=output_dir,
    )

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    if all_passed:
        print("All module boundaries match between HF and AD!")
    else:
        print(f"Divergence detected in module: {failing_module}")

    return all_passed, failing_module, results


# ============================================================================
# Main (for standalone usage)
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Module-Level Graph Comparison Tool")
    parser.add_argument(
        "--debug-dir",
        type=str,
        required=True,
        help="Directory containing debug dumps (from AD_DUMP_DEBUG_DIR)",
    )
    parser.add_argument(
        "--hf-model",
        type=str,
        default="cyankiwi/MiniMax-M2-BF16",
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="Number of layers in the model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save scatter plots for each block comparison",
    )
    args = parser.parse_args()

    debug_dir = Path(args.debug_dir)

    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("Module-Level Graph Comparison: AD vs HF")
    print("=" * 80)
    print(f"Debug dir: {debug_dir}")
    print(f"HF model: {args.hf_model}")
    print(f"Num layers: {args.num_layers}")
    print(f"Device: {args.device}")
    print(f"Output dir: {args.output_dir or '(not saving scatter plots)'}")

    # Load debug artifacts
    print("\nLoading debug artifacts...")
    final_gm, final_metadata, inputs = load_debug_artifacts(debug_dir, "final")

    if final_gm is None:
        print("ERROR: Could not load final GraphModule")
        return 1

    if final_metadata is None or not final_metadata:
        print("ERROR: Could not load final metadata")
        return 1

    # Load HF model using AD's custom model registry
    print("\nLoading HuggingFace model...")
    hf_model = _load_hf_model_with_custom_registry(args.hf_model, args.num_layers)

    # Prepare input
    if inputs is not None and "input_ids" in inputs:
        input_ids = inputs["input_ids"]
        ad_named_args = {"input_ids": input_ids}
    else:
        # Create dummy input
        print("Using dummy input (no saved inputs found)")
        input_ids = torch.randint(0, 1000, (1, 8))
        ad_named_args = {"input_ids": input_ids}

    # ========================================================================
    # Module-Level Comparison
    # ========================================================================
    all_passed, failing_module, _ = stage1_coarse_comparison(
        hf_model,
        final_gm,
        final_metadata,
        input_ids,
        ad_named_args=ad_named_args,
        device=args.device,
        output_dir=args.output_dir,
    )

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if all_passed:
        print("All module boundaries match between HF and AD!")
    else:
        print(f"Divergence detected in module: {failing_module}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
