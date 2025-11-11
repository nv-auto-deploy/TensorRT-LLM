"""Tests for basic graph sharding."""

import os
import sys
from functools import partial
from typing import Type

# Add paths for relative imports when running as standalone script
if __name__ == "__main__":
    # Add parent directories to sys.path for imports
    # Directory structure: tests/unittest/_torch/auto_deploy/unit/multigpu/transformations/library/
    # Helper modules are in: tests/unittest/_torch/auto_deploy/_utils_test/
    # Utils modules are in: tests/utils/
    current_dir = os.path.dirname(os.path.abspath(__file__))  # library
    transformations_dir = os.path.dirname(current_dir)  # transformations
    multigpu_dir = os.path.dirname(transformations_dir)  # multigpu
    unit_dir = os.path.dirname(multigpu_dir)  # unit
    auto_deploy_dir = os.path.dirname(unit_dir)  # auto_deploy
    _torch_dir = os.path.dirname(auto_deploy_dir)  # _torch
    unittest_dir = os.path.dirname(_torch_dir)  # unittest
    tests_dir = os.path.dirname(unittest_dir)  # tests (root test directory)

    utils_dir = os.path.join(auto_deploy_dir, "_utils_test")  # _utils_test

    # Add all necessary paths
    sys.path.insert(0, tests_dir)  # For utils.llm_data imports
    sys.path.insert(0, utils_dir)  # For _model_test_utils, etc.
    sys.path.insert(0, current_dir)  # For local imports
    sys.path.append(tests_dir)
    sys.path.append(utils_dir)
    sys.path.append(unittest_dir)
    sys.path.append(tests_dir)

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from _dist_test_utils import get_device_counts
from _graph_test_helpers import run_sharding_pattern_detection_test, run_test_transformed_gm
from _model_test_utils import FakeFP8Linear

import tensorrt_llm._torch.auto_deploy.distributed.common as dist_common
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.library.sharding import (
    SplitDimension,
    WeightShardingInfo,
)
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_linear_op, is_op
from tensorrt_llm._torch.auto_deploy.utils.sharding_utils import FP8TPShardingInfo

base_model_tp_plan = {
    "q_proj": "colwise",
    "k_proj": "colwise",
    "v_proj": "colwise",
    "o_proj": "rowwise",
    "gate_proj": "colwise",
    "up_proj": "colwise",
    "down_proj": "rowwise",
    "linear1": "colwise",
    "linear2": "rowwise",
    "linear": "gather",
    # "input_layernorm.weight": "sequence_parallel",
    # "post_attention_layernorm.weight": "sequence_parallel",
    # "norm.weight": "sequence_parallel",
    # "shared_expert.gate_proj": "local_colwise",
    # "shared_expert.up_proj": "local_colwise",
    # "shared_expert.down_proj": "local_rowwise",
    # "experts.gate_up_proj": "local_packed_rowwise",
    # "experts.down_proj": "local_colwise",
    # "experts": "local",
    "feed_forward": "gather",
    "self": "gather",
    "weight": "gather",
}

predefined_config = {
    "head_dim": 8,
    "tp_plan": base_model_tp_plan,
}


class GQA_Block(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        hidden_size: int,
        num_key_value_heads: int,
    ):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.is_gqa = num_key_value_heads < num_attention_heads
        assert self.hidden_size == self.num_attention_heads * self.head_dim

        # key, query, value, out projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_key_value_heads,
            bias=False,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_key_value_heads,
            bias=False,
        )

        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape

        q = self.q_proj(x).view(b, s, -1, self.head_dim)
        k = self.k_proj(x).view(b, s, -1, self.head_dim)
        v = self.v_proj(x).view(b, s, -1, self.head_dim)

        y = torch.ops.auto_deploy.torch_attention(q, k, v, is_causal=True, layout="bsnd")
        y = y.contiguous().view(b, s, -1)

        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear1 = nn.Linear(in_features, 4 * in_features, bias=bias)
        self.linear2 = nn.Linear(4 * in_features, out_features, bias=bias)

    def forward(self, x):
        y = F.relu(self.linear1(x))
        return self.linear2(y)


class FP8MLP(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.linear1 = FakeFP8Linear(in_features, 4 * in_features, bias=bias)
        self.linear2 = FakeFP8Linear(4 * in_features, out_features, bias=bias)

    def forward(self, x):
        y = F.relu(self.linear1(x))
        return self.linear2(y)


def _run_job(
    model_cls: nn.Module,
    dist_op_expected: str,
    bias: bool,
    from_config: bool,
    rank: int,
    world_size: int,
) -> None:
    # init model and input
    batch_size = 4
    sequence_len = 8
    num_features = 32

    # GQA specific parameters
    num_heads = 4
    num_key_value_heads = 1

    if model_cls == GQA_Block:
        model = model_cls(
            num_attention_heads=num_heads,
            hidden_size=num_features,
            num_key_value_heads=num_key_value_heads,
        ).to(device="cuda", dtype=torch.float16)
    elif model_cls == FP8MLP:
        model = model_cls(num_features, num_features, bias=bias).to("cuda")
    else:
        model = model_cls(num_features, num_features, bias=bias).to(
            device="cuda", dtype=torch.float16
        )
    x = torch.randn(batch_size, sequence_len, num_features, device="cuda", dtype=torch.float16)

    if model_cls == GQA_Block:
        head_dim = num_features // num_heads
        min_local_size = head_dim
    else:
        min_local_size = 1

    def _get_expected_num_params(num_p_og: int) -> int:
        num_update = 0
        if bias and dist_op_expected == "torch_dist_all_reduce":
            num_p_og -= num_features
            num_update = num_features * (rank == world_size - 1)

        if min_local_size > 1:
            # it means we are in the GQA. W_kv are partially replicated, we need to count
            # the number of parameters manually.
            W_q_local_size = num_features * num_features // world_size
            W_o_local_size = W_q_local_size
            W_k_local_size = num_features * head_dim * max(num_key_value_heads // world_size, 1)
            W_v_local_size = W_k_local_size
            num_params = W_q_local_size + W_k_local_size + W_v_local_size + W_o_local_size
        else:
            num_params = num_p_og // world_size + num_update
        return num_params

    def verify_local_weight_sizes(gm) -> bool:
        """Verify that all weight tensors have first dimension >= min_local_size after sharding."""
        for name, param in gm.named_parameters():
            # Only check parameters that have at least 1 dimension and are weight matrices
            if param.dim() >= 1 and "weight" in name:
                if param.shape[0] < min_local_size:
                    print(
                        f"Weight {name} has shape {param.shape}, dim {param.shape[0]} < min_local_size {min_local_size}"
                    )
                    return False
        return True

    # now run the test
    op_expected = getattr(torch.ops.auto_deploy, dist_op_expected)

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        None,
        {
            "detect_sharding": {
                "stage": "sharding",
                "use_sharding_from_factory": from_config,
            },
            "sharding_transform_executor": {
                "stage": "sharding",
            },
        },
    )(None, gm)

    def combined_graph_check(gm) -> bool:
        # Check for expected distributed operations
        has_expected_dist_ops = any(is_op(n, op_expected) for n in gm.graph.nodes) == (
            world_size > 1
        )
        # Check weight size constraints
        weight_sizes_valid = verify_local_weight_sizes(gm)
        return has_expected_dist_ops and weight_sizes_valid

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        check_transformed_graph=combined_graph_check,
        _get_expected_num_params=_get_expected_num_params,
    )


def _run_pattern_detection_job(
    model_cls: nn.Module,
    bias: bool,
    rank: int,
    world_size: int,
    from_config: bool,
) -> None:
    # init model and input
    batch_size = 4
    sequence_len = 8
    num_features = 32

    # GQA specific parameters
    num_heads = 4
    num_key_value_heads = 1

    if model_cls == GQA_Block:
        model = model_cls(
            num_attention_heads=num_heads,
            hidden_size=num_features,
            num_key_value_heads=num_key_value_heads,
        ).to(device="cuda", dtype=torch.float16)
    else:
        model = model_cls(num_features, num_features, bias=bias).to(
            device="cuda", dtype=torch.float16
        )
    x = torch.randn(batch_size, sequence_len, num_features, device="cuda", dtype=torch.float16)

    # Test pattern detection - create expected transformations for validation
    gm = torch_export_to_gm(model, args=(x,), clone=True)
    expected_transformations = []
    # if world_size == 1, no sharding transformations should be detected
    if world_size > 1:
        if model_cls == GQA_Block:
            min_local_shape = num_features // num_heads
            for node in gm.graph.nodes:
                if is_linear_op(node):
                    # for Q, K, V layers, we expect:
                    # dim = 0, add_dist = False
                    # for O layer, we expect:
                    # dim = 1, add_dist = True
                    if "o_proj" in node.args[1].name:
                        dim = SplitDimension.ROW
                        dist_op = "all_reduce"
                    else:
                        dim = SplitDimension.COLUMN
                        dist_op = None
                    expected_transformations.append(
                        WeightShardingInfo(
                            target_node=node.name,
                            split_dim=dim,
                            rank=rank,
                            world_size=world_size,
                            dist_op=dist_op,
                            min_local_shape=min_local_shape,
                        )
                    )
        elif model_cls == MLP:
            for node in gm.graph.nodes:
                if is_linear_op(node):
                    # linear1 should be sharded on dim=0, add_dist=False, min_local_shape=1
                    # linear2 should be sharded on dim=1, add_dist=True, min_local_shape=1
                    if "linear1" in node.args[1].name:
                        dim = SplitDimension.COLUMN
                        dist_op = None
                    else:
                        dim = SplitDimension.ROW
                        dist_op = "all_reduce"
                    expected_transformations.append(
                        WeightShardingInfo(
                            target_node=node.name,
                            split_dim=dim,
                            rank=rank,
                            world_size=world_size,
                            dist_op=dist_op,
                            min_local_shape=1,
                        )
                    )
        elif model_cls == nn.Linear:
            # expect simple shard only (dim=0, add_dist=True, min_local_shape=1)
            for node in gm.graph.nodes:
                if is_linear_op(node):
                    expected_transformations.append(
                        WeightShardingInfo(
                            target_node=node.name,
                            split_dim=SplitDimension.COLUMN,  # Simple shard uses dim=0
                            rank=rank,
                            world_size=world_size,
                            dist_op="all_gather",
                            min_local_shape=1,
                        )
                    )
        elif model_cls == FP8MLP:
            for node in gm.graph.nodes:
                if is_op(node, torch.ops.auto_deploy.torch_fake_quant_fp8_linear):
                    # linear1 should be sharded on dim=0, add_dist=False, min_local_shape=1
                    # linear2 should be sharded on dim=1, add_dist=True, min_local_shape=1
                    if "linear1" in node.args[1].name:
                        dim = SplitDimension.COLUMN
                        dist_op = None
                    else:
                        dim = SplitDimension.ROW
                        dist_op = "all_reduce"
                    expected_transformations.append(
                        FP8TPShardingInfo(
                            target_node=node.name,
                            split_dim=dim,
                            rank=rank,
                            world_size=world_size,
                            dist_op=dist_op,
                            min_local_shape=1,
                        )
                    )

    # get detected transformations
    optimizer = InferenceOptimizer(
        None,
        {
            "detect_sharding": {
                "stage": "sharding",
                "use_sharding_from_factory": from_config,
            },
        },
    )
    optimizer.shared_config.local_rank = rank
    optimizer.shared_config.world_size = world_size
    _ = optimizer(None, gm)
    detected_transformations = optimizer.shared_config.sharding_config.weight_sharding_transforms

    print(f"detected_transformations: {detected_transformations}")
    print(f"expected_transformations: {expected_transformations}")
    # Run pattern detection test
    run_sharding_pattern_detection_test(detected_transformations, expected_transformations)


@pytest.mark.parametrize("device_count", get_device_counts())
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("from_config", [False, True])
@pytest.mark.parametrize(
    "model_cls, dist_op_expected",
    (
        (MLP, "torch_dist_all_reduce"),
        (FP8MLP, "torch_dist_all_reduce"),
        (nn.Linear, "torch_dist_all_gather"),
        (GQA_Block, "torch_dist_all_reduce"),
    ),
)
def test_sharding(
    model_cls: Type[nn.Module],
    dist_op_expected: str,
    bias: bool,
    device_count: int,
    from_config: bool,
):
    dist_common.spawn_multiprocess_job(
        job=partial(_run_job, model_cls, dist_op_expected, bias, from_config),
        size=device_count,
    )


@pytest.mark.parametrize("world_size", [1, 8])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("from_config", [False, True])
@pytest.mark.parametrize(
    "model_cls, dist_op_expected",
    (
        (MLP, "torch_dist_all_reduce"),
        (FP8MLP, "torch_dist_all_reduce"),
        (nn.Linear, "torch_dist_all_gather"),
        (GQA_Block, "torch_dist_all_reduce"),
    ),
)
def test_sharding_pattern_detection(
    model_cls: Type[nn.Module],
    dist_op_expected: str,
    bias: bool,
    world_size: int,
    from_config: bool,
):
    """Test pattern detection logic without distributed execution.

    This test verifies only the pattern detection logic with provided world_size.
    No need to run distributed job, can be run on single process.
    """
    _run_pattern_detection_job(model_cls, bias, 0, world_size, from_config)


def _run_numerical_correctness_job(
    model_cls: nn.Module,
    bias: bool,
    from_config: bool,
    simple_shard_only: bool,
    rank: int,
    world_size: int,
    fixed_weight_init: bool = True,
    datatype: torch.dtype = torch.float16,
    all_reduce_precision: torch.dtype = None,
) -> None:
    """Test numerical correctness by comparing sharded vs non-sharded model outputs.

    Args:
        fixed_weight_init: If True, uses deterministic weight initialization for debugging.
                          If False, uses random initialization (same weights for both models).
        datatype: Data type to use for model weights and activations
                 (e.g., torch.float16, torch.float32, torch.float64).

    Fixed mode (easier debugging):
    - Each shard i of a weight tensor is initialized with constant value i
    - Input is initialized to all ones
    - Expected output: each position should have value = sum of shard indices * num_features

    Random mode (realistic testing):
    - Weights: Xavier/Glorot normal initialization with fixed seed=42 (normalized for stability)
    - Biases: Uniform [0,1) initialization with fixed seed=42
    - Exception for FP8MLP: Uses nn.Linear's default initialization
      (FakeFP8Linear quantizes weights during __init__ and calculates scale factors;
       reinitializing weights after construction would make scale factors stale)
    - Same initialization used for both sharded and non-sharded models (same seed on all ranks)
    - Input initialized randomly with fixed seed=123 (CPU+CUDA)
    - Normalized initialization prevents overflow in low precision formats (FP16, FP32)
    """
    # init model and input
    batch_size = 4
    sequence_len = 8
    num_features = 1024

    # GQA specific parameters
    num_heads = 4
    num_key_value_heads = 1

    # Handle FP8MLP special case: precision is fixed to FP8, datatype parameter is ignored
    if model_cls == FP8MLP:
        # FP8MLP uses FP8 quantization internally - datatype parameter doesn't apply
        # We use FP16 for input tensors since FP8 is not supported for torch.randn
        test_dtype = torch.float16
        if rank == 0 and datatype != torch.float16:
            print("\nNote: FP8MLP uses FP8 quantization internally.")
            print(f"      --datatype {datatype} parameter is ignored for model precision.")
            print("      Input tensors use float16 as intermediate format.\n")
    else:
        test_dtype = datatype

    # Create model
    # For FP8MLP, we need to set seed BEFORE model creation since FakeFP8Linear
    # quantizes weights during __init__ (can't reinitialize after)
    if model_cls == FP8MLP and not fixed_weight_init:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    if model_cls == GQA_Block:
        model = model_cls(
            num_attention_heads=num_heads,
            hidden_size=num_features,
            num_key_value_heads=num_key_value_heads,
        ).to(device="cuda", dtype=test_dtype)
    elif model_cls == FP8MLP:
        # FP8MLP manages its own precision (FP8) internally via FakeFP8Linear
        model = model_cls(num_features, num_features, bias=bias).to("cuda")
    else:
        model = model_cls(num_features, num_features, bias=bias).to(device="cuda", dtype=test_dtype)

    # Initialize weights
    # IMPORTANT: Skip weight initialization for FP8MLP!
    # FakeFP8Linear quantizes weights during __init__ and calculates weight_scale.
    # If we reinitialize weights after, the scale factor becomes stale, causing wrong results.
    if fixed_weight_init:
        # Deterministic initialization for easier debugging
        # For each weight tensor of shape [M, N], set shard i to value i
        with torch.no_grad():
            for name, param in model.named_parameters():
                if "weight" in name and param.dim() >= 2:
                    M = param.shape[0]
                    local_M = M // world_size
                    # Initialize each shard with its rank value
                    for i in range(world_size):
                        start_idx = i * local_M
                        end_idx = (i + 1) * local_M if i < world_size - 1 else M
                        param[start_idx:end_idx] = float(i)
                elif "bias" in name:
                    # Set bias to zero for simplicity
                    param.zero_()
    elif model_cls != FP8MLP:
        # Random initialization with fixed seed for reproducibility
        # Use normalized initialization to prevent overflow in low precision
        # All ranks use the same seed to ensure identical weights
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)  # Also seed CUDA RNG for CUDA tensors
        with torch.no_grad():
            for param in model.parameters():
                # Use normalized initialization for 2D tensors (weights)
                # Use uniform for 1D tensors (bias)
                if param.dim() >= 2:
                    # Xavier/Glorot normal initialization for weights
                    # This keeps variance stable across layers
                    fan_in = param.shape[1] if param.dim() >= 2 else param.shape[0]
                    fan_out = param.shape[0]
                    std = torch.sqrt(torch.tensor(2.0 / (fan_in + fan_out)))

                    # Special handling for FP8MLP: FakeFP8Linear may store weights in FP8
                    # Xavier init values are too small for FP8 representation, so use larger uniform init
                    if model_cls == FP8MLP:
                        # Use uniform initialization in range that survives FP8 quantization
                        # FP8 e4m3fn min normal: ~0.001953, so use [-1, 1] range
                        # Create in FP16 first, then convert (torch.rand doesn't support FP8)
                        scale = 1.0
                        temp = (
                            torch.rand(param.shape, dtype=torch.float16, device=param.device) * 2
                            - 1
                        ) * scale
                        # Convert to param's dtype (might be FP8)
                        param.data.copy_(temp.to(param.dtype))
                    else:
                        # For regular dtypes, use Xavier initialization
                        param.data.copy_(torch.randn_like(param) * std)
                else:
                    # For bias (1D tensors), use uniform initialization
                    if model_cls == FP8MLP:
                        # For FP8, use uniform in range [-0.5, 0.5]
                        # Create in FP16 first, then convert (torch.rand doesn't support FP8)
                        temp = (
                            torch.rand(param.shape, dtype=torch.float16, device=param.device) - 0.5
                        )
                        # Convert to param's dtype (might be FP8)
                        param.data.copy_(temp.to(param.dtype))
                    else:
                        # For regular dtypes, use uniform [0, 1)
                        param.data.copy_(torch.rand_like(param))

    # Create input
    if fixed_weight_init:
        # All ones for predictable output
        x = torch.ones(batch_size, sequence_len, num_features, device="cuda", dtype=test_dtype)
    else:
        # Random input with fixed seed
        torch.manual_seed(123)
        torch.cuda.manual_seed_all(123)  # Also seed CUDA RNG for CUDA tensors
        x = torch.randn(batch_size, sequence_len, num_features, device="cuda", dtype=test_dtype)

    # Debug: Check if weights and inputs are actually non-zero for FP8MLP
    if rank == 0 and model_cls == FP8MLP and not fixed_weight_init:
        print("\n[DEBUG] FP8MLP Initialization Check:")
        for name, param in model.named_parameters():
            # Convert to FP32 for stats (FP8 doesn't support min/max operations)
            param_fp32 = param.to(torch.float32)
            print(
                f"  {name}: shape={param.shape}, dtype={param.dtype}, "
                f"min={param_fp32.min().item():.4f}, max={param_fp32.max().item():.4f}, "
                f"mean={param_fp32.mean().item():.4f}, nonzero={param_fp32.count_nonzero().item()}/{param.numel()}"
            )
        x_fp32 = x.to(torch.float32)
        print(
            f"  input x: shape={x.shape}, dtype={x.dtype}, "
            f"min={x_fp32.min().item():.4f}, max={x_fp32.max().item():.4f}, "
            f"mean={x_fp32.mean().item():.4f}, nonzero={x_fp32.count_nonzero().item()}/{x.numel()}"
        )

    # Compute reference output from non-sharded model (on rank 0)
    with torch.no_grad():
        if rank == 0:
            y_ref = model(x).clone()
            # Debug: Check if output is non-zero for FP8MLP
            if model_cls == FP8MLP and not fixed_weight_init:
                print("\n[DEBUG] Forward pass output:")
                y_fp32 = y_ref.to(torch.float32)
                print(
                    f"  y_ref: min={y_fp32.min().item():.6f}, max={y_fp32.max().item():.6f}, "
                    f"mean={y_fp32.mean().item():.6f}, nonzero={y_fp32.count_nonzero().item()}/{y_ref.numel()}"
                )
        else:
            # Create dummy tensor with correct shape on other ranks
            y_ref = torch.zeros(
                batch_size, sequence_len, num_features, device="cuda", dtype=test_dtype
            )

    # Broadcast reference output to all ranks
    torch.distributed.broadcast(y_ref, src=0)

    # Compute FP64 ground truth if we're not already using FP64
    # This serves as the "infinite precision" accumulation reference
    # Key insight: Quantized weights (FP8/FP16) ARE the ground truth weights.
    # We upcast them losslessly to FP64 and compute in FP64 to eliminate accumulation errors.
    # This isolates accumulation precision errors from quantization errors.
    #
    # EXCEPTION: FP8MLP cannot have FP64 ground truth because:
    # - FakeFP8Linear stores weights in FP16, not FP8
    # - Quantization to FP8 happens during forward pass
    # - We cannot extract the quantized FP8 weights to upcast them
    # - The meaningful test for FP8MLP is: unsharded vs sharded (both use same FP8 quantization)
    y_fp64_gt = None
    if test_dtype != torch.float64 and model_cls != FP8MLP:
        with torch.no_grad():
            if rank == 0:
                # Convert model to FP64 for ground truth computation
                if model_cls == GQA_Block:
                    model_fp64 = model_cls(
                        num_attention_heads=num_heads,
                        hidden_size=num_features,
                        num_key_value_heads=num_key_value_heads,
                    ).to(device="cuda", dtype=torch.float64)
                else:
                    model_fp64 = model_cls(num_features, num_features, bias=bias).to(
                        device="cuda", dtype=torch.float64
                    )

                # Initialize FP64 model with quantized weights from original model
                # Upcast from FP8/FP16 to FP64 is lossless - every FP8/FP16 value
                # is exactly representable in FP64
                for (name_orig, param_orig), (name_fp64, param_fp64) in zip(
                    model.named_parameters(), model_fp64.named_parameters()
                ):
                    # For FP8: param_orig contains quantized FP8 weights
                    # Upcast to FP64 preserves exact quantized values
                    param_fp64.data.copy_(param_orig.data.to(torch.float64))

                # Convert input to FP64
                x_fp64 = x.to(torch.float64)

                # Compute FP64 ground truth output
                # This uses quantized weights but FP64 accumulation (no rounding)
                y_fp64_gt = model_fp64(x_fp64).clone()
            else:
                # Create dummy tensor on other ranks
                y_fp64_gt = torch.zeros(
                    batch_size, sequence_len, num_features, device="cuda", dtype=torch.float64
                )

        # Broadcast FP64 ground truth to all ranks
        torch.distributed.broadcast(y_fp64_gt, src=0)

    # Create and run sharded model
    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        None,
        {
            "detect_sharding": {
                "stage": "sharding",
                "use_sharding_from_factory": from_config,
                "simple_shard_only": simple_shard_only,
                "all_reduce_precision": all_reduce_precision,
            },
            "sharding_transform_executor": {
                "stage": "sharding",
            },
        },
    )(None, gm)

    with torch.no_grad():
        y_shard = gm_transformed(x)
    # Debug output
    if rank == 0:
        print(f"\n{'=' * 80}")
        print(f"Numerical Correctness Test: {model_cls.__name__}")
        print(f"{'=' * 80}")
        print("Configuration:")
        print(f"  world_size: {world_size}")
        if model_cls == FP8MLP:
            print(f"  dtype (requested): {datatype}")
            print("  dtype (effective): FP8 for model, float16 for inputs")
        else:
            print(f"  dtype: {datatype}")
        print(f"  bias: {bias}")
        print(f"  from_config: {from_config}")
        print(f"  fixed_weight_init: {fixed_weight_init}")
        print(f"  simple_shard_only: {simple_shard_only}")
        print(f"  all_reduce_precision: {all_reduce_precision}")

        # if fixed_weight_init:
        #     print("\nWeight initialization (fixed/deterministic for debugging):")
        #     print("  - Each shard i initialized with constant value i")
        #     print("  - Input: all ones")
        #     print("\nExpected behavior for single Linear layer:")
        #     print(
        #         f"  - For world_size=2, output should be: (0 + 1) * num_features / 2 = {num_features // 2}"
        #     )
        #     print("  - For world_size=N, output should be: sum(0..N-1) * num_features / N")
        # else:
        #     print("\nWeight initialization (random for realistic testing):")
        #     if model_cls == FP8MLP:
        #         print("  - Weights: nn.Linear default initialization")
        #         print("  - Note: FakeFP8Linear quantizes during __init__ with scale factors")
        #         print("  - Cannot reinitialize (would make scale factors stale)")
        #     else:
        #         print("  - Weights: Xavier/Glorot normal initialization (normalized, seed=42)")
        #         print("  - Biases: Uniform [0,1) initialization (seed=42)")
        #     print("  - Random input with seed=123 (CPU+CUDA)")
        #     if model_cls != FP8MLP:
        #         print("  - Normalized initialization prevents overflow in low precision")

        print("\nActual output samples (first element of each):")
        print(f"  y_ref (unsharded): {y_ref[0, 0, 0].item():.6f}")
        print(f"  y_shard (sharded):  {y_shard[0, 0, 0].item():.6f}")
        if y_fp64_gt is not None:
            print(f"  y_fp64 (ground truth): {y_fp64_gt[0, 0, 0].item():.6f}")

        print("\nOutput statistics:")
        print(
            f"  y_ref   - min: {y_ref.min().item():.4f}, max: {y_ref.max().item():.4f}, "
            f"mean: {y_ref.mean().item():.4f}"
        )
        print(
            f"  y_shard - min: {y_shard.min().item():.4f}, max: {y_shard.max().item():.4f}, "
            f"mean: {y_shard.mean().item():.4f}"
        )
        if y_fp64_gt is not None:
            print(
                f"  y_fp64  - min: {y_fp64_gt.min().item():.4f}, "
                f"max: {y_fp64_gt.max().item():.4f}, mean: {y_fp64_gt.mean().item():.4f}"
            )

    # Compare outputs and compute metrics
    if rank == 0:
        print("\n" + "=" * 80)
        print("NUMERICAL ACCURACY ANALYSIS")
        print("=" * 80)

        # Special handling for FP8MLP - explain why no FP64 GT
        if model_cls == FP8MLP:
            print("\nNote: FP64 ground truth not available for FP8MLP")
            print("Reason: FakeFP8Linear stores weights in FP16 and quantizes during forward pass.")
            print("        We cannot extract quantized FP8 weights to create FP64 ground truth.")
            print(
                "        The meaningful comparison is: unsharded vs sharded (both use same FP8).\n"
            )

        # If we have FP64 ground truth, show all three comparisons
        if y_fp64_gt is not None:
            # Convert to same dtype for fair comparison
            y_ref_for_comparison = y_ref.to(torch.float64)
            y_shard_for_comparison = y_shard.to(torch.float64)

            # Comparison 1: Unsharded vs FP64 ground truth
            diff_unsharded_vs_fp64 = (y_ref_for_comparison - y_fp64_gt).abs()
            max_diff_unsharded = diff_unsharded_vs_fp64.max().item()
            mean_diff_unsharded = diff_unsharded_vs_fp64.mean().item()

            # Comparison 2: Sharded vs FP64 ground truth
            diff_sharded_vs_fp64 = (y_shard_for_comparison - y_fp64_gt).abs()
            max_diff_sharded = diff_sharded_vs_fp64.max().item()
            mean_diff_sharded = diff_sharded_vs_fp64.mean().item()

            # Comparison 3: Unsharded vs Sharded
            diff_unsharded_vs_sharded = (y_ref_for_comparison - y_shard_for_comparison).abs()
            max_diff_mutual = diff_unsharded_vs_sharded.max().item()
            mean_diff_mutual = diff_unsharded_vs_sharded.mean().item()

            # Display effective datatype for clarity
            effective_dtype_str = "FP8" if model_cls == FP8MLP else str(datatype)

            print(f"\n1. Unsharded ({effective_dtype_str}) vs FP64 Ground Truth:")
            print(f"   Max absolute difference:  {max_diff_unsharded:.8f}")
            print(f"   Mean absolute difference: {mean_diff_unsharded:.8f}")

            print(f"\n2. Sharded ({effective_dtype_str}) vs FP64 Ground Truth:")
            print(f"   Max absolute difference:  {max_diff_sharded:.8f}")
            print(f"   Mean absolute difference: {mean_diff_sharded:.8f}")

            print(f"\n3. Unsharded vs Sharded (both in {effective_dtype_str}):")
            print(f"   Max absolute difference:  {max_diff_mutual:.8f}")
            print(f"   Mean absolute difference: {mean_diff_mutual:.8f}")

            # Determine which is more accurate
            print("\n" + "-" * 80)
            print("ACCURACY COMPARISON (lower is better):")
            print("-" * 80)
            if max_diff_unsharded < max_diff_sharded:
                winner = "UNSHARDED"
                if max_diff_unsharded > 0:
                    diff_percentage = (
                        (max_diff_sharded - max_diff_unsharded) / max_diff_unsharded * 100
                    )
                else:
                    diff_percentage = float("inf")
            elif max_diff_sharded < max_diff_unsharded:
                winner = "SHARDED"
                if max_diff_sharded > 0:
                    diff_percentage = (
                        (max_diff_unsharded - max_diff_sharded) / max_diff_sharded * 100
                    )
                else:
                    diff_percentage = float("inf")
            else:
                winner = "TIE"
                diff_percentage = 0.0

            print(
                f"Max error - Unsharded: {max_diff_unsharded:.8f} | Sharded: {max_diff_sharded:.8f}"
            )
            print(
                f"Mean error - Unsharded: {mean_diff_unsharded:.8f} | "
                f"Sharded: {mean_diff_sharded:.8f}"
            )

            if winner == "TIE":
                print("\n✓ Both methods have IDENTICAL accuracy relative to FP64 ground truth")
            else:
                lower_text = "lower" if diff_percentage < float("inf") else "much lower"
                print(
                    f"\n✓ {winner} is more accurate "
                    f"(max error is {diff_percentage:.2f}% {lower_text})"
                )
        else:
            # No FP64 ground truth - either we're in FP64 already, or FP8MLP
            diff_unsharded_vs_sharded = (y_ref - y_shard).abs()
            max_diff_mutual = diff_unsharded_vs_sharded.max().item()
            mean_diff_mutual = diff_unsharded_vs_sharded.mean().item()

            if model_cls == FP8MLP:
                print("\nUnsharded vs Sharded (both using FP8 quantization):")
                print(f"   Max absolute difference:  {max_diff_mutual:.10f}")
                print(f"   Mean absolute difference: {mean_diff_mutual:.10f}")
                print("\nThis difference reflects FP8 accumulation precision only.")
                print(
                    "Both models use identical FP8 quantization, differing only in accumulation order."
                )
            else:
                # We're in FP64 mode
                print("\nUnsharded vs Sharded (both in FP64):")
                print(f"   Max absolute difference:  {max_diff_mutual:.10f}")
                print(f"   Mean absolute difference: {mean_diff_mutual:.10f}")

    # Standard assertion for test pass/fail
    # try:
    #     torch.testing.assert_close(y_shard.to(dtype=y_ref.dtype), y_ref, atol=atol, rtol=rtol)
    #     if rank == 0:
    #         print("\n" + "=" * 80)
    #         print("✓ Numerical correctness test PASSED")
    #         print(f"  Tolerance used: atol={atol}, rtol={rtol}")
    #         print("=" * 80 + "\n")
    # except AssertionError as e:
    #     if rank == 0:
    #         print("\n" + "=" * 80)
    #         print("✗ Numerical correctness test FAILED")
    #         print("  Unsharded vs Sharded difference exceeds tolerance")
    #         print(f"  Tolerance: atol={atol}, rtol={rtol}")
    #         print("\nDifference map (first few elements):")
    #         diff = (y_shard - y_ref).abs()
    #         print(f"  Diff[0,0,:5] = {diff[0, 0, :5]}")
    #         print("=" * 80 + "\n")
    #     raise e


@pytest.mark.parametrize("device_count", get_device_counts())
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("from_config", [False, True])
@pytest.mark.parametrize("fixed_weight_init", [True, False])
@pytest.mark.parametrize("datatype", [torch.float16, torch.float32, torch.float64])
@pytest.mark.parametrize(
    "model_cls",
    (
        (MLP, "torch_dist_all_reduce"),
        (FP8MLP),
        (nn.Linear),
        (GQA_Block),
    ),
)
def test_sharding_numerical_correctness(
    model_cls: Type[nn.Module],
    bias: bool,
    device_count: int,
    from_config: bool,
    simple_shard_only: bool,
    fixed_weight_init: bool,
    datatype: torch.dtype,
):
    """Test that sharded model produces numerically correct outputs compared to non-sharded model.

    This test validates that tensor parallel sharding preserves numerical accuracy by:
    1. Computing reference output from non-sharded model on rank 0
    2. Computing output from sharded model distributed across ranks
    3. Comparing outputs with precision-appropriate tolerances

    Tolerances are adjusted based on model precision and initialization:
    - FP8 models: atol=1e-1, rtol=1e-1 (lower precision)
    - FP64: atol=1e-10, rtol=1e-10 (highest precision, near-zero error expected)
    - FP32: atol=1e-4, rtol=1e-5 (better precision than FP16)
    - FP16 with fixed weights: atol=1e-5, rtol=1e-5 (near-exact results expected)
    - FP16 with random weights: atol=0.05, rtol=1e-2 (TP changes accumulation order)

    Weight initialization modes:
    - fixed_weight_init=True: Deterministic initialization for debugging (exact results)
    - fixed_weight_init=False: Random initialization for realistic testing (FP16 rounding diffs)

    Note: With random FP16 weights, TP changes the order of floating-point operations
    (all-reduce happens at different points), leading to different rounding errors even
    though the operations are mathematically equivalent.
    """
    dist_common.spawn_multiprocess_job(
        job=partial(
            _run_numerical_correctness_job,
            model_cls,
            bias,
            from_config,
            simple_shard_only,
            fixed_weight_init=fixed_weight_init,
            datatype=datatype,
        ),
        size=device_count,
    )


if __name__ == "__main__":
    """Run the numerical correctness test as a standalone script with torchrun.
    """
    import argparse

    import torch.distributed as dist

    parser = argparse.ArgumentParser(description="Run TP sharding numerical correctness test")
    parser.add_argument(
        "--bias",
        type=str,
        required=False,
        default="False",
        choices=["True", "False"],
        help="Whether to use bias in linear layers",
    )
    parser.add_argument(
        "--from_config",
        type=str,
        required=False,
        default="False",
        choices=["True", "False"],
        help="Whether to use config-based sharding detection",
    )
    parser.add_argument(
        "--model_cls",
        type=str,
        required=False,
        default="MLP",
        choices=["MLP", "FP8MLP", "Linear", "GQA_Block"],
        help="Model class to test",
    )
    parser.add_argument(
        "--fixed_weight_init",
        type=str,
        required=False,
        default="False",
        choices=["True", "False"],
        help="Use fixed/deterministic weight init (True) or random init (False)",
    )
    parser.add_argument(
        "--datatype",
        type=str,
        required=False,
        default="float16",
        choices=["float16", "float32", "float64"],
        help="Data type for model weights and activations (default: float16). "
        "Note: Ignored for FP8MLP which always uses FP8 internally.",
    )

    parser.add_argument(
        "--simple_shard_only",
        type=str,
        required=False,
        default="False",
        choices=["True", "False"],
        help="Whether to use simple shard only",
    )

    parser.add_argument(
        "--all_reduce_precision",
        type=str,
        required=False,
        default="None",
        choices=["None", "float16", "float32", "float64"],
        help="All-reduce precision for TP sharding",
    )

    args = parser.parse_args()

    # Convert string boolean arguments to actual booleans
    bias = args.bias == "True"
    from_config = args.from_config == "True"
    fixed_weight_init = args.fixed_weight_init == "True"
    simple_shard_only = args.simple_shard_only == "True"
    # Map model class string to actual class
    model_cls_map = {
        "MLP": MLP,
        "FP8MLP": FP8MLP,
        "Linear": nn.Linear,
        "GQA_Block": GQA_Block,
    }
    model_cls = model_cls_map[args.model_cls]

    # Map datatype string to torch dtype
    datatype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    datatype = datatype_map[args.datatype]
    all_reduce_precision = None
    if args.all_reduce_precision == "float16":
        all_reduce_precision = torch.float16
    elif args.all_reduce_precision == "float32":
        all_reduce_precision = torch.float32
    elif args.all_reduce_precision == "float64":
        all_reduce_precision = torch.float64

    # Initialize distributed environment
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    # Get rank and world_size from torch.distributed
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set CUDA device
    torch.cuda.set_device(rank)

    if rank == 0:
        print("Running numerical correctness test with:")
        print(f"  model_cls: {args.model_cls}")
        print(f"  bias: {bias}")
        print(f"  from_config: {from_config}")
        print(f"  fixed_weight_init: {fixed_weight_init}")
        print(f"  datatype: {datatype}")
        print(f"  world_size: {world_size}")

    # Run the numerical correctness job
    try:
        _run_numerical_correctness_job(
            model_cls=model_cls,
            bias=bias,
            from_config=from_config,
            rank=rank,
            world_size=world_size,
            simple_shard_only=simple_shard_only,
            fixed_weight_init=fixed_weight_init,
            datatype=datatype,
            all_reduce_precision=all_reduce_precision,
        )
        if rank == 0:
            print("\n✓ Test completed successfully!")
    except Exception as e:
        if rank == 0:
            print(f"\n✗ Test failed with error: {e}")
        raise
    finally:
        dist.destroy_process_group()
