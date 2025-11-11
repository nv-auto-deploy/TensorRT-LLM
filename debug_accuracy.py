"""Debug script to isolate source of numerical errors in TP sharding.

This script simulates TP sharding on a single GPU to test whether numerical
differences come from:
1. Different matmul kernel behavior with smaller matrices
2. Independent FP16 rounding in intermediate activations
3. All-reduce arithmetic operations

No tensorrt_llm imports - pure PyTorch implementation.
"""

from typing import Dict, List, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPDebug(nn.Module):
    """MLP with methods to simulate sharded execution."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Create weights in FP64 for ground truth
        # linear1: [in_features, 4*in_features]
        # linear2: [4*in_features, out_features]
        self.W1 = nn.Parameter(torch.randn(4 * in_features, in_features, dtype=torch.float64))
        self.W2 = nn.Parameter(torch.randn(out_features, 4 * in_features, dtype=torch.float64))

        # Initialize with Xavier/Glorot for stability
        with torch.no_grad():
            fan_in, fan_out = in_features, 4 * in_features
            std = torch.sqrt(torch.tensor(2.0 / (fan_in + fan_out)))
            self.W1.data.mul_(std)

            fan_in, fan_out = 4 * in_features, out_features
            std = torch.sqrt(torch.tensor(2.0 / (fan_in + fan_out)))
            self.W2.data.mul_(std)

    def forward_unsharded(self, x: torch.Tensor, dtype: torch.dtype) -> Tuple[torch.Tensor, Dict]:
        """Standard unsharded forward pass in specified dtype."""
        # Convert weights to target dtype
        W1 = self.W1.to(dtype)
        W2 = self.W2.to(dtype)
        x = x.to(dtype)

        # Forward pass
        y1 = x @ W1.T  # [B, S, 4D]
        y1_relu = F.relu(y1)
        y2 = y1_relu @ W2.T  # [B, S, D]

        intermediates = {
            "y1": y1,
            "y1_relu": y1_relu,
            "y2": y2,
        }

        return y2, intermediates

    def forward_sharded(
        self, x: torch.Tensor, world_size: int, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, Dict]:
        """Simulate column-row TP sharding in specified dtype."""
        # Convert input to target dtype
        x = x.to(dtype)

        # Calculate chunk size for sharding
        hidden_size = 4 * self.in_features
        chunk_size = hidden_size // world_size

        # Store intermediate activations for each rank
        y1_locals = []
        y1_relu_locals = []
        y2_locals = []

        # Simulate computation on each rank
        for rank in range(world_size):
            start_idx = rank * chunk_size
            end_idx = (rank + 1) * chunk_size

            # Column shard linear1: split output dimension
            W1_local = self.W1[start_idx:end_idx, :].to(dtype)
            y1_local = x @ W1_local.T  # [B, S, chunk_size]

            # Apply ReLU
            y1_relu_local = F.relu(y1_local)

            # Row shard linear2: split input dimension
            W2_local = self.W2[:, start_idx:end_idx].to(dtype)
            y2_local = y1_relu_local @ W2_local.T  # [B, S, out_features]

            y1_locals.append(y1_local)
            y1_relu_locals.append(y1_relu_local)
            y2_locals.append(y2_local)

        # Simulate all_reduce in FP64 (like all_reduce_mp)
        y2_sum_fp64 = sum(y2.to(torch.float64) for y2 in y2_locals)
        y2_final = y2_sum_fp64.to(dtype)

        intermediates = {
            "y1_locals": y1_locals,
            "y1_relu_locals": y1_relu_locals,
            "y2_locals": y2_locals,
            "y2_before_allreduce": y2_locals,
            "y2_after_allreduce": y2_final,
        }

        return y2_final, intermediates

    def forward_fp64_reference(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Ground truth computation in FP64."""
        x = x.to(torch.float64)

        y1 = x @ self.W1.T
        y1_relu = F.relu(y1)
        y2 = y1_relu @ self.W2.T

        intermediates = {
            "y1": y1,
            "y1_relu": y1_relu,
            "y2": y2,
        }

        return y2, intermediates


def compute_error_stats(tensor1: torch.Tensor, tensor2: torch.Tensor) -> Tuple[float, float]:
    """Compute max and mean absolute error between two tensors."""
    diff = (tensor1.to(torch.float64) - tensor2.to(torch.float64)).abs()
    return diff.max().item(), diff.mean().item()


def run_experiment(
    model: MLPDebug,
    x: torch.Tensor,
    world_size: int,
    dtype: torch.dtype,
) -> List[Dict]:
    """Run experiment for one configuration and collect error statistics."""
    results = []
    dtype_str = str(dtype).split(".")[-1]

    # Get FP64 ground truth
    y_ref_fp64, inter_ref_fp64 = model.forward_fp64_reference(x)

    # Run unsharded version in target dtype
    y_unsharded, inter_unsharded = model.forward_unsharded(x, dtype)

    # Run sharded version in target dtype
    y_sharded, inter_sharded = model.forward_sharded(x, world_size, dtype)

    # === Compare unsharded intermediate steps to FP64 reference ===

    # After linear1
    max_err, mean_err = compute_error_stats(inter_unsharded["y1"], inter_ref_fp64["y1"])
    results.append(
        {
            "world_size": world_size,
            "dtype": dtype_str,
            "mode": "unsharded",
            "step": "linear1",
            "rank": None,
            "max_error": max_err,
            "mean_error": mean_err,
            "comparison": "vs_fp64_ref",
        }
    )

    # After ReLU
    max_err, mean_err = compute_error_stats(inter_unsharded["y1_relu"], inter_ref_fp64["y1_relu"])
    results.append(
        {
            "world_size": world_size,
            "dtype": dtype_str,
            "mode": "unsharded",
            "step": "relu",
            "rank": None,
            "max_error": max_err,
            "mean_error": mean_err,
            "comparison": "vs_fp64_ref",
        }
    )

    # Final output
    max_err, mean_err = compute_error_stats(y_unsharded, y_ref_fp64)
    results.append(
        {
            "world_size": world_size,
            "dtype": dtype_str,
            "mode": "unsharded",
            "step": "final",
            "rank": None,
            "max_error": max_err,
            "mean_error": mean_err,
            "comparison": "vs_fp64_ref",
        }
    )

    # === Compare sharded intermediate steps to FP64 reference ===

    # For each rank, compare y1_local to corresponding slice of FP64 reference
    hidden_size = 4 * model.in_features
    chunk_size = hidden_size // world_size

    for rank in range(world_size):
        start_idx = rank * chunk_size
        end_idx = (rank + 1) * chunk_size

        # After linear1 for this rank
        y1_ref_slice = inter_ref_fp64["y1"][:, :, start_idx:end_idx]
        max_err, mean_err = compute_error_stats(inter_sharded["y1_locals"][rank], y1_ref_slice)
        results.append(
            {
                "world_size": world_size,
                "dtype": dtype_str,
                "mode": "sharded",
                "step": "linear1",
                "rank": rank,
                "max_error": max_err,
                "mean_error": mean_err,
                "comparison": "vs_fp64_ref_slice",
            }
        )

        # After ReLU for this rank
        y1_relu_ref_slice = inter_ref_fp64["y1_relu"][:, :, start_idx:end_idx]
        max_err, mean_err = compute_error_stats(
            inter_sharded["y1_relu_locals"][rank], y1_relu_ref_slice
        )
        results.append(
            {
                "world_size": world_size,
                "dtype": dtype_str,
                "mode": "sharded",
                "step": "relu",
                "rank": rank,
                "max_error": max_err,
                "mean_error": mean_err,
                "comparison": "vs_fp64_ref_slice",
            }
        )

        # After linear2 for this rank (before all_reduce)
        max_err, mean_err = compute_error_stats(inter_sharded["y2_locals"][rank], y_ref_fp64)
        results.append(
            {
                "world_size": world_size,
                "dtype": dtype_str,
                "mode": "sharded",
                "step": "linear2_pre_allreduce",
                "rank": rank,
                "max_error": max_err,
                "mean_error": mean_err,
                "comparison": "vs_fp64_ref",
            }
        )

    # After all_reduce
    max_err, mean_err = compute_error_stats(inter_sharded["y2_after_allreduce"], y_ref_fp64)
    results.append(
        {
            "world_size": world_size,
            "dtype": dtype_str,
            "mode": "sharded",
            "step": "after_allreduce",
            "rank": None,
            "max_error": max_err,
            "mean_error": mean_err,
            "comparison": "vs_fp64_ref",
        }
    )

    # === Compare sharded vs unsharded ===
    max_err, mean_err = compute_error_stats(y_sharded, y_unsharded)
    results.append(
        {
            "world_size": world_size,
            "dtype": dtype_str,
            "mode": "comparison",
            "step": "sharded_vs_unsharded",
            "rank": None,
            "max_error": max_err,
            "mean_error": mean_err,
            "comparison": "sharded_vs_unsharded",
        }
    )

    return results


def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Test configuration (matching numerical test)
    batch_size = 4
    seq_len = 8
    num_features = 1024

    world_sizes = [1, 2, 4, 8, 16]
    dtypes = [torch.float16, torch.float32, torch.float64]

    # Create model and input on CPU for controlled testing
    device = "cpu"
    model = MLPDebug(num_features, num_features).to(device)
    x = torch.randn(batch_size, seq_len, num_features, dtype=torch.float64, device=device)

    # Run all experiments
    print("Running debug accuracy experiments...")
    print(f"Model: MLP({num_features} -> {4 * num_features} -> {num_features})")
    print(f"Input shape: [{batch_size}, {seq_len}, {num_features}]")
    print(f"World sizes: {world_sizes}")
    print(f"Dtypes: {[str(d).split('.')[-1] for d in dtypes]}")
    print("\n" + "=" * 80 + "\n")

    all_results = []
    for dtype in dtypes:
        for world_size in world_sizes:
            results = run_experiment(model, x, world_size, dtype)
            all_results.extend(results)

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Analysis and reporting
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS - KEY FINDINGS")
    print("=" * 80 + "\n")

    # 1. Compare sharded vs unsharded final output
    print("1. SHARDED vs UNSHARDED COMPARISON")
    print("-" * 80)
    comparison_df = df[df["comparison"] == "sharded_vs_unsharded"]
    pivot = comparison_df.pivot_table(
        values="max_error", index="dtype", columns="world_size", aggfunc="first"
    )
    print("Max absolute error between sharded and unsharded:")
    print(pivot)
    print()

    # 2. Check if error is consistent across world sizes
    print("2. DOES ERROR SCALE WITH WORLD_SIZE?")
    print("-" * 80)
    for dtype_val in df["dtype"].unique():
        dtype_data = comparison_df[comparison_df["dtype"] == dtype_val]
        errors = dtype_data["max_error"].values
        print(f"{dtype_val}: {[f'{e:.6f}' for e in errors]}")
        if len(errors) > 1:
            std = dtype_data["max_error"].std()
            print(f"  -> Std dev: {std:.8f} (consistent if low)")
    print()

    # 3. Check where error first appears in sharded path
    print("3. ERROR ACCUMULATION IN SHARDED PATH")
    print("-" * 80)
    for dtype_val in ["float16", "float32"]:
        for ws in [2, 8]:
            print(f"\n{dtype_val}, world_size={ws}:")
            subset = df[
                (df["dtype"] == dtype_val)
                & (df["world_size"] == ws)
                & (df["mode"] == "sharded")
                & (df["rank"].isna() | (df["rank"] == 0))
            ][["step", "rank", "max_error", "comparison"]]

            # Show progression through pipeline
            for step in ["linear1", "relu", "linear2_pre_allreduce", "after_allreduce"]:
                row = subset[subset["step"] == step]
                if not row.empty:
                    error = row.iloc[0]["max_error"]
                    print(f"  {step:25s}: {error:.8f}")

    # 4. Compare unsharded errors
    print("\n4. UNSHARDED vs FP64 REFERENCE")
    print("-" * 80)
    unsharded_final = df[(df["mode"] == "unsharded") & (df["step"] == "final")][
        ["dtype", "world_size", "max_error"]
    ].drop_duplicates()

    for dtype_val in unsharded_final["dtype"].unique():
        error = unsharded_final[unsharded_final["dtype"] == dtype_val].iloc[0]["max_error"]
        print(f"{dtype_val}: {error:.8f}")

    # 5. Key hypothesis tests
    print("\n" + "=" * 80)
    print("HYPOTHESIS VALIDATION")
    print("=" * 80 + "\n")

    # Hypothesis 1: Error appears in linear1 (before all_reduce)
    print("H1: Error appears in intermediate activations (linear1) before all_reduce?")
    for dtype_val in ["float16", "float32"]:
        subset = df[
            (df["dtype"] == dtype_val)
            & (df["step"] == "linear1")
            & (df["mode"] == "sharded")
            & (df["rank"] == 0)
        ]
        if not subset.empty:
            avg_error = subset["max_error"].mean()
            print(f"  {dtype_val}: Avg max error at linear1 = {avg_error:.8f}")
            print(f"    -> {'YES' if avg_error > 1e-10 else 'NO'}: Error present in linear1")

    # Hypothesis 2: Error is independent of world_size
    print("\nH2: Is error magnitude independent of world_size?")
    for dtype_val in ["float16", "float32"]:
        subset = comparison_df[comparison_df["dtype"] == dtype_val]
        if len(subset) > 1:
            std = subset["max_error"].std()
            mean = subset["max_error"].mean()
            cv = std / mean if mean > 0 else 0
            print(f"  {dtype_val}: Mean={mean:.8f}, StdDev={std:.8f}, CV={cv:.4f}")
            print(
                f"{'YES' if cv < 0.1 else 'NO'}: Error is {'consistent' if cv < 0.1 else 'varies'} across world_sizes"
            )

    # Hypothesis 3: All-reduce (FP64) doesn't introduce error
    print("\nH3: Does all_reduce (in FP64) introduce additional error?")
    for dtype_val in ["float16", "float32"]:
        for ws in [2, 8]:
            before_ar = df[
                (df["dtype"] == dtype_val)
                & (df["world_size"] == ws)
                & (df["step"] == "linear2_pre_allreduce")
                & (df["rank"] == 0)
            ]
            after_ar = df[
                (df["dtype"] == dtype_val)
                & (df["world_size"] == ws)
                & (df["step"] == "after_allreduce")
            ]

            if not before_ar.empty and not after_ar.empty:
                err_before = before_ar.iloc[0]["max_error"]
                err_after = after_ar.iloc[0]["max_error"]
                print(f"  {dtype_val}, ws={ws}: Before={err_before:.8f}, After={err_after:.8f}")
                # Note: before all_reduce, each rank has partial sum, so error is different scale

    # Save full results
    output_file = "debug_accuracy_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n\nFull results saved to: {output_file}")
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
