import torch


def test_summation_order(ws: int = 2):
    """Test if different summation orders cause numerical differences."""
    torch.manual_seed(42)

    B, S, D = 4, 8, 1024
    hidden = 4 * D

    # Create Y1_relu and W2
    Y1_relu = torch.randn(B, S, hidden, dtype=torch.float16)
    W2 = torch.randn(D, hidden, dtype=torch.float16)

    # Method 1: Full matmul (unsharded)
    Y2_full = Y1_relu @ W2.T  # [B, S, D]

    # Method 2: Split computation + sum in FP16 (mimics sharded with FP16 all_reduce)
    chunk = hidden // ws
    Y2_partial_fp16 = []
    for rank in range(ws):
        start, end = rank * chunk, (rank + 1) * chunk
        Y1_local = Y1_relu[:, :, start:end]
        W2_local = W2[:, start:end]
        Y2_local = Y1_local @ W2_local.T
        Y2_partial_fp16.append(Y2_local)

    Y2_split_fp16 = sum(Y2_partial_fp16)

    # Method 3: Split computation + sum in FP64 (mimics sharded with FP64 all_reduce)
    Y2_partial_fp64 = [p.to(torch.float64) for p in Y2_partial_fp16]
    Y2_split_fp64 = sum(Y2_partial_fp64).to(torch.float16)

    # Compare
    diff_fp16 = (Y2_split_fp16 - Y2_full).abs()
    diff_fp64 = (Y2_split_fp64 - Y2_full).abs()

    print("=" * 80)
    print("TEST 2: Does hierarchical summation cause numerical differences?")
    print("=" * 80)
    print(f"WORLD SIZE {ws}: Full matmul vs split-sum-FP16:")
    print(f"  Max diff: {diff_fp16.max().item():.8f}")
    print(f"  Mean diff: {diff_fp16.mean().item():.8f}")
    print()
    print("Full matmul vs split-sum-FP64:")
    print(f"  Max diff: {diff_fp64.max().item():.8f}")
    print(f"  Mean diff: {diff_fp64.mean().item():.8f}")


if __name__ == "__main__":
    for ws in [1, 2, 4, 8, 16, 32]:
        test_summation_order(ws)
