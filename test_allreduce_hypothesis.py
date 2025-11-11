import torch


def test_matmul_equivalence():
    """Test if partial matmuls equal slices of full matmul on CPU."""
    torch.manual_seed(42)

    B, S, D = 4, 8, 1024
    hidden = 4 * D

    # Create data in FP64, then cast
    X = torch.randn(B, S, D, dtype=torch.float64)
    W1 = torch.randn(hidden, D, dtype=torch.float64)

    dtype = torch.float16
    X_fp16 = X.to(dtype)
    W1_fp16 = W1.to(dtype)

    # Full matmul
    Y1_full = X_fp16 @ W1_fp16.T  # [B, S, hidden]

    # Partial matmuls
    ws = 2
    chunk = hidden // ws

    Y1_part0 = X_fp16 @ W1_fp16[0:chunk, :].T
    Y1_part1 = X_fp16 @ W1_fp16[chunk : 2 * chunk, :].T

    # Compare
    diff0 = (Y1_part0 - Y1_full[:, :, 0:chunk]).abs().max().item()
    diff1 = (Y1_part1 - Y1_full[:, :, chunk : 2 * chunk]).abs().max().item()

    print("=" * 80)
    print("TEST 1: Are partial matmuls identical to slices of full matmul?")
    print("=" * 80)
    print(f"Max diff for part 0: {diff0}")
    print(f"Max diff for part 1: {diff1}")
    print(f"Result: {'IDENTICAL' if diff0 == 0 and diff1 == 0 else 'DIFFERENT'}")
    print()

    return Y1_full, [Y1_part0, Y1_part1]


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
    # print()
    # print(f"Conclusion: Hierarchical summation {'DOES' if diff_fp64.max() > 1e-7 else 'DOES NOT'} cause differences")
    # print(f"            even with FP64 accumulation in all_reduce")
    # print()


def test_elementwise_vs_matmul():
    """Test whether difference comes from matmul internal accumulation vs final sum."""
    torch.manual_seed(42)

    B, S, D = 4, 8, 1024
    hidden = 4 * D

    Y1_relu = torch.randn(B, S, hidden, dtype=torch.float16)
    W2 = torch.randn(D, hidden, dtype=torch.float16)

    # Method 1: Full matmul
    Y2_full = Y1_relu @ W2.T

    # Method 2: Manual element-wise computation to isolate accumulation
    # For one output element: Y2[i,j] = sum_k(Y1[i,k] * W2[j,k])
    # Y2_manual = torch.zeros(B, S, D, dtype=torch.float16)

    # Compute first output element manually
    i, j = 0, 0

    # Full accumulation
    full_sum = sum(Y1_relu[i, j, k] * W2[0, k] for k in range(hidden))

    # Hierarchical accumulation (ws=2)
    chunk = hidden // 2
    partial_0 = sum(Y1_relu[i, j, k] * W2[0, k] for k in range(0, chunk))
    partial_1 = sum(Y1_relu[i, j, k] * W2[0, k] for k in range(chunk, hidden))
    hierarchical_fp16 = partial_0 + partial_1
    hierarchical_fp64 = partial_0.to(torch.float64) + partial_1.to(torch.float64)
    hierarchical_fp64 = hierarchical_fp64.to(torch.float16)

    print("=" * 80)
    print("TEST 3: Element-wise accumulation analysis")
    print("=" * 80)
    print(f"Full matmul result [0,0,0]: {Y2_full[0, 0, 0].item():.8f}")
    print(f"Full accumulation:           {full_sum.item():.8f}")
    print(f"Hierarchical (FP16):         {hierarchical_fp16.item():.8f}")
    print(f"Hierarchical (FP64):         {hierarchical_fp64.item():.8f}")
    print()
    print(f"Diff (full vs hier-FP16):    {abs(full_sum.item() - hierarchical_fp16.item()):.8f}")
    print(f"Diff (full vs hier-FP64):    {abs(full_sum.item() - hierarchical_fp64.item()):.8f}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("ISOLATING THE SOURCE OF NUMERICAL DIFFERENCES")
    print("=" * 80 + "\n")

    # # Test 1: Are partial matmuls identical?
    # Y1_full, Y1_parts = test_matmul_equivalence()

    # Test 2: Does hierarchical summation matter?
    for ws in [1, 2, 4, 8, 16, 32]:
        test_summation_order(ws)

    # # Test 3: Element-wise analysis
    # test_elementwise_vs_matmul()

    # print("\n" + "="*80)
    # print("SUMMARY")
    # print("="*80)
    # print()
    # print("If Test 1 shows IDENTICAL: Partial matmuls are equivalent on CPU")
    # print("If Test 2 shows difference: The hierarchical summation is the cause")
    # print("If Test 3 shows difference: Even FP64 all_reduce can't prevent errors")
    # print("                            because partials are already computed in FP16")
    # print()
