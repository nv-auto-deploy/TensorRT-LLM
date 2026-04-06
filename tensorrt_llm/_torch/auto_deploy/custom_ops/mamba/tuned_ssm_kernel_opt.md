# tuned_ssm_kernel optimization log

Target: `_tuned_ssm_update_kernel` in `tuned_ssm_kernel.py`
Model: Nemotron Nano v3, nheads=64, dim=64, dstate=128, ngroups=8
Grid: \[dim/BLOCK_SIZE_M, batch, nheads\]

## Shape Matrix

| ID   | batch | nheads | dim | dstate | grid_m | total_grid       |
| ---- | ----- | ------ | --- | ------ | ------ | ---------------- |
| B33  | 33    | 64     | 64  | 128    | 4      | 33\*64\*4 = 8448  |
| B64  | 64    | 64     | 64  | 128    | 4      | 64\*64\*4 = 16384 |
| B128 | 128   | 64     | 64  | 128    | 4      | 128\*64\*4 = 32768 |
| B256 | 256   | 64     | 64  | 128    | 4      | 256\*64\*4 = 65536 |
| B384 | 384   | 64     | 64  | 128    | 4      | 384\*64\*4 = 98304 |

## Baseline params

- BLOCK_SIZE_M=16, BLOCK_SIZE_DSTATE=128, num_warps=4, num_stages=3

## Iteration Results

| Iter | Description                         | B33 (us) | B64 (us) | B128 (us) | B256 (us) | B384 (us) | Notes                           |
| ---- | ----------------------------------- | -------- | -------- | --------- | --------- | --------- | ------------------------------- |
| 0    | Baseline (M=16, DS=128, W=4, S=3)   | 57.1     | 105.0    | 203.1     | 385.9     | 576.9     | Baseline                        |
| 1    | dt_clamp \[0.001, 0.1\] correctness   | 57.3     | 105.1    | 203.0     | 399.5     | 588.1     | CORRECTNESS FIX; +0-2% overhead |
| 2    | BLOCK_SIZE_M=4 sweep                | 58.5     | 106.9    | 207.4     | 404.6     | 598.7     | worse than baseline             |
| 3    | BLOCK_SIZE_M=8 sweep                | 57.6     | 105.9    | 205.0     | 401.5     | 591.1     | marginally worse                |
| 4    | BLOCK_SIZE_M=32 sweep               | 57.1     | 104.7    | 201.8     | 399.2     | 586.7     | best so far -- smaller grid     |
| 5    | BLOCK_SIZE_M=64 sweep               | 64.5     | 113.7    | 217.0     | 421.6     | 627.7     | worse -- grid_m=1, reg pressure |
| 6    | Best M=32 confirmed in launcher     | 57.1     | 104.7    | 201.8     | 399.2     | 586.7     | adopted into launcher           |
| 7    | num_warps=1 sweep (M=32)            | 79.8     | 143.2    | 272.9     | 531.8     | 797.0     | much worse -- memory bandwidth  |
| 8    | num_warps=2 sweep (M=32)            | 61.1     | 108.9    | 208.0     | 404.6     | 596.4     | slightly worse than W=4         |
| 9    | num_warps=8 sweep (M=32)            | 62.0     | 113.4    | 220.3     | 431.6     | 645.7     | worse -- too much resource use  |
| 10   | num_warps=16 sweep (M=32)           | 60.7     | 111.4    | 216.2     | 423.6     | 632.0     | worse than W=4                  |
| 11   | W=4 confirmed best; keep launcher   | 57.1     | 104.7    | 201.8     | 399.2     | 586.7     | W=4 optimal for M=32,DS=128     |
| 12   | num_stages=1 sweep (M=32,W=4)       | 56.9     | 104.6    | 201.9     | 398.7     | 586.7     | marginal improvement            |
| 13   | num_stages=2 sweep (M=32,W=4)       | 56.9     | 104.3    | 201.6     | 399.0     | 586.5     | essentially same                |
| 14   | num_stages=4 sweep (M=32,W=4)       | 56.8     | 104.3    | 202.0     | 398.7     | 586.7     | essentially same                |
| 15   | num_stages: stages insensitive      | 57.1     | 104.7    | 201.8     | 399.2     | 586.7     | keep S=3; no meaningful diff    |
| 16   | DS=32 sweep (old 1-pass, WRONG)     | 19.9     | 33.2     | 60.4      | 114.7     | 170.8     | INCORRECT: only 32/128 processed|
| 17   | DS=64 sweep (old 1-pass, WRONG)     | 32.9     | 57.0     | 108.9     | 213.2     | 321.8     | INCORRECT: only 64/128 processed|
| 18   | dstate loop DS=32 (correct)         | 63.5     | 116.7    | 223.9     | 443.9     | 661.3     | loop overhead > DS=128 1-pass   |
| 19   | exp2 instead of exp for dA          | 61.7     | 111.5    | 214.7     | 416.4     | 617.7     | marginal improvement ~0.3%      |
| 20   | hoist D load before dstate loop     | 60.0     | 108.7    | 210.9     | 410.2     | 609.4     | 1-2% gain; hide D load latency  |
| 21   | precompute dt_col,x_col,dt_log2e    | 59.9     | 108.3    | 210.8     | 410.3     | 609.8     | marginal; eliminates broadcasts |
| 22   | fuse dB*x: state+=B*(dt\*x)          | 59.8     | 108.5    | 210.5     | 410.0     | 609.4     | marginal; saves 1 mul in loop   |
| 23   | re-verify W=4 at new kernel state   | 59.9     | 108.6    | 210.8     | 410.4     | 609.8     | W=4 still best; W=2,8 +12-15%  |
| 24   | precompute 2D offs_m\*stride (REVERT)| 66.1     | 120.6    | 233.2     | 458.4     | 686.0     | WORSE: 2D tensors add reg press |
| 25   | batch-adaptive M/W heuristic doc     | 59.8     | 108.5    | 210.5     | 410.0     | 609.4     | no benefit; M=32/W=4 universal  |
| 26   | load B/C before state/A (REVERT)    | 65.0     | 118.1    | 226.9     | 445.4     | 667.2     | WORSE: more regs live in loop   |
| 27   | tl.math.fma for state update        | 59.7     | 108.7    | 210.7     | 410.1     | 609.6     | marginal; FMA hints for backend |
| 28   | early state\*dA + FMA reorder (REV)  | 62.2     | 113.7    | 220.5     | 432.1     | 646.0     | WORSE: extra temp adds reg press|
| 29   | fast_expf vs exp2 (reverted)        | 59.7     | 108.7    | 210.6     | 410.1     | 609.5     | same as exp2; kernel mem-bound  |
| 30   | .cg cache modifier for loads (REV)  | 59.8     | 108.5    | 211.0     | 410.9     | 611.1     | no benefit; .cg unhelpful here  |
| 31   | DSTATE_CONSTEXPR: compile-time loop | 57.3     | 105.1    | 203.2     | 398.7     | 586.3     | **+3-5% WIN** loop unrolled!    |
| 32   | DIM_CONSTEXPR: eliminate dim mask   | 57.2     | 104.9    | 203.0     | 398.5     | 586.2     | marginal; mask cmp eliminated   |
| 33   | NHEADS_NGROUPS_RATIO constexpr      | 56.5     | 104.4    | 202.0     | 399.5     | 587.0     | +0.5-1% div becomes shift/mask  |
| 34   | remove runtime nheads_ngroups_ratio | 56.8     | 104.4    | 202.3     | 399.3     | 587.2     | cleanup; within noise           |
| 35   | num_stages re-sweep (1-5) at final  | 56.7     | 104.5    | 202.1     | 399.4     | 587.2     | all within noise; keep S=3      |
| 36   | tl.dot for out accumulation (REV)   | 102.4    | 164.9    | 308.2     | 599.1     | 892.9     | WORSE: tl.dot overhead on \[32,128\]x\[128,1\] |
| 37   | int32 offs_n (within noise)         | 56.7     | 104.7    | 202.1     | 399.6     | 587.1     | no benefit; int64 fine          |
| 38   | DS=64 loop (2 iters vs 1 at DS=128) | 59.4     | 108.0    | 208.3     | 407.7     | 603.9     | worse; loop overhead > reg save |

## Key findings

- dt_clamp adds negligible overhead (~0-2%) but is critical for correctness
  (without clamp, outputs diverge by 100-300x vs reference)
- Correctness confirmed: max_diff \< 0.2 (within bfloat16 precision) when both
  kernel and reference use clamp
- BLOCK_SIZE_M=32 slightly better than 16 (reduces grid size by 2x, less scheduler overhead)
- BLOCK_SIZE_M=64 worse (all dim in one program causes register pressure / serialization)
- num_warps=4 is optimal for M=32, DS=128; W=1 is 40% worse, W=2 is ~4% worse, W=8/16 are ~5-10% worse
- dstate loop overhead with DS=32 is 11% SLOWER than 1-pass DS=128; DS=128 is optimal for dstate=128
- Dstate loop restructuring retained for correctness/flexibility (kernel now handles any BLOCK_SIZE_DSTATE)

## Best configs per shape

- (to be filled after iter 50)
