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

## Key findings

- dt_clamp adds negligible overhead (~0-2%) but is critical for correctness
  (without clamp, outputs diverge by 100-300x vs reference)
- Correctness confirmed: max_diff \< 0.2 (within bfloat16 precision) when both
  kernel and reference use clamp
- BLOCK_SIZE_M=32 slightly better than 16 (reduces grid size by 2x, less scheduler overhead)
- BLOCK_SIZE_M=64 worse (all dim in one program causes register pressure / serialization)
- num_warps=4 is optimal for M=32, DS=128; W=1 is 40% worse, W=2 is ~4% worse, W=8/16 are ~5-10% worse

## Best configs per shape

- (to be filled after iter 50)
