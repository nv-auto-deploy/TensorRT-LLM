# tuned_ssm_kernel optimization log

Target: `_tuned_ssm_update_kernel` in `tuned_ssm_kernel.py`
Model: Nemotron Nano v3, nheads=64, dim=64, dstate=128, ngroups=8
Grid: \[dim/BLOCK_SIZE_M, batch, nheads\]

## Shape Matrix

| ID   | batch | nheads | dim | dstate | grid_m | total_grid        |
|------|-------|--------|-----|--------|--------|-------------------|
| B33  | 33    | 64     | 64  | 128    | 4      | 33*64*4 = 8448    |
| B64  | 64    | 64     | 64  | 128    | 4      | 64*64*4 = 16384   |
| B128 | 128   | 64     | 64  | 128    | 4      | 128*64*4 = 32768  |
| B256 | 256   | 64     | 64  | 128    | 4      | 256*64*4 = 65536  |
| B384 | 384   | 64     | 64  | 128    | 4      | 384*64*4 = 98304  |

## Baseline params

- BLOCK_SIZE_M=16, BLOCK_SIZE_DSTATE=128, num_warps=4, num_stages=3

## Iteration Results

| Iter | Description                        | B33 (us) | B64 (us) | B128 (us) | B256 (us) | B384 (us) | Notes |
|------|------------------------------------|----------|----------|-----------|-----------|-----------|-------|
| 0    | Baseline (M=16, DS=128, W=4)       | TBD      | TBD      | TBD       | TBD       | TBD       | Baseline |
| 1    | dt_clamp \[0.001, 0.1\] correctness  | TBD      | TBD      | TBD       | TBD       | TBD       | CORRECTNESS FIX |

## Key findings

- (to be filled)

## Best configs per shape

- (to be filled after iter 50)
