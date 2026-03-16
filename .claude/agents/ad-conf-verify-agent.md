---
name: ad-conf-verify-agent
description: Run sanity check on winning AutoDeploy config by comparing generation output with baseline
tools: Read, Grep, Glob, Bash, Write, Edit, gpu-shell
model: sonnet
---

Run a sanity check on the winning AutoDeploy config by running `build_and_run_ad.py` and comparing generation output with the baseline.

## Inputs (from caller)

- **model**: HuggingFace model ID or local path
- **winner_config_yaml**: Path to the winning config YAML
- **baseline_log**: Path to the baseline `build_and_run_ad.py` output log
- **session_dir**: Path to the session directory
- **world_size**: Number of GPUs needed

If any required inputs are missing, ask the caller.

## Workflow

### 0. GPU Selection

Follow the same GPU selection pattern as `ad-run-agent`:

1. Run via gpu-shell:
   ```bash
   nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
   ```
2. A GPU is **free** if memory usage < 1000 MiB and utilization is 0%.
3. Select `world_size` contiguous free GPUs (prefer lowest indices).
4. If not enough free GPUs: report which are busy, wait 60 seconds, check again.

### 1. Run build_and_run_ad.py with Winner Config

Run via gpu-shell:
```bash
CUDA_VISIBLE_DEVICES=<GPUs> python examples/auto_deploy/build_and_run_ad.py \
  --model <MODEL> \
  --args.yaml-extra <winner_config_yaml> \
  2>&1 | tee <session_dir>/verify/verify.log
```

Create `<session_dir>/verify/` directory if it doesn't exist.

### 2. Compare Generation Output

Read both the baseline log and the verify log. Extract the generated text from each.

**Quality checks:**
1. **Not empty** — verify the model produced non-empty output
2. **Not garbled** — check for random character sequences, encoding artifacts
3. **Not repetitive** — check for excessive repetition (same phrase repeated 5+ times)
4. **Coherent** — the output should be roughly similar in quality to the baseline (same language, on-topic for the prompt)

**Note:** The outputs don't need to be identical — different configs may produce slightly different text. The check is for catastrophic degradation (garbled, empty, or repetitive output).

### 3. Report Result

**PASS** if:
- Model compiled and ran successfully
- Generated text passes all quality checks

**FAIL** if:
- Compilation error
- Runtime error
- Generated text fails any quality check (empty, garbled, or excessively repetitive)

Update `<session_dir>/session_log.md`:
```markdown
## Verification: <winner_config_yaml>

**Status:** PASS / FAIL
**Log:** <session_dir>/verify/verify.log
**Notes:** <brief description of comparison result>
```

### 4. Return to Caller

Return:
- **status**: `PASS` or `FAIL`
- **verify_log_path**: Path to the verify log
- **notes**: Brief description of the result (e.g., "Generation quality matches baseline" or "Output is garbled — possible numerical issue with this config")
