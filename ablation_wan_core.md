# Strange-attractor stack ablation — preset `wan-core`

_Generated: 2026-04-26 22:37:50_

| combo | runs | mean (s) | ±stdev | speedup vs baseline | VRAM peak | params |
|---|---|---|---|---|---|---|
| baseline | 1/1 | 689.0 | – | 1.00x | 8.65 GB | enable_drift_gate=False, enable_sigma_streak=False, enable_twin_borrow=False |
| gate-only | 0/1 | – | – | – | 9.68 GB | enable_drift_gate=True, enable_sigma_streak=False, enable_twin_borrow=False |
| streak-only | 1/1 | 592.7 | – | 1.16x | 8.65 GB | enable_drift_gate=False, enable_sigma_streak=True, enable_twin_borrow=False |
| gate+streak | 0/1 | – | – | – | 9.68 GB | enable_drift_gate=True, enable_sigma_streak=True, enable_twin_borrow=False |

## Failures

- **gate-only**: wait_timeout_or_error
- **gate+streak**: wait_timeout_or_error
