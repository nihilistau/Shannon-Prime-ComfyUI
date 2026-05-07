# Strange-attractor stack ablation — preset `voxtral-tail`

_Generated: 2026-04-27 01:53:15_

| combo | runs | mean (s) | ±stdev | speedup vs baseline | VRAM peak | params |
|---|---|---|---|---|---|---|
| vox-baseline | 0/1 | – | – | – | – | k_band_bits=5,5,4,3 |
| vox-ternary-tail | 0/1 | – | – | – | – | k_band_bits=5,5,4,3, k_ternary_bands=3 |

## Failures

- **vox-baseline**: Prompt outputs failed validation | node 2: Bad linked input, must be a length-2 list of [node_id, slot_index]
- **vox-ternary-tail**: Prompt outputs failed validation | node 2: Bad linked input, must be a length-2 list of [node_id, slot_index]
