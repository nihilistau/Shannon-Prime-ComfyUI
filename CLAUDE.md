# CLAUDE.md — shannon-prime-comfyui

## What This Repo Is

ComfyUI integration for Shannon-Prime VHT2 cross-attention caching on Wan
video generation models. Contains ComfyUI custom node definitions and the
Wan-aware cache wrappers. Core library lives in the `shannon-prime` repo as
a submodule at `lib/shannon-prime/`; never duplicate core code here, import
it.

The underlying transform is **VHT2** (Vilenkin-Hartley Transform) — the
single orthonormal staged Hartley transform that reduces to the classical
Walsh-Hadamard butterfly at n=2^k and extends to primes {2,3,5,7,11} at
sqfree-padded dims. Both the
ship cache (`ShannonPrimeWanCache`) and the aggressive sqfree+spinor variant
(`ShannonPrimeWanCacheSqfree`) use it.

## Wan Architecture — READ THIS FIRST

Wan 2.1/2.2 cross-attention K/V do **NOT** have RoPE — they are vanilla
linear projections of T5 text embeddings. This means:

1. K and V have SIMILAR spectral profiles (no asymmetry).
2. Both K and V get Möbius reorder (unlike self-attention where only K does).
3. Both get the same bit allocation (5/4/4/3, not 5/5/4/3 for K and flat 3 for V).

Wan 2.2 MoE has TWO experts with DIFFERENT cross_attn_k/cross_attn_v weights.
Cache MUST be keyed by (expert_id, block_index). Expert switch boundaries:
- T2V: σ = 0.875
- I2V: σ = 0.900

## Structure

```
shannon-prime-comfyui/
├── lib/
│   └── shannon-prime/              Submodule → shannon-prime repo
├── nodes/
│   ├── __init__.py                 ComfyUI node registration
│   └── shannon_prime_nodes.py      ShannonPrimeWanCache (ship) +
│                                   ShannonPrimeWanCacheSqfree (aggressive)
├── src/                            (empty — wrappers live in submodule tools/)
├── workflows/
│   ├── wan22_ti2v_5b_vht2.json     Wan 2.2 TI2V-5B ship-path example
│   └── wan22_ti2v_5b_sqfree.json   Wan 2.2 TI2V-5B sqfree+spinor example
├── scripts/
│   └── run_workflow.py             HTTP client for ComfyUI API
├── outputs/                        Reference webp outputs
├── ppl_logs/                       Reference runtime logs
├── tests/                          25 tests (in lib/shannon-prime/tests/test_comfyui.py)
├── docs/
│   └── INTEGRATION.md              Architecture + setup guide
└── README.md
```

## Rules

- Core math is in `lib/shannon-prime/`. NEVER copy-paste it here.
- The Wan cache wrappers (`lib/shannon-prime/tools/shannon_prime_comfyui.py`
  and `shannon_prime_comfyui_sqfree.py`) understand Wan 2.1, 2.2 MoE, TI2V-5B.
- Custom nodes (nodes/) expose the cache to ComfyUI's node graph.
- `ShannonPrimeWanCacheSqfree` is the opt-in aggressive variant; it uses
  `VHT2SqfreeCrossAttentionCache` + `WanSqfreeCrossAttnCachingLinear` from
  the sqfree module. Note: the sqfree wrapper currently caches only the first
  (position, head) slot as a demonstration — full per-(pos, head) per-expert
  sqfree compression for cross-attn K/V is a follow-up once `SqfreeShadowCache`
  slot capacity is parameterised.
- Test with `python tests/test_comfyui.py` — 25 tests must pass.

## Wan Model Configs (from source)

| Model | dim | heads | layers | head_dim | MoE |
|-------|-----|-------|--------|----------|-----|
| Wan 2.1 14B | 5120 | 40 | 40 | 128 | No |
| Wan 2.1 1.3B | 1536 | 12 | 30 | 128 | No |
| Wan 2.2 A14B | 5120 | 40 | 40 | 128 | Yes (2 experts) |
| Wan 2.2 TI2V-5B | 3072 | 24 | 30 | 128 | No |

## Quick Validation

```bash
# Start ComfyUI with the junction / symlink pointing at this repo
python main.py --listen 127.0.0.1 --port 8188

# Queue the ship workflow
python /path/to/shannon-prime-comfyui/scripts/run_workflow.py \
    /path/to/shannon-prime-comfyui/workflows/wan22_ti2v_5b_vht2.json

# Or the sqfree+spinor aggressive workflow
python .../scripts/run_workflow.py .../workflows/wan22_ti2v_5b_sqfree.json
```

Expect `[Shannon-Prime] patched 30/30 Wan blocks (head_dim=128, ...)` in the
server log, followed by an 8-step KSampler run (~65s on RTX 2060 for the
ship path) and a `sp_wan_vht2_00001_.webp` / `sp_wan_sqfree_00001_.webp`
output.
