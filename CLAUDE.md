# CLAUDE.md — Project Instructions for AI Agents

## What This Repo Is

This is the ComfyUI integration for Shannon-Prime VHT2 cross-attention caching.
It contains ComfyUI custom node definitions and the Wan-aware cache wrapper.

The core library lives in the `shannon-prime` repo (submodule at `lib/shannon-prime/`).
Never duplicate core code here. Import it.

## Wan Architecture — READ THIS FIRST

Wan 2.1/2.2 cross-attention K/V do NOT have RoPE. They are vanilla linear projections
of T5 text embeddings. This means:

1. K and V have SIMILAR spectral profiles (no asymmetry)
2. Both K and V get Möbius reorder (unlike self-attention where only K does)
3. Both get the same bit allocation (5/4/4/3, not 5/5/4/3 for K and flat 3 for V)

Wan 2.2 MoE has TWO experts with DIFFERENT cross_attn_k/cross_attn_v weights.
Cache MUST be keyed by (expert_id, block_index). Expert switch boundaries:
- T2V: σ = 0.875
- I2V: σ = 0.900

## Structure

```
shannon-prime-comfyui/
├── lib/
│   └── shannon-prime/            Git submodule → shannon-prime repo
├── nodes/
│   ├── __init__.py               ComfyUI node registration
│   └── shannon_prime_nodes.py    Custom node definitions
├── src/
│   └── shannon_prime_comfyui.py  Wan-aware cache (from shannon-prime/tools/)
├── workflows/
│   ├── wan21_vht2.json           Wan 2.1 example workflow
│   └── wan22_moe_vht2.json       Wan 2.2 MoE example workflow
├── tests/
│   └── test_comfyui.py           25 tests
├── docs/
│   └── INTEGRATION.md            Full guide with Wan architecture diagrams
├── CLAUDE.md
├── LICENSE
└── README.md
```

## Rules

- The core math is in `lib/shannon-prime/`. NEVER copy-paste it here.
- The Wan cache wrapper (src/) understands Wan 2.1, 2.2 MoE, and TI2V-5B.
- Custom nodes (nodes/) expose the cache to ComfyUI's node graph.
- Test with `python tests/test_comfyui.py` — 25 tests must pass.
- The hook point is WanAttentionBlock's cross_attn_k and cross_attn_v linear layers.
- WanCrossAttnCachingLinear is a drop-in nn.Module replacement for those layers.

## Wan Model Configs (from source)

| Model | dim | heads | layers | head_dim | MoE |
|-------|-----|-------|--------|----------|-----|
| Wan 2.1 14B | 5120 | 40 | 40 | 128 | No |
| Wan 2.1 1.3B | 1536 | 12 | 30 | 128 | No |
| Wan 2.2 A14B | 5120 | 40 | 40 | 128 | Yes (2 experts) |
| Wan 2.2 TI2V-5B | 3072 | 24 | 30 | 128 | No |
