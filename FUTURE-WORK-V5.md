# Future Work — v5 and Beyond

This document records the pieces of the *Music of the Spheres* framework
that haven't shipped yet, and the ambitions that go beyond the current
paper. The working principle: file ideas here so they don't get lost,
then build them when the prerequisites are in place.

---

## Status as of v6

Most of what was originally listed in v5 priority queue has shipped:

- ✅ **Strict 1D-circle Granite reconstruction** — landed v5 (`enable_one_dim_granite`)
- ✅ **Cross-tier energy borrowing** — landed v5 (`enable_cross_tier_borrow`)
- ✅ **Per-token VHT2 skeleton fraction** — landed v5 (`enable_per_token_skeleton`)
- ✅ **Higher-order integrators** — AB2 in v4, AB3 in v5 (`harmonic_order=3_ab3`); RK4 still future
- ✅ **Goldbach gap-8/10/12** — landed v5 (`goldbach_max_gap`)
- ✅ **Direct Lyapunov-spectrum measurement** — landed v6 (`ShannonPrimeWanLyapunovSnapshot` + `scripts/sp_lyapunov_analyze.py`)

What's still genuinely future:

- ❌ **Decode-path 1.58-bit ternary spatial sketch** (depends on per-token skel
  validation)
- ❌ **RK4 / symplectic integrators** beyond AB3
- ❌ **Goldbach extension to gap-14+** (trivial — `_goldbach_pairs(n, gaps=...)`,
  but needs a reason)
- ❌ **Cross-architecture generalization** to SSMs, RNNs (research)
- ❌ **Closed-form regime boundary prediction** from architecture (research)
- ❌ **Hardware-accelerated VHT2+Möbius+banded fused kernel** for sub-1B real-time
  models (engineering, ~2 weeks)

---

## The big new ambition — Multimodal Phase-Sync

Filed for the record because the metaphor demands it but the engineering
needs to be honest about what's hard.

The proposal: link Voxtral TTS and Shannon-Prime Wan video at the arithmetical
level via a shared phase anchor. Voice spectral flux drives video phase
velocity; voice plosives trigger video Cauchy resets; voice fundamental
frequency F₀ stabilizes the video Granite pillars. The two pipelines stop
being "separately running and then synced" and become one manifold expressing
itself through two shadows — pixel and phoneme.

### The mechanism the metaphor predicts

```
θ_video(t) = θ_base + ∫₀ᵗ Ψ(Audio(τ)) dτ
```

where Ψ is a coupling function that maps audio spectral flux into per-step
phase velocity for the video. High audio flux (sibilance) accelerates Sally
through the Jazz layers; deep stable vowels phase-lock the video to its
Radix-7 pillars. The Hamiltonian sentinel reinterprets audio energy as
arithmetical kinetic energy. The Cauchy reset fires on plosives.

A proposed C-side connector:

```cpp
typedef struct {
    float spectral_flux;     // d|FFT|/dt of the audio buffer
    float fundamental_f0;    // pitch anchor
    float phase_velocity;    // resultant θ-shift
    bool  trigger_reset;     // plosive/stop signal
} VoxtralPhaseState;
```

passed via pointer from the Voxtral C++ engine to a `vht2_wan_kernel`
extension that ingests the state on every CUDA call.

### What's tractable about this

- **Aligned generation timestamps.** TTS generates tokens at known sample
  rates; video diffusion has known step counts. A coarse "audio frame N
  drives video step M" mapping is mechanical and measurable.

- **Plosive-triggered resets.** Detecting plosives in the audio buffer is
  signal processing 101 (high-pass + envelope follower). Wiring that
  trigger to the existing temporal Cauchy reset on the video side is a
  ~50-line addition once the audio pipeline emits a flag.

- **Coarse spectral coupling.** Audio FFT bin energies modulating video
  cache thresholds is a real and testable hypothesis. If it produces
  better lip-sync or motion-to-music coherence, that's a concrete win.

### What's hard or speculative about this

- **The shared spectral basis assumption.** Audio Hartley-frequency space
  and video VHT2 head-dim space are not the same manifold in any
  rigorous sense. The metaphor wants them to be twin shadows of one
  attractor; the math wants them to be unrelated until proven otherwise.
  The empirical question is whether spectral-coupling-as-modulation
  produces measurably better output than naive audio-conditioning. It
  might. It might not.

- **Microsecond-level sync isn't the bottleneck.** ComfyUI's video
  diffusion has step times in the hundreds of milliseconds. Coupling
  audio with microsecond resolution doesn't help if the video can only
  react every 200ms. The latency budget is in the wrong place.

- **C-pointer passing across Python/CUDA boundaries.** The proposal
  bypasses Python by sharing a C struct between Voxtral and the video
  kernel. That's operationally fragile — version skew between the two
  C++ libraries, lifetime management of the shared struct, threading
  hazards. Worth doing only if there's a measured win that justifies
  the friction.

- **The "1D circle is universal" claim is the load-bearing one.** If
  audio and video really do share a 1D phase circle in some prime-
  harmonic basis, the v5 strict 1D-circle Granite work has already
  shipped half of it. Validate that on video first; come back to
  multimodal coupling once we know whether the video side's 1D
  approximation actually works in practice.

### The right order of operations

1. Validate v5's strict 1D-circle Granite on real Wan workflows. If
   granite λ ≈ 0 in the v6 Lyapunov measurement, the 1D approximation
   is empirically justified — green light for spectral coupling.
2. Build a Python-side audio→video coupling first (no C-pointer
   surgery): expose spectral flux from Voxtral as an input to a
   ComfyUI node, modulate the video's `harmonic_strength` or
   `curvature_threshold` by it. Two days of work, easy to back out.
3. If the Python coupling produces measurable lip-sync / motion
   coherence improvements, *then* drop into the C++ layer for the
   pointer-sharing version.
4. The "shared 1D phase circle" theoretical claim is paper-level
   work; the engineering work is the modulation hookup.

### Storage and architecture sketch (when the time comes)

- New ComfyUI node: `ShannonPrimeMultimodalPhaseSync` that accepts an
  AUDIO input (the Voxtral output) and emits a `PHASE_STREAM` output
  carrying the four-field VoxtralPhaseState equivalent.
- BlockSkip and BlockSkip-adjacent nodes accept `PHASE_STREAM` as an
  optional input. When present, they modulate per-step:
  - `harmonic_strength` ← scaled by `spectral_flux`
  - `temporal_cauchy` ← OR'd with `trigger_reset`
  - drift-gate thresholds ← shifted by `fundamental_f0` proximity
- The C-pointer version is a v∞ optimization once the Python version
  is validated.

This is the destination if the framework is correct. It's also at
least a six-month research project with three different fail modes.
File it. Don't build it tonight.

---

## V5 priority queue (now mostly shipped — kept for archival)

### 1. Strict 1D-circle Granite reconstruction

The headline compression claim from §3.3 / §10 of the theory paper. Granite
blocks (L00–L03) sit at cos_sim > 0.999 across many steps; their cache
trajectory is well-approximated by a single-parameter family of vectors
generated by phase rotation on a fixed prime-harmonic skeleton. Storing
*only* the scalar phase θ per Granite block would yield ~100×+ compression
on the Granite tier specifically.

**What to build:**

- A reference vector V₀ per (block_idx, generation), captured at the first
  miss step
- A phase-extraction operator: given a candidate y, recover the scalar θ
  that best aligns y with R(θ)·V₀. Numerically this is `θ = arg max_θ
  ⟨y, R(θ)·V₀⟩` — one-dimensional optimization, can be Newton-step
- A reconstruction operator: y_recon = R(θ̂) · V₀ where R is the block-
  diagonal RoPE-style rotation

**Storage shape:**

  Granite cache becomes: { θ: float, V0_id: int } per (block, position).
  Compared to current 23 spectral coefficients × 2 bytes/coef = 46 bytes
  per token, the new representation is 4 bytes (one fp32 θ) + V₀ shared.
  A 10×+ reduction on Granite, conservatively; the headline 100× requires
  V₀ shared across many positions which works because Granite is also
  spatially uniform.

**Risk:**
  The phase-extraction must be robust to the cache content actually NOT
  fitting the 1D-circle assumption (e.g., when a Granite block is mid-
  transition between basins). Need a residual norm ‖y − R(θ̂)·V₀‖ that, if
  too large, signals "fall back to skeleton" mode.

**Estimated scope:** ~400 lines, two-week experiment cycle, requires bench
validation against quality regression tests.


### 2. Cross-tier twin-prime energy borrowing

The metaphor's "bidirectional flow" from Gemini's notes — when a Jazz block
fails its sentinel, it can borrow stabilizing skeleton coefficients from a
Granite block at the same spatial position.

**What to build:**

- Per-tier averaged-skeleton buffers (one per tier, updated from misses)
- A donation policy: when a block's drift gate or curvature gate fires, it
  reads the corresponding-tier-OR-stabler-tier averaged skeleton, blends it
  with its own at small α, and writes the result as the new local cache

**Risk:**
  Cross-tier blending could introduce *backwards* drift — Jazz pulling
  Granite toward instability. Donation must be one-directional (stabler
  tier → less stable tier only).

**Estimated scope:** ~250 lines + ablation. Important to compare against
the existing Cauchy reset which is the simpler "just refresh" answer to
the same problem.


### 3. Per-token VHT2 skeleton fraction (v3.1)

V3 implemented per-token harmonic correction. The natural next step is
per-token *skeleton fraction* — subject tokens decompressed from a richer
spectral basis, background tokens from a sparser one. This is the full
realization of the foveated-heatmap claim from §8 of the paper.

**What to build:**

- Spatial-aware mask downsampling: given (T, H_p, W_p) patch grid, resize
  the user-provided mask to that shape exactly. Currently v3 uses 1D
  positional interpolation which loses spatial structure.
- Dual-skeleton storage: subject tokens stored with `granite_skel_frac`
  worth of coefficients, background tokens with a separate (lower)
  `background_skel_frac`. Shape: each cached block has TWO subset masks
  rather than one.
- Decode-path branching: depending on per-token mask, scatter into the
  appropriate skeleton size before inverse VHT2.

**Risk:**
  The cache store/load path becomes more complex; TURBO-mode (cache_ffn)
  needs a parallel split. Memory footprint slightly increases (need to
  remember which tokens are subject) until per-token compression delivers
  the offsetting reduction.

**Estimated scope:** ~500 lines + bench. The largest of the v5 pieces.


### 4. Higher-order geodesic integrators beyond AB2

V4 implemented Adams-Bashforth-2 (AB2). Natural extensions:

- AB3 / AB4: more previous y values, higher-order accurate, more cache
- Verlet (symplectic): preserves "energy" (Hamiltonian) exactly to order 2
- RK4 (Runge-Kutta): single-step but evaluates velocity 4× per call

**Why these matter:** AB2 is exact for quadratic trajectories. Granite
trajectories may be even *more* slowly-varying than that, in which case
even AB2 is over-engineered; Sand and Jazz may be where higher-order
helps most.

**Estimated scope:** ~100 lines per integrator, ablation needed.


### 5. Goldbach gap extension to gap-8/10/12+

V2's Goldbach extension covers gap-2/4/6. The Goldbach conjecture says any
even gap ≥ 2 contains valid prime pairs (with decreasing density at larger
gaps). At head_dim=128:

- gap-8 cousins: 5-13, 11-19, 23-31, 53-61, 89-97, ...
- gap-10: 3-13, 7-17, 13-23, 19-29, ...
- gap-12: 5-17, 7-19, 11-23, ...

α auto-scales as 2/gap so gap-12 borrows at α/6.

**What to build:** trivial — `_goldbach_pairs(n, gaps=(2,4,6,8,10,12))`.
The `enable_goldbach_pairs` toggle would gain a `goldbach_max_gap`
parameter (default 6, can extend to 12+).

**Estimated scope:** ~30 lines, ablation needed.


### 6. Direct Lyapunov-spectrum measurement

The strange-attractor view predicts positive Lyapunov exponent in at least
one direction. Direct measurement is feasible: pair two generations from
slightly different prompts, track divergence of cosine similarity over
denoising steps, fit exponential. Per-block / per-channel Lyapunov gives
a curvature map of the manifold.

**What to build:**
- A `ShannonPrimeWanLyapunov` diagnostic node that runs two parallel
  generations and emits a per-block divergence timeline
- Optional integration into the curvature gate's threshold tuning

**Estimated scope:** ~300 lines. Diagnostic, not load-bearing for inference.


### 7. Decode-path 1.58-bit ternary spatial sketch

For the V3 foveated-mask line specifically: store the ENTIRE background
region's cached y as a ternary {−1, 0, +1} bit-mask + a single scaling
factor per block. This is the literal "1-bit Sketch" claim from §8.

**Risk:** ternary at 1.58 bits is the limit; quality on background regions
will be roughly the same as 4-bit quantization but at ~3× compression.

**Estimated scope:** depends on whether per-token VHT2 (#3) lands first.


---

## Beyond v5: speculative pieces

These are research questions, not engineering tasks.

- **Closed-form regime boundary prediction**: given an architecture's
  rotation-frequency spacing and depth, predict where Granite/Sand/Jazz
  boundaries fall without empirical observation. Currently the boundaries
  are observed.

- **Cross-architecture generalization**: does the strange-attractor /
  10D-manifold framing transfer to state-space models (Mamba), recurrent
  networks with positional structure, or other non-attention architectures?
  The theory predicts yes.

- **Conserved quantities in multi-modal cache trajectories**: when the
  same model generates video and audio simultaneously (e.g., V2A
  pipelines), is there a coupling invariant that the trajectory respects?
  Goldbach + temporal Cauchy hint at a yes.

- **Hardware acceleration of the prime-harmonic basis**: a custom CUDA
  kernel that fuses VHT2 + Möbius reorder + banded quantize in a single
  pass on real-time-targeted small models (sub-1B). Could enable
  real-time arbitrary-resolution video on consumer hardware.

---

## Roadmap discipline

The intent of this document is operational: when v4 has been validated on
the bench and the operator wants the next move, this list is the menu.
Each piece is sized so it can be landed cleanly on a v5-numbered branch.

The framework is stable enough that the *order* of pieces matters less
than the *commitment* to land them. The stack is the kind of structure
where each piece compounds with the others. Strict 1D-circle wants
per-token VHT2 wants Lyapunov-tuned thresholds wants Goldbach-extended
borrow wants AB3 integration. The end state is the manifold operationally
realized end-to-end.

---

*Maintained by the Shannon-Prime project. Updates welcome via PR or
issue. Companion theory paper: `music_of_the_spheres.md` at the workspace
root.*
