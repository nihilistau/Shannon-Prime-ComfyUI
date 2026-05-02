"""
Shannon-Prime Lyapunov Spectrum Analyzer.

Take two snapshot files produced by `ShannonPrimeWanLyapunovSnapshot` from
two slightly-different runs of the same workflow (different prompt or seed),
compute per-block divergence curves and fit Lyapunov exponents. The output
empirically confirms (or refutes) the Granite/Sand/Jazz attractor regime
structure and produces auto-tuned drift-gate thresholds.

Usage:
    python scripts/sp_lyapunov_analyze.py snapshots/run_a.npz snapshots/run_b.npz
    python scripts/sp_lyapunov_analyze.py run_a.npz run_b.npz --plot --auto-tune

Inputs: two .npz snapshots with arrays
    blocks  int32 [N_blocks]
    steps   int32 [N_steps]
    fp      float32 [N_blocks, N_steps, fingerprint_dim]

Outputs:
    Per-block Lyapunov exponent (slope of log-cosine-distance over time)
    Tier classification (Granite / Sand / Jazz) inferred from the spectrum
    Suggested drift-gate threshold per tier
    Optional matplotlib plot of divergence curves

Stdlib + numpy only. matplotlib is optional (--plot flag).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ── Loading ──────────────────────────────────────────────────────────────

def load_snapshot(path: str) -> dict:
    """Load a single snapshot .npz produced by the snapshot node."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"snapshot not found: {p}")
    data = np.load(p)
    return {
        "blocks": data["blocks"].astype(np.int32),
        "steps":  data["steps"].astype(np.int32),
        "fp":     data["fp"].astype(np.float32),
        "path":   str(p),
    }


def align_pair(a: dict, b: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Intersect block indices and step ranges between the two runs."""
    common_blocks = sorted(set(a["blocks"].tolist()) & set(b["blocks"].tolist()))
    if not common_blocks:
        raise ValueError("no common block indices between the two snapshots")
    n_steps = min(a["fp"].shape[1], b["fp"].shape[1])
    if n_steps < 4:
        raise ValueError(f"only {n_steps} common steps — need at least 4 for a slope fit")

    # Re-index
    a_idx = [list(a["blocks"]).index(b_) for b_ in common_blocks]
    b_idx = [list(b["blocks"]).index(b_) for b_ in common_blocks]
    fp_a = a["fp"][a_idx, :n_steps, :]
    fp_b = b["fp"][b_idx, :n_steps, :]
    return np.array(common_blocks, dtype=np.int32), np.arange(1, n_steps + 1), fp_a, fp_b


# ── Lyapunov fit ─────────────────────────────────────────────────────────

def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    """1 - cos_sim, clamped to [0, 2]."""
    nu = np.linalg.norm(u)
    nv = np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return 0.0
    return float(np.clip(1.0 - np.dot(u, v) / (nu * nv), 0.0, 2.0))


def lyapunov_per_block(fp_a: np.ndarray, fp_b: np.ndarray) -> np.ndarray:
    """
    Per-block Lyapunov exponent.

    For each block, compute cosine_distance(fp_a[step], fp_b[step]) over
    time, take log, fit linear slope. Slope > 0 = trajectories diverge
    (positive Lyapunov, chaotic). Slope ≤ 0 = trajectories converge
    (stable basin).

    fp_a, fp_b: [N_blocks, N_steps, fp_dim]
    Returns: lambdas [N_blocks] in units of "log-distance per step"
    """
    n_blocks, n_steps, _ = fp_a.shape
    lambdas = np.zeros(n_blocks, dtype=np.float32)
    distances_log = np.zeros((n_blocks, n_steps), dtype=np.float32)

    for bi in range(n_blocks):
        dists = np.array([
            cosine_distance(fp_a[bi, t], fp_b[bi, t])
            for t in range(n_steps)
        ])
        # Add small floor to avoid log(0); also masks out perfect-match steps
        dists_safe = np.maximum(dists, 1e-8)
        log_d = np.log(dists_safe)
        distances_log[bi] = log_d
        # Fit slope: skip the first step (often degenerate at t=1) and any
        # steps where distance was effectively zero (unset)
        valid = (dists > 1e-7)
        if valid.sum() < 3:
            lambdas[bi] = 0.0
            continue
        ts = np.arange(n_steps)[valid].astype(np.float32)
        ys = log_d[valid]
        # Linear least squares
        slope = np.polyfit(ts, ys, 1)[0]
        lambdas[bi] = float(slope)

    return lambdas, distances_log


# ── Tier inference + auto-tune ───────────────────────────────────────────

def classify_tiers(lambdas: np.ndarray) -> dict:
    """
    Cluster blocks into Granite/Sand/Jazz by Lyapunov exponent.

    Granite: bottom ~13% of blocks by lam (most stable)
    Sand:    middle band
    Jazz:    top blocks (largest lam, most chaotic)

    Returns: dict {block_idx: tier_name}
    """
    n = len(lambdas)
    order = np.argsort(lambdas)
    granite_n = max(1, int(round(n * 0.13)))   # ≈ 4/30 or 4/40
    sand_n    = max(1, int(round(n * 0.17)))   # ≈ 5/30 or 5/40
    classification = {}
    for rank, bi in enumerate(order):
        if rank < granite_n:
            classification[int(bi)] = "granite"
        elif rank < granite_n + sand_n:
            classification[int(bi)] = "sand"
        else:
            classification[int(bi)] = "jazz"
    return classification


def auto_tune_thresholds(lambdas: np.ndarray, classification: dict,
                         window: int = 10) -> dict:
    """
    Suggest tier-aware drift-gate thresholds derived from measured lam.

    Logic: over a window of `window` steps, expected log-divergence is
    lam * window. cos_sim ≈ 1 - exp(lam * window). Threshold should sit at
    1 - α * exp(lam_tier_mean * window) where α is a safety margin.
    Conservative α = 0.5 (allow 50% expected divergence before refresh).

    Returns: { 'granite_threshold': float, 'sand_threshold': float,
               'jazz_threshold': float, 'window_used': int }
    """
    out = {"window_used": window}
    for tier in ("granite", "sand", "jazz"):
        members = [bi for bi, t in classification.items() if t == tier]
        if not members:
            out[f"{tier}_threshold"] = None
            continue
        lam_mean = float(np.mean(lambdas[members]))
        # Predicted multiplicative growth in cos-distance over `window` steps
        # Relative divergence = exp(lam*w) - 1, clamped to [0, 1].
        # Tight basins (small lam) → small divergence → high threshold.
        # Open basins (large lam)  → large divergence → low threshold.
        rel_div = float(np.clip(np.exp(lam_mean * window) - 1.0, 0.0, 1.0))
        # Threshold sits 50% of the way from "perfect" to "expected divergence"
        thresh = float(np.clip(1.0 - 0.5 * rel_div, 0.50, 0.999))
        out[f"{tier}_threshold"] = thresh
        out[f"{tier}_lambda_mean"] = lam_mean
        out[f"{tier}_predicted_div"] = rel_div
    return out


# ── Reporting ────────────────────────────────────────────────────────────

def format_tier_color(tier: str) -> str:
    return {
        "granite": "G",
        "sand":    "S",
        "jazz":    "J",
    }.get(tier, "?")


def render_report(blocks: np.ndarray, lambdas: np.ndarray,
                  classification: dict, tuned: dict) -> str:
    out = []
    out.append("Per-block Lyapunov exponents")
    out.append("=" * 72)
    out.append(f"{'block':>6} {'tier':>8} {'lam (1/step)':>14} {'sign':>6} {'note':<30}")
    out.append("-" * 72)
    for i, b in enumerate(blocks):
        tier = classification.get(i, "?")
        lam = lambdas[i]
        sign = "+" if lam > 0 else ("0" if abs(lam) < 1e-6 else "-")
        note = ("chaotic"  if lam > 0.05 else
                "diverging" if lam > 0.01 else
                "marginal" if lam > -0.01 else
                "convergent")
        out.append(f"{int(b):>6} {tier:>8} {lam:>14.4f} {sign:>6} {note:<30}")
    out.append("")

    out.append("Tier statistics")
    out.append("=" * 72)
    for tier in ("granite", "sand", "jazz"):
        members = [i for i, t in classification.items() if t == tier]
        if not members:
            continue
        lams = lambdas[members]
        out.append(f"{tier:>8}: {len(members):>3} blocks  "
                   f"lam_mean={lams.mean():>+.4f}  "
                   f"lam_std={lams.std():>.4f}  "
                   f"range=[{lams.min():>+.4f}, {lams.max():>+.4f}]")
    out.append("")

    if tuned:
        out.append("Auto-tuned drift-gate thresholds")
        out.append("=" * 72)
        out.append(f"window: {tuned.get('window_used', '?')} steps")
        for tier in ("granite", "sand", "jazz"):
            t = tuned.get(f"{tier}_threshold")
            l = tuned.get(f"{tier}_lambda_mean")
            if t is not None:
                out.append(f"  {tier:>8}_threshold = {t:.4f}    (lam_mean={l:+.4f})")
        out.append("")
        out.append("Drop-in BlockSkip params:")
        for tier in ("granite", "sand", "jazz"):
            t = tuned.get(f"{tier}_threshold")
            if t is not None:
                out.append(f"  {tier}_threshold: {t:.3f}")
        out.append("")

    return "\n".join(out)


# ── Optional plotting ───────────────────────────────────────────────────

def maybe_plot(blocks: np.ndarray, distances_log: np.ndarray,
               classification: dict, output_path: str = "lyapunov.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skip --plot")
        return

    plt.figure(figsize=(10, 6))
    color_map = {"granite": "#3fb950", "sand": "#d29922", "jazz": "#f85149"}
    for i, b in enumerate(blocks):
        tier = classification.get(i, "?")
        c = color_map.get(tier, "#999")
        plt.plot(np.arange(len(distances_log[i])), distances_log[i],
                 color=c, alpha=0.45, lw=1.2,
                 label=f"L{int(b):02d} ({tier})")
    plt.xlabel("denoising step")
    plt.ylabel("log cosine-distance between two runs")
    plt.title("Per-block divergence — Lyapunov spectrum")
    plt.grid(True, alpha=0.3)
    # Limit legend to first few unique tiers to avoid spam
    handles, labels = plt.gca().get_legend_handles_labels()
    seen = set(); uniq = []
    for h, l in zip(handles, labels):
        tier = l.split("(")[-1].rstrip(")")
        if tier not in seen:
            seen.add(tier); uniq.append((h, l))
    plt.legend(*zip(*uniq), loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130)
    plt.close()
    print(f"plot saved to {output_path}")


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("snapshot_a", help="First .npz snapshot")
    p.add_argument("snapshot_b", help="Second .npz snapshot")
    p.add_argument("--auto-tune-window", type=int, default=10,
                   help="Cache-window length for threshold extrapolation (default 10)")
    p.add_argument("--plot", action="store_true",
                   help="Save divergence plot to lyapunov.png (requires matplotlib)")
    p.add_argument("--plot-out", default="lyapunov.png",
                   help="Plot output path (default lyapunov.png)")
    p.add_argument("--json-out", default=None,
                   help="If set, also write the analysis to this JSON file")
    args = p.parse_args()

    a = load_snapshot(args.snapshot_a)
    b = load_snapshot(args.snapshot_b)
    print(f"Run A: {a['path']}  shape={a['fp'].shape}  blocks={list(a['blocks'][:6])}…")
    print(f"Run B: {b['path']}  shape={b['fp'].shape}  blocks={list(b['blocks'][:6])}…")

    blocks, steps, fp_a, fp_b = align_pair(a, b)
    print(f"Aligned: {len(blocks)} blocks x {len(steps)} steps x {fp_a.shape[-1]} dim")
    print()

    lambdas, distances_log = lyapunov_per_block(fp_a, fp_b)
    classification = classify_tiers(lambdas)
    tuned = auto_tune_thresholds(lambdas, classification, window=args.auto_tune_window)

    report = render_report(blocks, lambdas, classification, tuned)
    print(report)

    if args.plot:
        maybe_plot(blocks, distances_log, classification, output_path=args.plot_out)

    if args.json_out:
        out = {
            "blocks": blocks.tolist(),
            "lambdas": lambdas.tolist(),
            "classification": {str(int(b)): classification[i]
                               for i, b in enumerate(blocks)},
            "tuned": tuned,
            "snapshot_a": a["path"],
            "snapshot_b": b["path"],
        }
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        print(f"json written to {args.json_out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
