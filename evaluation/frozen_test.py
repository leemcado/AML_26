# evaluation/frozen_test.py — Observe hyperbolic structure of a trained model (frozen test)
# Load checkpoint -> project pred x_0 onto Poincare ball -> d_H(center, z_t) vs t
# Hypothesis: t up (noise) -> d_H down (origin), t down (data) -> d_H up (boundary)
#
# Usage:
#   python evaluation/frozen_test.py \
#       --ckpt results/004-DiT-B-2/checkpoints/0100000.pt \
#       --model DiT-B/2

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import json
from pathlib import Path

import torch
import numpy as np
import argparse

import torch.nn as nn

from models.dit import DiT_models
from models.hit import HiT_models
from models.hyperbolic import PoincareBallOps
from diffusion import create_diffusion
from data.latent_dataset import LatentDataset, DummyLatentDataset

try:
    from scipy.stats import spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    """Load model from checkpoint (EMA weights preferred)."""
    all_models = {**DiT_models, **HiT_models}
    is_hit = args.model in HiT_models
    model_kwargs = dict(input_size=args.image_size // 8, num_classes=args.num_classes)
    if is_hit:
        model_kwargs["curvature"] = args.curvature
    model = all_models[args.model](**model_kwargs)

    state_dict = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    if "ema" in state_dict:
        model.load_state_dict(state_dict["ema"])
        print(f"EMA weights loaded: {args.ckpt}")
    elif "model" in state_dict:
        model.load_state_dict(state_dict["model"])
        print(f"Model weights loaded: {args.ckpt}")
    else:
        model.load_state_dict(state_dict)
        print(f"Raw state_dict loaded: {args.ckpt}")

    model = model.to(device)
    model.eval()
    return model


def load_proj_head(
    proj_ckpt: str,
    proj_dim: int,
    flat_dim: int,
    device: torch.device,
) -> nn.Linear:
    """Load projection head from checkpoint."""
    ckpt = torch.load(proj_ckpt, map_location=device, weights_only=False)
    has_bias = "bias" in ckpt["proj_head"]
    proj_head = nn.Linear(flat_dim, proj_dim, bias=has_bias).to(device)
    proj_head.load_state_dict(ckpt["proj_head"])
    proj_head.eval()
    print(f"proj_head loaded ({flat_dim}->{proj_dim}): {proj_ckpt}")
    return proj_head


def collect_hyperbolic_stats(
    model: torch.nn.Module,
    diffusion: object,
    poincare: PoincareBallOps,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    num_samples: int = 1000,
    num_timestep_bins: int = 20,
    proj_head: nn.Linear | None = None,
) -> dict:
    """Project pred x_0 onto Poincare ball and collect d_H statistics.

    eps-prediction -> x_0 recovery -> 1/sqrt(d) scaling -> expmap0 -> d_H(0, z_t).
    Also measures rho w.r.t. manifold center.
    """
    raw_model = model.module if hasattr(model, "module") else model
    in_channels = getattr(raw_model, "in_channels", 4)
    has_learn_sigma = getattr(raw_model, "learn_sigma", True)

    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    timestep_values = np.linspace(0, diffusion.num_timesteps - 1, num_timestep_bins, dtype=int)

    all_t = []
    all_d_origin = []
    all_l2_norm = []
    all_hyp_points = []

    collected = 0
    with torch.no_grad():
        for x, y in loader:
            if collected >= num_samples:
                break
            x, y = x.to(device), y.to(device)

            for t_val in timestep_values:
                t = torch.full((x.shape[0],), t_val, device=device, dtype=torch.long)
                x_t = diffusion.q_sample(x, t)
                eps_pred = raw_model(x_t, t, y)

                if has_learn_sigma:
                    eps_pred = eps_pred[:, :in_channels]

                pred_x0 = diffusion._predict_xstart_from_eps(x_t, t, eps_pred)

                B, C, H, W = pred_x0.shape
                out_flat = pred_x0.reshape(B, -1)
                if proj_head is not None:
                    out_flat = proj_head(out_flat)
                scale = math.sqrt(out_flat.shape[1])
                out_scaled = poincare.tangent_clip(out_flat / scale, alpha=0.95)
                out_hyp = poincare.hard_clip(poincare.expmap0(out_scaled))

                d_origin = poincare.distance_to_origin(out_hyp)
                l2_norm = out_flat.norm(dim=-1)

                all_t.append(np.full(B, t_val))
                all_d_origin.append(d_origin.cpu().numpy())
                all_l2_norm.append(l2_norm.cpu().numpy())
                all_hyp_points.append(out_hyp.cpu())

            collected += x.shape[0]

    t_arr = np.concatenate(all_t)
    d_arr = np.concatenate(all_d_origin)
    l2_arr = np.concatenate(all_l2_norm)
    hyp_points = torch.cat(all_hyp_points, dim=0)

    # Manifold center (Euclidean mean)
    manifold_center = hyp_points.mean(dim=0, keepdim=True)
    manifold_center_norm = manifold_center.norm().item()

    # Distance from manifold center
    d_center_arr = poincare.poincare_distance(
        hyp_points, manifold_center.expand_as(hyp_points)
    ).numpy()

    # Per-timestep-bin statistics
    bins = []
    for t_val in timestep_values:
        mask = t_arr == t_val
        if mask.sum() > 0:
            bins.append({
                "t": int(t_val),
                "mean_d_origin": float(d_arr[mask].mean()),
                "std_d_origin": float(d_arr[mask].std()),
                "mean_d_center": float(d_center_arr[mask].mean()),
                "std_d_center": float(d_center_arr[mask].std()),
                "mean_l2_norm": float(l2_arr[mask].mean()),
                "n": int(mask.sum()),
            })

    # Spearman correlation (origin-based + center-based)
    rho_origin, p_origin = None, None
    rho_center, p_center = None, None
    if HAS_SCIPY:
        rho_origin, p_origin = spearmanr(t_arr, d_arr)
        rho_center, p_center = spearmanr(t_arr, d_center_arr)

    return {
        "bins": bins,
        "spearman_rho_origin": float(rho_origin) if rho_origin is not None else None,
        "spearman_p_origin": float(p_origin) if p_origin is not None else None,
        "spearman_rho_center": float(rho_center) if rho_center is not None else None,
        "spearman_p_center": float(p_center) if p_center is not None else None,
        "manifold_center_norm": manifold_center_norm,
        "raw_t": t_arr,
        "raw_d_origin": d_arr,
        "raw_d_center": d_center_arr,
        "raw_l2_norm": l2_arr,
    }


def print_report(stats: dict, model_name: str) -> None:
    """Print quantitative results."""
    print(f"\n{'='*70}")
    print(f"  Frozen Hyperbolic Structure Test (pred x_0): {model_name}")
    print(f"{'='*70}")

    if stats.get("spearman_rho_center") is not None:
        rho_c = stats["spearman_rho_center"]
        p_c = stats["spearman_p_center"]
        rho_o = stats["spearman_rho_origin"]
        p_o = stats["spearman_p_origin"]
        center_norm = stats.get("manifold_center_norm", 0)
        print(f"\n  Manifold center ||c||: {center_norm:.4f}")
        print(f"  Spearman rho (center): {rho_c:.4f}  (p={p_c:.2e})  <- primary")
        print(f"  Spearman rho (origin): {rho_o:.4f}  (p={p_o:.2e})")
    else:
        print("\n  [scipy not available, Spearman skipped]")

    print(f"\n  {'t':>6s}  {'d_center':>10s}  {'d_origin':>10s}  {'E[L2]':>10s}  {'n':>6s}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*6}")
    for b in stats["bins"]:
        d_c = b.get("mean_d_center", b["mean_d_origin"])
        print(f"  {b['t']:>6d}  {d_c:>10.4f}  {b['mean_d_origin']:>10.4f}  "
              f"{b['mean_l2_norm']:>10.2f}  {b['n']:>6d}")
    print()


def plot_results(
    stats: dict,
    save_dir: Path,
    model_name: str,
    poincare: PoincareBallOps,
) -> None:
    """Plot d_H vs t graph + scatter visualization."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir.mkdir(parents=True, exist_ok=True)

    # d_H(0, z_t) vs timestep
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ts = [b["t"] for b in stats["bins"]]
    means = [b["mean_d_origin"] for b in stats["bins"]]
    stds = [b["std_d_origin"] for b in stats["bins"]]

    ax = axes[0]
    ax.errorbar(ts, means, yerr=stds, fmt="o-", capsize=3, color="dodgerblue",
                label="E[d_H(0, z_t)]")
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("Mean Hyperbolic Distance to Origin")
    ax.set_title(f"{model_name} (frozen, pred x_0) — d_H vs Timestep")
    if stats["spearman_rho_center"] is not None:
        ax.text(0.02, 0.98, f"Spearman rho = {stats['spearman_rho_center']:.4f}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    ax.legend()
    ax.grid(True, alpha=0.3)

    # L2 norm vs timestep
    l2_means = [b["mean_l2_norm"] for b in stats["bins"]]
    ax2 = axes[1]
    ax2.plot(ts, l2_means, "s-", color="coral", label="E[||z_t||_2]")
    ax2.set_xlabel("Timestep t")
    ax2.set_ylabel("Mean Euclidean L2 Norm")
    ax2.set_title(f"{model_name} (frozen, pred x_0) — L2 Norm vs Timestep")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path1 = save_dir / "frozen_d_origin_vs_t.png"
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path1}")

    # Full sample scatter
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(stats["raw_t"], stats["raw_d_origin"], c=stats["raw_t"],
                    cmap="viridis", s=2, alpha=0.3, rasterized=True)
    plt.colorbar(sc, ax=ax, label="Timestep t")
    ax.set_xlabel("Timestep t")
    ax.set_ylabel("d_H(0, z_t)")
    ax.set_title(f"{model_name} (frozen, pred x_0) — All Samples Scatter")
    ax.grid(True, alpha=0.3)

    path2 = save_dir / "frozen_scatter.png"
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path2}")


def main(args: argparse.Namespace) -> None:
    """Frozen test main."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    model = load_model(args, device)
    poincare = PoincareBallOps(c=args.curvature)
    diffusion = create_diffusion(timestep_respacing="")

    # Projection head (optional)
    proj_head = None
    if args.proj_ckpt:
        raw_model = model.module if hasattr(model, "module") else model
        latent_size = args.image_size // 8
        flat_dim = raw_model.in_channels * latent_size * latent_size
        proj_head = load_proj_head(args.proj_ckpt, args.proj_dim, flat_dim, device)

    # Dataset
    if args.latent_dir and os.path.isdir(args.latent_dir):
        dataset = LatentDataset(args.latent_dir)
        print(f"Using real latents: {args.latent_dir} ({len(dataset):,} samples)")
    else:
        dataset = DummyLatentDataset(
            num_samples=args.num_samples, num_classes=args.num_classes
        )
        print(f"Using dummy latents ({args.num_samples} samples)")

    # Collect statistics
    print(f"Collecting hyperbolic statistics (curvature={args.curvature})...")
    stats = collect_hyperbolic_stats(
        model, diffusion, poincare, dataset, device,
        num_samples=args.num_samples,
        num_timestep_bins=args.num_timestep_bins,
        proj_head=proj_head,
    )

    print_report(stats, args.model)

    # Save JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_data = {
        "model": args.model,
        "ckpt": args.ckpt,
        "curvature": args.curvature,
        "num_samples": args.num_samples,
        "spearman_rho_origin": stats.get("spearman_rho_origin"),
        "spearman_p_origin": stats.get("spearman_p_origin"),
        "spearman_rho_center": stats.get("spearman_rho_center"),
        "spearman_p_center": stats.get("spearman_p_center"),
        "manifold_center_norm": stats.get("manifold_center_norm"),
        "bins": stats["bins"],
    }
    json_path = output_dir / "frozen_test_results.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved: {json_path}")

    plot_results(stats, output_dir, args.model, poincare)

    # UMAP visualization
    print("\nRunning UMAP visualization...")
    from evaluation.visualize_poincare import extract_embeddings, plot_poincare_umap
    embeddings, timesteps, labels, d_origins = extract_embeddings(
        model, diffusion, poincare, dataset, device,
        num_samples=min(args.num_samples, 500),
        num_timestep_bins=args.num_timestep_bins,
        proj_head=proj_head,
    )
    plot_poincare_umap(
        embeddings, timesteps, labels, d_origins,
        save_path=str(output_dir / "frozen_poincare_umap.png"),
        poincare=poincare,
    )

    print(f"\nAll results saved to: {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Observe hyperbolic structure of trained DiT/HiT (frozen test)"
    )
    parser.add_argument("--model", type=str, default="DiT-B/2")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--num-timestep-bins", type=int, default=20)
    parser.add_argument("--latent-dir", type=str, default=None,
                        help="Cached latent directory (uses dummy if absent)")
    parser.add_argument("--output-dir", type=str, default="frozen_test_results")
    parser.add_argument("--proj-ckpt", type=str, default=None,
                        help="Checkpoint containing proj_head state_dict")
    parser.add_argument("--proj-dim", type=int, default=256,
                        help="Projection output dimension")
    args = parser.parse_args()
    main(args)
