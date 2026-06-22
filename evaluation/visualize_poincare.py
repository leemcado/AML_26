# evaluation/visualize_poincare.py — Hyperbolic space UMAP 2D visualization
# Project pred x_0 onto Poincare ball -> UMAP 2D -> color-coded by timestep/class
#
# Usage:
#   python evaluation/visualize_poincare.py \
#       --ckpt results/001-phase2-HiT-B-2/checkpoints/0050000.pt

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse

from models.dit import DiT_models
from models.hit import HiT_models
from models.hyperbolic import PoincareBallOps
from diffusion import create_diffusion
from data.latent_dataset import DummyLatentDataset

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


def extract_embeddings(model, diffusion, poincare, dataset, device,
                       num_samples=500, num_timestep_bins=10, proj_head=None):
    """Extract Poincare ball embeddings of pred x_0.

    eps-prediction -> x_0 recovery -> 1/sqrt(d) scaling -> expmap0 -> d_H.
    """
    model.eval()
    all_embeddings = []
    all_timesteps = []
    all_labels = []
    all_d_origins = []

    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    collected = 0

    raw_model = model.module if hasattr(model, 'module') else model
    in_channels = getattr(raw_model, 'in_channels', 4)
    has_learn_sigma = getattr(raw_model, 'learn_sigma', True)

    with torch.no_grad():
        for x, y in loader:
            if collected >= num_samples:
                break
            x, y = x.to(device), y.to(device)

            for t_val in np.linspace(0, diffusion.num_timesteps - 1, num_timestep_bins, dtype=int):
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

                all_embeddings.append(out_hyp.cpu().numpy())
                all_timesteps.append(np.full(B, t_val))
                all_labels.append(y.cpu().numpy())
                all_d_origins.append(d_origin.cpu().numpy())

            collected += x.shape[0]

    embeddings = np.concatenate(all_embeddings, axis=0)
    timesteps = np.concatenate(all_timesteps, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    d_origins = np.concatenate(all_d_origins, axis=0)

    return embeddings, timesteps, labels, d_origins


def plot_poincare_umap(embeddings, timesteps, labels, d_origins,
                       save_path="poincare_umap.png", poincare=None):
    """Visualize Poincare disk via UMAP 2D projection."""
    if not HAS_UMAP:
        print("WARNING: umap-learn not installed, falling back to PCA")
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        emb_2d = reducer.fit_transform(embeddings)
    else:
        reducer = umap.UMAP(n_components=2, metric='euclidean', random_state=42)
        emb_2d = reducer.fit_transform(embeddings)

    # Normalize to fit unit disk
    max_norm = np.max(np.linalg.norm(emb_2d, axis=1))
    if max_norm > 0:
        emb_2d = emb_2d / (max_norm * 1.1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Color by timestep
    ax = axes[0]
    circle = plt.Circle((0, 0), 1.0, fill=False, color='black', linewidth=2)
    ax.add_patch(circle)
    sc = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=timesteps, cmap='viridis',
                    s=5, alpha=0.6, rasterized=True)
    plt.colorbar(sc, ax=ax, label='Timestep t')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.set_title('Hyperbolic UMAP — Colored by Timestep')
    ax.plot(0, 0, 'r+', markersize=15, markeredgewidth=3, label='Origin')
    ax.legend()

    # d_origin vs timestep
    ax2 = axes[1]
    unique_t = np.unique(timesteps)
    mean_d = [d_origins[timesteps == t].mean() for t in unique_t]
    std_d = [d_origins[timesteps == t].std() for t in unique_t]
    ax2.errorbar(unique_t, mean_d, yerr=std_d, fmt='o-', capsize=3,
                 color='dodgerblue', label='E[d_H(0, z_t)]')
    ax2.set_xlabel('Timestep t')
    ax2.set_ylabel('Mean Hyperbolic Distance to Origin')
    ax2.set_title('d_H(0, z_t) vs Timestep')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")
    return save_path


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model
    all_models = {**DiT_models, **HiT_models}
    is_hit = args.model in HiT_models
    model_kwargs = dict(input_size=args.image_size // 8, num_classes=args.num_classes)
    if is_hit:
        model_kwargs["curvature"] = args.curvature
    model = all_models[args.model](**model_kwargs).to(device)

    if args.ckpt:
        state_dict = torch.load(args.ckpt, map_location=device, weights_only=True)
        if "ema" in state_dict:
            model.load_state_dict(state_dict["ema"])
        elif "model" in state_dict:
            model.load_state_dict(state_dict["model"])
        else:
            model.load_state_dict(state_dict)
        print(f"Checkpoint loaded: {args.ckpt}")

    poincare = PoincareBallOps(c=args.curvature)
    diffusion = create_diffusion(timestep_respacing="")

    # Projection head (optional)
    proj_head = None
    if args.proj_ckpt:
        latent_size = args.image_size // 8
        flat_dim = model.in_channels * latent_size * latent_size
        ckpt = torch.load(args.proj_ckpt, map_location=device, weights_only=True)
        has_bias = "bias" in ckpt["proj_head"]
        proj_head = nn.Linear(flat_dim, args.proj_dim, bias=has_bias).to(device)
        proj_head.load_state_dict(ckpt["proj_head"])
        proj_head.eval()
        print(f"proj_head loaded ({flat_dim}->{args.proj_dim}): {args.proj_ckpt}")

    dataset = DummyLatentDataset(num_samples=args.num_samples,
                                 num_classes=args.num_classes)

    embeddings, timesteps, labels, d_origins = extract_embeddings(
        model, diffusion, poincare, dataset, device,
        num_samples=args.num_samples,
        proj_head=proj_head,
    )

    plot_poincare_umap(embeddings, timesteps, labels, d_origins,
                       save_path=args.output, poincare=poincare)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poincare UMAP visualization")
    parser.add_argument("--model", type=str, default="HiT-B/2")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--output", type=str, default="poincare_umap.png")
    parser.add_argument("--proj-ckpt", type=str, default=None,
                        help="Checkpoint containing proj_head state_dict")
    parser.add_argument("--proj-dim", type=int, default=256,
                        help="Projection output dimension")
    args = parser.parse_args()
    main(args)
