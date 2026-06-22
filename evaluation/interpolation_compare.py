# evaluation/interpolation_compare.py — Interpolation comparison: Euclidean / Geodesic / SLERP
# Generate two endpoints -> 4 interpolation methods -> VAE decode -> visualize
# Row 4: convert geodesic midpoint d_origin to noise level -> SDEdit denoising
#
# Usage:
#   python evaluation/interpolation_compare.py \
#       --ckpt results/003-phase1-DiT-L-2/checkpoints/0100000.pt \
#       --same-class 207 --cross-classes 207,949

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

from models.dit import DiT_models
from models.hit import HiT_models
from models.hyperbolic import PoincareBallOps
from diffusion import create_diffusion


ALPHAS = [i / 20 for i in range(21)]  # 0.0, 0.05, ..., 1.0


# -- Interpolation methods ---------------------------------------------------


def lerp(x1: torch.Tensor, x2: torch.Tensor, alphas: list[float]) -> torch.Tensor:
    """Linear interpolation."""
    return torch.stack([(1 - a) * x1 + a * x2 for a in alphas])


def geodesic_interp_latent(
    x1: torch.Tensor,
    x2: torch.Tensor,
    alphas: list[float],
    poincare: PoincareBallOps,
) -> tuple[torch.Tensor, list[float]]:
    """Hyperbolic geodesic interpolation — interpolate clean VAE latents in hyperbolic space."""
    shape = x1.shape
    d = x1.numel()
    scale = math.sqrt(d)

    x1_flat = x1.reshape(1, -1) / scale
    x2_flat = x2.reshape(1, -1) / scale
    x1_flat = poincare.tangent_clip(x1_flat, alpha=0.95)
    x2_flat = poincare.tangent_clip(x2_flat, alpha=0.95)
    x1_hyp = poincare.hard_clip(poincare.expmap0(x1_flat)).squeeze(0)
    x2_hyp = poincare.hard_clip(poincare.expmap0(x2_flat)).squeeze(0)

    interps, d_origins = [], []
    for a in alphas:
        pt = poincare.geodesic(x1_hyp, x2_hyp, a)
        d_origins.append(poincare.distance_to_origin(pt.unsqueeze(0)).item())
        v = poincare.logmap0(pt.unsqueeze(0)).squeeze(0)
        interps.append((v * scale).reshape(shape))

    return torch.stack(interps), d_origins


def slerp_interp_latent(
    x1: torch.Tensor,
    x2: torch.Tensor,
    alphas: list[float],
    poincare: PoincareBallOps,
) -> tuple[torch.Tensor, list[float]]:
    """SLERP interpolation — preserves L2 norm (approx. hyperbolic radius)."""
    shape = x1.shape
    x1_flat = x1.reshape(-1).double()
    x2_flat = x2.reshape(-1).double()

    n1, n2 = x1_flat.norm(), x2_flat.norm()
    cos_theta = (torch.dot(x1_flat, x2_flat) / (n1 * n2 + 1e-8)).clamp(-1, 1)
    theta = torch.acos(cos_theta)
    sin_theta = torch.sin(theta)

    interps = []
    for a in alphas:
        if sin_theta.abs() < 1e-6:
            v = ((1 - a) * x1_flat + a * x2_flat).float()
        else:
            w1 = torch.sin((1 - a) * theta) / sin_theta
            w2 = torch.sin(a * theta) / sin_theta
            v = (w1 * x1_flat + w2 * x2_flat).float()
        interps.append(v.reshape(shape))

    latents = torch.stack(interps)
    d_origins = compute_eucl_d_origins(latents, poincare)
    return latents, d_origins


def compute_eucl_d_origins(
    latents: torch.Tensor, poincare: PoincareBallOps,
) -> list[float]:
    """Project latents onto Poincare ball and compute d_origin."""
    d_origins = []
    for x in latents:
        x_flat = x.reshape(1, -1) / math.sqrt(x.numel())
        x_clipped = poincare.tangent_clip(x_flat, alpha=0.95)
        x_hyp = poincare.hard_clip(poincare.expmap0(x_clipped))
        d_origins.append(poincare.distance_to_origin(x_hyp).item())
    return d_origins


# -- Noise level estimation + partial denoise --------------------------------


def estimate_t_equiv(
    d_origins: list[float],
    d_endpoint: float | None = None,
    num_diffusion_steps: int = 1000,
) -> list[int]:
    """Convert d_origin trajectory to equivalent noise levels.

    Based on linear regularization target d_n = 2(1-t/T):
    t_equiv = T * max(0, 1 - d_i / d_endpoint)
    """
    if d_endpoint is None:
        d_endpoint = (d_origins[0] + d_origins[-1]) / 2
    t_equivs = []
    for d in d_origins:
        ratio = d / (d_endpoint + 1e-8)
        t_frac = max(0.0, 1.0 - ratio)
        t_equivs.append(min(int(t_frac * num_diffusion_steps), num_diffusion_steps - 1))
    return t_equivs


def _find_nearest_spaced_index(diffusion, t_orig: int) -> int:
    """Find nearest index in respaced timestep schedule."""
    best_idx = 0
    best_diff = abs(diffusion.timestep_map[0] - t_orig)
    for i in range(1, diffusion.num_timesteps):
        diff = abs(diffusion.timestep_map[i] - t_orig)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    return best_idx


def partial_denoise_cfg(
    model: torch.nn.Module,
    diffusion,
    x0_interp: torch.Tensor,
    t_equiv: int,
    device: torch.device,
    class_label: int,
    num_classes: int = 1000,
    cfg_scale: float = 4.0,
    noise_seed: int = 0,
) -> torch.Tensor:
    """SDEdit-style partial denoise: x_0 -> inject noise(t_equiv) -> DDIM reverse."""
    if t_equiv <= 0:
        return x0_interp

    spaced_idx = _find_nearest_spaced_index(diffusion, t_equiv)
    if diffusion.timestep_map[spaced_idx] <= 0:
        return x0_interp

    torch.manual_seed(noise_seed)
    x0_batch = x0_interp.unsqueeze(0).to(device)
    t_tensor = torch.tensor([spaced_idx], device=device)
    x_t = diffusion.q_sample(x0_batch, t_tensor)

    # CFG: [cond, uncond]
    x_t = torch.cat([x_t, x_t], dim=0)
    y = torch.cat([
        torch.tensor([class_label], device=device),
        torch.tensor([num_classes], device=device),
    ])
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    img = x_t
    for i in range(spaced_idx, -1, -1):
        t = torch.tensor([i, i], device=device)
        with torch.no_grad():
            out = diffusion.ddim_sample(
                model.forward_with_cfg, img, t,
                clip_denoised=False, model_kwargs=model_kwargs, eta=0.0,
            )
            img = out["sample"]

    result, _ = img.chunk(2, dim=0)
    return result.squeeze(0)


# -- Generation + decoding ---------------------------------------------------


def generate_sample(
    model: torch.nn.Module,
    diffusion,
    class_label: int,
    device: torch.device,
    cfg_scale: float = 4.0,
    num_classes: int = 1000,
    seed: int = 0,
    image_size: int = 256,
) -> torch.Tensor:
    """Generate a single clean x_0 latent (full denoising)."""
    torch.manual_seed(seed)
    latent_size = image_size // 8
    z = torch.randn(1, 4, latent_size, latent_size, device=device)
    z = torch.cat([z, z], 0)
    y = torch.cat([
        torch.tensor([class_label], device=device),
        torch.tensor([num_classes], device=device),
    ])
    model_kwargs = dict(y=y, cfg_scale=cfg_scale)

    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z,
        clip_denoised=False, model_kwargs=model_kwargs,
        progress=True, device=device,
    )
    samples, _ = samples.chunk(2, dim=0)
    return samples.squeeze(0)


def decode_latents(latents: torch.Tensor, vae) -> torch.Tensor:
    """VAE decode."""
    if vae is None:
        return latents[:, :3]
    with torch.no_grad():
        imgs = vae.decode(latents / 0.18215).sample
    return imgs


def tensor_to_numpy(img: torch.Tensor) -> np.ndarray:
    """(3,H,W) [-1,1] -> (H,W,3) [0,1] numpy."""
    return (img.clamp(-1, 1).permute(1, 2, 0).cpu().numpy() + 1) / 2


# -- Visualization ------------------------------------------------------------


def plot_interpolation(
    eucl_imgs: list[np.ndarray],
    geo_imgs: list[np.ndarray],
    slerp_imgs: list[np.ndarray],
    eucl_denoise_imgs: list[np.ndarray],
    geo_denoise_imgs: list[np.ndarray],
    slerp_denoise_imgs: list[np.ndarray],
    d_origins_eucl: list[float],
    d_origins_geo: list[float],
    d_origins_slerp: list[float],
    t_equivs_eucl: list[int],
    t_equivs_geo: list[int],
    t_equivs_slerp: list[int],
    title: str,
    save_path: str,
) -> None:
    """6-row image grid + d_origin/t_equiv trajectory plots."""
    n = len(eucl_imgs)
    img_size = 0.85
    fig_w = n * img_size + 2.2
    fig_h = img_size * 6 + 4.5
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        7, n, height_ratios=[1, 1, 1, 1, 1, 1, 1.2],
        hspace=0.40, wspace=0.05,
        left=0.08, right=0.97, top=0.95, bottom=0.04,
    )

    rows_data = [
        eucl_imgs, eucl_denoise_imgs,
        geo_imgs, geo_denoise_imgs,
        slerp_imgs, slerp_denoise_imgs,
    ]
    row_labels = [
        "Euclidean", "Eucl+Denoise",
        "Geodesic", "Geo+Denoise",
        "SLERP", "SLERP+Denoise",
    ]
    row_t_equivs = [
        None, t_equivs_eucl,
        None, t_equivs_geo,
        None, t_equivs_slerp,
    ]
    row_colors = [
        None, "blue",
        None, "red",
        None, "green",
    ]

    for row_idx, imgs in enumerate(rows_data):
        for i in range(n):
            ax = fig.add_subplot(gs[row_idx, i])
            ax.imshow(imgs[i])
            ax.axis("off")
            if row_idx == 0:
                if i == 0:
                    ax.set_title("Src", fontsize=6)
                elif i == n - 1:
                    ax.set_title("Tgt", fontsize=6)
                elif ALPHAS[i] in (0.25, 0.5, 0.75):
                    ax.set_title(f"{int(ALPHAS[i]*100)}%", fontsize=6)
            t_list = row_t_equivs[row_idx]
            if t_list is not None and t_list[i] > 0:
                ax.text(
                    0.5, -0.02, f"t={t_list[i]}",
                    fontsize=4, ha="center", va="top",
                    transform=ax.transAxes,
                    color=row_colors[row_idx] or "red",
                )

    for row_idx, label in enumerate(row_labels):
        mid = n // 2
        ax_ref = fig.axes[row_idx * n + mid]
        pos = ax_ref.get_position()
        fig.text(0.02, (pos.y0 + pos.y1) / 2, label,
                 fontsize=8, rotation=90, va="center", ha="center",
                 fontweight="bold")

    gs_bottom = gs[6, :].subgridspec(1, 2, wspace=0.30)

    ax_d = fig.add_subplot(gs_bottom[0, 0])
    ax_d.plot(ALPHAS, d_origins_eucl, "b-o", label="Euclidean",
              linewidth=1.5, markersize=2)
    ax_d.plot(ALPHAS, d_origins_geo, "r-o", label="Geodesic",
              linewidth=1.5, markersize=2)
    ax_d.plot(ALPHAS, d_origins_slerp, "g-o", label="SLERP",
              linewidth=1.5, markersize=2)
    ax_d.set_xlabel("alpha", fontsize=9)
    ax_d.set_ylabel("d_H(0, z)", fontsize=9)
    ax_d.legend(fontsize=7, loc="lower center")
    ax_d.grid(True, alpha=0.3)
    ax_d.set_title("Distance to Origin", fontsize=9)

    ax_t = fig.add_subplot(gs_bottom[0, 1])
    ax_t.plot(ALPHAS, t_equivs_eucl, "b-o", label="Euclidean",
              linewidth=1.5, markersize=2)
    ax_t.plot(ALPHAS, t_equivs_geo, "r-o", label="Geodesic",
              linewidth=1.5, markersize=2)
    ax_t.plot(ALPHAS, t_equivs_slerp, "g-o", label="SLERP",
              linewidth=1.5, markersize=2)
    ax_t.set_xlabel("alpha", fontsize=9)
    ax_t.set_ylabel("t_equiv", fontsize=9)
    ax_t.legend(fontsize=7, loc="upper center")
    ax_t.grid(True, alpha=0.3)
    ax_t.set_title("Estimated Noise Level (gauge)", fontsize=9)

    fig.suptitle(title, fontsize=12)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


# -- Main pipeline -------------------------------------------------------------


def run_interpolation(
    model: torch.nn.Module,
    diffusion,
    vae,
    class_label_1: int,
    class_label_2: int,
    poincare: PoincareBallOps,
    device: torch.device,
    seed: int,
    cfg_scale: float,
    num_classes: int,
    output_dir: Path,
    image_size: int = 256,
) -> None:
    """Run interpolation experiment with all 4 methods."""
    same_class = (class_label_1 == class_label_2)
    tag = (f"class{class_label_1}" if same_class
           else f"class{class_label_1}_to_{class_label_2}")

    # Generate endpoints
    print(f"[{tag}] Generating endpoint 1 (class={class_label_1})...")
    x0_1 = generate_sample(
        model, diffusion, class_label_1, device, cfg_scale, num_classes,
        seed=seed, image_size=image_size,
    )
    print(f"[{tag}] Generating endpoint 2 (class={class_label_2})...")
    x0_2 = generate_sample(
        model, diffusion, class_label_2, device, cfg_scale, num_classes,
        seed=seed + 1, image_size=image_size,
    )

    # Euclidean interpolation
    print(f"[{tag}] Euclidean interpolation...")
    eucl_latents = lerp(x0_1, x0_2, ALPHAS)
    d_origins_eucl = compute_eucl_d_origins(eucl_latents, poincare)

    # Geodesic interpolation
    print(f"[{tag}] Geodesic interpolation...")
    geo_latents, d_origins_geo = geodesic_interp_latent(
        x0_1, x0_2, ALPHAS, poincare,
    )

    # SLERP interpolation
    print(f"[{tag}] SLERP interpolation...")
    slerp_latents, d_origins_slerp = slerp_interp_latent(
        x0_1, x0_2, ALPHAS, poincare,
    )

    # Denoise rows
    d_ref = (d_origins_geo[0] + d_origins_geo[-1]) / 2
    t_equivs_eucl = estimate_t_equiv(d_origins_eucl, d_endpoint=d_ref)
    t_equivs_geo = estimate_t_equiv(d_origins_geo, d_endpoint=d_ref)
    t_equivs_slerp = estimate_t_equiv(d_origins_slerp, d_endpoint=d_ref)

    def _denoise_row(latents, t_equivs, label, noise_seed_base):
        print(f"[{tag}] {label} + Denoise (t_equiv max={max(t_equivs)})...")
        results = []
        for i, (lat, t_eq) in enumerate(zip(latents, t_equivs)):
            cls = class_label_1 if (same_class or ALPHAS[i] < 0.5) else class_label_2
            results.append(partial_denoise_cfg(
                model, diffusion, lat, t_eq, device,
                cls, num_classes, cfg_scale, noise_seed=noise_seed_base,
            ))
            if t_eq > 0 and i % 5 == 0:
                print(f"  alpha={ALPHAS[i]:.2f} -> t={t_eq}")
        return torch.stack(results)

    eucl_denoise = _denoise_row(eucl_latents, t_equivs_eucl, "Euclidean", seed + 2000)
    geo_denoise = _denoise_row(geo_latents, t_equivs_geo, "Geodesic", seed + 2000)
    slerp_denoise = _denoise_row(slerp_latents, t_equivs_slerp, "SLERP", seed + 2000)

    # VAE decode
    print(f"[{tag}] Decoding {len(ALPHAS)} x 6 latents...")
    eucl_imgs = decode_latents(eucl_latents, vae)
    geo_imgs = decode_latents(geo_latents, vae)
    slerp_imgs = decode_latents(slerp_latents, vae)
    eucl_denoise_imgs = decode_latents(eucl_denoise, vae)
    geo_denoise_imgs = decode_latents(geo_denoise, vae)
    slerp_denoise_imgs = decode_latents(slerp_denoise, vae)

    eucl_np = [tensor_to_numpy(img) for img in eucl_imgs]
    geo_np = [tensor_to_numpy(img) for img in geo_imgs]
    slerp_np = [tensor_to_numpy(img) for img in slerp_imgs]
    eucl_dn_np = [tensor_to_numpy(img) for img in eucl_denoise_imgs]
    geo_dn_np = [tensor_to_numpy(img) for img in geo_denoise_imgs]
    slerp_dn_np = [tensor_to_numpy(img) for img in slerp_denoise_imgs]

    # Plot
    title = (f"Same-class Interpolation (class={class_label_1})"
             if same_class else
             f"Cross-class Interpolation ({class_label_1} -> {class_label_2})")

    save_path = output_dir / f"interpolation_{tag}.png"
    plot_interpolation(
        eucl_np, geo_np, slerp_np,
        eucl_dn_np, geo_dn_np, slerp_dn_np,
        d_origins_eucl, d_origins_geo, d_origins_slerp,
        t_equivs_eucl, t_equivs_geo, t_equivs_slerp,
        title, str(save_path),
    )

    # Save metrics
    data = {
        "class_1": class_label_1,
        "class_2": class_label_2,
        "alphas": ALPHAS,
        "d_origins_euclidean": d_origins_eucl,
        "d_origins_geodesic": d_origins_geo,
        "d_origins_slerp": d_origins_slerp,
        "t_equivs_euclidean": t_equivs_eucl,
        "t_equivs_geodesic": t_equivs_geo,
        "t_equivs_slerp": t_equivs_slerp,
        "d_ref": d_ref,
    }
    json_path = output_dir / f"d_origins_{tag}.json"
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {json_path}")


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    poincare = PoincareBallOps(c=args.curvature)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    all_models = {**DiT_models, **HiT_models}
    print(f"Loading {args.model}...")
    latent_size = args.image_size // 8
    model_kwargs = dict(input_size=latent_size, num_classes=args.num_classes)
    if args.model in HiT_models:
        model_kwargs["curvature"] = args.curvature
    model = all_models[args.model](**model_kwargs).to(device)

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    state = ckpt.get("ema", ckpt.get("model", ckpt))
    model.load_state_dict(state)
    model.eval()
    print(f"Checkpoint loaded: {args.ckpt}")

    diffusion = create_diffusion(str(args.num_sampling_steps))

    # VAE
    vae = None
    try:
        from diffusers.models import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-{args.vae}",
        ).to(device)
        vae.eval()
        print("VAE loaded")
    except Exception as e:
        print(f"WARNING: VAE load failed ({e}), using raw latents")

    # Same-class interpolation
    print(f"\n=== Same-class interpolation (class={args.same_class}) ===")
    run_interpolation(
        model, diffusion, vae,
        args.same_class, args.same_class,
        poincare, device, args.seed, args.cfg_scale, args.num_classes,
        output_dir, image_size=args.image_size,
    )

    # Cross-class interpolation
    if args.cross_classes:
        c1, c2 = [int(x) for x in args.cross_classes.split(",")]
        print(f"\n=== Cross-class interpolation ({c1} -> {c2}) ===")
        run_interpolation(
            model, diffusion, vae,
            c1, c2,
            poincare, device, args.seed + 100, args.cfg_scale, args.num_classes,
            output_dir, image_size=args.image_size,
        )

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interpolation comparison: Euclidean / Geodesic / SLERP / +Denoise",
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--model", type=str, default="DiT-L/2")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--curvature", type=float, default=1.0)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--same-class", type=int, default=207)
    parser.add_argument("--cross-classes", type=str, default="207,949")
    parser.add_argument("--output-dir", type=str, default="interpolation_results")
    args = parser.parse_args()
    main(args)
