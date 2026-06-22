# training/train_hit.py — HiT training
# DiT + hyperbolic distance loss + Stiefel proj_head + origin linear regularization
#
# === Hyperbolic extension pipeline ===
#   1. DiT backbone -> Euclidean eps prediction (adaLN-Zero)
#   2. proj_head (Linear 4096->256, Stiefel) low-dimensional projection
#   3. 1/sqrt(d) scale + tangent_clip + expmap0 -> Poincare ball
#   4. d_H^2 loss (or hybrid MSE + lambda * d_H^2)
#   5. QR retraction to maintain Stiefel constraint
#
# === Origin-based linear regularization (optional) ===
#   Enforce d_origin(t) ~ r_max * (1 - t/T)
#   t=0 (clean) -> boundary, t=T (noise) -> origin. corr(d,t)<0 = success
#
# === DDP structure ===
#   - Backbone: DDP (NCCL gradient all-reduce automatic)
#   - proj_head: outside DDP -> manual gradient all-reduce + post-QR broadcast
#   - EMA: backbone only (proj_head excluded)
#
# Usage:
#   torchrun --nproc_per_node=4 training/train_hit.py --config configs/hit_b.yaml
#   python training/train_hit.py --config configs/hit_b.yaml --dummy

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from copy import deepcopy
from glob import glob
from time import time
import argparse
import yaml

from models.hit import build_hit
from models.hyperbolic import PoincareBallOps
from diffusion import create_diffusion
from data.latent_dataset import LatentDataset, DummyLatentDataset
from training.utils import (
    update_ema, requires_grad, create_logger,
    save_checkpoint, get_grad_norm,
)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def load_config(path: str) -> dict:
    """Load YAML and flatten nested sections."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    cfg = {}
    for key, val in raw.items():
        if isinstance(val, dict):
            cfg.update(val)
        else:
            cfg[key] = val
    return cfg


def setup_distributed() -> tuple[int, int, torch.device, int]:
    """Initialize NCCL DDP. Falls back to single GPU without torchrun."""
    use_ddp = "RANK" in os.environ
    if use_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device_id = rank % torch.cuda.device_count()
    else:
        rank = 0
        world_size = 1
        device_id = 0

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device_id)
    return rank, world_size, device, device_id


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def compute_geometric_metrics(
    model: nn.Module,
    poincare: PoincareBallOps,
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    diffusion,
    device: torch.device,
    proj_head: nn.Linear | None = None,
) -> dict[str, float]:
    """Compute hyperbolic geometry metrics (for wandb logging)."""
    raw_model = model.module if hasattr(model, "module") else model
    raw_model.eval()

    metrics = {}
    with torch.no_grad():
        x_t = diffusion.q_sample(x, t, noise=torch.randn_like(x))
        model_out = raw_model(x_t, t, y)
        if raw_model.learn_sigma:
            model_out = model_out[:, :raw_model.in_channels]

        B, C, H, W = model_out.shape
        out_flat = model_out.reshape(B, -1)
        if proj_head is not None:
            out_flat = proj_head(out_flat)
        scale = math.sqrt(out_flat.shape[1])
        out_hyp = poincare.hard_clip(poincare.expmap0(
            poincare.tangent_clip(out_flat / scale, alpha=0.95)
        ))
        d_origin = poincare.distance_to_origin(out_hyp)

        metrics["geo/mean_d_origin"] = d_origin.mean().item()
        metrics["geo/std_d_origin"] = d_origin.std().item()
        metrics["geo/max_d_origin"] = d_origin.max().item()
        metrics["geo/mean_l2_norm"] = out_flat.norm(dim=-1).mean().item()

        # Per-timestep-bin d_origin
        for bin_idx in range(10):
            t_low, t_high = bin_idx * 100, (bin_idx + 1) * 100
            mask = (t >= t_low) & (t < t_high)
            if mask.sum() > 0:
                metrics[f"geo/d_origin_t{t_low}-{t_high}"] = d_origin[mask].mean().item()

    raw_model.train()
    return metrics


def main(cfg: dict) -> None:
    # ---- DDP setup ----
    rank, world_size, device, device_id = setup_distributed()
    seed = cfg.get("seed", 0) * world_size + rank
    torch.manual_seed(seed)

    # ---- Experiment directory ----
    results_dir = cfg.get("results_dir", "results")
    reg_mode = cfg.get("mode", "none")
    hybrid = cfg.get("hybrid", False)

    if reg_mode == "linear":
        cutoff = cfg.get("cutoff", 1000)
        scope = "allt" if cutoff >= 1000 else f"t{cutoff}"
        phase_tag = f"linreg-{scope}"
    elif hybrid:
        phase_tag = "hybrid"
    elif cfg.get("enabled", False):
        phase_tag = "proj"
    else:
        phase_tag = "HiT"

    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)
        experiment_index = len(glob(f"{results_dir}/*"))
        size = cfg["size"]
        ps = cfg.get("patch_size", 2)
        experiment_dir = f"{results_dir}/{experiment_index:03d}-{phase_tag}-HiT-{size}-{ps}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
    else:
        experiment_dir = checkpoint_dir = None
        logger = create_logger(None)

    logger.info(f"[{phase_tag.upper()}] rank={rank}, seed={seed}, world_size={world_size}")

    # ---- Model ----
    latent_size = cfg.get("image_size", 256) // 8
    curvature = cfg.get("curvature", 1.0)
    model = build_hit(
        size=cfg["size"],
        patch_size=cfg.get("patch_size", 2),
        curvature=curvature,
        input_size=latent_size,
        num_classes=cfg.get("num_classes", 1000),
    )
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    # DDP: automatic backbone gradient all-reduce
    if dist.is_initialized():
        model = DDP(model.to(device), device_ids=[device_id])
    else:
        model = model.to(device)

    poincare = PoincareBallOps(c=curvature)
    diffusion = create_diffusion(
        timestep_respacing="",
        use_poincare=True,
        curvature=curvature,
    )

    raw_model = model.module if hasattr(model, "module") else model
    param_count = sum(p.numel() for p in raw_model.parameters())
    logger.info(f"HiT-{cfg['size']}/{cfg.get('patch_size', 2)} — {param_count:,} params, c={curvature}")

    # ---- Projection head (Stiefel manifold) ----
    # Projects eps prediction (4096-dim) to low-dim (256) before hyperbolic mapping
    # Weight: orthogonal matrix constraint (restored via QR retraction each step)
    proj_head = None
    use_proj = cfg.get("enabled", False)
    proj_dim = cfg.get("dim", 256)
    if use_proj:
        flat_dim = raw_model.in_channels * latent_size * latent_size
        proj_head = nn.Linear(flat_dim, proj_dim, bias=False).to(device)
        nn.init.orthogonal_(proj_head.weight)
        # Ensure identical initialization across all ranks
        if dist.is_initialized():
            dist.broadcast(proj_head.weight.data, src=0)
        logger.info(f"proj_head: {flat_dim} -> {proj_dim} (Stiefel)")

    # ---- Optimizer ----
    params = list(raw_model.parameters())
    if proj_head is not None:
        params += list(proj_head.parameters())
    opt = torch.optim.AdamW(
        params, lr=cfg.get("lr", 1e-4), weight_decay=cfg.get("weight_decay", 0.0),
    )

    # ---- Data ----
    if cfg.get("dummy", False):
        batch_size = cfg.get("global_batch_size", 256)
        dataset = DummyLatentDataset(
            num_samples=max(batch_size * 10, 1000),
            num_classes=cfg.get("num_classes", 1000),
        )
        logger.info("Using dummy dataset")
    else:
        dataset = LatentDataset(cfg.get("latent_dir", "data/cached_latents"))
        logger.info(f"Dataset: {len(dataset):,} latents")

    if dist.is_initialized():
        sampler = DistributedSampler(
            dataset, num_replicas=world_size, rank=rank,
            shuffle=True, seed=cfg.get("seed", 0),
        )
    else:
        sampler = None

    local_batch_size = cfg.get("global_batch_size", 256) // world_size
    loader = DataLoader(
        dataset, batch_size=local_batch_size, shuffle=(sampler is None),
        sampler=sampler, num_workers=cfg.get("num_workers", 4),
        pin_memory=True, drop_last=True,
        persistent_workers=cfg.get("num_workers", 4) > 0,
    )

    # ---- Resume from checkpoint ----
    start_step = 0
    if cfg.get("resume"):
        ckpt = torch.load(cfg["resume"], map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"], strict=False)
        if "ema" in ckpt:
            ema.load_state_dict(ckpt["ema"], strict=False)
        if "opt" in ckpt and not use_proj:
            opt.load_state_dict(ckpt["opt"])
        elif "opt" in ckpt and "proj_head" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        if proj_head is not None and "proj_head" in ckpt:
            proj_head.load_state_dict(ckpt["proj_head"])
            # Restore Stiefel constraint
            with torch.no_grad():
                Q, _ = torch.linalg.qr(proj_head.weight.data.T)
                proj_head.weight.data.copy_(Q.T)
            if dist.is_initialized():
                dist.broadcast(proj_head.weight.data, src=0)
            logger.info("proj_head loaded (Stiefel re-projected)")
        start_step = ckpt.get("train_steps", 0)
        logger.info(f"Resuming from step {start_step}")

    # ---- WandB ----
    if rank == 0 and HAS_WANDB and not cfg.get("no_wandb", False):
        run_name = cfg.get("wandb_run_name") or f"{phase_tag}-HiT-{cfg['size']}/{cfg.get('patch_size', 2)}"
        wandb.init(
            project=cfg.get("wandb_project", "project-hit"),
            name=run_name,
            config=cfg,
        )

    # ---- Training loop ----
    update_ema(ema, raw_model, decay=0)
    model.train()
    ema.eval()

    train_steps = start_step
    log_steps = 0
    running_loss = 0.0
    running_reg_loss = 0.0
    running_mse_monitor = 0.0
    running_poincare = 0.0
    running_reg_corr = 0.0
    running_hyp_norm = 0.0
    start_time = time()
    max_steps = cfg.get("max_steps", 100000)
    log_every = cfg.get("log_every", 100)
    ckpt_every = cfg.get("ckpt_every", 10000)
    hybrid_lambda = cfg.get("hybrid_lambda", 0.1)

    logger.info(f"Starting training for {max_steps} steps")
    epoch = 0
    if dist.is_initialized() and sampler is not None:
        sampler.set_epoch(epoch)
    data_iter = iter(loader)

    while train_steps < max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            epoch += 1
            if dist.is_initialized() and sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(loader)
            x, y = next(data_iter)

        x = x.to(device)
        y = y.to(device)
        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)

        loss_dict = diffusion.training_losses(
            model, x, t, dict(y=y), proj_head=proj_head,
        )

        # Loss: hybrid (MSE + lambda * d_H^2) or pure hyperbolic
        if hybrid:
            mse_loss = loss_dict["mse_monitor"].mean()
            poincare_loss = loss_dict["poincare"].mean()
            loss = mse_loss + hybrid_lambda * poincare_loss
        else:
            loss = loss_dict["loss"].mean()

        # === Origin-based linear regularization ===
        reg_loss_val = 0.0
        reg_corr_val = 0.0
        if reg_mode == "linear" and "pred_x0" in loss_dict:
            t_norm = t.float() / diffusion.num_timesteps

            pred_x0 = loss_dict["pred_x0"]
            x0_flat = pred_x0.reshape(pred_x0.shape[0], -1)
            x0_proj = proj_head(x0_flat)
            scale = math.sqrt(x0_proj.shape[1])
            x0_hyp = poincare.hard_clip(
                poincare.expmap0(
                    poincare.tangent_clip(x0_proj / scale, alpha=0.95)
                )
            )

            d_origin = poincare.distance_to_origin(x0_hyp)

            cutoff = cfg.get("cutoff", 1000)
            mask = (t <= cutoff).float()

            rmax = cfg.get("rmax", 2.0)
            target = rmax * (1.0 - t_norm)
            reg = ((d_origin - target) ** 2 * mask).sum() / mask.sum().clamp(min=1.0)

            reg_lambda = cfg.get("reg_lambda", 0.05)
            reg_loss = reg_lambda * reg

            with torch.no_grad():
                dd, tt = d_origin.detach(), t.float()
                reg_corr_val = (
                    ((dd - dd.mean()) * (tt - tt.mean())).mean()
                    / (dd.std() * tt.std() + 1e-8)
                ).item()

            loss = loss + reg_loss
            reg_loss_val = reg_loss.item()

        opt.zero_grad()
        loss.backward()

        if dist.is_initialized() and proj_head is not None:
            for p in proj_head.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                    p.grad.div_(world_size)

        clip_params = list(raw_model.parameters())
        if proj_head is not None:
            clip_params += list(proj_head.parameters())
        torch.nn.utils.clip_grad_norm_(clip_params, max_norm=1.0)

        opt.step()

        if proj_head is not None:
            with torch.no_grad():
                Q, _ = torch.linalg.qr(proj_head.weight.data.T)
                proj_head.weight.data.copy_(Q.T)
            if dist.is_initialized():
                dist.broadcast(proj_head.weight.data, src=0)

        update_ema(ema, raw_model)

        # ---- Accumulate metrics ----
        running_loss += loss.item()
        running_reg_loss += reg_loss_val
        running_reg_corr += reg_corr_val
        if "mse_monitor" in loss_dict:
            running_mse_monitor += loss_dict["mse_monitor"].mean().item()
        if "poincare" in loss_dict:
            running_poincare += loss_dict["poincare"].mean().item()
        if "pred_hyp" in loss_dict:
            with torch.no_grad():
                d_orig = poincare.distance_to_origin(loss_dict["pred_hyp"])
                running_hyp_norm += d_orig.mean().item()
        log_steps += 1
        train_steps += 1

        # ---- Logging ----
        if train_steps % log_every == 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time() - start_time
            steps_per_sec = log_steps / elapsed
            avg_loss = running_loss / log_steps
            avg_reg = running_reg_loss / log_steps

            if dist.is_initialized():
                loss_tensor = torch.tensor(avg_loss, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                avg_loss = loss_tensor.item() / world_size

            avg_mse_monitor = running_mse_monitor / log_steps
            avg_poincare = running_poincare / log_steps
            avg_reg_corr = running_reg_corr / log_steps

            logger.info(
                f"(step={train_steps:07d}) "
                f"Loss: {avg_loss:.4f}, "
                f"MSE: {avg_mse_monitor:.4f}, "
                f"Poinc: {avg_poincare:.4f}, "
                f"Reg: {avg_reg:.4f}, "
                f"corr(d,t): {avg_reg_corr:+.3f}, "
                f"Steps/s: {steps_per_sec:.2f}"
            )

            if rank == 0 and HAS_WANDB and not cfg.get("no_wandb", False):
                avg_hyp_norm = running_hyp_norm / log_steps
                log_dict = {
                    "train/loss": avg_loss,
                    "train/mse_monitor": avg_mse_monitor,
                    "train/poincare": avg_poincare,
                    "train/reg_loss": avg_reg,
                    "train/reg_corr": avg_reg_corr,
                    "train/hyp_d_origin": avg_hyp_norm,
                    "train/steps_per_sec": steps_per_sec,
                    "train/step": train_steps,
                    "train/grad_norm": get_grad_norm(raw_model),
                }

                if train_steps % (log_every * 5) == 0:
                    geo_metrics = compute_geometric_metrics(
                        model, poincare, x, t, y, diffusion, device,
                        proj_head=proj_head,
                    )
                    log_dict.update(geo_metrics)

                wandb.log(log_dict, step=train_steps)

            running_loss = 0.0
            running_reg_loss = 0.0
            running_mse_monitor = 0.0
            running_poincare = 0.0
            running_reg_corr = 0.0
            running_hyp_norm = 0.0
            log_steps = 0
            start_time = time()

        # ---- Checkpoint ----
        if train_steps % ckpt_every == 0 and train_steps > 0:
            if rank == 0:
                extra = None
                if proj_head is not None:
                    extra = {"proj_head": proj_head.state_dict()}
                save_checkpoint(
                    raw_model, ema, opt, cfg, train_steps,
                    checkpoint_dir, logger, extra_state=extra,
                )
            if dist.is_initialized():
                dist.barrier()

    model.eval()
    logger.info("Training complete")
    if rank == 0 and HAS_WANDB and not cfg.get("no_wandb", False):
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HiT training")
    parser.add_argument("--config", type=str, required=True,
                        help="YAML config file path")
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path (resume training)")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--dummy", action="store_true",
                        help="Debug with dummy data")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override max_steps from YAML")
    cli = parser.parse_args()

    cfg = load_config(cli.config)
    if cli.resume:
        cfg["resume"] = cli.resume
    if cli.no_wandb:
        cfg["no_wandb"] = True
    if cli.dummy:
        cfg["dummy"] = True
    if cli.max_steps is not None:
        cfg["max_steps"] = cli.max_steps

    main(cfg)
