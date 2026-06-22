# training/train_dit.py — DiT baseline training
# adaLN-Zero conditional latent diffusion + CFG + DDP
#
# Usage:
#   torchrun --nproc_per_node=4 training/train_dit.py --config configs/dit_b.yaml
#   python training/train_dit.py --config configs/dit_b.yaml --dummy

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from copy import deepcopy
from glob import glob
from time import time
import argparse
import yaml

from models.dit import build_dit
from diffusion import create_diffusion
from data.latent_dataset import LatentDataset, DummyLatentDataset
from training.utils import (
    update_ema, requires_grad, create_logger,
    save_checkpoint, load_checkpoint, get_grad_norm,
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


def main(cfg: dict) -> None:
    # ---- DDP setup ----
    rank, world_size, device, device_id = setup_distributed()
    seed = cfg.get("seed", 0) * world_size + rank
    torch.manual_seed(seed)

    # ---- Experiment directory (rank 0 only) ----
    results_dir = cfg.get("results_dir", "results")
    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)
        experiment_index = len(glob(f"{results_dir}/*"))
        size = cfg["size"]
        ps = cfg.get("patch_size", 2)
        experiment_dir = f"{results_dir}/{experiment_index:03d}-DiT-{size}-{ps}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
    else:
        experiment_dir = checkpoint_dir = None
        logger = create_logger(None)

    logger.info(f"rank={rank}, seed={seed}, world_size={world_size}")

    # ---- Model ----
    latent_size = cfg.get("image_size", 256) // 8
    model = build_dit(
        size=cfg["size"],
        patch_size=cfg.get("patch_size", 2),
        input_size=latent_size,
        num_classes=cfg.get("num_classes", 1000),
    )

    # EMA: moving average weights for stable sampling (no gradient computation)
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)

    # DDP: auto gradient all-reduce via NCCL during backward()
    if dist.is_initialized():
        model = DDP(model.to(device), device_ids=[device_id])
    else:
        model = model.to(device)

    diffusion = create_diffusion(timestep_respacing="")
    raw_model = model.module if hasattr(model, "module") else model
    param_count = sum(p.numel() for p in raw_model.parameters())
    logger.info(f"DiT-{cfg['size']}/{cfg.get('patch_size', 2)} — {param_count:,} params")

    # ---- Optimizer ----
    opt = torch.optim.AdamW(
        raw_model.parameters(),
        lr=cfg.get("lr", 1e-4),
        weight_decay=cfg.get("weight_decay", 0.0),
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

    # DistributedSampler: non-overlapping data shards per rank
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
        start_step = load_checkpoint(cfg["resume"], raw_model, ema, opt, device)
        logger.info(f"Resuming from step {start_step}")

    # ---- WandB ----
    if rank == 0 and HAS_WANDB and not cfg.get("no_wandb", False):
        wandb.init(
            project=cfg.get("wandb_project", "project-hit"),
            name=cfg.get("wandb_run_name") or f"DiT-{cfg['size']}/{cfg.get('patch_size', 2)}",
            config=cfg,
        )

    # ---- Training loop ----
    update_ema(ema, raw_model, decay=0)  # Initialize EMA (copy model weights)
    model.train()
    ema.eval()

    train_steps = start_step
    log_steps = 0
    running_loss = 0.0
    start_time = time()
    max_steps = cfg.get("max_steps", 100000)
    log_every = cfg.get("log_every", 100)
    ckpt_every = cfg.get("ckpt_every", 10000)

    logger.info(f"Starting training for {max_steps} steps")
    epoch = 0
    if dist.is_initialized() and sampler is not None:
        sampler.set_epoch(epoch)
    data_iter = iter(loader)

    while train_steps < max_steps:
        # Infinite iterator: advance epoch on StopIteration
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

        loss_dict = diffusion.training_losses(model, x, t, dict(y=y))
        loss = loss_dict["loss"].mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        update_ema(ema, raw_model)

        running_loss += loss.item()
        log_steps += 1
        train_steps += 1

        # ---- Logging ----
        if train_steps % log_every == 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time() - start_time
            steps_per_sec = log_steps / elapsed
            avg_loss = running_loss / log_steps

            if dist.is_initialized():
                loss_tensor = torch.tensor(avg_loss, device=device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                avg_loss = loss_tensor.item() / world_size

            logger.info(
                f"(step={train_steps:07d}) "
                f"Loss: {avg_loss:.4f}, "
                f"Steps/s: {steps_per_sec:.2f}"
            )

            if rank == 0 and HAS_WANDB and not cfg.get("no_wandb", False):
                log_dict = {
                    "train/loss": avg_loss,
                    "train/steps_per_sec": steps_per_sec,
                    "train/step": train_steps,
                    "train/grad_norm": get_grad_norm(raw_model),
                }
                if "mse" in loss_dict:
                    log_dict["train/mse"] = loss_dict["mse"].mean().item()
                if "vb" in loss_dict:
                    log_dict["train/vb"] = loss_dict["vb"].mean().item()
                wandb.log(log_dict, step=train_steps)

            running_loss = 0.0
            log_steps = 0
            start_time = time()

        # ---- Checkpoint ----
        if train_steps % ckpt_every == 0 and train_steps > 0:
            if rank == 0:
                save_checkpoint(
                    raw_model, ema, opt, cfg, train_steps, checkpoint_dir, logger,
                )
            if dist.is_initialized():
                dist.barrier()

    model.eval()
    logger.info("Training complete")
    if rank == 0 and HAS_WANDB and not cfg.get("no_wandb", False):
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiT training")
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
