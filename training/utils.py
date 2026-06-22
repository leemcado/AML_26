# training/utils.py — EMA, checkpoint, logging utilities

import os
import logging
import torch
import torch.distributed as dist
from collections import OrderedDict


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Update EMA weights — exponential moving average toward current model."""
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """Set requires_grad for all model parameters."""
    for p in model.parameters():
        p.requires_grad = flag


def create_logger(logging_dir):
    """Create file+stdout logger (rank 0 only)."""
    if dist.is_initialized() and dist.get_rank() != 0:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
        return logger

    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f"{logging_dir}/log.txt"),
        ] if logging_dir else [logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def save_checkpoint(model, ema, opt, args, train_steps, checkpoint_dir, logger,
                    extra_state=None):
    """Save checkpoint.

    Args:
        extra_state: Additional state_dicts (e.g. {"proj_head": proj_head.state_dict()}).
    """
    checkpoint = {
        "model": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        "ema": ema.state_dict(),
        "opt": opt.state_dict(),
        "args": vars(args) if hasattr(args, '__dict__') else args,
        "train_steps": train_steps,
    }
    if extra_state:
        checkpoint.update(extra_state)
    checkpoint_path = os.path.join(checkpoint_dir, f"{train_steps:07d}.pt")
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, model, ema=None, opt=None, device='cpu'):
    """Load checkpoint — restore model/EMA/optimizer state."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_state = checkpoint["model"]
    if hasattr(model, 'module'):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)
    if ema is not None and "ema" in checkpoint:
        ema.load_state_dict(checkpoint["ema"])
    if opt is not None and "opt" in checkpoint:
        opt.load_state_dict(checkpoint["opt"])
    train_steps = checkpoint.get("train_steps", 0)
    return train_steps


def get_grad_norm(model):
    """Compute total gradient L2 norm (for monitoring)."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5
