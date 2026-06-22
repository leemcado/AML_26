# evaluation/fid.py — 50K sample generation + FID measurement
# DDP multi-GPU parallel generation supported
#
# Usage:
#   torchrun --nproc_per_node=4 evaluation/fid.py \
#       --ckpt results/000-DiT-B-2/checkpoints/0100000.pt \
#       --model DiT-B/2 --ref-path data/imagenet_val
#
# Debug:
#   python evaluation/fid.py --dummy --num-fid-samples 64 --per-proc-batch-size 8

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from contextlib import nullcontext

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.distributed as dist
import math
import argparse
from pathlib import Path
from torchvision.utils import save_image

from models.dit import DiT_models
from models.hit import HiT_models
from diffusion import create_diffusion


def create_model(
    model_name: str,
    latent_size: int,
    num_classes: int,
    curvature: float,
) -> tuple[torch.nn.Module, bool]:
    """Create model instance."""
    all_models = {**DiT_models, **HiT_models}
    if model_name not in all_models:
        raise ValueError(
            f"Unknown model: {model_name}. Available: {list(all_models.keys())}"
        )
    is_hit = model_name in HiT_models
    kwargs = dict(input_size=latent_size, num_classes=num_classes)
    if is_hit:
        kwargs["curvature"] = curvature
    model = all_models[model_name](**kwargs)
    return model, is_hit


def load_ema_weights(
    model: torch.nn.Module,
    ckpt_path: str,
    device: torch.device,
) -> None:
    """Load EMA weights from checkpoint (fallback to model weights)."""
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "ema" in state_dict:
        model.load_state_dict(state_dict["ema"])
    elif "model" in state_dict:
        model.load_state_dict(state_dict["model"])
    else:
        model.load_state_dict(state_dict)


def load_vae(device: torch.device) -> torch.nn.Module | None:
    """Load Stable Diffusion VAE decoder."""
    try:
        from diffusers.models import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse"
        ).to(device)
        vae.eval()
        return vae
    except Exception as e:
        print(f"WARNING: VAE load failed ({e}), saving raw latents")
        return None


def generate_samples(
    model: torch.nn.Module,
    diffusion: object,
    vae: torch.nn.Module | None,
    output_dir: Path,
    num_samples: int,
    batch_size: int,
    num_classes: int,
    cfg_scale: float,
    image_size: int,
    seed: int,
    rank: int,
    world_size: int,
    device: torch.device,
) -> int:
    """DDP sample generation + PNG save.

    Uniform class distribution: 50 samples/class (for 50K total).
    """
    latent_size = image_size // 8
    samples_per_rank = math.ceil(num_samples / world_size)
    global_offset = rank * samples_per_rank

    actual_samples = min(samples_per_rank, num_samples - global_offset)
    if actual_samples <= 0:
        return 0

    torch.manual_seed(seed + rank)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    while generated < actual_samples:
        current_batch = min(batch_size, actual_samples - generated)

        # Uniform class distribution
        global_indices = torch.arange(
            global_offset + generated,
            global_offset + generated + current_batch,
        )
        class_labels = (global_indices % num_classes).to(device)

        z = torch.randn(
            current_batch, 4, latent_size, latent_size,
            device=device,
        )

        # CFG: [cond, uncond]
        z = torch.cat([z, z], dim=0)
        y_null = torch.full_like(class_labels, num_classes)
        y = torch.cat([class_labels, y_null], dim=0)
        model_kwargs = dict(y=y, cfg_scale=cfg_scale)

        samples = diffusion.p_sample_loop(
            model.forward_with_cfg,
            z.shape,
            z,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=device,
        )
        samples, _ = samples.chunk(2, dim=0)

        # VAE decode
        if vae is not None:
            with torch.no_grad():
                samples = vae.decode(samples.float() / 0.18215).sample

        for i in range(samples.shape[0]):
            global_idx = global_offset + generated + i
            save_image(
                samples[i],
                output_dir / f"{global_idx:05d}.png",
                normalize=True,
                value_range=(-1, 1),
            )

        generated += current_batch
        if rank == 0:
            total_done = generated * world_size
            print(
                f"\r[Rank 0] Generated: {min(total_done, num_samples)}/{num_samples}",
                end="", flush=True,
            )

    if rank == 0:
        print()

    return generated


def compute_fid(generated_dir: str, ref_path: str, device_str: str) -> float:
    """Compute FID using pytorch-fid."""
    from pytorch_fid.fid_score import calculate_fid_given_paths

    fid_value = calculate_fid_given_paths(
        [generated_dir, ref_path],
        batch_size=50,
        device=device_str,
        dims=2048,
    )
    return fid_value


def main(args: argparse.Namespace) -> None:
    """FID measurement main."""
    # DDP setup
    use_ddp = "RANK" in os.environ
    if use_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    else:
        rank = 0
        world_size = 1
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    torch.set_grad_enabled(False)

    if rank == 0:
        print(f"FID evaluation: {args.model}, "
              f"samples={args.num_fid_samples}, "
              f"world_size={world_size}")

    # Model
    model, is_hit = create_model(
        args.model,
        latent_size=args.image_size // 8,
        num_classes=args.num_classes,
        curvature=args.curvature,
    )

    if not args.dummy:
        load_ema_weights(model, args.ckpt, device)
    model = model.to(device)
    model.eval()

    # torch.compile optimization
    use_autocast = torch.cuda.is_available()
    if not args.no_compile:
        try:
            model = torch.compile(model)
            if rank == 0:
                print("torch.compile applied")
        except Exception:
            if rank == 0:
                print("torch.compile unavailable, using eager mode")

    # Diffusion
    diffusion = create_diffusion(
        str(args.num_sampling_steps),
        use_poincare=is_hit,
        curvature=args.curvature if is_hit else 1e-5,
    )

    # VAE
    if not args.dummy:
        vae = load_vae(device)
    else:
        vae = None
        if rank == 0:
            print("Dummy mode: skipping VAE")

    # Output directory
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if use_ddp:
        dist.barrier()

    # Generate
    if rank == 0:
        print(f"Generating {args.num_fid_samples} samples -> {output_dir}/")

    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else nullcontext()
    with ctx:
        n_generated = generate_samples(
            model=model,
            diffusion=diffusion,
            vae=vae,
            output_dir=output_dir,
            num_samples=args.num_fid_samples,
            batch_size=args.per_proc_batch_size,
            num_classes=args.num_classes,
            cfg_scale=args.cfg_scale,
            image_size=args.image_size,
            seed=args.seed,
            rank=rank,
            world_size=world_size,
            device=device,
        )

    if use_ddp:
        dist.barrier()

    # FID computation
    if rank == 0:
        total_files = len(list(output_dir.glob("*.png")))
        print(f"Total PNGs: {total_files}")

        if args.ref_path:
            print(f"Computing FID (reference: {args.ref_path})...")
            fid = compute_fid(
                str(output_dir),
                args.ref_path,
                device_str="cuda" if torch.cuda.is_available() else "cpu",
            )
            print(f"FID: {fid:.2f}")
        else:
            print("No --ref-path provided, skipping FID")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sample generation + FID measurement"
    )

    parser.add_argument("--model", type=str, default="DiT-B/2")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--curvature", type=float, default=1e-5)

    parser.add_argument("--num-fid-samples", type=int, default=50000)
    parser.add_argument("--per-proc-batch-size", type=int, default=32)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--ref-path", type=str, default=None,
        help="ImageNet val folder or .npz statistics file",
    )
    parser.add_argument("--output-dir", type=str, default="fid_samples")

    parser.add_argument(
        "--dummy", action="store_true",
        help="Quick test without checkpoint/VAE",
    )
    parser.add_argument(
        "--no-compile", action="store_true",
        help="Disable torch.compile",
    )

    args = parser.parse_args()

    if not args.dummy and args.ckpt is None:
        parser.error("--ckpt is required unless --dummy is set")

    main(args)
