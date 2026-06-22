# evaluation/sample.py — DiT/HiT sample image generation
# CFG sampling + VAE decoding
#
# Usage:
#   python evaluation/sample.py --ckpt results/.../checkpoints/0100000.pt --model DiT-B/2

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torchvision.utils import save_image
import argparse

from models.dit import DiT_models
from models.hit import HiT_models
from diffusion import create_diffusion


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    all_models = {**DiT_models, **HiT_models}
    is_hit = args.model in HiT_models

    latent_size = args.image_size // 8
    model_kwargs_init = dict(input_size=latent_size, num_classes=args.num_classes)
    if is_hit:
        model_kwargs_init["curvature"] = args.curvature

    model = all_models[args.model](**model_kwargs_init).to(device)

    # Load checkpoint
    if args.ckpt:
        state_dict = torch.load(args.ckpt, map_location=device, weights_only=True)
        if "ema" in state_dict:
            model.load_state_dict(state_dict["ema"])
        elif "model" in state_dict:
            model.load_state_dict(state_dict["model"])
        else:
            model.load_state_dict(state_dict)

    model.eval()

    use_poincare = is_hit
    diffusion = create_diffusion(
        str(args.num_sampling_steps),
        use_poincare=use_poincare,
        curvature=args.curvature if is_hit else 1e-5,
    )

    # VAE decoder
    vae = None
    try:
        from diffusers.models import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
        vae.eval()
    except Exception:
        print("WARNING: VAE load failed, saving raw latents")

    # CFG setup
    class_labels = [int(x) for x in args.class_labels.split(",")]
    n = len(class_labels)
    z = torch.randn(n, 4, latent_size, latent_size, device=device)
    y = torch.tensor(class_labels, device=device)

    z = torch.cat([z, z], 0)
    y_null = torch.tensor([args.num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

    # Sampling
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False,
        model_kwargs=model_kwargs, progress=True, device=device,
    )
    samples, _ = samples.chunk(2, dim=0)

    # VAE decode
    if vae is not None:
        samples = vae.decode(samples / 0.18215).sample

    save_image(samples, args.output, nrow=4, normalize=True, value_range=(-1, 1))
    print(f"Samples saved: {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DiT/HiT sample generation")
    parser.add_argument("--model", type=str, default="DiT-B/2")
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--curvature", type=float, default=1e-5)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--class-labels", type=str, default="207,360,387,974,88,979,417,279")
    parser.add_argument("--output", type=str, default="sample.png")
    args = parser.parse_args()
    main(args)
