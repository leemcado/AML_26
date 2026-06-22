# data/extract_latents.py — ImageNet -> VAE latent extraction
# Run once before training to create latent cache (removes VAE overhead)
#
# Usage:
#   python data/extract_latents.py --data-path /path/to/imagenet --output-dir /path/to/latents
#   torchrun --nproc_per_node=2 data/extract_latents.py --data-path ... --output-dir ...

import os
import argparse
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from diffusers.models import AutoencoderKL


def center_crop_arr(pil_image, image_size):
    """Center-crop image to square."""
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def main(args):
    # DDP setup (optional)
    use_ddp = 'RANK' in os.environ
    if use_ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = rank % torch.cuda.device_count()
    else:
        rank = 0
        world_size = 1
        device = 0 if torch.cuda.is_available() else 'cpu'

    torch.cuda.set_device(device) if torch.cuda.is_available() else None

    # Load VAE
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae.eval()

    # Dataset
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    dataset = ImageFolder(args.data_path, transform=transform)

    if use_ddp:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        sampler = None

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        sampler=sampler, num_workers=args.num_workers, pin_memory=True,
    )

    # Output directory
    class_dirs = {}
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        for class_idx in range(len(dataset.classes)):
            class_dir = os.path.join(args.output_dir, str(class_idx))
            os.makedirs(class_dir, exist_ok=True)
            class_dirs[class_idx] = class_dir
    if use_ddp:
        dist.barrier()
    for class_idx in range(len(dataset.classes)):
        class_dirs[class_idx] = os.path.join(args.output_dir, str(class_idx))

    # Extract latents
    global_idx = rank
    pbar = tqdm(loader, disable=(rank != 0), desc="Extracting latents")

    for batch_imgs, batch_labels in pbar:
        batch_imgs = batch_imgs.to(device)
        with torch.no_grad():
            latents = vae.encode(batch_imgs).latent_dist.sample().mul_(0.18215)

        for i in range(latents.shape[0]):
            label = batch_labels[i].item()
            latent = latents[i].cpu().to(torch.bfloat16)
            save_path = os.path.join(class_dirs[label], f"{global_idx:07d}.pt")
            torch.save({"latent": latent, "label": label}, save_path)
            global_idx += world_size

    if rank == 0:
        print(f"Done! Latents saved to: {args.output_dir}")

    if use_ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ImageNet -> VAE latent extraction")
    parser.add_argument("--data-path", type=str, required=True, help="ImageNet directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Latent output path")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    args = parser.parse_args()
    main(args)
