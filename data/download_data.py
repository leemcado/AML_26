# data/download_data.py — Download VAE latents from HuggingFace
# Supports ImageNet-1K latent, ImageNet-100, CIFAR-100
#
# Usage:
#   python data/download_data.py --dataset imagenet-1k --output-dir data/cached_latents
#   python data/download_data.py --dataset cifar100 --output-dir data/cached_latents
#
# Prerequisites:
#   pip install datasets huggingface_hub
#   huggingface-cli login  (for ImageNet access)

import os
import argparse
import torch
from tqdm import tqdm


def download_imagenet_latents(output_dir, subset_classes=None, max_samples=None):
    """Download pre-extracted ImageNet-1K latents from HuggingFace."""
    from datasets import load_dataset

    print("Loading ImageNet-1K latents from HuggingFace...")
    print("(Requires HuggingFace login + dataset access agreement)")

    ds = load_dataset("Forbu14/imagenet-1k-latent", split="train", streaming=True)

    os.makedirs(output_dir, exist_ok=True)
    class_counters = {}
    total_saved = 0

    for sample in tqdm(ds, desc="Downloading"):
        label = sample["label"]

        if subset_classes is not None and label not in subset_classes:
            continue

        class_dir = os.path.join(output_dir, str(label))
        os.makedirs(class_dir, exist_ok=True)

        idx = class_counters.get(label, 0)
        class_counters[label] = idx + 1

        if "latent" in sample:
            latent = torch.tensor(sample["latent"], dtype=torch.bfloat16)
        elif "image" in sample:
            print("WARNING: raw image format, use extract_latents.py instead")
            return False
        else:
            keys = list(sample.keys())
            print(f"Available keys: {keys}")
            return False

        save_path = os.path.join(class_dir, f"{idx:07d}.pt")
        torch.save({"latent": latent, "label": label}, save_path)
        total_saved += 1

        if max_samples and total_saved >= max_samples:
            break

    print(f"Done! {total_saved} latents, {len(class_counters)} classes -> {output_dir}")
    return True


def download_imagenet_raw_and_extract(output_dir, image_size=256, vae_type="ema",
                                      subset_classes=None, max_samples=None):
    """Download raw ImageNet + VAE encode to extract latents. DDP supported."""
    from datasets import load_dataset
    from diffusers.models import AutoencoderKL
    from torchvision import transforms
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group("nccl")
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if rank == 0:
        print(f"Loading VAE (sd-vae-ft-{vae_type}), {world_size} GPUs...")

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_type}").to(device)
    vae.eval()

    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    if rank == 0:
        print("Streaming ImageNet-1K from HuggingFace...")

    ds = load_dataset("ILSVRC/imagenet-1k", split="train", streaming=True, token=True)

    if world_size > 1:
        ds = ds.shard(num_shards=world_size, index=rank)

    os.makedirs(output_dir, exist_ok=True)
    class_counters = {}
    total_saved = 0
    batch_imgs = []
    batch_labels = []
    batch_size = 128

    iterator = tqdm(ds, desc=f"Rank {rank} extracting") if rank == 0 else ds

    for sample in iterator:
        label = sample["label"]
        img = sample["image"]

        if subset_classes is not None and label not in subset_classes:
            continue

        if img.mode != "RGB":
            img = img.convert("RGB")

        img_tensor = transform(img)
        batch_imgs.append(img_tensor)
        batch_labels.append(label)

        if len(batch_imgs) >= batch_size:
            batch = torch.stack(batch_imgs).to(device)
            with torch.no_grad():
                latents = vae.encode(batch).latent_dist.sample().mul_(0.18215)

            for i in range(latents.shape[0]):
                lbl = batch_labels[i]
                class_dir = os.path.join(output_dir, str(lbl))
                os.makedirs(class_dir, exist_ok=True)

                idx = class_counters.get(lbl, 0)
                class_counters[lbl] = idx + 1

                latent = latents[i].cpu().to(torch.bfloat16)
                save_path = os.path.join(class_dir, f"{idx:07d}_r{rank}.pt")
                torch.save({"latent": latent, "label": lbl}, save_path)
                total_saved += 1

            batch_imgs = []
            batch_labels = []

            if max_samples and total_saved >= max_samples:
                break

    # Process remaining batch
    if batch_imgs and (not max_samples or total_saved < max_samples):
        batch = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            latents = vae.encode(batch).latent_dist.sample().mul_(0.18215)
        for i in range(latents.shape[0]):
            lbl = batch_labels[i]
            class_dir = os.path.join(output_dir, str(lbl))
            os.makedirs(class_dir, exist_ok=True)
            idx = class_counters.get(lbl, 0)
            class_counters[lbl] = idx + 1
            latent = latents[i].cpu().to(torch.bfloat16)
            save_path = os.path.join(class_dir, f"{idx:07d}_r{rank}.pt")
            torch.save({"latent": latent, "label": lbl}, save_path)
            total_saved += 1

    print(f"[Rank {rank}] Done! {total_saved} saved")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()
    return True


def download_cifar100_and_extract(output_dir, vae_type="ema"):
    """Download CIFAR-100 -> upscale to 256x256 -> extract VAE latents."""
    from torchvision.datasets import CIFAR100
    from torchvision import transforms
    from diffusers.models import AutoencoderKL

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading VAE ({device})...")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{vae_type}").to(device)
    vae.eval()

    transform = transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    print("Downloading CIFAR-100...")
    dataset = CIFAR100(root="data/cifar100_raw", train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    os.makedirs(output_dir, exist_ok=True)
    class_counters = {}
    total_saved = 0

    for batch_imgs, batch_labels in tqdm(loader, desc="Extracting CIFAR-100 latents"):
        batch_imgs = batch_imgs.to(device)
        with torch.no_grad():
            latents = vae.encode(batch_imgs).latent_dist.sample().mul_(0.18215)

        for i in range(latents.shape[0]):
            label = batch_labels[i].item()
            class_dir = os.path.join(output_dir, str(label))
            os.makedirs(class_dir, exist_ok=True)
            idx = class_counters.get(label, 0)
            class_counters[label] = idx + 1

            latent = latents[i].cpu().to(torch.bfloat16)
            save_path = os.path.join(class_dir, f"{idx:07d}.pt")
            torch.save({"latent": latent, "label": label}, save_path)
            total_saved += 1

    print(f"Done! {total_saved} CIFAR-100 latents -> {output_dir}")
    return True


IMAGENET_100_CLASSES = set(range(100))


def main(args):
    if args.dataset == "imagenet-1k-latent":
        success = download_imagenet_latents(
            args.output_dir,
            max_samples=args.max_samples,
        )
        if not success:
            print("Pre-extracted latent download failed. Falling back to raw extraction...")
            download_imagenet_raw_and_extract(args.output_dir, max_samples=args.max_samples)

    elif args.dataset == "imagenet-1k":
        download_imagenet_raw_and_extract(
            args.output_dir,
            image_size=args.image_size,
            vae_type=args.vae,
            max_samples=args.max_samples,
        )

    elif args.dataset == "imagenet-100":
        download_imagenet_raw_and_extract(
            args.output_dir,
            image_size=args.image_size,
            vae_type=args.vae,
            subset_classes=IMAGENET_100_CLASSES,
            max_samples=args.max_samples,
        )

    elif args.dataset == "cifar100":
        download_cifar100_and_extract(
            args.output_dir,
            vae_type=args.vae,
        )

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Summary
    total_files = sum(
        len(files) for _, _, files in os.walk(args.output_dir) if files
    )
    total_size_mb = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fns in os.walk(args.output_dir)
        for f in fns
    ) / (1024 * 1024)
    print(f"\n=== Data Summary ===")
    print(f"Directory: {args.output_dir}")
    print(f"Total files: {total_files:,}")
    print(f"Total size: {total_size_mb:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and prepare training data")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["imagenet-1k-latent", "imagenet-1k", "imagenet-100", "cifar100"],
                        help="Dataset to download")
    parser.add_argument("--output-dir", type=str, default="data/cached_latents",
                        help="Output path for cached latents")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Max number of samples (for testing)")
    args = parser.parse_args()
    main(args)
