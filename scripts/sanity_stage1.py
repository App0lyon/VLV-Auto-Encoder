import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from src.vlv.data.coco import CocoImagesOnlyDataset, collate_images_only
from src.vlv.models.frozen_sd21 import FrozenSD21
from src.vlv.models.vision_to_textemb import VisionToTextEmb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sanity check for stage1 checkpoint.")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to stage1 checkpoint.")
    parser.add_argument("--data_dir", type=str, default=None, help="Local COCO root.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--save_grid", type=str, default=None, help="Path to save image grid.")
    return parser.parse_args()


def save_grid(images: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    images = images.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    for ax, img in zip(axes.flat, images):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    dataset = CocoImagesOnlyDataset(
        split="val",
        data_dir=args.data_dir,
        subset_ratio=0.01,
        seed=args.seed,
        image_size=args.image_size,
        use_hf=True,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_images_only,
    )

    images = next(iter(dataloader))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = images.to(device)

    frozen_sd = FrozenSD21(device=device)
    model = VisionToTextEmb().to(device)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        latents = frozen_sd.encode_images_to_latents(images)
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0,
            frozen_sd.scheduler.config.num_train_timesteps,
            (latents.shape[0],),
            device=latents.device,
            dtype=torch.long,
        )
        noisy_latents = frozen_sd.add_noise(latents, noise, timesteps)
        cond = model(images)
        pred = frozen_sd.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=cond,
        ).sample
        loss = F.mse_loss(pred, noise).item()

    mean = cond.mean().item()
    std = cond.std().item()

    print(f"batch_loss={loss:.6f}")
    print(f"embeddings mean={mean:.6f} std={std:.6f}")

    if args.save_grid:
        save_grid(images, args.save_grid)
        print(f"saved grid to {args.save_grid}")


if __name__ == "__main__":
    main()
