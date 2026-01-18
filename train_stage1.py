import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader

from src.vlv.data.coco import CocoImagesOnlyDataset, collate_images_only
from src.vlv.models.frozen_sd21 import FrozenSD21
from src.vlv.models.vision_to_textemb import VisionToTextEmb


@dataclass
class TrainArgs:
    batch_size: int = 8
    lr: float = 1e-4
    epochs: int = 1
    subset_ratio: float = 0.1
    seed: int = 42
    output_dir: str = "outputs/stage1"
    mixed_precision: str = "no"
    log_wandb: bool = False
    clip_grad_norm: Optional[float] = None
    data_dir: Optional[str] = None
    image_size: int = 512
    sd_model_id: str = "stabilityai/stable-diffusion-2-1-base"
    hf_token: Optional[str] = None


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Stage1 distillation training.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--subset_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/stage1")
    parser.add_argument("--mixed_precision", type=str, default="no")
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--clip_grad_norm", type=float, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--sd_model_id", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--hf_token", type=str, default=None)
    args = parser.parse_args()
    return TrainArgs(**vars(args))


def save_checkpoint(
    accelerator: Accelerator, model: torch.nn.Module, args: TrainArgs, step: int
) -> None:
    if not accelerator.is_main_process:
        return
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"vision_encoder_{step}.pt")
    torch.save(accelerator.get_state_dict(model), ckpt_path)
    with open(os.path.join(args.output_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(args), f, indent=2)


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = parse_args()
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    torch.manual_seed(args.seed)

    if args.log_wandb and accelerator.is_main_process:
        try:
            import wandb
        except Exception:
            raise RuntimeError("wandb is not installed but --log_wandb was set.")
        wandb.init(project="vlv-stage1", config=asdict(args))

    dataset = CocoImagesOnlyDataset(
        split="train",
        data_dir=args.data_dir,
        subset_ratio=args.subset_ratio,
        seed=args.seed,
        image_size=args.image_size,
        use_hf=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        collate_fn=collate_images_only,
        pin_memory=True,
    )

    if args.mixed_precision == "fp16":
        sd_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        sd_dtype = torch.bfloat16
    else:
        sd_dtype = torch.float32

    frozen_sd = FrozenSD21(
        device=accelerator.device,
        model_id=args.sd_model_id,
        dtype=sd_dtype,
        use_auth_token=args.hf_token,
    )
    vision_encoder = VisionToTextEmb()
    # vision_encoder = torch.compile(vision_encoder)
    optimizer = torch.optim.AdamW(vision_encoder.parameters(), lr=args.lr)

    vision_encoder, optimizer, dataloader = accelerator.prepare(
        vision_encoder, optimizer, dataloader
    )

    global_step = 0
    total_steps = len(dataloader) * args.epochs
    start_time = time.time()
    progress_every = 10
    for epoch in range(args.epochs):
        vision_encoder.train()
        running_loss = 0.0
        num_steps = 0

        for images in dataloader:
            images = images.to(accelerator.device, dtype=frozen_sd.dtype, non_blocking=True)
            with torch.no_grad():
                latents = frozen_sd.encode_images_to_latents(images)
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    frozen_sd.scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                    dtype=torch.long,
                )
                noisy_latents = frozen_sd.add_noise(latents, noise, timesteps)

            cond = vision_encoder(images).to(dtype=frozen_sd.dtype)
            pred = frozen_sd.unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states=cond,
            ).sample
            loss = F.mse_loss(pred, noise)

            accelerator.backward(loss)
            if args.clip_grad_norm is not None:
                accelerator.clip_grad_norm_(vision_encoder.parameters(), args.clip_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            running_loss += loss.detach().float().item()
            num_steps += 1
            global_step += 1

            if accelerator.is_main_process and global_step % progress_every == 0:
                elapsed = time.time() - start_time
                steps_done = max(1, global_step)
                steps_left = max(0, total_steps - global_step)
                rate = steps_done / max(elapsed, 1e-6)
                eta_sec = steps_left / max(rate, 1e-6)
                progress = 100.0 * global_step / max(total_steps, 1)
                print(
                    f"progress={progress:.1f}% step={global_step}/{total_steps} "
                    f"eta={eta_sec/60:.1f}m"
                )

            if accelerator.is_main_process and global_step % 50 == 0:
                avg_loss = running_loss / max(1, num_steps)
                print(f"epoch={epoch} step={global_step} loss={avg_loss:.6f}")
                if args.log_wandb:
                    import wandb

                    wandb.log({"loss": avg_loss, "step": global_step, "epoch": epoch})

        save_checkpoint(accelerator, vision_encoder, args, global_step)

    if args.log_wandb and accelerator.is_main_process:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
