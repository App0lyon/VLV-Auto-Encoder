import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Optional

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from src.vlv.data.coco import CocoImageCaptionDataset, collate_images_and_captions
from src.vlv.models.emb_to_caption import EmbeddingCaptioner
from src.vlv.models.vision_to_textemb import VisionToTextEmb


@dataclass
class TrainArgs:
    stage1_ckpt: str
    llm_name: str = "gpt2-medium"
    batch_size: int = 8
    lr: float = 1e-4
    epochs: int = 1
    subset_ratio: float = 0.1
    output_dir: str = "outputs/stage2"
    max_length: int = 64
    seed: int = 42
    data_dir: Optional[str] = None
    image_size: int = 512
    mixed_precision: str = "no"
    finetune_encoder: bool = False


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="Stage2 captioning training.")
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--llm_name", type=str, default="distilgpt2")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--subset_ratio", type=float, default=0.1)
    parser.add_argument("--output_dir", type=str, default="outputs/stage2")
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--mixed_precision", type=str, default="no")
    parser.add_argument("--finetune_encoder", action="store_true")
    args = parser.parse_args()
    return TrainArgs(**vars(args))


def save_checkpoint(
    accelerator: Accelerator,
    captioner: EmbeddingCaptioner,
    args: TrainArgs,
    step: int,
) -> None:
    if not accelerator.is_main_process:
        return
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_path = os.path.join(args.output_dir, f"captioner_{step}.pt")
    torch.save(accelerator.get_state_dict(captioner), ckpt_path)
    captioner.tokenizer.save_pretrained(args.output_dir)
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(args), f, indent=2)


def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = parse_args()
    accelerator = Accelerator(mixed_precision=args.mixed_precision)
    torch.manual_seed(args.seed)

    dataset = CocoImageCaptionDataset(
        split="train",
        data_dir=args.data_dir,
        subset_ratio=args.subset_ratio,
        seed=args.seed,
        image_size=args.image_size,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=6,
        collate_fn=collate_images_and_captions,
        pin_memory=True,
    )

    vision_encoder = VisionToTextEmb()
    state = torch.load(args.stage1_ckpt, map_location="cpu")
    vision_encoder.load_state_dict(state, strict=False)
    if not args.finetune_encoder:
        for p in vision_encoder.parameters():
            p.requires_grad = False
        vision_encoder.eval()

    captioner = EmbeddingCaptioner(model_name=args.llm_name, max_length=args.max_length)
    params = list(captioner.parameters())
    if args.finetune_encoder:
        params += list(vision_encoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    vision_encoder, captioner, optimizer, dataloader = accelerator.prepare(
        vision_encoder, captioner, optimizer, dataloader
    )

    global_step = 0
    total_steps = len(dataloader) * args.epochs
    start_time = time.time()
    progress_every = 100
    for epoch in range(args.epochs):
        captioner.train()
        if args.finetune_encoder:
            vision_encoder.train()

        running_loss = 0.0
        num_steps = 0
        for images, captions in dataloader:
            images = images.to(accelerator.device, non_blocking=True)
            with torch.set_grad_enabled(args.finetune_encoder):
                if args.finetune_encoder:
                    emb = vision_encoder(images)
                else:
                    with torch.no_grad():
                        emb = vision_encoder(images)
            out = captioner(emb, captions=captions)
            loss = out.loss

            accelerator.backward(loss)
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

        save_checkpoint(accelerator, captioner, args, global_step)


if __name__ == "__main__":
    main()
