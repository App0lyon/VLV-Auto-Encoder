import argparse
import json
import os

import torch
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionPipeline
from transformers import AutoTokenizer

from src.vlv.models.emb_to_caption import EmbeddingCaptioner
from src.vlv.models.vision_to_textemb import VisionToTextEmb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference: image -> caption.")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage2_ckpt", type=str, required=True)

    parser.add_argument(
        "--prompt",
        type=str,
        default=(
            "Write a long, highly detailed, factual caption of the image."
        ),
    )

    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--min_new_tokens", type=int, default=60)

    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--length_penalty", type=float, default=1.2)
    parser.add_argument(
        "--early_stopping",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop when all beams hit EOS (usually speeds up). Use --no-early_stopping to disable.",
    )

    parser.add_argument("--repetition_penalty", type=float, default=1.1)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)

    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)

    parser.add_argument("--image_size", type=int, default=128)

    parser.add_argument("--regen_image", action="store_true")
    parser.add_argument("--sd_model_id", type=str, default="Manojb/stable-diffusion-2-1-base")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_image", type=str, default="outputs/regenerated.png")
    return parser.parse_args()


def build_transform(image_size: int = 512):
    return transforms.Compose(
        [
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def load_stage2_config(stage2_ckpt: str) -> dict:
    ckpt_dir = os.path.dirname(stage2_ckpt) or "."
    cfg_path = os.path.join(ckpt_dir, "config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = Image.open(args.image_path).convert("RGB")
    image_tensor = build_transform(args.image_size)(image).unsqueeze(0).to(device)

    vision_encoder = VisionToTextEmb().to(device)
    state = torch.load(args.stage1_ckpt, map_location="cpu")
    vision_encoder.load_state_dict(state, strict=False)
    vision_encoder.eval()

    cfg = load_stage2_config(args.stage2_ckpt)
    llm_name = cfg.get("llm_name", "distilgpt2")
    max_length = cfg.get("max_length", 64)

    captioner = EmbeddingCaptioner(model_name=llm_name, max_length=max_length).to(device)
    cap_state = torch.load(args.stage2_ckpt, map_location="cpu")
    captioner.load_state_dict(cap_state, strict=False)
    captioner.eval()

    ckpt_dir = os.path.dirname(args.stage2_ckpt) or "."
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with torch.no_grad():
        prefix = vision_encoder(image_tensor)
        prefix = captioner.prefix_proj(prefix).to(device)
        inputs_embeds = prefix

        prompt = args.prompt.strip()
        if prompt:
            prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            prompt_embeds = captioner.llm.get_input_embeddings()(prompt_ids)
            inputs_embeds = torch.cat([prefix, prompt_embeds], dim=1)

        attention = torch.ones((1, inputs_embeds.shape[1]), device=device, dtype=torch.long)
        dummy_ids = torch.full(
            (1, inputs_embeds.shape[1]),
            tokenizer.pad_token_id,
            device=device,
            dtype=torch.long,
        )

        gen_kwargs = {
            "input_ids": dummy_ids,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention,
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": args.min_new_tokens,
            "num_beams": args.num_beams,
            "length_penalty": args.length_penalty,
            "early_stopping": args.early_stopping,
            "do_sample": args.do_sample,
            "repetition_penalty": args.repetition_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if args.do_sample:
            gen_kwargs.update(
                {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "top_k": args.top_k,
                }
            )

        outputs = captioner.llm.generate(**gen_kwargs)

    generated = outputs[:, inputs_embeds.shape[1] :]
    text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
    print(text)

    if args.regen_image:
        sd_dtype = torch.float16 if device.type == "cuda" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            args.sd_model_id,
            torch_dtype=sd_dtype,
        )
        pipe = pipe.to(device)
        generator = None
        if args.seed >= 0:
            generator = torch.Generator(device=device).manual_seed(args.seed)
        image = pipe(
            prompt=text,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
        ).images[0]
        os.makedirs(os.path.dirname(args.out_image) or ".", exist_ok=True)
        image.save(args.out_image)
        print(f"Saved regenerated image to: {args.out_image}")


if __name__ == "__main__":
    main()
