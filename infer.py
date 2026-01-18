import argparse
import json
import os

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from src.vlv.models.emb_to_caption import EmbeddingCaptioner
from src.vlv.models.vision_to_textemb import VisionToTextEmb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference: image -> caption.")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage2_ckpt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--num_beams", type=int, default=1)
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
    image_tensor = build_transform(512)(image).unsqueeze(0).to(device)

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
        attention = torch.ones((1, prefix.shape[1]), device=device, dtype=torch.long)
        dummy_ids = torch.full(
            (1, prefix.shape[1]),
            tokenizer.pad_token_id,
            device=device,
            dtype=torch.long,
        )
        outputs = captioner.llm.generate(
            input_ids=dummy_ids,
            inputs_embeds=prefix,
            attention_mask=attention,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[:, prefix.shape[1] :]
    text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
    print(text)


if __name__ == "__main__":
    main()
