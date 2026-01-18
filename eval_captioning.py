import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, CLIPModel, CLIPProcessor

from src.vlv.models.emb_to_caption import EmbeddingCaptioner
from src.vlv.models.vision_to_textemb import VisionToTextEmb

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mini captioning evaluation on COCO val.")
    parser.add_argument("--stage1_ckpt", type=str, required=True)
    parser.add_argument("--stage2_ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=2000)
    parser.add_argument("--subset_ratio", type=float, default=1.0)
    parser.add_argument("--out_jsonl", type=str, default="outputs/coco_val_preds.jsonl")
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", type=str, choices=["clip", "bleu"], default="clip")
    parser.add_argument("--image_size", type=int, default=512)
    return parser.parse_args()


def build_transform(image_size: int) -> transforms.Compose:
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


def load_samples(
    data_dir: Optional[str],
    subset_ratio: float,
    num_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    samples: List[Dict[str, Any]] = []

    if load_dataset is not None and data_dir is None:
        ds = load_dataset("coco_captions", "2017", split="validation")
        indices = list(range(len(ds)))
        rng.shuffle(indices)
        if subset_ratio < 1.0:
            indices = indices[: max(1, int(len(indices) * subset_ratio))]
        if num_samples > 0:
            indices = indices[: num_samples]
        for i in indices:
            row = ds[i]
            samples.append(
                {
                    "image": row["image"],
                    "caption": row.get("caption", ""),
                    "image_id": row.get("image_id", i),
                }
            )
        return samples

    if data_dir is None:
        raise RuntimeError("data_dir is required when datasets is unavailable.")

    ann_path = os.path.join(data_dir, "annotations", "captions_val2017.json")
    img_dir = os.path.join(data_dir, "val2017")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"COCO annotations not found at {ann_path}")
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"COCO images folder not found at {img_dir}")

    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)
    id_to_file = {img["id"]: img["file_name"] for img in ann.get("images", [])}

    indices = list(range(len(ann.get("annotations", []))))
    rng.shuffle(indices)
    if subset_ratio < 1.0:
        indices = indices[: max(1, int(len(indices) * subset_ratio))]
    if num_samples > 0:
        indices = indices[: num_samples]

    for idx in indices:
        a = ann["annotations"][idx]
        file_name = id_to_file.get(a["image_id"])
        if not file_name:
            continue
        samples.append(
            {
                "image": os.path.join(img_dir, file_name),
                "caption": a.get("caption", ""),
                "image_id": a["image_id"],
            }
        )
    return samples


def simple_bleu(pred: str, ref: str) -> float:
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()
    if not pred_tokens:
        return 0.0
    pred_counts = {}
    for t in pred_tokens:
        pred_counts[t] = pred_counts.get(t, 0) + 1
    ref_counts = {}
    for t in ref_tokens:
        ref_counts[t] = ref_counts.get(t, 0) + 1
    overlap = sum(min(pred_counts[t], ref_counts.get(t, 0)) for t in pred_counts)
    precision = overlap / max(1, len(pred_tokens))
    bp = 1.0 if len(pred_tokens) >= len(ref_tokens) else torch.exp(
        torch.tensor(1 - len(ref_tokens) / max(1, len(pred_tokens)))
    ).item()
    return precision * bp


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = load_samples(args.data_dir, args.subset_ratio, args.num_samples, args.seed)
    transform = build_transform(args.image_size)

    vision_encoder = VisionToTextEmb().to(device)
    vision_encoder.load_state_dict(torch.load(args.stage1_ckpt, map_location="cpu"), strict=False)
    vision_encoder.eval()

    cfg = load_stage2_config(args.stage2_ckpt)
    llm_name = cfg.get("llm_name", "distilgpt2")
    max_length = cfg.get("max_length", 64)

    captioner = EmbeddingCaptioner(model_name=llm_name, max_length=max_length).to(device)
    captioner.load_state_dict(torch.load(args.stage2_ckpt, map_location="cpu"), strict=False)
    captioner.eval()

    ckpt_dir = os.path.dirname(args.stage2_ckpt) or "."
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    clip_model = None
    clip_processor = None
    if args.metric == "clip":
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    out_f = open(args.out_jsonl, "w", encoding="utf-8")

    total_score = 0.0
    count = 0

    batch_size = 8
    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        images = []
        captions = []
        image_ids = []
        for item in batch:
            img = item["image"]
            if isinstance(img, Image.Image):
                pil = img.convert("RGB")
            else:
                pil = Image.open(img).convert("RGB")
            images.append(pil)
            captions.append(item["caption"])
            image_ids.append(item["image_id"])

        image_tensor = torch.stack([transform(im) for im in images]).to(device)
        with torch.no_grad():
            prefix = vision_encoder(image_tensor)
            prefix = captioner.prefix_proj(prefix).to(device)
            attention = torch.ones((prefix.shape[0], prefix.shape[1]), device=device, dtype=torch.long)
            dummy_ids = torch.full(
                (prefix.shape[0], prefix.shape[1]),
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
            pred_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            pred_texts = [t.strip() for t in pred_texts]

        if args.metric == "clip":
            assert clip_model is not None and clip_processor is not None
            inputs = clip_processor(text=pred_texts, images=images, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                image_features = clip_model.get_image_features(**inputs)
                text_features = clip_model.get_text_features(**inputs)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            scores = (image_features * text_features).sum(dim=-1).detach().cpu().tolist()
        else:
            scores = [simple_bleu(p, r) for p, r in zip(pred_texts, captions)]

        for image_id, pred, gt, score in zip(image_ids, pred_texts, captions, scores):
            out_f.write(json.dumps({"image_id": image_id, "pred_caption": pred, "gt_caption": gt}) + "\n")
            total_score += score
            count += 1

    out_f.close()
    if count > 0:
        print(f"{args.metric}_score={total_score / count:.4f}")


if __name__ == "__main__":
    main()
