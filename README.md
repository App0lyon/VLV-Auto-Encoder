VLV-Auto-Encoder (mini)

Mini reimplementation of the paper (see `2507.07104v2.pdf`) focused on a two-stage
pipeline: (1) distill Stable Diffusion conditioning into a vision encoder, then
(2) train a captioner on top of the learned image-to-text-embedding prefix.

This README explains how to run the code and how the architecture is organized.

Contents
- Overview
- Install
- Data
- Training
- Inference
- Evaluation
- Outputs and checkpoints
- Architecture (code map)

Overview
- Stage 1: learn a vision encoder that maps an image -> a sequence of text-like
  embeddings. The supervision signal comes from Stable Diffusion 2.1: we predict
  the diffusion noise given the image latents and the learned condition.
- Stage 2: learn a captioner that takes the prefix embeddings and generates text
  (teacher forcing over COCO captions). The LLM is frozen by default.

Install
Create env and install deps:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
Data
The training/eval code loads COCO 2017 from local folders (data_dir required).

Local COCO layout (data_dir):
- data_dir/
  - annotations/captions_train2017.json
  - annotations/captions_val2017.json
  - train2017/*.jpg
  - val2017/*.jpg

Notes:
- Stage 1 uses images only (no captions).
- Stage 2 uses image + caption pairs.

Downloading dataset:
```bash
python download_coco2017.py --all --out_dir data/coco2017
```

Training
Stage 1: distill conditioning from Stable Diffusion 2.1
```bash
python train_stage1.py --subset_ratio 1.0 --epochs 3 --batch_size 32 --lr 1e-4 --mixed_precision fp16 --image_size 128 --data_dir data/coco2017/train2017 --sd_model_id Manojb/stable-diffusion-2-1-base 
```

Stage 2: captioner training
```bash
python train_stage2.py --stage1_ckpt .\outputs\stage1\vision_encoder_*.pt --subset_ratio 1.0 --epochs 3 --batch_size 8 --lr 5e-5 --mixed_precision fp16 --image_size 128 --output_dir outputs/stage2 --data_dir data/coco2017/train2017
```

To finetune the vision encoder jointly:
```bash
python train_stage2.py --stage1_ckpt ... --finetune_encoder
```

Inference
Generate a caption for a single image:
```bash
python infer.py --image_path path\to\image.jpg --stage1_ckpt outputs\stage1\vision_encoder_*.pt --stage2_ckpt outputs\stage2\captioner_*.pt --num_beams 5 --length_penalty 1.2 --max_new_tokens 220 --min_new_tokens 70
```

Regenerate an image from the generated caption:
```bash
python infer.py --image_path path\to\image.jpg --stage1_ckpt outputs\stage1\vision_encoder_*.pt --stage2_ckpt outputs\stage2\captioner_*.pt --regen_image --out_image outputs\regenerated.png
```

Evaluation
Quick captioning eval on COCO val (CLIP score or simple BLEU):
```bash
python eval_captioning.py--stage1_ckpt outputs\stage1\vision_encoder_*.pt--stage2_ckpt outputs\stage2\captioner_*.pt--metric clip--num_samples 2000--out_jsonl outputs\coco_val_preds.json
```

Outputs and checkpoints
- Stage 1 writes:
  - outputs/stage1/vision_encoder_{step}.pt
  - outputs/stage1/args.json
- Stage 2 writes:
  - outputs/stage2/captioner_{step}.pt
  - outputs/stage2/config.json
  - outputs/stage2/tokenizer files (from `save_pretrained`)

`config.json` is used at inference time to reload the LLM name and max_length.

Architecture (code map)
Data
- `src/vlv/data/coco.py`: COCO datasets + transforms (local only). Two dataset
  classes:
  - `CocoImagesOnlyDataset` for stage 1 (images only)
  - `CocoImageCaptionDataset` for stage 2 (images + captions)

Models
- `src/vlv/models/vision_to_textemb.py`: vision encoder that maps an image to a
  (B, 77, 1024) prefix. Backbone is ResNet50, then a projection to sequence
  length with optional positional embeddings and LayerNorm.
- `src/vlv/models/frozen_sd21.py`: frozen Stable Diffusion 2.1 VAE+UNet wrapper
  used for stage 1 distillation. Provides encode_images_to_latents and
  add_noise. No gradients in the SD modules.
- `src/vlv/models/emb_to_caption.py`: captioner that takes the prefix embeddings
  and prepends them to token embeddings of an LLM (default distilgpt2). Uses a
  linear prefix projection and standard LM loss. LLM can be frozen, with optional
  LoRA if `peft` is installed.

Training scripts
- `train_stage1.py`: constructs COCO images-only dataset, frozen SD21, and
  trains `VisionToTextEmb` by predicting diffusion noise (MSE) for random
  timesteps. Saves `vision_encoder_*.pt`.
- `train_stage2.py`: loads stage1 checkpoint, builds captioner, and trains on
  COCO captions. Optionally fine-tunes the vision encoder.

Inference and eval
- `infer.py`: image -> prefix -> LLM generate. Loads tokenizer from stage2 output.
- `eval_captioning.py`: runs mini eval on COCO val with CLIP score or BLEU.

Conceptual flow
1) Image -> VisionToTextEmb -> prefix embeddings (77 x 1024).
2) Stage 1: prefix conditions SD21 UNet to predict noise (distillation).
3) Stage 2: prefix projected into LLM embedding space; LLM generates caption.
