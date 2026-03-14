#!/usr/bin/env python3
"""Run text-prompt inference on a single image and save the best mask as PNG."""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import argparse

import numpy as np
import torch
from PIL import Image

from efficientsam.model_builder import build_efficientsam3_image_model
from efficientsam.sam3_image_processor import Sam3Processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EfficientSAM3 text-prompt inference on a single image and save mask.png"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pt/.pth)")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", required=True, help="Text prompt (e.g. person)")
    parser.add_argument("--output", default="mask.png", help="Output mask PNG path")
    parser.add_argument(
        "--backbone-type",
        default="tinyvit",
        choices=["efficientvit", "repvit", "tinyvit"],
    )
    parser.add_argument("--model-name", default="21m")
    parser.add_argument(
        "--text-encoder-type",
        default="MobileCLIP-S1",
        help="Student text encoder type. Ignored when --use-teacher-text-encoder is set.",
    )
    parser.add_argument(
        "--use-teacher-text-encoder",
        action="store_true",
        help="Use the original SAM3 text encoder instead of student MobileCLIP.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Inference device",
    )
    parser.add_argument("--compile", action="store_true", help="Enable torch compile")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Mask presence threshold. Lower values keep more candidates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = build_efficientsam3_image_model(
        checkpoint_path=args.checkpoint,
        backbone_type=args.backbone_type,
        model_name=args.model_name,
        text_encoder_type=None if args.use_teacher_text_encoder else args.text_encoder_type,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        eval_mode=True,
        compile=args.compile,
        device=args.device,
    )
    processor = Sam3Processor(
        model,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )

    image = Image.open(args.image).convert("RGB")

    with torch.inference_mode():
        state = processor.set_image(image)
        state = processor.set_text_prompt(prompt=args.prompt, state=state)

    masks = state["masks"]
    scores = state["scores"]
    if scores.numel() == 0:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        empty_mask = np.zeros((image.height, image.width), dtype=np.uint8)
        Image.fromarray(empty_mask).save(out_path)
        print(f"saved empty mask to: {out_path}")
        print("no detections for the given prompt")
        return

    best_idx = int(torch.argmax(scores).item())
    mask = masks[best_idx].detach().to("cpu").numpy()
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    mask_u8 = (mask > 0).astype(np.uint8) * 255

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_u8).save(out_path)

    print(f"saved mask to: {out_path}")
    print(f"best score: {scores[best_idx].item():.4f}")


if __name__ == "__main__":
    main()
