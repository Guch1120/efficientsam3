#!/usr/bin/env python3
"""Compare text-prompt results between SAM3 and EfficientSAM3 on one image."""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import argparse

import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_efficientsam3_image_model, build_sam3_image_model


def _summarize(name: str, scores) -> None:
    count = int(scores.numel())
    if count == 0:
        print(f"{name}: detections=0 top_score=None")
        return
    top_score = float(scores.max().item())
    print(f"{name}: detections={count} top_score={top_score:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare SAM3 and EfficientSAM3 text-prompt outputs on one image"
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument(
        "--eff-checkpoint",
        required=True,
        help="EfficientSAM3 merged checkpoint",
    )
    parser.add_argument(
        "--sam3-checkpoint",
        default=None,
        help="Optional original SAM3 checkpoint. If omitted, the builder default is used.",
    )
    parser.add_argument("--backbone-type", default="tinyvit", choices=["efficientvit", "repvit", "tinyvit"])
    parser.add_argument("--model-name", default="21m")
    parser.add_argument("--text-encoder-type", default="MobileCLIP-S1")
    parser.add_argument("--confidence-threshold", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Image.open(args.image).convert("RGB")

    sam3_model = build_sam3_image_model(
        checkpoint_path=args.sam3_checkpoint,
        load_from_HF=args.sam3_checkpoint is None,
        device=args.device,
    )
    sam3_proc = Sam3Processor(
        sam3_model,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )

    eff_student_model = build_efficientsam3_image_model(
        checkpoint_path=args.eff_checkpoint,
        backbone_type=args.backbone_type,
        model_name=args.model_name,
        text_encoder_type=args.text_encoder_type,
        device=args.device,
    )
    eff_student_proc = Sam3Processor(
        eff_student_model,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )

    eff_teacher_text_model = build_efficientsam3_image_model(
        checkpoint_path=args.eff_checkpoint,
        backbone_type=args.backbone_type,
        model_name=args.model_name,
        text_encoder_type=None,
        device=args.device,
    )
    eff_teacher_text_proc = Sam3Processor(
        eff_teacher_text_model,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )

    with torch.inference_mode():
        sam3_state = sam3_proc.set_image(image)
        sam3_state = sam3_proc.set_text_prompt(args.prompt, sam3_state)

        eff_student_state = eff_student_proc.set_image(image)
        eff_student_state = eff_student_proc.set_text_prompt(args.prompt, eff_student_state)

        eff_teacher_state = eff_teacher_text_proc.set_image(image)
        eff_teacher_state = eff_teacher_text_proc.set_text_prompt(args.prompt, eff_teacher_state)

    print(f"prompt={args.prompt} threshold={args.confidence_threshold}")
    _summarize("sam3", sam3_state["scores"])
    _summarize("efficientsam3_student_text", eff_student_state["scores"])
    _summarize("efficientsam3_teacher_text", eff_teacher_state["scores"])


if __name__ == "__main__":
    main()
