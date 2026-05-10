#!/usr/bin/env python3
"""
Text prompt finetune checkpoint の学習済み trunk / neck を merged EfficientSAM3 checkpoint に戻す。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert text-prompt finetune checkpoint into merged EfficientSAM3 checkpoint"
    )
    parser.add_argument("--finetune-ckpt", required=True, help="Path to pilot finetune checkpoint")
    parser.add_argument("--pretrained", required=True, help="Path to merged EfficientSAM3 checkpoint")
    parser.add_argument(
        "--output",
        default=None,
        help="Output path. Defaults to <pretrained>_textprompt.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pretrained_path = Path(args.pretrained)
    output_path = (
        Path(args.output)
        if args.output is not None
        else pretrained_path.with_suffix("").with_name(pretrained_path.stem + "_textprompt.pt")
    )

    finetune_ckpt = torch.load(args.finetune_ckpt, map_location="cpu")
    finetune_sd = finetune_ckpt.get("model", finetune_ckpt)
    if "student_trunk" in finetune_sd:
        student_trunk_sd = finetune_sd["student_trunk"]
    else:
        student_trunk_sd = finetune_sd
    student_neck_sd = finetune_sd.get("student_neck", {})

    pretrained_ckpt = torch.load(args.pretrained, map_location="cpu")
    wrap_in_model = isinstance(pretrained_ckpt, dict) and "model" in pretrained_ckpt
    pretrained_sd = pretrained_ckpt["model"] if wrap_in_model else pretrained_ckpt

    replacements: dict[str, torch.Tensor] = {}
    for key, value in student_trunk_sd.items():
        if key.startswith("model."):
            replacements[f"detector.backbone.vision_backbone.trunk.{key}"] = value
        else:
            replacements[f"detector.backbone.vision_backbone.trunk.model.{key}"] = value
    neck_prefix = "detector.backbone.vision_backbone.convs."
    for key, value in student_neck_sd.items():
        replacements[f"{neck_prefix}{key}"] = value

    replaced = 0
    for key, value in replacements.items():
        if key in pretrained_sd and pretrained_sd[key].shape == value.shape:
            pretrained_sd[key] = value
            replaced += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if wrap_in_model:
        torch.save({"model": pretrained_sd}, output_path)
    else:
        torch.save(pretrained_sd, output_path)

    print(f"replaced_weights={replaced}")
    print(f"output_path={output_path}")


if __name__ == "__main__":
    main()
