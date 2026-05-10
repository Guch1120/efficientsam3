#!/usr/bin/env python3
"""Compare baseline / finetuned EfficientSAM3 checkpoints against SAM3 teacher."""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import subprocess

from PIL import Image
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate text-prompt checkpoints with the single-image mask script"
    )
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--finetuned-checkpoint", required=True)
    parser.add_argument(
        "--teacher-checkpoint",
        default=None,
        help="Optional full SAM3 teacher checkpoint used as the reference target.",
    )
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument("--prompt", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone-type", default="repvit", choices=["efficientvit", "repvit", "tinyvit"])
    parser.add_argument("--model-name", default="m1.1")
    parser.add_argument("--text-encoder-type", default="MobileCLIP-S1")
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder-feature-downsample", type=int, default=2)
    parser.add_argument("--selection-mode", default="topk_nms", choices=["best", "all", "topk_nms"])
    parser.add_argument("--max-detections", type=int, default=8)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.6)
    parser.add_argument("--roi-refine-method", default="geometric_box")
    parser.add_argument("--refine-score-threshold", type=float, default=0.25)
    parser.add_argument("--refine-fill-threshold", type=float, default=0.55)
    parser.add_argument("--geometric-refine-expand-ratio", type=float, default=0.18)
    parser.add_argument("--max-refine-rois", type=int, default=8)
    return parser.parse_args()


def run_case(
    *,
    checkpoint: str | None,
    image_path: str,
    prompt: str,
    output_dir: Path,
    prefix: str,
    args: argparse.Namespace,
    use_sam3_model: bool = False,
) -> dict[str, object]:
    output_mask = output_dir / f"{prefix}_mask.png"
    output_overlay = output_dir / f"{prefix}_overlay.png"
    cmd = [
        "python3",
        "sam3/efficientsam3_examples/save_text_prompt_mask.py",
        "--image",
        image_path,
        "--prompt",
        prompt,
        "--output",
        str(output_mask),
        "--overlay-output",
        str(output_overlay),
        "--confidence-threshold",
        str(args.confidence_threshold),
        "--device",
        args.device,
    ]
    if use_sam3_model:
        if checkpoint is None:
            raise ValueError("teacher checkpoint is required when use_sam3_model=True")
        cmd.extend(
            [
                "--use-sam3-model",
                "--sam3-checkpoint",
                checkpoint,
                "--selection-mode",
                args.selection_mode,
                "--max-detections",
                str(args.max_detections),
                "--nms-iou-threshold",
                str(args.nms_iou_threshold),
            ]
        )
    else:
        if checkpoint is None:
            raise ValueError("checkpoint is required for EfficientSAM3 evaluation")
        cmd.extend(
            [
                "--checkpoint",
                checkpoint,
                "--backbone-type",
                args.backbone_type,
                "--model-name",
                args.model_name,
                "--text-encoder-type",
                args.text_encoder_type,
                "--encoder-feature-downsample",
                str(args.encoder_feature_downsample),
                "--selection-mode",
                args.selection_mode,
                "--max-detections",
                str(args.max_detections),
                "--nms-iou-threshold",
                str(args.nms_iou_threshold),
                "--roi-refine-method",
                args.roi_refine_method,
                "--refine-rois",
                "--refine-score-threshold",
                str(args.refine_score_threshold),
                "--refine-fill-threshold",
                str(args.refine_fill_threshold),
                "--geometric-refine-expand-ratio",
                str(args.geometric_refine_expand_ratio),
                "--max-refine-rois",
                str(args.max_refine_rois),
            ]
        )
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)

    mask = np.array(Image.open(output_mask).convert("L")) > 0
    return {
        "mask_path": str(output_mask),
        "overlay_path": str(output_overlay),
        "mask_pixels": int(mask.sum()),
        "stdout": proc.stdout,
    }


def _compute_mask_iou(mask_a_path: str, mask_b_path: str) -> float:
    mask_a = np.array(Image.open(mask_a_path).convert("L")) > 0
    mask_b = np.array(Image.open(mask_b_path).convert("L")) > 0
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 1.0
    intersection = np.logical_and(mask_a, mask_b).sum()
    return float(intersection / union)


def main() -> None:
    args = parse_args()
    if len(args.image) != len(args.prompt):
        raise ValueError("--image and --prompt must have the same length")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []

    for index, (image_path, prompt) in enumerate(zip(args.image, args.prompt)):
        case_name = f"case{index}_{Path(image_path).stem}_{prompt}"
        teacher = None
        if args.teacher_checkpoint is not None:
            teacher = run_case(
                checkpoint=args.teacher_checkpoint,
                image_path=image_path,
                prompt=prompt,
                output_dir=output_dir,
                prefix=f"{case_name}_teacher",
                args=args,
                use_sam3_model=True,
            )
        baseline = run_case(
            checkpoint=args.baseline_checkpoint,
            image_path=image_path,
            prompt=prompt,
            output_dir=output_dir,
            prefix=f"{case_name}_baseline",
            args=args,
        )
        finetuned = run_case(
            checkpoint=args.finetuned_checkpoint,
            image_path=image_path,
            prompt=prompt,
            output_dir=output_dir,
            prefix=f"{case_name}_finetuned",
            args=args,
        )
        if teacher is not None:
            baseline["iou_to_teacher"] = _compute_mask_iou(
                baseline["mask_path"], teacher["mask_path"]
            )
            finetuned["iou_to_teacher"] = _compute_mask_iou(
                finetuned["mask_path"], teacher["mask_path"]
            )
        results.append(
            {
                "case": case_name,
                "image": image_path,
                "prompt": prompt,
                "teacher": teacher,
                "baseline": baseline,
                "finetuned": finetuned,
            }
        )

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
