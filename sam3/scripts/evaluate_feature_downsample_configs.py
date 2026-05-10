#!/usr/bin/env python3
"""Compare encoder feature downsample factors across multiple image/prompt cases."""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import argparse
import statistics
import time

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from efficientsam.model_builder import build_efficientsam3_image_model
from efficientsam.sam3_image_processor import Sam3Processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate feature downsample stability across cases"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder-onnx", required=True)
    parser.add_argument("--backbone-type", default="repvit", choices=["efficientvit", "repvit", "tinyvit"])
    parser.add_argument("--model-name", default="m1.1")
    parser.add_argument("--text-encoder-type", default="MobileCLIP-S1")
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument(
        "--factor",
        type=int,
        action="append",
        default=[],
        help="Feature downsample factor. Repeatable. Default: 1 and 2.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Case in the form image_path::prompt . Repeatable.",
    )
    parser.add_argument(
        "--adaptive",
        action="store_true",
        help="Also evaluate heuristic adaptive factor selection between 2 and 3.",
    )
    parser.add_argument(
        "--adaptive-feature-threshold",
        type=float,
        default=0.27,
        help="If complexity is above this threshold, use factor 2, else factor 3.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path.cwd() / "feature_downsample_results"),
        help="Directory to save mask and overlay images.",
    )
    return parser.parse_args()


def get_ort_providers() -> list[str | tuple[str, dict[str, object]]]:
    available = ort.get_available_providers()
    trt_provider = (
        "TensorrtExecutionProvider",
        {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": "/tmp/ort_trt_cache",
        },
    )
    if "TensorrtExecutionProvider" in available:
        return [trt_provider, "CUDAExecutionProvider", "CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def maybe_downsample_last_feature_level(
    model: torch.nn.Module,
    backbone_out: dict,
    factor: int,
) -> dict:
    if factor <= 1:
        return backbone_out

    features = list(backbone_out["backbone_fpn"])
    last_feat = features[-1]
    if last_feat.shape[-1] % factor != 0 or last_feat.shape[-2] % factor != 0:
        raise ValueError(
            f"Last feature shape {tuple(last_feat.shape[-2:])} is not divisible by factor {factor}"
        )

    down_feat = F.avg_pool2d(last_feat, kernel_size=factor, stride=factor)
    down_pos = model.backbone.vision_backbone.position_encoding(down_feat).to(down_feat.dtype)
    features[-1] = down_feat

    pos_enc = list(backbone_out["vision_pos_enc"])
    pos_enc[-1] = down_pos

    return {
        **backbone_out,
        "vision_features": down_feat,
        "vision_pos_enc": pos_enc,
        "backbone_fpn": features,
    }


def prepare_image_input(processor: Sam3Processor, image: Image.Image) -> tuple[np.ndarray, int, int]:
    image_np = np.array(image, copy=True)
    image_tensor = processor.transform(torch.from_numpy(image_np).permute(2, 0, 1))
    image_np = image_tensor.unsqueeze(0).numpy().astype(np.float32)
    width, height = image.size
    return image_np, width, height


def compute_iou(mask_a: torch.Tensor | None, mask_b: torch.Tensor | None) -> float | None:
    if mask_a is None or mask_b is None:
        return None
    inter = torch.logical_and(mask_a > 0, mask_b > 0).sum().item()
    union = torch.logical_or(mask_a > 0, mask_b > 0).sum().item()
    if union == 0:
        return 1.0
    return inter / union


def select_adaptive_factor(backbone_out: dict, threshold: float) -> tuple[int, float]:
    last_feat = backbone_out["backbone_fpn"][-1]
    grad_x = (last_feat[:, :, :, 1:] - last_feat[:, :, :, :-1]).abs().mean().item()
    grad_y = (last_feat[:, :, 1:, :] - last_feat[:, :, :-1, :]).abs().mean().item()
    complexity = grad_x + grad_y
    factor = 2 if complexity >= threshold else 3
    return factor, complexity


def sanitize_label(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)


def save_result_images(
    *,
    image: Image.Image,
    mask: torch.Tensor | None,
    box: np.ndarray | None,
    output_dir: Path,
    stem: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_np = np.array(image, copy=True)
    if mask is None:
        mask_u8 = np.zeros((image.height, image.width), dtype=np.uint8)
    else:
        mask_np = mask.numpy()
        if mask_np.ndim == 3 and mask_np.shape[0] == 1:
            mask_np = mask_np[0]
        mask_u8 = (mask_np > 0).astype(np.uint8) * 255
    overlay = image_np.copy()
    overlay[mask_u8 > 0] = (
        0.65 * overlay[mask_u8 > 0] + 0.35 * np.array([255, 64, 64], dtype=np.float32)
    ).astype(np.uint8)
    Image.fromarray(mask_u8).save(output_dir / f"{stem}_mask.png")
    overlay_image = Image.fromarray(overlay)
    if box is not None:
        draw = ImageDraw.Draw(overlay_image)
        x0, y0, x1, y1 = [float(v) for v in box.tolist()]
        draw.rectangle((x0, y0, x1, y1), outline=(64, 255, 64), width=3)
    overlay_image.save(output_dir / f"{stem}_overlay.png")


def run_case(
    *,
    model: torch.nn.Module,
    processor: Sam3Processor,
    session: ort.InferenceSession,
    input_name: str,
    image: Image.Image,
    prompt: str,
    factor: int,
    runs: int,
    warmup_runs: int,
    cached_text_outputs: dict[str, dict[str, torch.Tensor]],
    adaptive_threshold: float | None = None,
) -> dict[str, object]:
    image_np, width, height = prepare_image_input(processor, image)
    timings: list[float] = []
    final_state = None

    total_runs = warmup_runs + runs
    for idx in range(total_runs):
        start = time.perf_counter()
        image_embed = session.run(None, {input_name: image_np})[0]
        image_embed_t = torch.from_numpy(image_embed).to(processor.device)

        vb = model.backbone.vision_backbone
        active_convs = list(vb.convs)
        scalp = int(getattr(model.backbone, "scalp", 0))
        if scalp > 0:
            active_convs = active_convs[:-scalp]

        sam3_features: list[torch.Tensor] = []
        sam3_pos: list[torch.Tensor] = []
        for conv in active_convs:
            feat = conv(image_embed_t)
            pos = vb.position_encoding(feat).to(feat.dtype)
            sam3_features.append(feat)
            sam3_pos.append(pos)

        state = {
            "original_height": height,
            "original_width": width,
            "backbone_out": {
                "vision_features": sam3_features[-1],
                "vision_pos_enc": sam3_pos,
                "backbone_fpn": sam3_features,
                "sam2_backbone_out": None,
            },
        }
        effective_factor = factor
        complexity = None
        if adaptive_threshold is not None:
            effective_factor, complexity = select_adaptive_factor(
                state["backbone_out"], adaptive_threshold
            )
        state["backbone_out"] = maybe_downsample_last_feature_level(
            model, state["backbone_out"], effective_factor
        )

        with torch.inference_mode():
            state["backbone_out"].update(cached_text_outputs[prompt])
            state["geometric_prompt"] = model._get_dummy_prompt()
            state = processor._forward_grounding(state)

        if processor.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        if idx >= warmup_runs:
            timings.append(elapsed)
        final_state = state

    scores = final_state["scores"]
    if scores.numel() == 0:
        best_mask = None
        best_box = None
        top_score = None
    else:
        best_idx = int(torch.argmax(scores).item())
        best_mask = final_state["masks"][best_idx].detach().cpu()
        best_box = final_state["boxes"][best_idx].detach().cpu().numpy()
        top_score = float(scores.max().item())

    return {
        "avg_ms": statistics.mean(timings) * 1000.0,
        "std_ms": statistics.pstdev(timings) * 1000.0 if len(timings) > 1 else 0.0,
        "fps": len(timings) / sum(timings),
        "detections": int(scores.numel()),
        "top_score": top_score,
        "best_mask": best_mask,
        "best_box": best_box,
        "selected_factor": effective_factor,
        "complexity": complexity,
    }


def main() -> None:
    args = parse_args()
    factors = args.factor or [1, 2]
    cases = args.case or [
        "/ros2_ws/efficientsam3/test_image.jpg::children",
        "/ros2_ws/efficientsam3/groceries.jpg::object",
        "/ros2_ws/efficientsam3/groceries.jpg::apple",
    ]

    output_dir = Path(args.output_dir)

    model = build_efficientsam3_image_model(
        checkpoint_path=args.checkpoint,
        backbone_type=args.backbone_type,
        model_name=args.model_name,
        text_encoder_type=args.text_encoder_type,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        eval_mode=True,
        device=args.device,
    )
    processor = Sam3Processor(
        model,
        resolution=args.resolution,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )

    session = ort.InferenceSession(args.encoder_onnx, providers=get_ort_providers())
    input_name = session.get_inputs()[0].name

    prompts = sorted({case.split("::", 1)[1] for case in cases})
    cached_text_outputs = {}
    with torch.inference_mode():
        for prompt in prompts:
            cached_text_outputs[prompt] = model.backbone.forward_text([prompt], device=args.device)

    for case in cases:
        image_path, prompt = case.split("::", 1)
        image = Image.open(image_path).convert("RGB")
        print(f"CASE image={Path(image_path).name} prompt={prompt}")
        baseline = None
        for factor in factors:
            result = run_case(
                model=model,
                processor=processor,
                session=session,
                input_name=input_name,
                image=image,
                prompt=prompt,
                factor=factor,
                runs=args.runs,
                warmup_runs=args.warmup_runs,
                cached_text_outputs=cached_text_outputs,
            )
            if factor == 1:
                baseline = result
                mask_iou = None
            else:
                mask_iou = compute_iou(baseline["best_mask"], result["best_mask"])
            top_score = result["top_score"]
            top_score_str = "None" if top_score is None else f"{top_score:.4f}"
            iou_str = "-" if mask_iou is None else f"{mask_iou:.4f}"
            print(
                f"  factor={factor} avg_ms={result['avg_ms']:.2f} std_ms={result['std_ms']:.2f} "
                f"fps={result['fps']:.2f} detections={result['detections']} "
                f"top_score={top_score_str} best_mask_iou_vs_base={iou_str}"
            )
            stem = f"{sanitize_label(Path(image_path).stem)}__{sanitize_label(prompt)}__factor{factor}"
            save_result_images(
                image=image,
                mask=result["best_mask"],
                box=result["best_box"],
                output_dir=output_dir,
                stem=stem,
            )
        if args.adaptive:
            adaptive = run_case(
                model=model,
                processor=processor,
                session=session,
                input_name=input_name,
                image=image,
                prompt=prompt,
                factor=1,
                runs=args.runs,
                warmup_runs=args.warmup_runs,
                cached_text_outputs=cached_text_outputs,
                adaptive_threshold=args.adaptive_feature_threshold,
            )
            adaptive_iou = compute_iou(baseline["best_mask"], adaptive["best_mask"])
            top_score = adaptive["top_score"]
            top_score_str = "None" if top_score is None else f"{top_score:.4f}"
            iou_str = "-" if adaptive_iou is None else f"{adaptive_iou:.4f}"
            print(
                f"  adaptive factor={adaptive['selected_factor']} complexity={adaptive['complexity']:.4f} "
                f"avg_ms={adaptive['avg_ms']:.2f} std_ms={adaptive['std_ms']:.2f} "
                f"fps={adaptive['fps']:.2f} detections={adaptive['detections']} "
                f"top_score={top_score_str} best_mask_iou_vs_base={iou_str}"
            )
            stem = f"{sanitize_label(Path(image_path).stem)}__{sanitize_label(prompt)}__adaptive"
            save_result_images(
                image=image,
                mask=adaptive["best_mask"],
                box=adaptive["best_box"],
                output_dir=output_dir,
                stem=stem,
            )


if __name__ == "__main__":
    main()
