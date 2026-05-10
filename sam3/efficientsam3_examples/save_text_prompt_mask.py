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
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.ops import nms

from efficientsam.model_builder import build_efficientsam3_image_model
from efficientsam.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor as TeacherSam3Processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run EfficientSAM3 text-prompt inference on a single image and save mask.png"
    )
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoint (.pt/.pth)")
    parser.add_argument(
        "--use-sam3-model",
        action="store_true",
        help="Run the full SAM3 teacher model instead of EfficientSAM3.",
    )
    parser.add_argument(
        "--sam3-checkpoint",
        default="/ros2_ws/src/sam3/sam3.pt",
        help="Checkpoint used when --use-sam3-model is set.",
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", required=True, help="Text prompt (e.g. person)")
    parser.add_argument("--output", default="mask.png", help="Output mask PNG path")
    parser.add_argument(
        "--overlay-output",
        default=None,
        help="Output overlay PNG path. Default is <output>_overlay.png",
    )
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
    parser.add_argument(
        "--encoder-feature-downsample",
        type=int,
        default=1,
        help="Average-pool the last encoder feature level by this factor before grounding.",
    )
    parser.add_argument(
        "--adaptive-encoder-feature-downsample",
        action="store_true",
        help="Choose factor 2 or 3 from the last feature-map complexity.",
    )
    parser.add_argument(
        "--adaptive-feature-threshold",
        type=float,
        default=0.27,
        help="If gradient complexity is above this threshold, use factor 2, else factor 3.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["best", "all", "topk_nms"],
        default="topk_nms",
        help="How to select detections for mask / overlay export.",
    )
    parser.add_argument(
        "--max-detections",
        type=int,
        default=8,
        help="Maximum number of detections to keep when selection-mode is all or topk_nms.",
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=0.6,
        help="IoU threshold for box NMS when selection-mode=topk_nms.",
    )
    parser.add_argument(
        "--refine-rois",
        action="store_true",
        help="Refine selected detections on cropped ROIs to improve mask coverage.",
    )
    parser.add_argument(
        "--refine-score-threshold",
        type=float,
        default=0.25,
        help="Refine detections whose score is below this threshold.",
    )
    parser.add_argument(
        "--refine-fill-threshold",
        type=float,
        default=0.5,
        help="Refine detections whose mask fill ratio inside the box is below this threshold.",
    )
    parser.add_argument(
        "--refine-padding-ratio",
        type=float,
        default=0.15,
        help="Expand each ROI by this ratio before crop refinement.",
    )
    parser.add_argument(
        "--refine-downsample-factor",
        type=int,
        default=1,
        help="Feature downsample factor to use inside ROI refinement.",
    )
    parser.add_argument(
        "--max-refine-rois",
        type=int,
        default=2,
        help="Maximum number of low-quality ROIs to refine. Set 0 or less for no limit.",
    )
    parser.add_argument(
        "--geometric-refine-expand-ratio",
        type=float,
        default=0.12,
        help="Expand the geometric prompt box by this ratio before box-guided refinement.",
    )
    parser.add_argument(
        "--roi-refine-method",
        choices=["none", "efficient_ensemble", "sam3_fallback", "geometric_box"],
        default="none",
        help="ROI refinement method for low-confidence detections.",
    )
    parser.add_argument(
        "--roi-refine-prompts",
        default=None,
        help="Comma separated prompts used during ROI refinement.",
    )
    parser.add_argument(
        "--roi-refine-auto-prompts",
        action="store_true",
        help="Use built-in prompt variants for ROI refinement.",
    )
    parser.add_argument(
        "--teacher-sam3-checkpoint",
        default="/ros2_ws/src/sam3/sam3.pt",
        help="Checkpoint for SAM3 ROI fallback.",
    )
    parser.add_argument(
        "--teacher-confidence-threshold",
        type=float,
        default=0.1,
        help="Confidence threshold for SAM3 ROI fallback.",
    )
    return parser.parse_args()


def _maybe_downsample_last_feature_level(
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


def _select_adaptive_downsample_factor(
    backbone_out: dict,
    threshold: float,
) -> tuple[int, float]:
    last_feat = backbone_out["backbone_fpn"][-1]
    grad_x = (last_feat[:, :, :, 1:] - last_feat[:, :, :, :-1]).abs().mean().item()
    grad_y = (last_feat[:, :, 1:, :] - last_feat[:, :, :-1, :]).abs().mean().item()
    complexity = grad_x + grad_y
    factor = 2 if complexity >= threshold else 3
    return factor, complexity


def _select_detection_indices(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    *,
    selection_mode: str,
    max_detections: int,
    nms_iou_threshold: float,
) -> torch.Tensor:
    if scores.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=scores.device)

    sorted_indices = torch.argsort(scores, descending=True)
    if selection_mode == "best":
        return sorted_indices[:1]

    if selection_mode == "all":
        return sorted_indices[:max_detections]

    keep = nms(boxes, scores, nms_iou_threshold)
    keep_scores = scores[keep]
    keep_order = torch.argsort(keep_scores, descending=True)
    return keep[keep_order[:max_detections]]


def _clip_box_to_image(
    box: np.ndarray,
    *,
    width: int,
    height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = [float(v) for v in box.tolist()]
    box_w = max(1.0, x1 - x0)
    box_h = max(1.0, y1 - y0)
    pad_x = box_w * padding_ratio
    pad_y = box_h * padding_ratio
    crop_x0 = max(0, int(np.floor(x0 - pad_x)))
    crop_y0 = max(0, int(np.floor(y0 - pad_y)))
    crop_x1 = min(width, int(np.ceil(x1 + pad_x)))
    crop_y1 = min(height, int(np.ceil(y1 + pad_y)))
    return crop_x0, crop_y0, crop_x1, crop_y1


def _box_xyxy_to_normalized_cxcywh(
    box: np.ndarray,
    *,
    width: int,
    height: int,
    expand_ratio: float = 0.0,
) -> list[float]:
    x0, y0, x1, y1 = [float(v) for v in box.tolist()]
    box_w_px = max(1.0, x1 - x0) * (1.0 + expand_ratio)
    box_h_px = max(1.0, y1 - y0) * (1.0 + expand_ratio)
    cx_px = (x0 + x1) * 0.5
    cy_px = (y0 + y1) * 0.5
    cx_px = min(max(cx_px, box_w_px * 0.5), width - box_w_px * 0.5)
    cy_px = min(max(cy_px, box_h_px * 0.5), height - box_h_px * 0.5)
    cx = cx_px / width
    cy = cy_px / height
    box_w = min(1.0, box_w_px / width)
    box_h = min(1.0, box_h_px / height)
    return [cx, cy, box_w, box_h]


def _compute_mask_box_metrics(
    mask_bool: np.ndarray,
    box: np.ndarray,
    *,
    image_width: int,
    image_height: int,
) -> dict[str, float]:
    x0, y0, x1, y1 = [float(v) for v in box.tolist()]
    box_w = max(1.0, x1 - x0)
    box_h = max(1.0, y1 - y0)
    box_area = box_w * box_h
    mask_area = float(mask_bool.sum())
    image_area = float(image_width * image_height)
    return {
        "fill_ratio": mask_area / box_area,
        "mask_area_ratio": mask_area / image_area,
        "box_area_ratio": box_area / image_area,
        "aspect_ratio": max(box_h / box_w, box_w / box_h),
    }


def _compute_detection_quality(
    *,
    mask_bool: np.ndarray,
    box: np.ndarray,
    score: float,
    image_width: int,
    image_height: int,
    overlap_bonus: float = 0.0,
) -> float:
    metrics = _compute_mask_box_metrics(
        mask_bool,
        box,
        image_width=image_width,
        image_height=image_height,
    )
    fill_ratio = min(metrics["fill_ratio"], 1.0)
    area_ratio = metrics["mask_area_ratio"]
    return (
        0.7 * score
        + 0.2 * fill_ratio
        + 0.1 * min(area_ratio * 40.0, 1.0)
        + 0.1 * overlap_bonus
    )


def _get_prompt_variants(
    prompt: str,
    custom_prompts: str | None,
    use_auto_prompts: bool,
) -> list[str]:
    if custom_prompts:
        prompts = [item.strip() for item in custom_prompts.split(",") if item.strip()]
        if prompts:
            return prompts

    if not use_auto_prompts:
        return [prompt]

    prompt_l = prompt.strip().lower()
    auto_map = {
        "children": ["children", "child", "kid", "kids", "person", "people"],
        "child": ["child", "children", "kid", "person"],
        "kid": ["kid", "kids", "child", "children", "person"],
        "person": ["person", "people", "human", "child"],
        "people": ["people", "person", "children", "human"],
        "object": ["object", "thing", "item"],
        "apple": ["apple", "fruit", "food"],
    }
    return auto_map.get(prompt_l, [prompt])


def _run_grounding_once(
    *,
    model: torch.nn.Module,
    processor: Sam3Processor,
    image: Image.Image,
    prompt: str,
    downsample_factor: int,
) -> dict:
    state = processor.set_image(image)
    state["backbone_out"] = _maybe_downsample_last_feature_level(
        model, state["backbone_out"], downsample_factor
    )
    return processor.set_text_prompt(prompt=prompt, state=state)


def _run_teacher_grounding_once(
    *,
    model: torch.nn.Module,
    processor: TeacherSam3Processor,
    image: Image.Image,
    prompt: str,
) -> dict:
    state = processor.set_image(image)
    return processor.set_text_prompt(prompt=prompt, state=state)


def _refine_selected_masks(
    *,
    model: torch.nn.Module,
    processor: Sam3Processor,
    teacher_model: torch.nn.Module | None,
    teacher_processor: TeacherSam3Processor | None,
    base_state: dict,
    image: Image.Image,
    prompt: str,
    selected_masks: list[np.ndarray],
    selected_boxes: list[np.ndarray],
    selected_scores: list[float],
    refine_score_threshold: float,
    refine_fill_threshold: float,
    refine_padding_ratio: float,
    refine_downsample_factor: int,
    roi_refine_method: str,
    roi_refine_prompts: list[str],
    max_refine_rois: int,
    geometric_refine_expand_ratio: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float], int]:
    refined_masks: list[np.ndarray] = []
    refined_boxes: list[np.ndarray] = []
    refined_scores: list[float] = []
    refined_count = 0

    refine_candidate_indices: list[int] = []
    for index, (mask_bool, box_np, score) in enumerate(
        zip(selected_masks, selected_boxes, selected_scores)
    ):
        metrics = _compute_mask_box_metrics(
            mask_bool,
            box_np,
            image_width=image.width,
            image_height=image.height,
        )
        should_refine = score < refine_score_threshold or metrics["fill_ratio"] < refine_fill_threshold
        if not should_refine:
            continue
        refine_candidate_indices.append(index)

    if max_refine_rois > 0 and len(refine_candidate_indices) > max_refine_rois:
        refine_candidate_indices = sorted(
            refine_candidate_indices,
            key=lambda idx: _compute_detection_quality(
                mask_bool=selected_masks[idx],
                box=selected_boxes[idx],
                score=selected_scores[idx],
                image_width=image.width,
                image_height=image.height,
            ),
        )[:max_refine_rois]
    refine_candidate_set = set(refine_candidate_indices)

    for index, (mask_bool, box_np, score) in enumerate(
        zip(selected_masks, selected_boxes, selected_scores)
    ):
        if index not in refine_candidate_set:
            refined_masks.append(mask_bool)
            refined_boxes.append(box_np)
            refined_scores.append(score)
            continue

        crop_x0, crop_y0, crop_x1, crop_y1 = _clip_box_to_image(
            box_np,
            width=image.width,
            height=image.height,
            padding_ratio=refine_padding_ratio,
        )
        if crop_x1 <= crop_x0 or crop_y1 <= crop_y0:
            refined_masks.append(mask_bool)
            refined_boxes.append(box_np)
            refined_scores.append(score)
            continue

        crop = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))
        original_quality = _compute_detection_quality(
            mask_bool=mask_bool,
            box=box_np,
            score=score,
            image_width=image.width,
            image_height=image.height,
            overlap_bonus=1.0,
        )

        best_refined_mask = None
        best_refined_box = None
        best_refined_score = None
        best_refined_quality = original_quality

        if roi_refine_method == "geometric_box":
            normalized_box = _box_xyxy_to_normalized_cxcywh(
                box_np,
                width=image.width,
                height=image.height,
                expand_ratio=geometric_refine_expand_ratio,
            )
            geometric_state = {
                "original_height": base_state["original_height"],
                "original_width": base_state["original_width"],
                "backbone_out": base_state["backbone_out"],
            }
            with torch.inference_mode():
                geometric_state = processor.add_geometric_prompt(
                    normalized_box,
                    True,
                    geometric_state,
                )
            candidate_scores = geometric_state["scores"]
            candidate_boxes = geometric_state["boxes"]
            candidate_masks = geometric_state["masks"]

            if candidate_scores.numel() > 0:
                original_box_t = torch.as_tensor(box_np, device=candidate_boxes.device).view(1, 4)
                inter_x0 = torch.maximum(candidate_boxes[:, 0], original_box_t[:, 0])
                inter_y0 = torch.maximum(candidate_boxes[:, 1], original_box_t[:, 1])
                inter_x1 = torch.minimum(candidate_boxes[:, 2], original_box_t[:, 2])
                inter_y1 = torch.minimum(candidate_boxes[:, 3], original_box_t[:, 3])
                inter = (inter_x1 - inter_x0).clamp(min=0) * (inter_y1 - inter_y0).clamp(min=0)
                area_a = (candidate_boxes[:, 2] - candidate_boxes[:, 0]).clamp(min=0) * (
                    candidate_boxes[:, 3] - candidate_boxes[:, 1]
                ).clamp(min=0)
                area_b = (original_box_t[:, 2] - original_box_t[:, 0]).clamp(min=0) * (
                    original_box_t[:, 3] - original_box_t[:, 1]
                ).clamp(min=0)
                ious = inter / (area_a + area_b - inter).clamp(min=1e-6)
                best_idx = int(torch.argmax(ious).item())
                refined_mask = candidate_masks[best_idx].detach().cpu().numpy()
                if refined_mask.ndim == 3 and refined_mask.shape[0] == 1:
                    refined_mask = refined_mask[0]
                refined_mask_bool = refined_mask > 0
                refined_score = float(candidate_scores[best_idx].item())
                refined_quality = _compute_detection_quality(
                    mask_bool=refined_mask_bool,
                    box=candidate_boxes[best_idx].detach().cpu().numpy(),
                    score=refined_score,
                    image_width=image.width,
                    image_height=image.height,
                    overlap_bonus=float(ious[best_idx].item()),
                )
                if refined_quality > best_refined_quality:
                    best_refined_mask = refined_mask_bool
                    best_refined_box = candidate_boxes[best_idx].detach().cpu().numpy()
                    best_refined_score = refined_score
                    best_refined_quality = refined_quality
        else:
            for refine_prompt in roi_refine_prompts:
                with torch.inference_mode():
                    if roi_refine_method == "sam3_fallback":
                        crop_state = _run_teacher_grounding_once(
                            model=teacher_model,
                            processor=teacher_processor,
                            image=crop,
                            prompt=refine_prompt,
                        )
                    else:
                        crop_state = _run_grounding_once(
                            model=model,
                            processor=processor,
                            image=crop,
                            prompt=refine_prompt,
                            downsample_factor=refine_downsample_factor,
                        )
                crop_scores = crop_state["scores"]
                if crop_scores.numel() == 0:
                    continue

                crop_best_idx = int(torch.argmax(crop_scores).item())
                crop_mask = crop_state["masks"][crop_best_idx].detach().cpu().numpy()
                if crop_mask.ndim == 3 and crop_mask.shape[0] == 1:
                    crop_mask = crop_mask[0]
                crop_mask_bool = crop_mask > 0
                crop_score = float(crop_scores[crop_best_idx].item())
                crop_quality = _compute_detection_quality(
                    mask_bool=crop_mask_bool,
                    box=crop_state["boxes"][crop_best_idx].detach().cpu().numpy(),
                    score=crop_score,
                    image_width=crop.width,
                    image_height=crop.height,
                    overlap_bonus=1.0,
                )

                if crop_quality <= best_refined_quality:
                    continue

                full_mask = np.zeros((image.height, image.width), dtype=bool)
                full_mask[crop_y0:crop_y1, crop_x0:crop_x1] = crop_mask_bool

                crop_box = crop_state["boxes"][crop_best_idx].detach().cpu().numpy()
                crop_box[[0, 2]] += crop_x0
                crop_box[[1, 3]] += crop_y0

                best_refined_mask = full_mask
                best_refined_box = crop_box
                best_refined_score = crop_score
                best_refined_quality = crop_quality

        if best_refined_mask is None:
            refined_masks.append(mask_bool)
            refined_boxes.append(box_np)
            refined_scores.append(score)
            continue

        refined_masks.append(best_refined_mask)
        refined_boxes.append(best_refined_box)
        refined_scores.append(best_refined_score)
        refined_count += 1

    return refined_masks, refined_boxes, refined_scores, refined_count


def main() -> None:
    args = parse_args()
    if not args.use_sam3_model and not args.checkpoint:
        raise ValueError("--checkpoint is required unless --use-sam3-model is set")

    if args.use_sam3_model:
        model = build_sam3_image_model(
            checkpoint_path=args.sam3_checkpoint,
            load_from_HF=False,
            device=args.device,
            eval_mode=True,
        )
        processor = TeacherSam3Processor(
            model,
            device=args.device,
            confidence_threshold=args.confidence_threshold,
        )
    else:
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
    teacher_model = None
    teacher_processor = None
    if not args.use_sam3_model and args.roi_refine_method == "sam3_fallback":
        teacher_model = build_sam3_image_model(
            checkpoint_path=args.teacher_sam3_checkpoint,
            load_from_HF=False,
            device=args.device,
            eval_mode=True,
        )
        teacher_processor = TeacherSam3Processor(
            teacher_model,
            device=args.device,
            confidence_threshold=args.teacher_confidence_threshold,
        )

    image = Image.open(args.image).convert("RGB")

    with torch.inference_mode():
        state = processor.set_image(image)
        selected_factor = 1
        complexity = None
        if not args.use_sam3_model:
            selected_factor = args.encoder_feature_downsample
            if args.adaptive_encoder_feature_downsample:
                selected_factor, complexity = _select_adaptive_downsample_factor(
                    state["backbone_out"], args.adaptive_feature_threshold
                )
            state["backbone_out"] = _maybe_downsample_last_feature_level(
                model, state["backbone_out"], selected_factor
            )
        state = processor.set_text_prompt(prompt=args.prompt, state=state)
    base_state = None
    if not args.use_sam3_model:
        base_state = {
            "original_height": state["original_height"],
            "original_width": state["original_width"],
            "backbone_out": state["backbone_out"],
        }

    masks = state["masks"]
    scores = state["scores"]
    boxes = state["boxes"]
    out_path = Path(args.output)
    overlay_path = (
        Path(args.overlay_output)
        if args.overlay_output is not None
        else out_path.with_name(f"{out_path.stem}_overlay{out_path.suffix}")
    )
    if scores.numel() == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        empty_mask = np.zeros((image.height, image.width), dtype=np.uint8)
        Image.fromarray(empty_mask).save(out_path)
        image.save(overlay_path)
        print(f"saved empty mask to: {out_path}")
        print(f"saved empty overlay to: {overlay_path}")
        print("no detections for the given prompt")
        return

    selected_indices = _select_detection_indices(
        boxes=boxes,
        scores=scores,
        selection_mode=args.selection_mode,
        max_detections=args.max_detections,
        nms_iou_threshold=args.nms_iou_threshold,
    )

    selected_masks = []
    selected_boxes = []
    selected_scores = []
    roi_refine_prompts = _get_prompt_variants(
        args.prompt,
        args.roi_refine_prompts,
        args.roi_refine_auto_prompts,
    )
    for idx in selected_indices.tolist():
        mask = masks[idx].detach().to("cpu").numpy()
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        selected_masks.append(mask > 0)
        selected_boxes.append(boxes[idx].detach().to("cpu").numpy())
        selected_scores.append(float(scores[idx].item()))

    refined_count = 0
    if not args.use_sam3_model and (args.refine_rois or args.roi_refine_method != "none"):
        (
            selected_masks,
            selected_boxes,
            selected_scores,
            refined_count,
        ) = _refine_selected_masks(
            model=model,
            processor=processor,
            teacher_model=teacher_model,
            teacher_processor=teacher_processor,
            base_state=base_state,
            image=image,
            prompt=args.prompt,
            selected_masks=selected_masks,
            selected_boxes=selected_boxes,
            selected_scores=selected_scores,
            refine_score_threshold=args.refine_score_threshold,
            refine_fill_threshold=args.refine_fill_threshold,
            refine_padding_ratio=args.refine_padding_ratio,
            refine_downsample_factor=args.refine_downsample_factor,
            roi_refine_method=args.roi_refine_method,
            roi_refine_prompts=roi_refine_prompts,
            max_refine_rois=args.max_refine_rois,
            geometric_refine_expand_ratio=args.geometric_refine_expand_ratio,
        )

    merged_mask = np.zeros((image.height, image.width), dtype=bool)
    for mask in selected_masks:
        merged_mask |= mask
    mask_u8 = merged_mask.astype(np.uint8) * 255

    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask_u8).save(out_path)

    overlay_np = np.array(image, copy=True)
    overlay_np[mask_u8 > 0] = (
        0.65 * overlay_np[mask_u8 > 0] + 0.35 * np.array([255, 64, 64], dtype=np.float32)
    ).astype(np.uint8)
    overlay_image = Image.fromarray(overlay_np)
    draw = ImageDraw.Draw(overlay_image)
    for box, score in zip(selected_boxes, selected_scores):
        x0, y0, x1, y1 = [float(v) for v in box.tolist()]
        draw.rectangle((x0, y0, x1, y1), outline=(64, 255, 64), width=3)
        draw.text((x0 + 2, max(0.0, y0 - 14.0)), f"{score:.2f}", fill=(64, 255, 64))
    overlay_image.save(overlay_path)

    print(f"saved mask to: {out_path}")
    print(f"saved overlay to: {overlay_path}")
    print(f"selection mode: {args.selection_mode}")
    print(f"selected detections: {len(selected_boxes)} / total {scores.numel()}")
    if args.refine_rois or args.roi_refine_method != "none":
        print(f"refined detections: {refined_count}")
        print(f"roi refine method: {args.roi_refine_method}")
        print(f"roi refine prompts: {roi_refine_prompts}")
    if selected_scores:
        print(f"top selected score: {max(selected_scores):.4f}")
        print(f"first selected box: {[round(v, 2) for v in selected_boxes[0].tolist()]}")
    if args.adaptive_encoder_feature_downsample:
        print(f"selected downsample factor: {selected_factor}")
        print(f"feature complexity: {complexity:.6f}")


if __name__ == "__main__":
    main()
