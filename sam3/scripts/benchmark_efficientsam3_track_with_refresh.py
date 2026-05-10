#!/usr/bin/env python3
"""EfficientSAM3 検出 + SAM3 tracker の周期補正 benchmark。"""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import argparse

import torch
from PIL import Image

from efficientsam.model_builder import build_efficientsam3_image_model
from efficientsam.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_tracker
from video_text_prompt_benchmark_utils import (
    Timer,
    finalize_summary,
    make_frame_record,
    print_summary,
    sync_cuda,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark EfficientSAM3 detection + SAM3 tracker with periodic refresh"
    )
    parser.add_argument("--input", required=True, help="画像ディレクトリ")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--checkpoint", required=True, help="EfficientSAM3 checkpoint")
    parser.add_argument("--sam3-checkpoint", required=True, help="Full SAM3 checkpoint")
    parser.add_argument(
        "--backbone-type",
        default="repvit",
        choices=["efficientvit", "repvit", "tinyvit"],
    )
    parser.add_argument("--model-name", default="m1.1")
    parser.add_argument("--text-encoder-type", default="MobileCLIP-S1")
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--refresh-interval", type=int, default=4)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-detections", type=int, default=8)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def list_frame_paths(input_dir: str) -> list[Path]:
    path = Path(input_dir)
    if not path.is_dir():
        raise ValueError("--input must be an image directory for this script")
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    frames = sorted(p for p in path.iterdir() if p.suffix.lower() in exts)
    if not frames:
        raise ValueError(f"no image frames found in {input_dir}")
    return frames


def build_detector(args: argparse.Namespace) -> Sam3Processor:
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
    return Sam3Processor(
        model,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )


def build_tracker_from_checkpoint(args: argparse.Namespace):
    tracker = build_tracker(
        apply_temporal_disambiguation=True,
        with_backbone=True,
    )
    ckpt = torch.load(args.sam3_checkpoint, map_location="cpu", weights_only=True)
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        ckpt = ckpt["model"]
    tracker_state = {
        key[len("tracker."):]: value
        for key, value in ckpt.items()
        if key.startswith("tracker.")
    }
    tracker_state.update(
        {
            key.replace("detector.backbone.", "backbone."): value
            for key, value in ckpt.items()
            if key.startswith("detector.backbone.")
        }
    )
    missing_keys, unexpected_keys = tracker.load_state_dict(tracker_state, strict=False)
    if missing_keys:
        print(f"tracker missing keys: {len(missing_keys)}")
    if unexpected_keys:
        print(f"tracker unexpected keys: {len(unexpected_keys)}")
    tracker.to(device=args.device)
    tracker.eval()
    return tracker


def detect_masks(
    processor: Sam3Processor,
    image: Image.Image,
    prompt: str,
    max_detections: int,
) -> tuple[list[torch.Tensor], int, float | None]:
    with torch.inference_mode():
        state = processor.set_image(image)
        state = processor.set_text_prompt(prompt, state)
    scores = state["scores"]
    masks = state["masks"]
    count = int(scores.numel())
    top_score = float(scores.max().item()) if count else None
    if count == 0:
        return [], 0, None

    order = torch.argsort(scores, descending=True)
    if max_detections > 0:
        order = order[:max_detections]
    selected_masks: list[torch.Tensor] = []
    for idx in order.tolist():
        mask = masks[idx].detach()
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        selected_masks.append(mask.to(dtype=torch.float32))
    return selected_masks, count, top_score


def main() -> None:
    args = parse_args()
    if args.refresh_interval <= 0:
        raise ValueError("refresh interval must be positive")

    frame_paths = list_frame_paths(args.input)
    end_frame = len(frame_paths) - 1
    if args.max_frames is not None:
        end_frame = min(args.start_frame + args.max_frames - 1, end_frame)

    detector = build_detector(args)
    tracker = build_tracker_from_checkpoint(args)
    tracker_state = tracker.init_state(video_path=args.input)

    frame_records = []
    refresh_count = 0
    current_frame = args.start_frame

    with Timer() as total_timer:
        while current_frame <= end_frame:
            segment_end = min(current_frame + args.refresh_interval - 1, end_frame)
            tracker.clear_all_points_in_video(tracker_state)

            image = Image.open(frame_paths[current_frame]).convert("RGB")
            with Timer() as detect_timer:
                detected_masks, raw_count, top_score = detect_masks(
                    detector,
                    image,
                    args.prompt,
                    args.max_detections,
                )
                sync_cuda(args.device)

            for obj_id, mask in enumerate(detected_masks):
                tracker.add_new_mask(
                    tracker_state,
                    frame_idx=current_frame,
                    obj_id=obj_id,
                    mask=mask,
                )
            sync_cuda(args.device)

            detect_output = {
                "out_probs": [top_score] * len(detected_masks) if top_score is not None else [],
                "out_binary_masks": [mask.cpu().numpy() > 0.5 for mask in detected_masks],
            }
            record = make_frame_record(
                frame_idx=current_frame,
                elapsed_sec=detect_timer.elapsed,
                output=detect_output if detected_masks else None,
                phase="refresh_detect",
                refresh_index=refresh_count,
            )
            record["raw_detection_count"] = raw_count
            record["used_detection_count"] = len(detected_masks)
            frame_records.append(record)

            track_span = segment_end - current_frame
            if track_span > 0 and detected_masks:
                for item in tracker.propagate_in_video(
                    tracker_state,
                    start_frame_idx=current_frame + 1,
                    max_frame_num_to_track=track_span - 1,
                    reverse=False,
                    propagate_preflight=True,
                ):
                    if len(item) == 5:
                        frame_idx, obj_ids, _low_res_masks, video_res_masks, obj_scores = item
                    else:
                        raise RuntimeError(f"unexpected tracker output length: {len(item)}")
                    sync_cuda(args.device)
                    probs = obj_scores.detach().float().sigmoid().cpu().numpy()
                    output = {
                        "out_probs": probs,
                        "out_binary_masks": video_res_masks.detach().cpu().numpy() > 0,
                    }
                    record = make_frame_record(
                        frame_idx=frame_idx,
                        elapsed_sec=0.0,
                        output=output,
                        phase="track",
                        refresh_index=refresh_count,
                    )
                    record["used_detection_count"] = len(obj_ids)
                    frame_records.append(record)

            refresh_count += 1
            current_frame = segment_end + 1

    summary = finalize_summary(
        mode="efficientsam3_detect_then_sam3_track_with_refresh",
        prompt=args.prompt,
        input_path=args.input,
        frame_records=frame_records,
        total_elapsed_sec=total_timer.elapsed,
        extra={
            "refresh_count": refresh_count,
            "refresh_interval": args.refresh_interval,
            "detector_checkpoint": args.checkpoint,
            "tracker_checkpoint": args.sam3_checkpoint,
            "max_detections": args.max_detections,
        },
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
