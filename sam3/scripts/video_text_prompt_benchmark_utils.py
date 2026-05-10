#!/usr/bin/env python3
"""SAM3 動画推論の benchmark 共通処理。"""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import argparse
import json
import time
from typing import Any

import numpy as np
import torch

from sam3.model_builder import build_sam3_video_model


def add_common_video_args(
    parser: argparse.ArgumentParser,
    *,
    include_refresh: bool = False,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--input",
        required=True,
        help="画像ディレクトリ・単画像・動画のいずれか",
    )
    parser.add_argument("--prompt", required=True, help="text prompt")
    parser.add_argument("--sam3-checkpoint", default=None, help="Path to sam3.pt")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--offload-video-to-cpu", action="store_true")
    parser.add_argument("--async-loading-frames", action="store_true")
    if include_refresh:
        parser.add_argument(
            "--refresh-interval",
            type=int,
            default=8,
            help="このフレーム間隔ごとに text prompt で再初期化する",
        )
    return parser


def build_video_model(args: argparse.Namespace):
    return build_sam3_video_model(
        checkpoint_path=args.sam3_checkpoint,
        load_from_HF=args.sam3_checkpoint is None,
        device=args.device,
        compile=args.compile,
    )


def init_video_state(model, args: argparse.Namespace) -> dict[str, Any]:
    state = model.init_state(
        resource_path=args.input,
        offload_video_to_cpu=args.offload_video_to_cpu,
        async_loading_frames=args.async_loading_frames,
    )
    total_frames = state["num_frames"]
    if args.start_frame < 0 or args.start_frame >= total_frames:
        raise ValueError(f"start frame is out of range: {args.start_frame} / {total_frames}")
    return state


def get_end_frame(state: dict[str, Any], args: argparse.Namespace) -> int:
    total_frames = state["num_frames"]
    if args.max_frames is None:
        return total_frames - 1
    return min(args.start_frame + args.max_frames - 1, total_frames - 1)


def sync_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize_output(output: dict[str, Any] | None) -> dict[str, Any]:
    if output is None:
        return {
            "num_objects": 0,
            "top_score": None,
            "nonzero_pixels": 0,
        }

    probs = np.asarray(output["out_probs"])
    masks = np.asarray(output["out_binary_masks"])
    num_objects = int(probs.shape[0])
    top_score = float(probs.max()) if num_objects > 0 else None
    nonzero_pixels = int(masks.sum()) if masks.size > 0 else 0
    return {
        "num_objects": num_objects,
        "top_score": top_score,
        "nonzero_pixels": nonzero_pixels,
    }


def make_frame_record(
    *,
    frame_idx: int,
    elapsed_sec: float,
    output: dict[str, Any] | None,
    phase: str,
    refresh_index: int | None = None,
) -> dict[str, Any]:
    record = summarize_output(output)
    record.update(
        {
            "frame_idx": int(frame_idx),
            "elapsed_ms": elapsed_sec * 1000.0,
            "phase": phase,
        }
    )
    if refresh_index is not None:
        record["refresh_index"] = int(refresh_index)
    return record


def finalize_summary(
    *,
    mode: str,
    prompt: str,
    input_path: str,
    frame_records: list[dict[str, Any]],
    total_elapsed_sec: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    processed_frames = len(frame_records)
    nonzero_frames = sum(record["nonzero_pixels"] > 0 for record in frame_records)
    detections = [record["num_objects"] for record in frame_records]
    top_scores = [
        record["top_score"] for record in frame_records if record["top_score"] is not None
    ]
    summary: dict[str, Any] = {
        "mode": mode,
        "prompt": prompt,
        "input": input_path,
        "processed_frames": processed_frames,
        "total_elapsed_ms": total_elapsed_sec * 1000.0,
        "avg_latency_ms": (total_elapsed_sec / processed_frames * 1000.0)
        if processed_frames
        else None,
        "throughput_fps": (processed_frames / total_elapsed_sec)
        if total_elapsed_sec > 0 and processed_frames > 0
        else None,
        "avg_num_objects": (sum(detections) / processed_frames) if processed_frames else 0.0,
        "max_num_objects": max(detections) if detections else 0,
        "nonzero_frames": nonzero_frames,
        "top_score_max": max(top_scores) if top_scores else None,
        "top_score_mean": (sum(top_scores) / len(top_scores)) if top_scores else None,
        "frames": frame_records,
    }
    if extra:
        summary.update(extra)
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2))


class Timer:
    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.end = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return self.end - self.start
