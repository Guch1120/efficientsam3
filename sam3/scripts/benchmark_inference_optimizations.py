#!/usr/bin/env python3
"""Benchmark EfficientSAM3 image encoder latency and memory footprint.

Compares eager/compile/mixed-precision/channels-last options on the visual trunk
and can run a preset profile for 8GB / 16GB VRAM planning.
"""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import argparse
import time
from dataclasses import dataclass

import torch
from torch import nn



@dataclass(frozen=True)
class ProfileConfig:
    name: str
    compile: bool
    amp: bool
    channels_last: bool


def _get_visual_trunk(model: nn.Module) -> nn.Module:
    """Return student visual trunk across legacy/current backbone field names."""
    backbone = getattr(model, "backbone", None)
    if backbone is None:
        raise AttributeError("model has no `backbone` attribute")

    vision_backbone = getattr(backbone, "vision_backbone", None)
    if vision_backbone is None:
        vision_backbone = getattr(backbone, "visual", None)  # legacy name fallback
    if vision_backbone is None:
        raise AttributeError(
            "Could not find vision backbone on model.backbone (expected `vision_backbone` or `visual`)."
        )

    trunk = getattr(vision_backbone, "trunk", None)
    if trunk is None:
        raise AttributeError("vision backbone has no `trunk` attribute")
    return trunk


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark EfficientSAM3 visual trunk")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--backbone-type",
        default="efficientvit",
        choices=["efficientvit", "repvit", "tinyvit"],
    )
    parser.add_argument("--model-name", default="b0")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--img-size", type=int, default=1008)
    parser.add_argument("--batch-size", type=int, default=1)

    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use autocast mixed precision")

    parser.add_argument(
        "--preset",
        choices=["single", "vram8", "vram16", "all"],
        default="single",
        help="single: run only flags above; vram8/vram16/all: run built-in config matrix",
    )
    parser.add_argument(
        "--vram-budget-gb",
        type=float,
        default=None,
        help="If set on CUDA, report whether each config fits this VRAM budget",
    )
    return parser.parse_args()


def _validate_checkpoint_path(checkpoint: str) -> Path:
    ckpt = Path(checkpoint).expanduser()
    if str(ckpt).startswith('/path/to/'):
        raise FileNotFoundError(
            f"Checkpoint path looks like a README placeholder: {checkpoint}. "
            "Please pass a real .pt/.pth path."
        )
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}. "
            "Please verify --checkpoint points to an existing file."
        )
    return ckpt


def _check_runtime_dependencies() -> None:
    # Keep this lightweight and explicit for common failure in mixed ROS/venv envs.
    __import__("einops")


def _run_once(args: argparse.Namespace, cfg: ProfileConfig, device: torch.device) -> tuple[float, float | None]:
    checkpoint_path = _validate_checkpoint_path(args.checkpoint)

    _check_runtime_dependencies()
    from efficientsam.model_builder import build_efficientsam3_image_model

    model = build_efficientsam3_image_model(
        checkpoint_path=checkpoint_path.as_posix(),
        backbone_type=args.backbone_type,
        model_name=args.model_name,
        enable_segmentation=False,
        enable_inst_interactivity=False,
        eval_mode=True,
        compile=cfg.compile,
        device=device,
    )
    encoder = _get_visual_trunk(model).eval()

    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
    if cfg.channels_last:
        encoder = encoder.to(memory_format=torch.channels_last)
        x = x.to(memory_format=torch.channels_last)

    amp_enabled = cfg.amp and device.type in {"cuda", "mps"}
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                _ = encoder(x)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                _ = encoder(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

    avg_ms = (elapsed / args.iters) * 1000.0
    peak_mb = None
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    return avg_ms, peak_mb


def _profile_configs(args: argparse.Namespace) -> list[ProfileConfig]:
    if args.preset == "single":
        return [
            ProfileConfig(
                name="single",
                compile=args.compile,
                amp=args.amp,
                channels_last=args.channels_last,
            )
        ]

    all_cfgs = [
        ProfileConfig("eager_fp32", compile=False, amp=False, channels_last=False),
        ProfileConfig("eager_amp", compile=False, amp=True, channels_last=False),
        ProfileConfig("eager_amp_cl", compile=False, amp=True, channels_last=True),
        ProfileConfig("compile_amp_cl", compile=True, amp=True, channels_last=True),
    ]

    if args.preset == "all":
        return all_cfgs

    if args.preset == "vram8":
        return [c for c in all_cfgs if c.amp]

    # vram16
    return all_cfgs


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    configs = _profile_configs(args)

    print(
        f"device={device} backbone={args.backbone_type}/{args.model_name} img={args.img_size} "
        f"bs={args.batch_size} preset={args.preset}"
    )

    print("name,compile,amp,channels_last,avg_latency_ms,peak_memory_mb,fits_budget")
    for cfg in configs:
        avg_ms, peak_mb = _run_once(args, cfg, device)

        fits_budget = "N/A"
        if peak_mb is not None and args.vram_budget_gb is not None:
            fits_budget = "yes" if peak_mb <= args.vram_budget_gb * 1024.0 else "no"

        peak_str = f"{peak_mb:.1f}" if peak_mb is not None else "N/A"
        print(
            f"{cfg.name},{cfg.compile},{cfg.amp},{cfg.channels_last},"
            f"{avg_ms:.3f},{peak_str},{fits_budget}"
        )


if __name__ == "__main__":
    main()
