#!/usr/bin/env python3
"""Benchmark EfficientSAM3 image encoder latency and memory footprint.

Compares eager/compile/mixed-precision/channels-last options on the visual trunk.
"""

from __future__ import annotations

import argparse
import time

import torch

from sam3.model_builder import build_efficientsam3_image_model


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark EfficientSAM3 visual trunk")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone-type", default="efficientvit", choices=["efficientvit", "repvit", "tinyvit"])
    parser.add_argument("--model-name", default="b0")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--img-size", type=int, default=1008)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use autocast mixed precision")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    model = build_efficientsam3_image_model(
        checkpoint_path=args.checkpoint,
        backbone_type=args.backbone_type,
        model_name=args.model_name,
        enable_segmentation=False,
        enable_inst_interactivity=False,
        eval_mode=True,
        compile=args.compile,
        device=device,
    )
    encoder = model.backbone.visual.trunk.eval()

    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
    if args.channels_last:
        encoder = encoder.to(memory_format=torch.channels_last)
        x = x.to(memory_format=torch.channels_last)

    amp_enabled = args.amp and device.type in {"cuda", "mps"}
    amp_dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                _ = encoder(x)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        for _ in range(args.iters):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                _ = encoder(x)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

    avg_ms = (elapsed / args.iters) * 1000.0
    print(f"avg_latency_ms={avg_ms:.3f}")

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"peak_memory_mb={peak_mb:.1f}")
    else:
        print("peak_memory_mb=N/A (CUDA only)")


if __name__ == "__main__":
    main()
