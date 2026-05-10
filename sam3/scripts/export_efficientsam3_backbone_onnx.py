#!/usr/bin/env python3
"""Export EfficientSAM3 visual backbone path to ONNX.

Input: image [B, 3, H, W]
Output: FPN features + positional encodings after `backbone.forward_image`.
"""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import argparse

import torch
from torch import nn


class _BackboneOnnxWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.backbone = model.backbone

    def forward(self, image: torch.Tensor):
        backbone_out = self.backbone.forward_image(image)
        feats = backbone_out["backbone_fpn"]
        poss = backbone_out["vision_pos_enc"]
        return (*feats, *poss)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export EfficientSAM3 visual backbone.forward_image to ONNX"
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--backbone-type",
        default="tinyvit",
        choices=["efficientvit", "repvit", "tinyvit"],
    )
    parser.add_argument("--model-name", default="21m")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--dynamic-batch", action="store_true")
    parser.add_argument("--img-size", type=int, default=1008)
    return parser.parse_args()


def _check_deps() -> None:
    for pkg in ("onnx", "onnxscript", "einops"):
        __import__(pkg)


def main() -> None:
    args = _parse_args()
    _check_deps()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    from efficientsam.model_builder import build_efficientsam3_image_model

    model = build_efficientsam3_image_model(
        checkpoint_path=ckpt.as_posix(),
        backbone_type=args.backbone_type,
        model_name=args.model_name,
        enable_segmentation=False,
        enable_inst_interactivity=False,
        eval_mode=True,
        compile=False,
        device="cpu",
    )

    wrapper = _BackboneOnnxWrapper(model).eval()
    dummy = torch.randn(1, 3, args.img_size, args.img_size, dtype=torch.float32)

    with torch.inference_mode():
        sample_outputs = wrapper(dummy)
    num_levels = len(sample_outputs) // 2
    output_names = [f"feat_l{i}" for i in range(num_levels)] + [
        f"pos_l{i}" for i in range(num_levels)
    ]

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"image": {0: "batch"}}
        for name in output_names:
            dynamic_axes[name] = {0: "batch"}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        out.as_posix(),
        input_names=["image"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"Exported backbone ONNX: {out}")


if __name__ == "__main__":
    main()
