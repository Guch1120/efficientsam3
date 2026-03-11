#!/usr/bin/env python3
"""Export EfficientSAM3 student image encoder to ONNX.

This exports the distilled visual trunk (student backbone + projection head)
which is typically the dominant compute block for image inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from sam3.model_builder import build_efficientsam3_image_model


class _EncoderOnnxWrapper(nn.Module):
    """Wrap EfficientSAM3 visual trunk and expose a tensor output."""

    def __init__(self, visual_trunk: nn.Module):
        super().__init__()
        self.visual_trunk = visual_trunk

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.visual_trunk(image)
        if isinstance(feats, (list, tuple)):
            return feats[0]
        return feats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export EfficientSAM3 student image encoder to ONNX"
    )
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument(
        "--output",
        required=True,
        help="Output ONNX file path (e.g. efficientsam3_encoder.onnx)",
    )
    parser.add_argument(
        "--backbone-type",
        default="efficientvit",
        choices=["efficientvit", "repvit", "tinyvit"],
    )
    parser.add_argument(
        "--model-name",
        default="b0",
        help="Backbone variant (e.g. b0, m1.1, 5m)",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="Enable dynamic batch axis for ONNX input/output",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=1008,
        help="Input square image size expected by SAM3 visual trunk",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_efficientsam3_image_model(
        checkpoint_path=args.checkpoint,
        backbone_type=args.backbone_type,
        model_name=args.model_name,
        enable_segmentation=False,
        enable_inst_interactivity=False,
        eval_mode=True,
        compile=False,
        device="cpu",
    )

    encoder = _EncoderOnnxWrapper(model.backbone.visual.trunk).eval()
    dummy = torch.randn(1, 3, args.img_size, args.img_size, dtype=torch.float32)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "image": {0: "batch"},
            "image_embed": {0: "batch"},
        }

    torch.onnx.export(
        encoder,
        dummy,
        output_path.as_posix(),
        input_names=["image"],
        output_names=["image_embed"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"Exported ONNX model: {output_path}")


if __name__ == "__main__":
    main()
