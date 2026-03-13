#!/usr/bin/env python3
"""Export EfficientSAM3 neck-decoder path to ONNX.

Input: image_embed [B, 1024, 72, 72] (from encoder ONNX output)
Output: FPN features + positional encodings (4 levels each).
"""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import argparse

import torch
from torch import nn


class _NeckDecoderWrapper(nn.Module):
    def __init__(self, vision_backbone: nn.Module):
        super().__init__()
        self.convs = vision_backbone.convs
        self.position_encoding = vision_backbone.position_encoding

    def forward(self, image_embed: torch.Tensor):
        outs: list[torch.Tensor] = []
        poss: list[torch.Tensor] = []
        for conv in self.convs:
            feat = conv(image_embed)
            pos = self.position_encoding(feat).to(feat.dtype)
            outs.append(feat)
            poss.append(pos)
        # 8 outputs: feat_l0..l3, pos_l0..l3
        return (*outs, *poss)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export EfficientSAM3 neck decoder to ONNX")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--backbone-type", default="tinyvit", choices=["efficientvit", "repvit", "tinyvit"])
    p.add_argument("--model-name", default="21m")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--dynamic-batch", action="store_true")
    return p.parse_args()


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

    vb = model.backbone.vision_backbone
    decoder = _NeckDecoderWrapper(vb).eval()

    dummy = torch.randn(1, 1024, 72, 72, dtype=torch.float32)
    output_names = [
        "feat_l0", "feat_l1", "feat_l2", "feat_l3",
        "pos_l0", "pos_l1", "pos_l2", "pos_l3",
    ]
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {"image_embed": {0: "batch"}}
        for n in output_names:
            dynamic_axes[n] = {0: "batch"}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        decoder,
        dummy,
        out.as_posix(),
        input_names=["image_embed"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"Exported decoder ONNX: {out}")


if __name__ == "__main__":
    main()
