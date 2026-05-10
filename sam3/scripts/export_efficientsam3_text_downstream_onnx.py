#!/usr/bin/env python3
"""Export fixed-text downstream path to ONNX.

Input: image_embed [B, 1024, 72, 72] from encoder ONNX.
Output: pred_masks, pred_logits, presence_logit_dec, pred_boxes.
"""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import argparse

import torch
from torch import nn


class _TextDownstreamWrapper(nn.Module):
    def __init__(self, model: nn.Module, text_prompt: str):
        super().__init__()
        self.model = model
        self.text_prompt = text_prompt

        text_out = self.model.backbone.forward_text([text_prompt], device="cpu")
        self.register_buffer(
            "language_features",
            text_out["language_features"].detach(),
            persistent=False,
        )
        self.register_buffer(
            "language_mask",
            text_out["language_mask"].detach(),
            persistent=False,
        )

        from sam3.model.data_misc import FindStage

        self.find_stage = FindStage(
            img_ids=torch.tensor([0], dtype=torch.long),
            text_ids=torch.tensor([0], dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        self.geometric_prompt = self.model._get_dummy_prompt()

        vision_backbone = self.model.backbone.vision_backbone
        convs = list(vision_backbone.convs)
        scalp = int(getattr(self.model.backbone, "scalp", 0))
        if scalp > 0:
            convs = convs[:-scalp]
        self.convs = nn.ModuleList(convs)
        self.position_encoding = vision_backbone.position_encoding

    def forward(self, image_embed: torch.Tensor):
        sam3_features: list[torch.Tensor] = []
        sam3_pos: list[torch.Tensor] = []
        for conv in self.convs:
            feat = conv(image_embed)
            pos = self.position_encoding(feat).to(feat.dtype)
            sam3_features.append(feat)
            sam3_pos.append(pos)

        backbone_out = {
            "vision_features": sam3_features[-1],
            "vision_pos_enc": sam3_pos,
            "backbone_fpn": sam3_features,
            "sam2_backbone_out": None,
            "language_features": self.language_features,
            "language_mask": self.language_mask,
        }

        out = self.model.forward_grounding(
            backbone_out=backbone_out,
            find_input=self.find_stage,
            find_target=None,
            geometric_prompt=self.geometric_prompt,
        )
        return (
            out["pred_masks"],
            out["pred_logits"],
            out["presence_logit_dec"],
            out["pred_boxes"],
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export fixed-text downstream ONNX")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--text-prompt", required=True)
    p.add_argument(
        "--backbone-type",
        default="tinyvit",
        choices=["efficientvit", "repvit", "tinyvit"],
    )
    p.add_argument("--model-name", default="21m")
    p.add_argument("--text-encoder-type", default="MobileCLIP-S1")
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
        text_encoder_type=args.text_encoder_type,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        eval_mode=True,
        compile=False,
        device="cpu",
    )

    wrapper = _TextDownstreamWrapper(model, args.text_prompt).eval()
    dummy = torch.randn(1, 1024, 72, 72, dtype=torch.float32)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "image_embed": {0: "batch"},
            "pred_masks": {0: "batch"},
            "pred_logits": {0: "batch"},
            "presence_logit_dec": {0: "batch"},
            "pred_boxes": {0: "batch"},
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        out.as_posix(),
        input_names=["image_embed"],
        output_names=[
            "pred_masks",
            "pred_logits",
            "presence_logit_dec",
            "pred_boxes",
        ],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"Exported text-downstream ONNX: {out}")


if __name__ == "__main__":
    main()
