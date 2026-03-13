#!/usr/bin/env python3
"""Export text-conditioned downstream path (decoder+mask head) to ONNX.

This export bakes a fixed text prompt into the graph and exposes an image input.
Input image is expected to be preprocessed float32 tensor [B, 3, 1008, 1008].
Outputs raw `pred_masks` and `pred_logits` from `forward_grounding`.
"""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import argparse

import torch
from torch import nn


class _TextGroundingWrapper(nn.Module):
    def __init__(self, model: nn.Module, text_prompt: str):
        super().__init__()
        self.model = model
        self.text_prompt = text_prompt

        text_out = self.model.backbone.forward_text([text_prompt], device="cpu")
        # register as buffers for ONNX trace/export reproducibility
        self.register_buffer("language_features", text_out["language_features"], persistent=False)
        self.register_buffer("language_mask", text_out["language_mask"], persistent=False)

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

    def forward(self, image: torch.Tensor):
        backbone_out = self.model.backbone.forward_image(image)
        backbone_out["language_features"] = self.language_features
        backbone_out["language_mask"] = self.language_mask

        out = self.model.forward_grounding(
            backbone_out=backbone_out,
            find_input=self.find_stage,
            find_target=None,
            geometric_prompt=self.geometric_prompt,
        )
        return out["pred_masks"], out["pred_logits"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export text-conditioned EfficientSAM3 path to ONNX")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--text-prompt", required=True, help="e.g. person")
    p.add_argument("--backbone-type", default="tinyvit", choices=["efficientvit", "repvit", "tinyvit"])
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

    wrapper = _TextGroundingWrapper(model, args.text_prompt).eval()
    dummy = torch.randn(1, 3, 1008, 1008, dtype=torch.float32)

    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {
            "image": {0: "batch"},
            "pred_masks": {0: "batch"},
            "pred_logits": {0: "batch"},
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        out.as_posix(),
        input_names=["image"],
        output_names=["pred_masks", "pred_logits"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=True,
    )
    print(f"Exported text-conditioned ONNX: {out}")


if __name__ == "__main__":
    main()
