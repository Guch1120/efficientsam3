#!/usr/bin/env python3
"""Export EfficientSAM3 grounding core to ONNX.

The exported graph starts after prompt encoding.
Inputs:
  - feat_l{i}, pos_l{i}: backbone FPN features and positional encodings
  - prompt, prompt_mask: output of `_encode_prompt`
Outputs:
  - pred_masks, pred_logits, presence_logit_dec, pred_boxes
"""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import argparse

import torch
from torch import nn


class _GroundingCoreWrapper(nn.Module):
    def __init__(self, model: nn.Module, num_levels: int):
        super().__init__()
        self.model = model
        self.num_levels = num_levels

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

    def forward(self, *inputs: torch.Tensor):
        feature_inputs = list(inputs[: self.num_levels])
        pos_inputs = list(inputs[self.num_levels : self.num_levels * 2])
        prompt = inputs[self.num_levels * 2]
        prompt_mask = inputs[self.num_levels * 2 + 1]

        backbone_out = {
            "vision_features": feature_inputs[-1],
            "vision_pos_enc": pos_inputs,
            "backbone_fpn": feature_inputs,
            "sam2_backbone_out": None,
        }

        backbone_out, encoder_out, _ = self.model._run_encoder(
            backbone_out,
            self.find_stage,
            prompt,
            prompt_mask,
        )
        out = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
            "prev_encoder_out": {
                "encoder_out": encoder_out,
                "backbone_out": backbone_out,
            },
        }
        out, hs = self.model._run_decoder(
            memory=out["encoder_hidden_states"],
            pos_embed=encoder_out["pos_embed"],
            src_mask=encoder_out["padding_mask"],
            out=out,
            prompt=prompt,
            prompt_mask=prompt_mask,
            encoder_out=encoder_out,
        )
        self.model._run_segmentation_heads(
            out=out,
            backbone_out=backbone_out,
            img_ids=self.find_stage.img_ids,
            vis_feat_sizes=encoder_out["vis_feat_sizes"],
            encoder_hidden_states=out["encoder_hidden_states"],
            prompt=prompt,
            prompt_mask=prompt_mask,
            hs=hs,
        )
        return (
            out["pred_masks"],
            out["pred_logits"],
            out["presence_logit_dec"],
            out["pred_boxes"],
        )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export EfficientSAM3 grounding core to ONNX")
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
    p.add_argument("--resolution", type=int, default=1008)
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

    from PIL import Image
    from torchvision.transforms import v2

    from efficientsam.model_builder import build_efficientsam3_image_model
    from efficientsam.sam3_image_processor import Sam3Processor

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
    processor = Sam3Processor(
        model,
        resolution=args.resolution,
        device="cpu",
        confidence_threshold=0.05,
    )

    dummy_image = Image.new("RGB", (args.resolution, args.resolution), color=0)
    image_tensor = v2.functional.to_image(dummy_image)
    image_np = processor.transform(image_tensor).unsqueeze(0).to(torch.float32)
    backbone_out = model.backbone.forward_image(image_np)
    backbone_out.update(model.backbone.forward_text([args.text_prompt], device="cpu"))
    dummy_prompt = model._get_dummy_prompt()
    prompt, prompt_mask, _ = model._encode_prompt(
        backbone_out,
        processor.find_stage,
        dummy_prompt,
    )

    features = list(backbone_out["backbone_fpn"])
    pos = list(backbone_out["vision_pos_enc"])
    num_levels = len(features)
    wrapper = _GroundingCoreWrapper(model, num_levels=num_levels).eval()
    dummy_inputs = tuple(features + pos + [prompt, prompt_mask])

    input_names = [f"feat_l{i}" for i in range(num_levels)] + [
        f"pos_l{i}" for i in range(num_levels)
    ] + ["prompt", "prompt_mask"]
    dynamic_axes = None
    if args.dynamic_batch:
        dynamic_axes = {}
        for name in input_names[:-2]:
            dynamic_axes[name] = {0: "batch"}
        dynamic_axes["prompt"] = {1: "batch"}
        dynamic_axes["prompt_mask"] = {0: "batch"}
        dynamic_axes["pred_masks"] = {0: "batch"}
        dynamic_axes["pred_logits"] = {0: "batch"}
        dynamic_axes["presence_logit_dec"] = {0: "batch"}
        dynamic_axes["pred_boxes"] = {0: "batch"}

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        out.as_posix(),
        input_names=input_names,
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
    print(f"Exported grounding core ONNX: {out}")


if __name__ == "__main__":
    main()
