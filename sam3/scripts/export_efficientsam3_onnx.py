#!/usr/bin/env python3
"""Export EfficientSAM3 student image encoder to ONNX.

This exports the distilled visual trunk (student backbone + projection head)
which is typically the dominant compute block for image inference.
"""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import argparse

import torch
from torch import nn

from efficientsam.model_builder import build_efficientsam3_image_model


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




def _infer_arch_from_checkpoint_name(checkpoint: Path) -> tuple[str, str] | None:
    name = checkpoint.name.lower()
    if "efficientvit" in name:
        if "_b0" in name or "_s" in name:
            return ("efficientvit", "b0")
        if "_b1" in name or "_m" in name:
            return ("efficientvit", "b1")
        if "_b2" in name or "_l" in name:
            return ("efficientvit", "b2")
    if "tinyvit" in name:
        if "_5m" in name:
            return ("tinyvit", "5m")
        if "_11m" in name or "_m_" in name:
            return ("tinyvit", "11m")
        if "_21m" in name or "_l_" in name:
            return ("tinyvit", "21m")
    if "repvit" in name:
        if "m0.9" in name or "_s" in name:
            return ("repvit", "m0.9")
        if "m1.1" in name or "_m" in name:
            return ("repvit", "m1.1")
        if "m2.3" in name or "_l" in name:
            return ("repvit", "m2.3")
    return None


def _validate_arch_args(checkpoint: Path, backbone_type: str, model_name: str) -> None:
    inferred = _infer_arch_from_checkpoint_name(checkpoint)
    if inferred is None:
        return
    inf_backbone, inf_model = inferred
    if (inf_backbone, inf_model) != (backbone_type, model_name):
        raise ValueError(
            "Checkpoint/argument architecture mismatch: "
            f"checkpoint '{checkpoint.name}' looks like ({inf_backbone}, {inf_model}) "
            f"but got --backbone-type {backbone_type} --model-name {model_name}. "
            "Use matching backbone/model flags."
        )

def _check_onnx_export_dependencies() -> None:
    missing: list[str] = []
    for pkg in ("onnx", "onnxscript"):
        try:
            __import__(pkg)
        except ModuleNotFoundError:
            missing.append(pkg)
    if missing:
        pkgs = " ".join(missing)
        raise ModuleNotFoundError(
            "Missing ONNX export dependencies: "
            f"{', '.join(missing)}. "
            "Install them with: "
            f"pip install {pkgs}"
        )


def main() -> None:
    args = _parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_path = _validate_checkpoint_path(args.checkpoint)
    _validate_arch_args(checkpoint_path, args.backbone_type, args.model_name)

    _check_onnx_export_dependencies()

    model = build_efficientsam3_image_model(
        checkpoint_path=checkpoint_path.as_posix(),
        backbone_type=args.backbone_type,
        model_name=args.model_name,
        enable_segmentation=False,
        enable_inst_interactivity=False,
        eval_mode=True,
        compile=False,
        device="cpu",
    )

    encoder = _EncoderOnnxWrapper(_get_visual_trunk(model)).eval()
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
