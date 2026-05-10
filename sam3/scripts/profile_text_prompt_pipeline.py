#!/usr/bin/env python3
"""Text prompt 推論パイプラインを段階ごとに計測する。"""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import argparse
import statistics
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

from efficientsam.model_builder import build_efficientsam3_image_model
from efficientsam.sam3_image_processor import Sam3Processor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile text-prompt inference pipeline")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder-onnx", default=None)
    parser.add_argument(
        "--backend",
        choices=["pytorch", "onnx_local"],
        default="onnx_local",
    )
    parser.add_argument(
        "--backbone-type",
        default="repvit",
        choices=["efficientvit", "repvit", "tinyvit"],
    )
    parser.add_argument("--model-name", default="m1.1")
    parser.add_argument("--text-encoder-type", default="MobileCLIP-S1")
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--cache-text-prompt", action="store_true")
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-encoder-layers", type=int, default=None)
    parser.add_argument("--max-decoder-layers", type=int, default=None)
    parser.add_argument("--encoder-feature-downsample", type=int, default=1)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def sync_cuda(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


class StageTimer:
    def __init__(self, metrics: dict[str, list[float]], name: str, device: str) -> None:
        self.metrics = metrics
        self.name = name
        self.device = device

    def __enter__(self):
        sync_cuda(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        sync_cuda(self.device)
        elapsed = time.perf_counter() - self.start
        self.metrics.setdefault(self.name, []).append(elapsed)


def build_model_and_processor(args: argparse.Namespace):
    model = build_efficientsam3_image_model(
        checkpoint_path=args.checkpoint,
        backbone_type=args.backbone_type,
        model_name=args.model_name,
        text_encoder_type=args.text_encoder_type,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        eval_mode=True,
        device=args.device,
    )
    apply_runtime_limits(
        model,
        max_queries=args.max_queries,
        max_encoder_layers=args.max_encoder_layers,
        max_decoder_layers=args.max_decoder_layers,
    )
    processor = Sam3Processor(
        model,
        resolution=args.resolution,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )
    return model, processor


def apply_runtime_limits(
    model: torch.nn.Module,
    *,
    max_queries: int | None,
    max_encoder_layers: int | None,
    max_decoder_layers: int | None,
) -> None:
    decoder = model.transformer.decoder
    if max_queries and max_queries < decoder.num_queries:
        query_weight = decoder.query_embed.weight[:max_queries].detach().clone()
        ref_weight = decoder.reference_points.weight[:max_queries].detach().clone()
        decoder.query_embed = nn.Embedding.from_pretrained(query_weight, freeze=True)
        decoder.reference_points = nn.Embedding.from_pretrained(ref_weight, freeze=True)
        decoder.num_queries = max_queries

    encoder = model.transformer.encoder
    if max_encoder_layers and max_encoder_layers < encoder.num_layers:
        encoder.layers = nn.ModuleList(list(encoder.layers[:max_encoder_layers]))
        encoder.num_layers = max_encoder_layers

    if max_decoder_layers and max_decoder_layers < decoder.num_layers:
        decoder.layers = nn.ModuleList(list(decoder.layers[:max_decoder_layers]))
        if isinstance(decoder.fine_layers, nn.ModuleList):
            decoder.fine_layers = nn.ModuleList(list(decoder.fine_layers[:max_decoder_layers]))
        else:
            decoder.fine_layers = list(decoder.fine_layers[:max_decoder_layers])
        decoder.num_layers = max_decoder_layers


def maybe_downsample_last_feature_level(
    model: torch.nn.Module,
    backbone_out: dict,
    factor: int,
) -> dict:
    if factor <= 1:
        return backbone_out

    features = list(backbone_out["backbone_fpn"])
    last_feat = features[-1]
    if last_feat.shape[-1] % factor != 0 or last_feat.shape[-2] % factor != 0:
        raise ValueError(
            f"Last feature shape {tuple(last_feat.shape[-2:])} is not divisible by factor {factor}"
        )

    down_feat = F.avg_pool2d(last_feat, kernel_size=factor, stride=factor)
    down_pos = model.backbone.vision_backbone.position_encoding(down_feat).to(down_feat.dtype)
    features[-1] = down_feat

    pos_enc = list(backbone_out["vision_pos_enc"])
    pos_enc[-1] = down_pos

    return {
        **backbone_out,
        "vision_features": down_feat,
        "vision_pos_enc": pos_enc,
        "backbone_fpn": features,
    }


def prepare_image_tensor(processor: Sam3Processor, image: Image.Image) -> np.ndarray:
    image_tensor = v2.functional.to_image(image)
    return processor.transform(image_tensor).unsqueeze(0).numpy().astype(np.float32)


def run_profile(args: argparse.Namespace) -> dict[str, list[float]]:
    image = Image.open(args.image).convert("RGB")
    model, processor = build_model_and_processor(args)
    image_np = prepare_image_tensor(processor, image)
    image_torch = torch.from_numpy(image_np).to(args.device)

    ort_sess = None
    ort_inp_name = None
    active_convs = list(model.backbone.vision_backbone.convs)
    scalp = int(getattr(model.backbone, "scalp", 0))
    if scalp > 0:
        active_convs = active_convs[:-scalp]
    if args.backend == "onnx_local":
        import onnxruntime as ort

        if not args.encoder_onnx:
            raise ValueError("--encoder-onnx is required for onnx_local profiling")
        ort_sess = ort.InferenceSession(
            args.encoder_onnx,
            providers=[
                (
                    "TensorrtExecutionProvider",
                    {
                        "trt_fp16_enable": True,
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": "/tmp/ort_trt_cache",
                    },
                ),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
        )
        ort_inp_name = ort_sess.get_inputs()[0].name

    cached_text_outputs = None
    if args.cache_text_prompt:
        with torch.inference_mode():
            cached_text_outputs = model.backbone.forward_text([args.prompt], device=args.device)

    metrics: dict[str, list[float]] = {}
    total_runs = args.warmup_runs + args.runs
    query_counts: list[int] = []

    for idx in range(total_runs):
        current_metrics: dict[str, list[float]] = {}

        with StageTimer(current_metrics, "total", args.device):
            if args.backend == "pytorch":
                with StageTimer(current_metrics, "image_encoder", args.device):
                    backbone_out = model.backbone.forward_image(image_torch)
                    backbone_out = maybe_downsample_last_feature_level(
                        model, backbone_out, args.encoder_feature_downsample
                    )
            else:
                with StageTimer(current_metrics, "encoder_onnx", args.device):
                    image_embed = ort_sess.run(None, {ort_inp_name: image_np})[0]
                image_embed_t = torch.from_numpy(image_embed).to(args.device)
                sam3_features = []
                sam3_pos = []
                with StageTimer(current_metrics, "neck_decoder", args.device):
                    for conv in active_convs:
                        feat = conv(image_embed_t)
                        pos = model.backbone.vision_backbone.position_encoding(feat).to(feat.dtype)
                        sam3_features.append(feat)
                        sam3_pos.append(pos)
                backbone_out = {
                    "vision_features": sam3_features[-1],
                    "vision_pos_enc": sam3_pos,
                    "backbone_fpn": sam3_features,
                    "sam2_backbone_out": None,
                }
                backbone_out = maybe_downsample_last_feature_level(
                    model, backbone_out, args.encoder_feature_downsample
                )

            with StageTimer(current_metrics, "text_encode", args.device):
                if cached_text_outputs is None:
                    text_outputs = model.backbone.forward_text([args.prompt], device=args.device)
                else:
                    text_outputs = cached_text_outputs
            backbone_out = {**backbone_out, **text_outputs}

            state = {
                "original_height": image.height,
                "original_width": image.width,
                "backbone_out": backbone_out,
                "geometric_prompt": model._get_dummy_prompt(),
            }

            with StageTimer(current_metrics, "prompt_encode", args.device):
                prompt, prompt_mask, state["backbone_out"] = model._encode_prompt(
                    state["backbone_out"],
                    processor.find_stage,
                    state["geometric_prompt"],
                )

            with StageTimer(current_metrics, "transformer_encoder", args.device):
                state["backbone_out"], encoder_out, _ = model._run_encoder(
                    state["backbone_out"],
                    processor.find_stage,
                    prompt,
                    prompt_mask,
                )

            out = {
                "encoder_hidden_states": encoder_out["encoder_hidden_states"],
                "prev_encoder_out": {
                    "encoder_out": encoder_out,
                    "backbone_out": state["backbone_out"],
                },
            }

            with StageTimer(current_metrics, "transformer_decoder", args.device):
                out, hs = model._run_decoder(
                    memory=out["encoder_hidden_states"],
                    pos_embed=encoder_out["pos_embed"],
                    src_mask=encoder_out["padding_mask"],
                    out=out,
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                    encoder_out=encoder_out,
                )

            query_counts.append(int(hs.shape[2]))

            with StageTimer(current_metrics, "segmentation_heads", args.device):
                model._run_segmentation_heads(
                    out=out,
                    backbone_out=state["backbone_out"],
                    img_ids=processor.find_stage.img_ids,
                    vis_feat_sizes=encoder_out["vis_feat_sizes"],
                    encoder_hidden_states=out["encoder_hidden_states"],
                    prompt=prompt,
                    prompt_mask=prompt_mask,
                    hs=hs,
                )

            with StageTimer(current_metrics, "postprocess", args.device):
                out_logits = out["pred_logits"]
                out_masks = out["pred_masks"]
                out_probs = out_logits.sigmoid()
                presence_score = out["presence_logit_dec"].sigmoid().unsqueeze(1)
                out_probs = (out_probs * presence_score).squeeze(-1)
                keep = out_probs > args.confidence_threshold
                kept_masks = out_masks[keep]
                _ = kept_masks.shape[0]

        if idx >= args.warmup_runs:
            for name, values in current_metrics.items():
                metrics.setdefault(name, []).extend(values)

    metrics["query_count"] = query_counts[args.warmup_runs:]
    return metrics


def main() -> None:
    args = parse_args()
    metrics = run_profile(args)
    query_count = metrics.pop("query_count", [])
    ordered_names = [
        "total",
        "image_encoder",
        "encoder_onnx",
        "neck_decoder",
        "text_encode",
        "prompt_encode",
        "transformer_encoder",
        "transformer_decoder",
        "segmentation_heads",
        "postprocess",
    ]
    for name in ordered_names:
        values = metrics.get(name)
        if not values:
            continue
        avg_ms = statistics.mean(values) * 1000.0
        pct = 0.0
        if name != "total" and metrics.get("total"):
            pct = statistics.mean(values) / statistics.mean(metrics["total"]) * 100.0
        print(f"{name}: avg_ms={avg_ms:.2f} pct_of_total={pct:.1f}")
    if query_count:
        print(f"query_count={int(statistics.mean(query_count))}")


if __name__ == "__main__":
    main()
