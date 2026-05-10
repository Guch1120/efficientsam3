#!/usr/bin/env python3
"""Benchmark text-prompt inference on a single image for PyTorch or ONNX server."""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import argparse
import io
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

from efficientsam.model_builder import build_efficientsam3_image_model
from efficientsam.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model

_ORT_PROVIDER_MODE = "auto"
_TRT_FP16_ENABLE = False
_TRT_ENGINE_CACHE_ENABLE = False
_TRT_ENGINE_CACHE_PATH = "/tmp/ort_trt_cache"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark single-image text-prompt inference"
    )
    parser.add_argument(
        "--backend",
        choices=[
            "pytorch",
            "sam3",
            "onnx_server",
            "onnx_local",
            "onnx_split",
            "onnx_backbone",
            "onnx_text_downstream",
            "onnx_grounding_core",
        ],
        default="pytorch",
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--resolution", type=int, default=1008)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--confidence-threshold", type=float, default=0.05)
    parser.add_argument("--server", default="http://127.0.0.1:18080")
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--encoder-onnx", default=None)
    parser.add_argument("--decoder-onnx", default=None)
    parser.add_argument("--backbone-onnx", default=None)
    parser.add_argument("--text-downstream-onnx", default=None)
    parser.add_argument("--grounding-core-onnx", default=None)
    parser.add_argument(
        "--ort-provider",
        choices=["auto", "cuda", "tensorrt"],
        default="auto",
    )
    parser.add_argument("--trt-fp16", action="store_true")
    parser.add_argument("--trt-engine-cache", action="store_true")
    parser.add_argument("--trt-engine-cache-path", default="/tmp/ort_trt_cache")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--sam3-checkpoint", default=None)
    parser.add_argument(
        "--backbone-type",
        default="tinyvit",
        choices=["efficientvit", "repvit", "tinyvit"],
    )
    parser.add_argument("--model-name", default="21m")
    parser.add_argument("--text-encoder-type", default="MobileCLIP-S1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Reduce decoder query count for runtime experiments.",
    )
    parser.add_argument("--max-encoder-layers", type=int, default=None)
    parser.add_argument("--max-decoder-layers", type=int, default=None)
    parser.add_argument(
        "--encoder-feature-downsample",
        type=int,
        default=1,
        help="Average-pool the last encoder feature level by this factor for runtime experiments.",
    )
    parser.add_argument(
        "--adaptive-encoder-feature-downsample",
        action="store_true",
        help="Choose factor 2 or 3 from the last feature-map complexity.",
    )
    parser.add_argument(
        "--adaptive-feature-threshold",
        type=float,
        default=0.27,
        help="If complexity is above this threshold, use factor 2, else factor 3.",
    )
    parser.add_argument(
        "--cache-text-prompt",
        action="store_true",
        help="Reuse text encoder outputs across runs for the same prompt.",
    )
    args = parser.parse_args()
    if args.backend in ("pytorch", "onnx_local", "onnx_split", "onnx_backbone") and not args.checkpoint:
        parser.error("--checkpoint is required when --backend pytorch/onnx_local/onnx_split/onnx_backbone")
    if args.backend in ("onnx_local", "onnx_split", "onnx_text_downstream") and not args.encoder_onnx:
        parser.error("--encoder-onnx is required when --backend onnx_local/onnx_split/onnx_text_downstream")
    if args.backend == "onnx_split" and not args.decoder_onnx:
        parser.error("--decoder-onnx is required when --backend onnx_split")
    if args.backend == "onnx_backbone" and not args.backbone_onnx:
        parser.error("--backbone-onnx is required when --backend onnx_backbone")
    if args.backend == "onnx_text_downstream" and not args.text_downstream_onnx:
        parser.error("--text-downstream-onnx is required when --backend onnx_text_downstream")
    if args.backend == "onnx_grounding_core" and not args.grounding_core_onnx:
        parser.error("--grounding-core-onnx is required when --backend onnx_grounding_core")
    if args.backend == "onnx_grounding_core" and not args.decoder_onnx:
        parser.error("--decoder-onnx is required when --backend onnx_grounding_core")
    return args


def _get_ort_providers() -> list[str]:
    import onnxruntime as ort

    available = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    trt_provider = (
        "TensorrtExecutionProvider",
        {
            "trt_fp16_enable": _TRT_FP16_ENABLE,
            "trt_engine_cache_enable": _TRT_ENGINE_CACHE_ENABLE,
            "trt_engine_cache_path": _TRT_ENGINE_CACHE_PATH,
        },
    )
    if _ORT_PROVIDER_MODE == "tensorrt":
        if "TensorrtExecutionProvider" in available:
            providers = [trt_provider, "CUDAExecutionProvider", "CPUExecutionProvider"]
        elif "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif _ORT_PROVIDER_MODE == "cuda":
        if "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        if "TensorrtExecutionProvider" in available:
            providers = [trt_provider, "CUDAExecutionProvider", "CPUExecutionProvider"]
        elif "CUDAExecutionProvider" in available:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return providers


def _build_model_and_processor(
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, Sam3Processor]:
    model = build_efficientsam3_image_model(
        checkpoint_path=args.checkpoint,
        backbone_type=args.backbone_type,
        model_name=args.model_name,
        text_encoder_type=args.text_encoder_type,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        eval_mode=True,
        compile=args.compile,
        device=args.device,
    )
    _apply_runtime_limits(
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


def _apply_runtime_limits(
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


def _maybe_downsample_last_feature_level(
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


def _select_adaptive_downsample_factor(
    backbone_out: dict,
    threshold: float,
) -> int:
    last_feat = backbone_out["backbone_fpn"][-1]
    grad_x = (last_feat[:, :, :, 1:] - last_feat[:, :, :, :-1]).abs().mean().item()
    grad_y = (last_feat[:, :, 1:, :] - last_feat[:, :, :-1, :]).abs().mean().item()
    complexity = grad_x + grad_y
    return 2 if complexity >= threshold else 3


def _build_sam3_model_and_processor(
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, Sam3Processor]:
    model = build_sam3_image_model(
        checkpoint_path=args.sam3_checkpoint,
        load_from_HF=args.sam3_checkpoint is None,
        device=args.device,
        eval_mode=True,
    )
    processor = Sam3Processor(
        model,
        resolution=args.resolution,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
    )
    return model, processor


def _prepare_image_input(
    processor: Sam3Processor,
    image: Image.Image,
) -> tuple[np.ndarray, int, int]:
    width, height = image.size
    image_tensor = v2.functional.to_image(image)
    image_np = processor.transform(image_tensor).unsqueeze(0).numpy().astype(np.float32)
    return image_np, width, height


def _run_pytorch(args: argparse.Namespace, image: Image.Image) -> tuple[list[float], int, float | None]:
    model, processor = _build_model_and_processor(args)
    cached_text_outputs = None
    if args.cache_text_prompt:
        with torch.inference_mode():
            cached_text_outputs = model.backbone.forward_text([args.prompt], device=args.device)

    timings: list[float] = []
    last_count = 0
    last_top_score: float | None = None

    total_runs = args.warmup_runs + args.runs
    for idx in range(total_runs):
        start = time.perf_counter()
        with torch.inference_mode():
            state = processor.set_image(image)
            factor = args.encoder_feature_downsample
            if args.adaptive_encoder_feature_downsample:
                factor = _select_adaptive_downsample_factor(
                    state["backbone_out"], args.adaptive_feature_threshold
                )
            state["backbone_out"] = _maybe_downsample_last_feature_level(
                model, state["backbone_out"], factor
            )
            if cached_text_outputs is None:
                state = processor.set_text_prompt(args.prompt, state)
            else:
                state["backbone_out"].update(cached_text_outputs)
                if "geometric_prompt" not in state:
                    state["geometric_prompt"] = model._get_dummy_prompt()
                state = processor._forward_grounding(state)
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        scores = state["scores"]
        last_count = int(scores.numel())
        last_top_score = float(scores.max().item()) if last_count else None
        if idx >= args.warmup_runs:
            timings.append(elapsed)

    return timings, last_count, last_top_score


def _run_sam3(
    args: argparse.Namespace, image: Image.Image
) -> tuple[list[float], int, float | None]:
    model, processor = _build_sam3_model_and_processor(args)
    cached_text_outputs = None
    if args.cache_text_prompt:
        with torch.inference_mode():
            cached_text_outputs = model.backbone.forward_text([args.prompt], device=args.device)

    timings: list[float] = []
    last_count = 0
    last_top_score: float | None = None

    total_runs = args.warmup_runs + args.runs
    for idx in range(total_runs):
        start = time.perf_counter()
        with torch.inference_mode():
            state = processor.set_image(image)
            if cached_text_outputs is None:
                state = processor.set_text_prompt(args.prompt, state)
            else:
                state["backbone_out"].update(cached_text_outputs)
                if "geometric_prompt" not in state:
                    state["geometric_prompt"] = model._get_dummy_prompt()
                state = processor._forward_grounding(state)
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        scores = state["scores"]
        last_count = int(scores.numel())
        last_top_score = float(scores.max().item()) if last_count else None
        if idx >= args.warmup_runs:
            timings.append(elapsed)

    return timings, last_count, last_top_score


def _run_onnx_server(args: argparse.Namespace, image: Image.Image) -> tuple[list[float], int, float | None]:
    image_np = np.asarray(image.convert("RGB"), dtype=np.uint8)
    payload = io.BytesIO()
    np.save(payload, image_np, allow_pickle=False)
    body = payload.getvalue()
    prompt = quote(args.prompt, safe="")
    url = f"{args.server.rstrip('/')}/segment_text?prompt={prompt}"

    timings: list[float] = []
    last_count = -1

    total_runs = args.warmup_runs + args.runs
    for idx in range(total_runs):
        req = Request(
            url=url,
            data=body,
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        start = time.perf_counter()
        with urlopen(req, timeout=args.timeout_sec) as resp:
            mask = np.load(io.BytesIO(resp.read()), allow_pickle=False)
        elapsed = time.perf_counter() - start
        last_count = int(np.count_nonzero(mask) > 0)
        if idx >= args.warmup_runs:
            timings.append(elapsed)

    return timings, last_count, None


def _run_onnx_local(
    args: argparse.Namespace, image: Image.Image
) -> tuple[list[float], int, float | None, list[str] | None]:
    import onnxruntime as ort

    sess = ort.InferenceSession(args.encoder_onnx, providers=_get_ort_providers())
    inp_name = sess.get_inputs()[0].name

    model, processor = _build_model_and_processor(args)
    image_np, width, height = _prepare_image_input(processor, image)
    cached_text_outputs = None
    if args.cache_text_prompt:
        with torch.inference_mode():
            cached_text_outputs = model.backbone.forward_text([args.prompt], device=args.device)

    vb = model.backbone.vision_backbone
    scalp = int(getattr(model.backbone, "scalp", 0))
    active_convs = list(vb.convs)
    if scalp > 0:
        active_convs = active_convs[:-scalp]

    timings: list[float] = []
    last_count = 0
    last_top_score: float | None = None

    total_runs = args.warmup_runs + args.runs
    for idx in range(total_runs):
        start = time.perf_counter()
        image_embed = sess.run(None, {inp_name: image_np})[0]
        image_embed_t = torch.from_numpy(image_embed).to(args.device)

        sam3_features: list[torch.Tensor] = []
        sam3_pos: list[torch.Tensor] = []
        for conv in active_convs:
            feat = conv(image_embed_t)
            pos = vb.position_encoding(feat).to(feat.dtype)
            sam3_features.append(feat)
            sam3_pos.append(pos)

        state = {
            "original_height": height,
            "original_width": width,
            "backbone_out": {
                "vision_features": sam3_features[-1],
                "vision_pos_enc": sam3_pos,
                "backbone_fpn": sam3_features,
                "sam2_backbone_out": None,
            },
        }
        factor = args.encoder_feature_downsample
        if args.adaptive_encoder_feature_downsample:
            factor = _select_adaptive_downsample_factor(
                state["backbone_out"], args.adaptive_feature_threshold
            )
        state["backbone_out"] = _maybe_downsample_last_feature_level(
            model, state["backbone_out"], factor
        )

        with torch.inference_mode():
            if cached_text_outputs is None:
                state = processor.set_text_prompt(args.prompt, state)
            else:
                state["backbone_out"].update(cached_text_outputs)
                if "geometric_prompt" not in state:
                    state["geometric_prompt"] = model._get_dummy_prompt()
                state = processor._forward_grounding(state)
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        scores = state["scores"]
        last_count = int(scores.numel())
        last_top_score = float(scores.max().item()) if last_count else None
        if idx >= args.warmup_runs:
            timings.append(elapsed)

    return timings, last_count, last_top_score, sess.get_providers()


def _run_onnx_split(
    args: argparse.Namespace, image: Image.Image
) -> tuple[list[float], int, float | None, list[str] | None]:
    import onnxruntime as ort

    providers = _get_ort_providers()
    enc_sess = ort.InferenceSession(args.encoder_onnx, providers=providers)
    dec_sess = ort.InferenceSession(args.decoder_onnx, providers=providers)
    enc_inp_name = enc_sess.get_inputs()[0].name
    dec_inp_name = dec_sess.get_inputs()[0].name

    model, processor = _build_model_and_processor(args)
    image_np, width, height = _prepare_image_input(processor, image)
    cached_text_outputs = None
    if args.cache_text_prompt:
        with torch.inference_mode():
            cached_text_outputs = model.backbone.forward_text([args.prompt], device=args.device)

    num_levels = len(dec_sess.get_outputs()) // 2

    timings: list[float] = []
    last_count = 0
    last_top_score: float | None = None

    total_runs = args.warmup_runs + args.runs
    for idx in range(total_runs):
        start = time.perf_counter()
        image_embed = enc_sess.run(None, {enc_inp_name: image_np})[0]
        decoder_outs = dec_sess.run(None, {dec_inp_name: image_embed})

        sam3_features = [
            torch.from_numpy(decoder_outs[i]).to(args.device) for i in range(num_levels)
        ]
        sam3_pos = [
            torch.from_numpy(decoder_outs[num_levels + i]).to(args.device)
            for i in range(num_levels)
        ]

        state = {
            "original_height": height,
            "original_width": width,
            "backbone_out": {
                "vision_features": sam3_features[-1],
                "vision_pos_enc": sam3_pos,
                "backbone_fpn": sam3_features,
                "sam2_backbone_out": None,
            },
        }
        factor = args.encoder_feature_downsample
        if args.adaptive_encoder_feature_downsample:
            factor = _select_adaptive_downsample_factor(
                state["backbone_out"], args.adaptive_feature_threshold
            )
        state["backbone_out"] = _maybe_downsample_last_feature_level(
            model, state["backbone_out"], factor
        )

        with torch.inference_mode():
            if cached_text_outputs is None:
                state = processor.set_text_prompt(args.prompt, state)
            else:
                state["backbone_out"].update(cached_text_outputs)
                if "geometric_prompt" not in state:
                    state["geometric_prompt"] = model._get_dummy_prompt()
                state = processor._forward_grounding(state)
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        scores = state["scores"]
        last_count = int(scores.numel())
        last_top_score = float(scores.max().item()) if last_count else None
        if idx >= args.warmup_runs:
            timings.append(elapsed)

    return timings, last_count, last_top_score, enc_sess.get_providers()


def _run_onnx_backbone(
    args: argparse.Namespace, image: Image.Image
) -> tuple[list[float], int, float | None, list[str] | None]:
    import onnxruntime as ort

    sess = ort.InferenceSession(args.backbone_onnx, providers=_get_ort_providers())
    inp_name = sess.get_inputs()[0].name

    model, processor = _build_model_and_processor(args)
    image_np, width, height = _prepare_image_input(processor, image)
    num_levels = len(sess.get_outputs()) // 2
    cached_text_outputs = None
    if args.cache_text_prompt:
        with torch.inference_mode():
            cached_text_outputs = model.backbone.forward_text([args.prompt], device=args.device)

    timings: list[float] = []
    last_count = 0
    last_top_score: float | None = None

    total_runs = args.warmup_runs + args.runs
    for idx in range(total_runs):
        start = time.perf_counter()
        backbone_outs = sess.run(None, {inp_name: image_np})

        sam3_features = [
            torch.from_numpy(backbone_outs[i]).to(args.device) for i in range(num_levels)
        ]
        sam3_pos = [
            torch.from_numpy(backbone_outs[num_levels + i]).to(args.device)
            for i in range(num_levels)
        ]

        state = {
            "original_height": height,
            "original_width": width,
            "backbone_out": {
                "vision_features": sam3_features[-1],
                "vision_pos_enc": sam3_pos,
                "backbone_fpn": sam3_features,
                "sam2_backbone_out": None,
            },
        }
        factor = args.encoder_feature_downsample
        if args.adaptive_encoder_feature_downsample:
            factor = _select_adaptive_downsample_factor(
                state["backbone_out"], args.adaptive_feature_threshold
            )
        state["backbone_out"] = _maybe_downsample_last_feature_level(
            model, state["backbone_out"], factor
        )

        with torch.inference_mode():
            if cached_text_outputs is None:
                state = processor.set_text_prompt(args.prompt, state)
            else:
                state["backbone_out"].update(cached_text_outputs)
                if "geometric_prompt" not in state:
                    state["geometric_prompt"] = model._get_dummy_prompt()
                state = processor._forward_grounding(state)
        if args.device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        scores = state["scores"]
        last_count = int(scores.numel())
        last_top_score = float(scores.max().item()) if last_count else None
        if idx >= args.warmup_runs:
            timings.append(elapsed)

    return timings, last_count, last_top_score, sess.get_providers()


def _run_onnx_text_downstream(
    args: argparse.Namespace, image: Image.Image
) -> tuple[list[float], int, float | None, list[str] | None]:
    import onnxruntime as ort

    providers = _get_ort_providers()
    enc_sess = ort.InferenceSession(args.encoder_onnx, providers=providers)
    down_sess = ort.InferenceSession(args.text_downstream_onnx, providers=providers)
    enc_inp_name = enc_sess.get_inputs()[0].name
    down_inp_name = down_sess.get_inputs()[0].name

    model, processor = _build_model_and_processor(args)
    image_np, _, _ = _prepare_image_input(processor, image)

    timings: list[float] = []
    last_count = 0
    last_top_score: float | None = None

    total_runs = args.warmup_runs + args.runs
    for idx in range(total_runs):
        start = time.perf_counter()
        image_embed = enc_sess.run(None, {enc_inp_name: image_np})[0]
        pred_masks, pred_logits, presence_logit_dec, _pred_boxes = down_sess.run(
            None, {down_inp_name: image_embed}
        )
        out_probs = 1.0 / (1.0 + np.exp(-pred_logits))
        presence = 1.0 / (1.0 + np.exp(-presence_logit_dec))
        out_probs = (out_probs * np.expand_dims(presence, axis=1)).squeeze(-1)
        keep = out_probs > args.confidence_threshold
        kept_scores = out_probs[keep]
        last_count = int(kept_scores.size)
        last_top_score = float(kept_scores.max()) if last_count else None
        elapsed = time.perf_counter() - start
        if idx >= args.warmup_runs:
            timings.append(elapsed)

    return timings, last_count, last_top_score, enc_sess.get_providers()


def _run_onnx_grounding_core(
    args: argparse.Namespace, image: Image.Image
) -> tuple[list[float], int, float | None, list[str] | None]:
    import onnxruntime as ort

    providers = _get_ort_providers()
    enc_sess = ort.InferenceSession(args.encoder_onnx, providers=providers)
    dec_sess = ort.InferenceSession(args.decoder_onnx, providers=providers)
    core_sess = ort.InferenceSession(args.grounding_core_onnx, providers=providers)
    enc_inp_name = enc_sess.get_inputs()[0].name
    dec_inp_name = dec_sess.get_inputs()[0].name
    core_input_names = [inp.name for inp in core_sess.get_inputs()]

    model, processor = _build_model_and_processor(args)
    image_np, _, _ = _prepare_image_input(processor, image)
    cached_text_outputs = None
    if args.cache_text_prompt:
        with torch.inference_mode():
            cached_text_outputs = model.backbone.forward_text([args.prompt], device=args.device)
    cached_geometric_prompt = model._get_dummy_prompt()

    num_levels = len(dec_sess.get_outputs()) // 2

    timings: list[float] = []
    last_count = 0
    last_top_score: float | None = None

    total_runs = args.warmup_runs + args.runs
    for idx in range(total_runs):
        start = time.perf_counter()
        image_embed = enc_sess.run(None, {enc_inp_name: image_np})[0]
        decoder_outs = dec_sess.run(None, {dec_inp_name: image_embed})

        sam3_features = [
            torch.from_numpy(decoder_outs[i]).to(args.device) for i in range(num_levels)
        ]
        sam3_pos = [
            torch.from_numpy(decoder_outs[num_levels + i]).to(args.device)
            for i in range(num_levels)
        ]
        backbone_out = {
            "vision_features": sam3_features[-1],
            "vision_pos_enc": sam3_pos,
            "backbone_fpn": sam3_features,
            "sam2_backbone_out": None,
        }
        if cached_text_outputs is None:
            text_outputs = model.backbone.forward_text([args.prompt], device=args.device)
        else:
            text_outputs = cached_text_outputs
        backbone_out.update(text_outputs)
        prompt, prompt_mask, _ = model._encode_prompt(
            backbone_out,
            processor.find_stage,
            cached_geometric_prompt,
        )

        ort_inputs = {}
        for level_idx in range(num_levels):
            ort_inputs[f"feat_l{level_idx}"] = sam3_features[level_idx].detach().cpu().numpy()
            ort_inputs[f"pos_l{level_idx}"] = sam3_pos[level_idx].detach().cpu().numpy()
        ort_inputs["prompt"] = prompt.detach().cpu().numpy()
        ort_inputs["prompt_mask"] = prompt_mask.detach().cpu().numpy()
        pred_masks, pred_logits, presence_logit_dec, _pred_boxes = core_sess.run(
            None,
            {name: ort_inputs[name] for name in core_input_names},
        )
        out_probs = 1.0 / (1.0 + np.exp(-pred_logits))
        presence = 1.0 / (1.0 + np.exp(-presence_logit_dec))
        out_probs = (out_probs * np.expand_dims(presence, axis=1)).squeeze(-1)
        keep = out_probs > args.confidence_threshold
        kept_scores = out_probs[keep]
        last_count = int(kept_scores.size)
        last_top_score = float(kept_scores.max()) if last_count else None
        elapsed = time.perf_counter() - start
        if idx >= args.warmup_runs:
            timings.append(elapsed)

    return timings, last_count, last_top_score, enc_sess.get_providers()


def main() -> None:
    args = parse_args()
    global _ORT_PROVIDER_MODE, _TRT_FP16_ENABLE, _TRT_ENGINE_CACHE_ENABLE, _TRT_ENGINE_CACHE_PATH
    _ORT_PROVIDER_MODE = args.ort_provider
    _TRT_FP16_ENABLE = args.trt_fp16
    _TRT_ENGINE_CACHE_ENABLE = args.trt_engine_cache
    _TRT_ENGINE_CACHE_PATH = args.trt_engine_cache_path
    image = Image.open(args.image).convert("RGB")

    if args.backend == "pytorch":
        timings, last_count, last_top_score = _run_pytorch(args, image)
        ort_providers = None
    elif args.backend == "sam3":
        timings, last_count, last_top_score = _run_sam3(args, image)
        ort_providers = None
    elif args.backend == "onnx_local":
        timings, last_count, last_top_score, ort_providers = _run_onnx_local(args, image)
    elif args.backend == "onnx_split":
        timings, last_count, last_top_score, ort_providers = _run_onnx_split(args, image)
    elif args.backend == "onnx_backbone":
        timings, last_count, last_top_score, ort_providers = _run_onnx_backbone(args, image)
    elif args.backend == "onnx_text_downstream":
        timings, last_count, last_top_score, ort_providers = _run_onnx_text_downstream(args, image)
    elif args.backend == "onnx_grounding_core":
        timings, last_count, last_top_score, ort_providers = _run_onnx_grounding_core(args, image)
    else:
        timings, last_count, last_top_score = _run_onnx_server(args, image)
        ort_providers = None

    avg_ms = sum(timings) / len(timings) * 1000.0
    min_ms = min(timings) * 1000.0
    max_ms = max(timings) * 1000.0
    fps = len(timings) / sum(timings)

    print(f"backend={args.backend}")
    print(f"prompt={args.prompt}")
    print(f"runs={args.runs} warmup_runs={args.warmup_runs}")
    print(f"confidence_threshold={args.confidence_threshold}")
    print(f"avg_latency_ms={avg_ms:.2f}")
    print(f"min_latency_ms={min_ms:.2f}")
    print(f"max_latency_ms={max_ms:.2f}")
    print(f"throughput_fps={fps:.2f}")
    print(f"last_detection_count_indicator={last_count}")
    print(f"last_top_score={last_top_score}")
    if ort_providers is not None:
        print(f"ort_providers={ort_providers}")


if __name__ == "__main__":
    main()
