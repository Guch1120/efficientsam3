#!/usr/bin/env python3
"""ONNX Runtime HTTP server for EfficientSAM3 encoder (+ optional text segmentation).

POST /encode
- Body: `.npy` float32 tensor [B,3,1008,1008]
- Returns: `.npy` float32 tensor [B,1024,72,72]

POST /segment_text?prompt=<text>  (requires --pytorch-checkpoint)
- Body: `.npy` image (H,W,3) uint8 or [3,H,W] float/uint8 or batched variants
- Returns: `.npy` uint8 mask (H,W), values {0,255}
"""

from __future__ import annotations

import argparse
import io
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ONNX Runtime encoder server")
    p.add_argument("--model", required=True, help="Path to encoder ONNX")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=18080)

    # Optional PyTorch path for text-prompt segmentation endpoint.
    p.add_argument("--pytorch-checkpoint", default=None)
    p.add_argument(
        "--backbone-type",
        default="tinyvit",
        choices=["efficientvit", "repvit", "tinyvit"],
    )
    p.add_argument("--model-name", default="21m")
    p.add_argument("--text-encoder-type", default=None)
    p.add_argument("--text-seg-onnx", default=None, help="Fixed-prompt text-seg ONNX path")
    return p.parse_args()


def _to_hwc_uint8_image(arr: np.ndarray) -> np.ndarray:
    x = arr
    if x.ndim == 4:
        x = x[0]
    if x.ndim != 3:
        raise ValueError(f"Expected 3D image tensor, got shape={x.shape}")

    # CHW -> HWC if needed
    if x.shape[0] == 3 and x.shape[-1] != 3:
        x = np.transpose(x, (1, 2, 0))

    if x.shape[-1] != 3:
        raise ValueError(f"Expected 3 channels, got shape={x.shape}")

    if x.dtype != np.uint8:
        if np.issubdtype(x.dtype, np.floating):
            # allow either [0,1] or [0,255]
            x = np.clip(x * 255.0 if x.max() <= 1.0 else x, 0, 255).astype(np.uint8)
        else:
            x = np.clip(x, 0, 255).astype(np.uint8)
    return x


class Handler(BaseHTTPRequestHandler):
    sess: ort.InferenceSession = None
    inp_name: str = "image"
    model = None
    processor = None
    text_seg_sess: ort.InferenceSession = None
    text_seg_inp: str = "image"

    def _read_npy(self) -> np.ndarray:
        n = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(n)
        return np.load(io.BytesIO(raw), allow_pickle=False)

    def _write_npy(self, arr: np.ndarray, code: int = 200) -> None:
        buf = io.BytesIO()
        np.save(buf, arr, allow_pickle=False)
        body = buf.getvalue()
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _write_text(self, msg: str, code: int = 400) -> None:
        body = msg.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/encode":
            try:
                x = self._read_npy()
                if x.dtype != np.float32:
                    x = x.astype(np.float32)
                y = self.sess.run(None, {self.inp_name: x})[0]
                self._write_npy(y, code=200)
            except Exception as e:
                self._write_text(str(e), code=400)
            return

        if parsed.path == "/segment_text_onnx":
            if self.text_seg_sess is None:
                self._write_text(
                    "Text ONNX endpoint is disabled. Start server with --text-seg-onnx.",
                    code=400,
                )
                return
            try:
                img = _to_hwc_uint8_image(self._read_npy())
                x = img.astype(np.float32) / 255.0
                x = (x - 0.5) / 0.5
                x = np.transpose(x, (2, 0, 1))[None, ...]
                pred_masks, pred_logits = self.text_seg_sess.run(None, {self.text_seg_inp: x})
                # expected shapes: [B,Q,H,W], [B,Q,1]
                score = 1.0 / (1.0 + np.exp(-pred_logits[0, :, 0]))
                idx = int(np.argmax(score))
                mask = (pred_masks[0, idx] > 0).astype(np.uint8) * 255
                self._write_npy(mask, code=200)
            except Exception as e:
                self._write_text(str(e), code=400)
            return

        if parsed.path == "/segment_text":
            if self.model is None or self.processor is None:
                self._write_text(
                    "Text segmentation endpoint is disabled. Start server with --pytorch-checkpoint.",
                    code=400,
                )
                return

            prompt = parse_qs(parsed.query).get("prompt", [""])[0].strip()
            if not prompt:
                prompt = self.headers.get("X-Text-Prompt", "").strip()
            if not prompt:
                self._write_text("Missing prompt. Use query ?prompt=... or X-Text-Prompt header.", code=400)
                return

            try:
                img = _to_hwc_uint8_image(self._read_npy())
                from PIL import Image
                import torch

                with torch.inference_mode():
                    state = self.processor.set_image(Image.fromarray(img))
                    state = self.processor.set_text_prompt(prompt=prompt, state=state)
                    masks = state["masks"]
                    scores = state["scores"]

                best_idx = int(np.argmax(scores))
                mask = (masks[best_idx] > 0).astype(np.uint8) * 255
                self._write_npy(mask, code=200)
            except Exception as e:
                self._write_text(str(e), code=400)
            return

        self.send_response(404)
        self.end_headers()


def main() -> None:
    args = parse_args()
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    Handler.sess = sess
    Handler.inp_name = sess.get_inputs()[0].name

    if args.text_seg_onnx:
        t_sess = ort.InferenceSession(args.text_seg_onnx, providers=["CPUExecutionProvider"])
        Handler.text_seg_sess = t_sess
        Handler.text_seg_inp = t_sess.get_inputs()[0].name

    if args.pytorch_checkpoint:
        from efficientsam.model_builder import build_efficientsam3_image_model
        from efficientsam.sam3_image_processor import Sam3Processor

        model = build_efficientsam3_image_model(
            checkpoint_path=args.pytorch_checkpoint,
            backbone_type=args.backbone_type,
            model_name=args.model_name,
            text_encoder_type=args.text_encoder_type,
            enable_inst_interactivity=False,
            eval_mode=True,
            compile=False,
            device="cpu",
        )
        Handler.model = model
        Handler.processor = Sam3Processor(model)

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Encoder server on http://{args.host}:{args.port}")
    print("POST /encode for encoder embedding")
    if Handler.text_seg_sess is not None:
        print("POST /segment_text_onnx for fixed-prompt ONNX text mask")
    if Handler.model is not None:
        print("POST /segment_text?prompt=person for text-prompt mask")
    srv.serve_forever()


if __name__ == "__main__":
    main()
