#!/usr/bin/env python3
"""Benchmark EfficientSAM3 text-prompt inference on a ROS2 image topic."""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import argparse
import io
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from PIL import Image
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from torchvision.transforms import v2

from efficientsam.model_builder import build_efficientsam3_image_model
from efficientsam.sam3_image_processor import Sam3Processor


class EfficientSam3Ros2Benchmark(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("efficientsam3_ros2_benchmark")
        self.args = args
        self.bridge = CvBridge()
        self.latest_msg: RosImage | None = None
        self.latest_rgb: np.ndarray | None = None
        self.image_count = 0
        self.processed_count = 0
        self.total_latency_sec = 0.0
        self.max_latency_sec = 0.0
        self.start_time = time.perf_counter()
        self.last_report_time = self.start_time

        self.mask_pub = None
        if args.output_topic:
            self.mask_pub = self.create_publisher(RosImage, args.output_topic, 10)
        self.overlay_pub = None
        if args.overlay_output_topic:
            self.overlay_pub = self.create_publisher(RosImage, args.overlay_output_topic, 10)

        self.create_subscription(
            RosImage,
            args.input_topic,
            self._on_image,
            10,
        )

        self.create_timer(1.0 / max(args.target_fps, 1e-6), self._process_latest_frame)

        self.backend = args.backend
        if self.backend in ("pytorch", "onnx_local", "onnx_split"):
            self.model = build_efficientsam3_image_model(
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
            self.processor = Sam3Processor(self.model, device=args.device)
            self.processor.set_confidence_threshold(args.confidence_threshold)
            self.ort_sess = None
            self.ort_input_name = None
            self.ort_decoder_sess = None
            self.ort_decoder_input_name = None
            self.active_convs = None
            if self.backend in ("onnx_local", "onnx_split"):
                import onnxruntime as ort

                providers = ["CPUExecutionProvider"]
                available = ort.get_available_providers()
                if "CUDAExecutionProvider" in available:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                self.ort_sess = ort.InferenceSession(args.encoder_onnx, providers=providers)
                self.ort_input_name = self.ort_sess.get_inputs()[0].name
                if self.backend == "onnx_local":
                    convs = list(self.model.backbone.vision_backbone.convs)
                    scalp = int(getattr(self.model.backbone, "scalp", 0))
                    if scalp > 0:
                        convs = convs[:-scalp]
                    self.active_convs = convs
                else:
                    self.ort_decoder_sess = ort.InferenceSession(
                        args.decoder_onnx,
                        providers=providers,
                    )
                    self.ort_decoder_input_name = self.ort_decoder_sess.get_inputs()[0].name
        else:
            self.model = None
            self.processor = None
            self.ort_sess = None
            self.ort_input_name = None
            self.ort_decoder_sess = None
            self.ort_decoder_input_name = None
            self.active_convs = None

        self.get_logger().info(
            f"Benchmark started. backend={args.backend} input_topic={args.input_topic} "
            f"prompt={args.prompt} target_fps={args.target_fps}"
        )

    def _on_image(self, msg: RosImage) -> None:
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        image_np = np.asarray(image)

        if image_np.ndim == 2:
            rgb = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.ndim == 3 and image_np.shape[2] == 3:
            if msg.encoding == "rgb8":
                rgb = image_np
            else:
                rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        elif image_np.ndim == 3 and image_np.shape[2] == 4:
            if msg.encoding == "rgba8":
                rgb = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
            else:
                rgb = cv2.cvtColor(image_np, cv2.COLOR_BGRA2RGB)
        else:
            self.get_logger().warning(
                f"Unsupported image shape={image_np.shape} encoding={msg.encoding}"
            )
            return

        self.latest_msg = msg
        self.latest_rgb = np.ascontiguousarray(rgb)
        self.image_count += 1

    def _run_pytorch_inference(self, rgb: np.ndarray) -> np.ndarray:
        pil_img = Image.fromarray(rgb)
        with torch.inference_mode():
            state = self.processor.set_image(pil_img)
            state = self.processor.set_text_prompt(prompt=self.args.prompt, state=state)

        masks = state["masks"]
        scores = state["scores"]
        best_idx = int(torch.argmax(scores).item())
        mask = masks[best_idx].detach().to("cpu").numpy()
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        return (mask > 0).astype(np.uint8) * 255

    def _run_onnx_local_inference(self, rgb: np.ndarray) -> np.ndarray:
        image = Image.fromarray(rgb)
        image_tensor = v2.functional.to_image(image)
        image_np = self.processor.transform(image_tensor).unsqueeze(0).numpy().astype(np.float32)

        image_embed = self.ort_sess.run(None, {self.ort_input_name: image_np})[0]
        image_embed_t = torch.from_numpy(image_embed).to(self.args.device)

        vb = self.model.backbone.vision_backbone
        sam3_features: list[torch.Tensor] = []
        sam3_pos: list[torch.Tensor] = []
        for conv in self.active_convs:
            feat = conv(image_embed_t)
            pos = vb.position_encoding(feat).to(feat.dtype)
            sam3_features.append(feat)
            sam3_pos.append(pos)

        state = {
            "original_height": rgb.shape[0],
            "original_width": rgb.shape[1],
            "backbone_out": {
                "vision_features": sam3_features[-1],
                "vision_pos_enc": sam3_pos,
                "backbone_fpn": sam3_features,
                "sam2_backbone_out": None,
            },
        }

        with torch.inference_mode():
            state = self.processor.set_text_prompt(prompt=self.args.prompt, state=state)

        masks = state["masks"]
        scores = state["scores"]
        if scores.numel() == 0:
            return np.zeros(rgb.shape[:2], dtype=np.uint8)

        best_idx = int(torch.argmax(scores).item())
        mask = masks[best_idx].detach().to("cpu").numpy()
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        return (mask > 0).astype(np.uint8) * 255

    def _run_onnx_split_inference(self, rgb: np.ndarray) -> np.ndarray:
        image = Image.fromarray(rgb)
        image_tensor = v2.functional.to_image(image)
        image_np = self.processor.transform(image_tensor).unsqueeze(0).numpy().astype(np.float32)

        image_embed = self.ort_sess.run(None, {self.ort_input_name: image_np})[0]
        decoder_outs = self.ort_decoder_sess.run(
            None,
            {self.ort_decoder_input_name: image_embed},
        )

        num_levels = len(decoder_outs) // 2
        sam3_features = [
            torch.from_numpy(decoder_outs[i]).to(self.args.device) for i in range(num_levels)
        ]
        sam3_pos = [
            torch.from_numpy(decoder_outs[num_levels + i]).to(self.args.device)
            for i in range(num_levels)
        ]

        state = {
            "original_height": rgb.shape[0],
            "original_width": rgb.shape[1],
            "backbone_out": {
                "vision_features": sam3_features[-1],
                "vision_pos_enc": sam3_pos,
                "backbone_fpn": sam3_features,
                "sam2_backbone_out": None,
            },
        }

        with torch.inference_mode():
            state = self.processor.set_text_prompt(prompt=self.args.prompt, state=state)

        masks = state["masks"]
        scores = state["scores"]
        if scores.numel() == 0:
            return np.zeros(rgb.shape[:2], dtype=np.uint8)

        best_idx = int(torch.argmax(scores).item())
        mask = masks[best_idx].detach().to("cpu").numpy()
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        return (mask > 0).astype(np.uint8) * 255

    def _run_onnx_server_inference(self, rgb: np.ndarray) -> np.ndarray:
        buf = io.BytesIO()
        np.save(buf, rgb.astype(np.uint8), allow_pickle=False)

        prompt = quote(self.args.prompt, safe="")
        req = Request(
            url=f"{self.args.server.rstrip('/')}/segment_text?prompt={prompt}",
            data=buf.getvalue(),
            headers={"Content-Type": "application/octet-stream"},
            method="POST",
        )
        with urlopen(req, timeout=self.args.timeout_sec) as resp:
            return np.load(io.BytesIO(resp.read()), allow_pickle=False).astype(np.uint8)

    def _publish_mask(self, mask: np.ndarray) -> None:
        if self.mask_pub is None or self.latest_msg is None:
            return
        out_msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
        out_msg.header = self.latest_msg.header
        self.mask_pub.publish(out_msg)

    def _build_overlay_image(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        overlay = rgb.copy()
        positive = mask > 0
        if positive.any():
            ys, xs = np.where(positive)
            x_min = int(xs.min())
            x_max = int(xs.max())
            y_min = int(ys.min())
            y_max = int(ys.max())

            # セグメント領域を半透明で重ね、bbox を同時に描画する。
            overlay[positive] = (
                0.6 * overlay[positive] + 0.4 * np.array([0, 255, 0], dtype=np.float32)
            ).astype(np.uint8)
            cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (255, 64, 64), 2)
        return overlay

    def _publish_overlay(self, rgb: np.ndarray, mask: np.ndarray) -> None:
        if self.overlay_pub is None or self.latest_msg is None:
            return
        overlay = self._build_overlay_image(rgb, mask)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        out_msg = self.bridge.cv2_to_imgmsg(overlay_bgr, encoding="bgr8")
        out_msg.header = self.latest_msg.header
        self.overlay_pub.publish(out_msg)

    def _process_latest_frame(self) -> None:
        if self.latest_rgb is None:
            return

        rgb = self.latest_rgb.copy()
        t0 = time.perf_counter()
        try:
            if self.backend == "pytorch":
                mask = self._run_pytorch_inference(rgb)
            elif self.backend == "onnx_local":
                mask = self._run_onnx_local_inference(rgb)
            elif self.backend == "onnx_split":
                mask = self._run_onnx_split_inference(rgb)
            else:
                mask = self._run_onnx_server_inference(rgb)
        except Exception as exc:
            self.get_logger().error(f"inference failed: {exc}")
            return

        latency = time.perf_counter() - t0
        self.processed_count += 1
        self.total_latency_sec += latency
        self.max_latency_sec = max(self.max_latency_sec, latency)
        self._publish_mask(mask)
        self._publish_overlay(rgb, mask)
        self._maybe_report()

    def _maybe_report(self) -> None:
        now = time.perf_counter()
        if now - self.last_report_time < self.args.report_interval_sec:
            return

        elapsed = max(now - self.start_time, 1e-9)
        avg_latency_ms = (self.total_latency_sec / max(self.processed_count, 1)) * 1000.0
        processed_fps = self.processed_count / elapsed
        input_fps = self.image_count / elapsed
        self.get_logger().info(
            "input_fps=%.2f processed_fps=%.2f avg_latency_ms=%.2f max_latency_ms=%.2f frames=%d"
            % (
                input_fps,
                processed_fps,
                avg_latency_ms,
                self.max_latency_sec * 1000.0,
                self.processed_count,
            )
        )
        self.last_report_time = now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark EfficientSAM3 on a ROS2 image topic")
    parser.add_argument("--backend", choices=["pytorch", "onnx_server", "onnx_local", "onnx_split"], default="pytorch")
    parser.add_argument("--checkpoint", default=None, help="Required for pytorch/onnx_local/onnx_split backend")
    parser.add_argument("--encoder-onnx", default=None, help="Required for onnx_local/onnx_split backend")
    parser.add_argument("--decoder-onnx", default=None, help="Required for onnx_split backend")
    parser.add_argument(
        "--backbone-type",
        default="tinyvit",
        choices=["efficientvit", "repvit", "tinyvit"],
    )
    parser.add_argument("--model-name", default="21m")
    parser.add_argument("--text-encoder-type", default="MobileCLIP-S1")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--server", default="http://127.0.0.1:18080")
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--input-topic", default="/camera/color/image_raw")
    parser.add_argument("--output-topic", default=None)
    parser.add_argument("--overlay-output-topic", default=None)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--target-fps", type=float, default=5.0)
    parser.add_argument("--report-interval-sec", type=float, default=5.0)
    args = parser.parse_args()

    if args.backend in ("pytorch", "onnx_local", "onnx_split") and not args.checkpoint:
        parser.error("--checkpoint is required when --backend pytorch/onnx_local/onnx_split")
    if args.backend in ("onnx_local", "onnx_split") and not args.encoder_onnx:
        parser.error("--encoder-onnx is required when --backend onnx_local/onnx_split")
    if args.backend == "onnx_split" and not args.decoder_onnx:
        parser.error("--decoder-onnx is required when --backend onnx_split")
    return args


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = EfficientSam3Ros2Benchmark(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
