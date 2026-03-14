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

        self.create_subscription(
            RosImage,
            args.input_topic,
            self._on_image,
            10,
        )

        self.create_timer(1.0 / max(args.target_fps, 1e-6), self._process_latest_frame)

        self.backend = args.backend
        if self.backend == "pytorch":
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
        else:
            self.model = None
            self.processor = None

        self.get_logger().info(
            f"Benchmark started. backend={args.backend} input_topic={args.input_topic} "
            f"prompt={args.prompt} target_fps={args.target_fps}"
        )

    def _on_image(self, msg: RosImage) -> None:
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.latest_msg = msg
        self.latest_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
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

    def _process_latest_frame(self) -> None:
        if self.latest_rgb is None:
            return

        rgb = self.latest_rgb.copy()
        t0 = time.perf_counter()
        try:
            if self.backend == "pytorch":
                mask = self._run_pytorch_inference(rgb)
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
    parser.add_argument("--backend", choices=["pytorch", "onnx_server"], default="pytorch")
    parser.add_argument("--checkpoint", default=None, help="Required for pytorch backend")
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
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--target-fps", type=float, default=5.0)
    parser.add_argument("--report-interval-sec", type=float, default=5.0)
    args = parser.parse_args()

    if args.backend == "pytorch" and not args.checkpoint:
        parser.error("--checkpoint is required when --backend pytorch")
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
