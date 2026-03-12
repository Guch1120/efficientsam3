#!/usr/bin/env python3
"""ROS2 wrapper node for EfficientSAM3 image segmentation.

Behavior:
- Subscribes image topic and caches the latest frame.
- Runs inference only when a request message arrives.
- Publishes mask as mono8 image.

Request message type: std_msgs/msg/String
- empty string: use --text-prompt if provided, otherwise center-point prompt.
- non-empty string: use message text as prompt.
"""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

import argparse

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from PIL import Image
from rclpy.node import Node
from sensor_msgs.msg import Image as RosImage
from std_msgs.msg import String

from efficientsam.model_builder import build_efficientsam3_image_model
from efficientsam.sam3_image_processor import Sam3Processor


class EfficientSam3Ros2Node(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("efficientsam3_ros2")
        self.bridge = CvBridge()
        self.device = args.device

        self.model = build_efficientsam3_image_model(
            checkpoint_path=args.checkpoint,
            backbone_type=args.backbone_type,
            model_name=args.model_name,
            enable_inst_interactivity=True,
            enable_segmentation=True,
            compile=args.compile,
            device=self.device,
            text_encoder_type=args.text_encoder_type,
        )
        self.processor = Sam3Processor(self.model)

        self.point_x = args.point_x
        self.point_y = args.point_y
        self.default_text_prompt = args.text_prompt

        self.latest_msg: RosImage | None = None
        self.latest_rgb: np.ndarray | None = None

        self.image_sub = self.create_subscription(
            RosImage,
            args.input_topic,
            self._on_image,
            10,
        )
        self.request_sub = self.create_subscription(
            String,
            args.request_topic,
            self._on_request,
            10,
        )
        self.pub = self.create_publisher(RosImage, args.output_topic, 10)

        self.get_logger().info(
            f"EfficientSAM3 ROS2 node initialized. image_topic={args.input_topic} "
            f"request_topic={args.request_topic} output_topic={args.output_topic}"
        )

    def _on_image(self, msg: RosImage) -> None:
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.latest_msg = msg
        self.latest_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _on_request(self, req: String) -> None:
        if self.latest_rgb is None or self.latest_msg is None:
            self.get_logger().warning("No image cached yet. Skip request.")
            return

        rgb = self.latest_rgb
        pil_img = Image.fromarray(rgb)

        h, w = rgb.shape[:2]
        x = self.point_x if self.point_x is not None else (w / 2.0)
        y = self.point_y if self.point_y is not None else (h / 2.0)
        point_coords = np.array([[x, y]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

        text_prompt = req.data.strip() if req.data else ""
        if not text_prompt:
            text_prompt = self.default_text_prompt or ""

        with torch.inference_mode():
            inference_state = self.processor.set_image(pil_img)
            if text_prompt:
                inference_state = self.processor.set_text_prompt(
                    prompt=text_prompt,
                    state=inference_state,
                )
                masks = inference_state["masks"]
                scores = inference_state["scores"]
            else:
                masks, scores, _ = self.model.predict_inst(
                    inference_state,
                    point_coords=point_coords,
                    point_labels=point_labels,
                    multimask_output=True,
                )

        best_idx = int(np.argmax(scores))
        mask = masks[best_idx]
        mask_u8 = (mask > 0).astype(np.uint8) * 255

        out_msg = self.bridge.cv2_to_imgmsg(mask_u8, encoding="mono8")
        out_msg.header = self.latest_msg.header
        self.pub.publish(out_msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EfficientSAM3 ROS2 wrapper node")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--backbone-type",
        default="efficientvit",
        choices=["efficientvit", "repvit", "tinyvit"],
    )
    parser.add_argument("--model-name", default="b0")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--input-topic", default="/camera/color/image_raw")
    parser.add_argument("--request-topic", default="/efficientsam3/request")
    parser.add_argument("--output-topic", default="/efficientsam3/mask")
    parser.add_argument("--point-x", type=float, default=None)
    parser.add_argument("--point-y", type=float, default=None)
    parser.add_argument("--text-prompt", default=None, help="default prompt (e.g. person)")
    parser.add_argument(
        "--text-encoder-type",
        default=None,
        help="Set student text encoder (e.g. MobileCLIP-S0/MobileCLIP-S1/MobileCLIP2-L)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = EfficientSam3Ros2Node(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
