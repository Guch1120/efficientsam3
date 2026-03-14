#!/usr/bin/env python3
"""Send one image to the ONNX encoder server text endpoint and save mask.png."""

from __future__ import annotations

from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import argparse
import io
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send one image to /segment_text and save returned mask.png"
    )
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument("--prompt", required=True, help="Text prompt (e.g. person)")
    parser.add_argument(
        "--server",
        default="http://127.0.0.1:18080",
        help="Base URL of onnx_encoder_server",
    )
    parser.add_argument("--output", default="mask.png", help="Output mask PNG path")
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image = Image.open(args.image).convert("RGB")
    image_np = np.asarray(image, dtype=np.uint8)

    buf = io.BytesIO()
    np.save(buf, image_np, allow_pickle=False)

    prompt = quote(args.prompt, safe="")
    req = Request(
        url=f"{args.server.rstrip('/')}/segment_text?prompt={prompt}",
        data=buf.getvalue(),
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=args.timeout_sec) as resp:
            mask = np.load(io.BytesIO(resp.read()), allow_pickle=False)
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"server returned HTTP {exc.code}")
        if body:
            print(body)
        raise

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(out_path)
    print(f"saved mask to: {out_path}")
    print(f"mask shape: {mask.shape}")


if __name__ == "__main__":
    main()
