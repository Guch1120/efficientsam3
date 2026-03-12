#!/usr/bin/env python3
"""Run exported decoder ONNX on saved encoder embedding (.npy)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run EfficientSAM3 decoder ONNX")
    p.add_argument("--model", required=True, help="decoder onnx")
    p.add_argument("--input", required=True, help="image_embed .npy")
    p.add_argument("--output", required=True, help="output .npz")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    x = np.load(a.input)
    if x.dtype != np.float32:
        x = x.astype(np.float32)

    sess = ort.InferenceSession(a.model, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    outs = sess.run(None, {inp: x})
    names = [o.name for o in sess.get_outputs()]
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.output, **{k: v for k, v in zip(names, outs)})
    print("Saved:", a.output)
    for k, v in zip(names, outs):
        print(k, v.shape, v.dtype)


if __name__ == "__main__":
    main()
