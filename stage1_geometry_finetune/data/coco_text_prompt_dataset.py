"""
COCO の画像・カテゴリ名・マスクを text prompt 微調整用に返すデータセット。
"""

from __future__ import annotations

import copy
import json
import os
from typing import Dict

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

from stage1.data.transforms import ResizeLongestSide


class COCOTextPromptDataset(torch.utils.data.Dataset):
    """COCO instance annotation を text prompt 付きサンプルへ展開する。"""

    def __init__(
        self,
        data_root: str,
        img_size: int = 1008,
        split: str = "train",
        num_samples: int = -1,
        pixel_mean: list[float] = [123.675, 116.28, 103.53],
        pixel_std: list[float] = [58.395, 57.12, 57.375],
        use_prompt_variants: bool = True,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.img_size = img_size
        self.split = split
        self.pixel_mean = torch.tensor(pixel_mean).view(-1, 1, 1)
        self.pixel_std = torch.tensor(pixel_std).view(-1, 1, 1)
        self.transform = ResizeLongestSide(img_size)
        self.num_samples = num_samples
        self.use_prompt_variants = use_prompt_variants
        self.data: list[dict] = []
        self.keys: list[str] = []
        self._prepare_data()

    def _prepare_data(self) -> None:
        anno_path = os.path.join(
            self.data_root, "annotations", f"instances_{self.split}2017.json"
        )
        if not os.path.exists(anno_path):
            raise FileNotFoundError(f"Annotation file not found: {anno_path}")

        with open(anno_path, "r", encoding="utf-8") as f:
            anno_json = json.load(f)

        categories = {
            item["id"]: item["name"].strip().lower()
            for item in anno_json["categories"]
        }
        images = {item["id"]: item for item in anno_json["images"]}

        counter = 0
        for anno in anno_json["annotations"]:
            if anno.get("iscrowd", 0):
                continue
            image_info = images.get(anno["image_id"])
            if image_info is None:
                continue
            category_name = categories.get(anno["category_id"])
            if not category_name:
                continue

            file_name = image_info.get("file_name")
            if file_name is None:
                continue

            prompt_variants = self._build_prompt_variants(category_name)
            for prompt_text in prompt_variants:
                self.data.append(
                    {
                        "image_info": image_info,
                        "annotation": anno,
                        "prompt_text": prompt_text,
                    }
                )
                safe_prompt = prompt_text.replace(" ", "_")
                self.keys.append(f"{anno['image_id']}_{anno['id']}_{safe_prompt}")
                counter += 1
                if self.num_samples > 0 and counter >= self.num_samples:
                    break
            if self.num_samples > 0 and counter >= self.num_samples:
                break

        if not self.data:
            raise RuntimeError("No COCO text-prompt samples were loaded")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        item = copy.deepcopy(self.data[index])
        image_info = item["image_info"]
        anno = item["annotation"]
        prompt_text = item["prompt_text"]

        image_path = self._resolve_image_path(image_info["file_name"])
        image = Image.open(image_path).convert("RGB")
        image = pil_to_tensor(image)
        original_size = image.shape[1:]

        gt_mask = self._decode_mask(anno["segmentation"], image_info["height"], image_info["width"])

        image = self.transform.apply_image_torch(image[None].float()).squeeze(0)
        gt_mask = self.transform.apply_masks_torch(gt_mask[None], original_size).squeeze(0)

        img_size_before_pad = torch.tensor(image.shape[1:], dtype=torch.int32)
        image = self._pad(self._norm(image))
        gt_mask = self._pad(gt_mask)

        return {
            "image": image,
            "prompt_text": prompt_text,
            "gt_mask": gt_mask.float(),
            "gt_box_cxcywh": self._build_gt_box(anno, original_size),
            "img_size_before_pad": img_size_before_pad,
            "key": self.keys[index],
        }

    def get_keys(self) -> list[str]:
        return self.keys

    def _decode_mask(self, segmentation, height: int, width: int) -> torch.Tensor:
        if isinstance(segmentation, list):
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles)
        elif isinstance(segmentation["counts"], list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rle = segmentation
        mask = mask_utils.decode(rle)
        if mask.ndim == 3:
            mask = np.any(mask, axis=2).astype(np.uint8)
        return torch.from_numpy(mask)

    def _build_gt_box(
        self,
        anno: dict,
        original_size: tuple[int, int],
    ) -> torch.Tensor:
        x, y, w, h = [float(v) for v in anno["bbox"]]
        gt_box_xyxy = torch.tensor([[x, y, x + w, y + h]], dtype=torch.float32)
        gt_box_xyxy = self.transform.apply_boxes_torch(gt_box_xyxy, original_size)
        gt_box_xyxy = gt_box_xyxy.squeeze(0)
        x0, y0, x1, y1 = gt_box_xyxy.tolist()
        cx = ((x0 + x1) * 0.5) / self.img_size
        cy = ((y0 + y1) * 0.5) / self.img_size
        box_w = (x1 - x0) / self.img_size
        box_h = (y1 - y0) / self.img_size
        return torch.tensor([cx, cy, box_w, box_h], dtype=torch.float32)

    def _build_prompt_variants(self, category_name: str) -> list[str]:
        if not self.use_prompt_variants:
            return [category_name]

        prompt_variants = [category_name, f"a {category_name}", f"the {category_name}"]
        synonym_map = {
            "person": ["human", "people"],
            "car": ["automobile", "vehicle"],
            "airplane": ["plane", "aircraft"],
            "motorcycle": ["motorbike", "bike"],
            "couch": ["sofa"],
            "tv": ["television", "monitor"],
            "cell phone": ["phone", "mobile phone"],
        }
        for synonym in synonym_map.get(category_name, []):
            prompt_variants.append(synonym)
            prompt_variants.append(f"a {synonym}")

        deduped: list[str] = []
        for prompt_text in prompt_variants:
            if prompt_text not in deduped:
                deduped.append(prompt_text)
        return deduped

    def _norm(self, image: torch.Tensor) -> torch.Tensor:
        return (image - self.pixel_mean) / self.pixel_std

    def _pad(self, tensor: torch.Tensor) -> torch.Tensor:
        height, width = tensor.shape[-2:]
        pad_height = self.img_size - height
        pad_width = self.img_size - width
        return F.pad(tensor, (0, pad_width, 0, pad_height))

    def _resolve_image_path(self, file_name: str) -> str:
        candidates = [
            os.path.join(self.data_root, "trainval", file_name),
            os.path.join(self.data_root, "train2017", file_name),
            os.path.join(self.data_root, "val2017", file_name),
            os.path.join(self.data_root, "images", "train2017", file_name),
            os.path.join(self.data_root, "images", "val2017", file_name),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            f"COCO image not found for {file_name}. Checked: {candidates}"
        )
