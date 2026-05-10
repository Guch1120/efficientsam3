"""
Text prompt 条件付き蒸留用のモデル。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from sam3.model.data_misc import FindStage


class TextPromptFinetuneModel(nn.Module):
    """軽量 trunk を text grounding 条件で微調整する。"""

    def __init__(
        self,
        student_backbone_name: str,
        sam3_checkpoint_path: str,
        stage1_checkpoint_path: str | None = None,
        embed_dim: int = 1024,
        embed_size: int = 72,
        img_size: int = 1008,
        train_neck: bool = False,
        student_text_encoder_type: str = "MobileCLIP-S1",
    ) -> None:
        super().__init__()
        if sam3_checkpoint_path is None:
            raise ValueError("sam3_checkpoint_path is required")

        from sam3.model_builder import build_sam3_image_model
        from efficientsam.model_builder import build_efficientsam3_image_model

        self.student_model = build_efficientsam3_image_model(
            checkpoint_path=stage1_checkpoint_path,
            load_from_HF=False,
            eval_mode=True,
            device="cpu",
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
            backbone_type=self._infer_backbone_type(student_backbone_name),
            model_name=self._infer_model_name(student_backbone_name),
            text_encoder_type=student_text_encoder_type,
        )
        self.student_trunk = deepcopy(self.student_model.backbone.vision_backbone.trunk)
        self.sam3 = build_sam3_image_model(
            checkpoint_path=sam3_checkpoint_path,
            load_from_HF=False,
            eval_mode=True,
            device="cpu",
            enable_segmentation=True,
            enable_inst_interactivity=False,
            compile=False,
            enable_text_encoder=True,
        )
        for param in self.sam3.parameters():
            param.requires_grad = False
        self.sam3.eval()
        for param in self.student_model.parameters():
            param.requires_grad = False
        self.student_model.eval()

        self.embed_size = embed_size
        self.img_size = img_size
        self.train_neck = train_neck

        vision_backbone = self.student_model.backbone.vision_backbone
        scalp = int(getattr(self.student_model.backbone, "scalp", 0))
        active_convs = list(vision_backbone.convs)
        if scalp > 0:
            active_convs = active_convs[:-scalp]
        self.frozen_convs = nn.ModuleList(active_convs)
        self.position_encoding = vision_backbone.position_encoding
        for param in self.frozen_convs.parameters():
            param.requires_grad = train_neck

    def forward_student_backbone(self, images: torch.Tensor) -> torch.Tensor:
        student_embedding = self.student_trunk(images)
        if isinstance(student_embedding, (list, tuple)):
            return student_embedding[0]
        return student_embedding

    @torch.no_grad()
    def forward_teacher(
        self, images: torch.Tensor, prompts: List[str]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        teacher_embedding = self.sam3.backbone.vision_backbone.trunk(images)[-1]
        backbone_out = self.sam3.backbone.forward_image(images)
        backbone_out.update(self.sam3.backbone.forward_text(prompts, device=images.device))
        outputs = self.sam3.forward_grounding(
            backbone_out=backbone_out,
            find_input=self._build_find_stage(images.shape[0], images.device),
            find_target=None,
            geometric_prompt=self.sam3._get_dummy_prompt(images.shape[0]),
        )
        return outputs, teacher_embedding

    def forward_student(
        self, images: torch.Tensor, prompts: List[str]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        student_embedding = self.forward_student_backbone(images)
        backbone_out = self._build_student_backbone_out(student_embedding)
        backbone_out.update(self.student_model.backbone.forward_text(prompts, device=images.device))
        outputs = self.student_model.forward_grounding(
            backbone_out=backbone_out,
            find_input=self._build_find_stage(images.shape[0], images.device),
            find_target=None,
            geometric_prompt=self.student_model._get_dummy_prompt(images.shape[0]),
        )
        return outputs, student_embedding

    def _build_student_backbone_out(
        self, student_embedding: torch.Tensor
    ) -> Dict[str, torch.Tensor | list[torch.Tensor] | None]:
        backbone_fpn: list[torch.Tensor] = []
        vision_pos_enc: list[torch.Tensor] = []
        for conv in self.frozen_convs:
            feat = conv(student_embedding)
            pos = self.position_encoding(feat).to(feat.dtype)
            backbone_fpn.append(feat)
            vision_pos_enc.append(pos)
        return {
            "vision_features": backbone_fpn[-1],
            "vision_pos_enc": vision_pos_enc,
            "backbone_fpn": backbone_fpn,
            "sam2_backbone_out": None,
        }

    def _build_find_stage(self, batch_size: int, device: torch.device) -> FindStage:
        batch_ids = torch.arange(batch_size, device=device, dtype=torch.long)
        return FindStage(
            img_ids=batch_ids,
            text_ids=batch_ids,
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

    def train(self, mode: bool = True):
        super().train(mode)
        self.sam3.eval()
        self.student_model.eval()
        return self

    def get_trainable_parameters(self):
        for param in self.student_trunk.parameters():
            yield param
        if self.train_neck:
            for param in self.frozen_convs.parameters():
                yield param

    def get_finetune_state_dict(self) -> Dict[str, Dict[str, torch.Tensor] | torch.Tensor]:
        """推論変換に必要な学習済み重みをまとめて返す。"""
        state_dict: Dict[str, Dict[str, torch.Tensor] | torch.Tensor] = {
            "student_trunk": self.student_trunk.state_dict()
        }
        if self.train_neck:
            state_dict["student_neck"] = self.frozen_convs.state_dict()
        return state_dict

    @staticmethod
    def _infer_backbone_type(student_backbone_name: str) -> str:
        if student_backbone_name.startswith("repvit"):
            return "repvit"
        if student_backbone_name.startswith("tiny_vit"):
            return "tinyvit"
        if student_backbone_name.startswith("efficientvit"):
            return "efficientvit"
        raise ValueError(f"Unsupported student backbone: {student_backbone_name}")

    @staticmethod
    def _infer_model_name(student_backbone_name: str) -> str:
        if student_backbone_name.startswith("repvit_"):
            return student_backbone_name.replace("repvit_", "").replace("_", ".")
        if student_backbone_name.startswith("tiny_vit_"):
            return student_backbone_name.replace("tiny_vit_", "").replace("_", "")
        if student_backbone_name.startswith("efficientvit_"):
            return student_backbone_name.replace("efficientvit_", "")
        raise ValueError(f"Unsupported student backbone: {student_backbone_name}")
