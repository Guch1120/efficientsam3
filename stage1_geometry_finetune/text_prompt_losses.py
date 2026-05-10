"""
Text prompt 条件付き蒸留で使う損失関数。
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from stage1_geometry_finetune.losses import dice_loss, masked_mse_loss, sigmoid_ce_loss


def build_valid_mask(
    img_size_before_pad: torch.Tensor,
    target_hw: Tuple[int, int],
    device: torch.device,
) -> torch.Tensor:
    """パディング領域を無視するための有効画素マスクを作る。"""
    batch_size = img_size_before_pad.shape[0]
    target_h, target_w = target_hw
    valid_mask = torch.zeros(batch_size, 1, target_h, target_w, device=device)
    for index in range(batch_size):
        image_h = int(img_size_before_pad[index, 0].item())
        image_w = int(img_size_before_pad[index, 1].item())
        scaled_h = max(1, round(image_h / 1008.0 * target_h))
        scaled_w = max(1, round(image_w / 1008.0 * target_w))
        valid_mask[index, :, :scaled_h, :scaled_w] = 1.0
    return valid_mask


def select_query_from_teacher(
    teacher_masks: torch.Tensor,
    teacher_scores: torch.Tensor,
    gt_masks: torch.Tensor,
) -> torch.Tensor:
    """GT IoU を使って teacher query を選ぶ。"""
    gt_masks = F.interpolate(
        gt_masks.unsqueeze(1),
        size=teacher_masks.shape[-2:],
        mode="nearest",
    ).squeeze(1)
    teacher_binary = teacher_masks.sigmoid() > 0.5
    gt_binary = gt_masks > 0.5

    intersection = (
        teacher_binary & gt_binary[:, None, :, :]
    ).flatten(2).sum(dim=-1).float()
    union = (
        teacher_binary | gt_binary[:, None, :, :]
    ).flatten(2).sum(dim=-1).float().clamp_min(1.0)
    iou = intersection / union
    fallback_idx = teacher_scores.argmax(dim=1)
    best_iou, best_idx = iou.max(dim=1)
    return torch.where(best_iou > 0.0, best_idx, fallback_idx)


def compute_text_prompt_distill_loss(
    student_outputs: Dict[str, torch.Tensor],
    teacher_outputs: Dict[str, torch.Tensor],
    gt_masks: torch.Tensor,
    gt_boxes_cxcywh: torch.Tensor,
    img_size_before_pad: torch.Tensor,
    student_embedding: torch.Tensor,
    teacher_embedding: torch.Tensor,
    embedding_weight: float,
    score_weight: float,
    gt_mask_weight: float,
    box_weight: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """teacher/student/GT を混ぜて損失を計算する。"""
    student_masks = student_outputs["pred_masks"]
    teacher_masks = teacher_outputs["pred_masks"]

    student_scores = _combine_scores(student_outputs)
    teacher_scores = _combine_scores(teacher_outputs)

    best_idx = select_query_from_teacher(teacher_masks, teacher_scores, gt_masks)
    batch_idx = torch.arange(student_masks.shape[0], device=student_masks.device)

    student_masks_selected = student_masks[batch_idx, best_idx].unsqueeze(1)
    teacher_masks_selected = teacher_masks[batch_idx, best_idx].unsqueeze(1)
    student_scores_selected = student_scores[batch_idx, best_idx]
    teacher_scores_selected = teacher_scores[batch_idx, best_idx]
    student_boxes_selected = student_outputs["pred_boxes"][batch_idx, best_idx]

    resized_gt_masks = F.interpolate(
        gt_masks.unsqueeze(1),
        size=student_masks_selected.shape[-2:],
        mode="nearest",
    )
    valid_mask = build_valid_mask(
        img_size_before_pad=img_size_before_pad,
        target_hw=student_masks_selected.shape[-2:],
        device=student_masks.device,
    )

    mask_bce = sigmoid_ce_loss(
        student_masks_selected,
        teacher_masks_selected,
        valid_mask=valid_mask,
        target_is_logit=True,
    )
    mask_dice = dice_loss(
        student_masks_selected,
        teacher_masks_selected,
        valid_mask=valid_mask,
        target_is_logit=True,
    )
    score_loss = F.mse_loss(student_scores_selected, teacher_scores_selected)
    gt_bce = sigmoid_ce_loss(
        student_masks_selected,
        resized_gt_masks,
        valid_mask=valid_mask,
    )
    gt_dice = dice_loss(
        student_masks_selected,
        resized_gt_masks,
        valid_mask=valid_mask,
    )
    box_loss = F.l1_loss(student_boxes_selected, gt_boxes_cxcywh)
    embed_loss = masked_mse_loss(student_embedding, teacher_embedding)

    total_loss = (
        mask_bce
        + mask_dice
        + score_weight * score_loss
        + embedding_weight * embed_loss
        + gt_mask_weight * (gt_bce + gt_dice)
        + box_weight * box_loss
    )
    loss_dict = {
        "loss_total": total_loss.detach(),
        "loss_mask_bce": mask_bce.detach(),
        "loss_mask_dice": mask_dice.detach(),
        "loss_score": score_loss.detach(),
        "loss_embed": embed_loss.detach(),
        "loss_gt_bce": gt_bce.detach(),
        "loss_gt_dice": gt_dice.detach(),
        "loss_box": box_loss.detach(),
    }
    return total_loss, loss_dict


def _combine_scores(outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
    pred_logits = outputs["pred_logits"].squeeze(-1)
    presence = outputs.get("presence_logit_dec")
    if presence is None:
        return pred_logits.sigmoid()
    if presence.ndim == 1:
        presence = presence[:, None]
    return pred_logits.sigmoid() * presence.sigmoid()
