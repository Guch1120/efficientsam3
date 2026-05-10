#!/usr/bin/env python3
"""
EdgeSAM 風の prompt-in-the-loop 蒸留を text prompt 向けに回す学習スクリプト。
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from stage1_geometry_finetune.data.coco_text_prompt_dataset import COCOTextPromptDataset
from stage1_geometry_finetune.text_prompt_losses import compute_text_prompt_distill_loss
from stage1_geometry_finetune.text_prompt_model import TextPromptFinetuneModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Text-prompt finetuning for EfficientSAM3")
    parser.add_argument("--data-root", required=True, help="COCO root directory")
    parser.add_argument(
        "--sam3-checkpoint",
        required=True,
        help="Full SAM3 teacher checkpoint. EfficientSAM3 merged checkpoints are not supported here.",
    )
    parser.add_argument("--stage1-checkpoint", default=None)
    parser.add_argument(
        "--student-text-encoder-type",
        default="MobileCLIP-S1",
        help="Text encoder type used by the EfficientSAM3 student runtime.",
    )
    parser.add_argument(
        "--student-backbone",
        required=True,
        help="Student trunk name such as tiny_vit_21m or repvit_m2_3",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val"],
        help="COCO split to use for this run.",
    )
    parser.add_argument("--img-size", type=int, default=1008)
    parser.add_argument("--embed-dim", type=int, default=1024)
    parser.add_argument("--embed-size", type=int, default=72)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--embedding-weight", type=float, default=0.0015)
    parser.add_argument("--score-weight", type=float, default=1.0)
    parser.add_argument("--gt-mask-weight", type=float, default=0.5)
    parser.add_argument("--box-weight", type=float, default=0.2)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument(
        "--disable-prompt-variants",
        action="store_true",
        help="Use only the raw COCO category name as text prompt.",
    )
    parser.add_argument(
        "--train-neck",
        action="store_true",
        help="Also fine-tune the lightweight neck convolutions on top of the student trunk.",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = COCOTextPromptDataset(
        data_root=args.data_root,
        img_size=args.img_size,
        split=args.split,
        num_samples=args.num_samples,
        use_prompt_variants=not args.disable_prompt_variants,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )

    model = TextPromptFinetuneModel(
        student_backbone_name=args.student_backbone,
        sam3_checkpoint_path=args.sam3_checkpoint,
        stage1_checkpoint_path=args.stage1_checkpoint,
        embed_dim=args.embed_dim,
        embed_size=args.embed_size,
        img_size=args.img_size,
        train_neck=args.train_neck,
        student_text_encoder_type=args.student_text_encoder_type,
    )
    _load_stage1_student_weights(model, args.stage1_checkpoint)
    model.to(args.device)

    optimizer = torch.optim.AdamW(
        list(model.get_trainable_parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and args.device == "cuda")

    history: list[dict] = []
    for epoch in range(args.epochs):
        metrics = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
        )
        metrics["epoch"] = epoch
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False))

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = output_dir / f"text_prompt_finetune_epoch_{epoch + 1}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.get_finetune_state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "metrics": metrics,
                },
                ckpt_path,
            )

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def train_one_epoch(
    model: TextPromptFinetuneModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    args: argparse.Namespace,
) -> Dict[str, float]:
    model.train()
    running: Dict[str, float] = {}
    count = 0
    start_time = time.perf_counter()
    progress = tqdm(
        loader,
        desc="train",
        leave=True,
        dynamic_ncols=True,
    )

    for batch in progress:
        images = batch["image"].to(args.device, non_blocking=True)
        gt_masks = batch["gt_mask"].to(args.device, non_blocking=True)
        gt_boxes_cxcywh = batch["gt_box_cxcywh"].to(args.device, non_blocking=True)
        img_size_before_pad = batch["img_size_before_pad"].to(args.device, non_blocking=True)
        prompts = list(batch["prompt_text"])

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=args.amp and args.device == "cuda"):
            with torch.no_grad():
                teacher_outputs, teacher_embedding = model.forward_teacher(images, prompts)
            student_outputs, student_embedding = model.forward_student(images, prompts)
            loss, loss_dict = compute_text_prompt_distill_loss(
                student_outputs=student_outputs,
                teacher_outputs=teacher_outputs,
                gt_masks=gt_masks,
                gt_boxes_cxcywh=gt_boxes_cxcywh,
                img_size_before_pad=img_size_before_pad,
                student_embedding=student_embedding,
                teacher_embedding=teacher_embedding,
                embedding_weight=args.embedding_weight,
                score_weight=args.score_weight,
                gt_mask_weight=args.gt_mask_weight,
                box_weight=args.box_weight,
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        count += 1
        for key, value in loss_dict.items():
            running[key] = running.get(key, 0.0) + float(value.item())
        elapsed = time.perf_counter() - start_time
        progress.set_postfix(
            loss=f"{float(loss.item()):.4f}",
            avg=f"{running['loss_total'] / count:.4f}",
            step=count,
            elapsed=f"{elapsed:.0f}s",
        )

    progress.close()

    return {key: value / max(count, 1) for key, value in running.items()}


def _load_stage1_student_weights(
    model: TextPromptFinetuneModel,
    checkpoint_path: str | None,
) -> None:
    if not checkpoint_path:
        return
    if getattr(model, "student_model", None) is not None:
        print("Stage1 checkpoint is already loaded through student_model; skip manual trunk load")
        return

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    if "student_trunk" in state_dict:
        state_dict = state_dict["student_trunk"]
    trunk_state: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("student_trunk."):
            trunk_state[key[len("student_trunk."):]] = value
        elif key.startswith("image_encoder."):
            trunk_state[key[len("image_encoder."):]] = value
        elif key.startswith("detector.backbone.vision_backbone.trunk.model."):
            mapped_key = key[len("detector.backbone.vision_backbone.trunk.model."):]
            trunk_state[f"backbone.{mapped_key}"] = value
    if not trunk_state:
        print(f"Warning: no student trunk weights found in {checkpoint_path}")
        return

    missing_keys, unexpected_keys = model.student_trunk.load_state_dict(
        trunk_state, strict=False
    )
    print(
        "Loaded Stage1 weights:",
        f"missing={len(missing_keys)}",
        f"unexpected={len(unexpected_keys)}",
    )


if __name__ == "__main__":
    main()
