# 実行ログ

## 2026-03-14
- `efficient_sam3_tinyvit_21m_mobileclip_s1.pth` の encoder ONNX export は成功
- `onnx_encoder_server.py` の `/segment_text` で 0 件検出時に 400 になっていたため、空マスク返却へ修正
- `Sam3Processor` の device 未指定により CPU/GPU 不一致が発生していたため、各呼び出し側で device を明示
- `children` prompt で本家 SAM3 は検出できる一方、EfficientSAM3 merged checkpoint は空マスク
- README を確認し、Stage 1 merged text encoder の性能低下は既知であることを確認
- 次の検証として、student text encoder を使わず本家 SAM3 text encoder を残したハイブリッド経路を優先する方針に変更
- `--use-teacher-text-encoder --confidence-threshold 0.1` でも `children` は空マスク
- student text encoder だけでなく、現状の EfficientSAM3 visual encoder と本家 grounding 系の整合も課題の可能性が高い
- 次段の診断用に `compare_text_prompt_models.py` を追加
- GPU 上で `compare_text_prompt_models.py` を実行し、`children` / threshold `0.1` で以下を確認
- `sam3`: detections=8, top_score=0.9580
- `efficientsam3_student_text`: detections=0
- `efficientsam3_teacher_text`: detections=0
- 現行 Stage 1 TinyViT-21M merged checkpoint では、teacher text encoder に戻しても本家 SAM3 相当の text grounding は再現できない
