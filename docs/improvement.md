# 改善実装計画

## 現在の課題
- `efficient_sam3_tinyvit_21m_mobileclip_s1.pth` では、本家 SAM3 で検出できる `children` が空マスクになる
- README 上も Stage 1 merged text encoder の性能低下は想定内と明記されている
- 目標は「可変 text prompt を維持しつつ軽量化・高速化」

## 改善方針
1. student text encoder と本家 SAM3 text encoder を切り替えて比較できるようにする
2. confidence threshold を CLI から調整できるようにする
3. `EfficientSAM3 visual encoder + 本家 SAM3 text encoder` のハイブリッド経路を優先評価する
4. その構成で品質が改善するなら、encoder ONNX + PyTorch downstream の構成で速度評価する

## 直近の実装項目
- `save_text_prompt_mask.py` に `--use-teacher-text-encoder` を追加
- `save_text_prompt_mask.py` に `--confidence-threshold` を追加
- `howtouse.md` にハイブリッド検証手順を追記
- 本家 SAM3 と EfficientSAM3 を同条件で比較する診断スクリプトを追加
