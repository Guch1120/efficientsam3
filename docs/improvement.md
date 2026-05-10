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
- ONNX server / ROS wrapper でも `confidence-threshold` を調整できるようにする
- ROS2 トピック未起動時にも進められるよう，単画像反復ベンチを追加する
- `onnx_encoder_server.py` が PyTorch 単体より遅い場合は，HTTP サーバ構成を見直し，同一プロセス化または後段 ONNX 化へ進む
- `onnxruntime-gpu` が使えるなら encoder ONNX の速度改善を先に確認し，その後 score calibration か decoder/downstream ONNX 化で精度回復を狙う
- same-process `onnx_local` の再構成は `model.backbone.scalp` を含めて本物の `forward_image()` と一致させる
- `encoder ONNX + PyTorch downstream` が PyTorch より遅い場合は，decoder/downstream ONNX 化を優先し，split 実行の限界を確認する
- full text-seg ONNX が exporter またはメモリ制約で重い場合は，固定 prompt 完全 ONNX を追わず，decoder 分割 export を優先する
- `onnx_split` 実測でなお遅い場合は，ORT 分割構成自体を主戦略から外し，call boundary 削減か TensorRT を検討する
- TensorRT ランタイム導入後は，`onnx_local` と `onnx_split` を CUDA EP と TensorRT EP の両方で比較する
- TensorRT EP で `onnx_local` が PyTorch を超えるなら，まず encoder ONNX + TensorRT を主候補に据える
- 画像側をより大きく fused した `onnx_backbone` も比較し，本当に大きい 1 graph が有利か確認する
- TensorRT では FP16 と engine cache を前提設定にし，その状態を基準値にする
- `tinyvit_21m` が 1008 固定なら，解像度 sweep ではなく lighter backbone sweep を優先する
- `repvit-m2.3` でも 9 FPS 未満なら，さらに小さい backbone か text 側の近似を検討する
- 毎フレーム text prompt 検出だけでなく，SAM3 の tracking API を使った B/C 方式も別スクリプトで追加する
- 方式Cでは，tracking 中に一定周期で text prompt を再実行して drift を補正する
- tracking benchmark は upstream video API を使い，per-frame / text-prompt-then-track / periodic-refresh を別ファイルで比較する
- `TinyViT-11M ft` も追加比較し，`RepViT-M1.1 ft` より良いかを確認する
- fine-tune については 16GB VRAM 前提の縮小条件を明記し，batch 1 / AMP / サンプル数削減を前提に再評価する
- full SAM3 video が遅いなら，EfficientSAM3 検出マスクを SAM3 tracker に渡す hybrid 方式を追加し，object 数と refresh interval の trade-off を測る
- 論文調査では，`MobileSAM` の decoupled distillation と `EdgeSAM` の prompt-in-the-loop distillation が最も既存構成に乗せやすい
- このリポジトリには image/text の Stage 1 蒸留と geometry prompt の Stage 2 があるため，次は text prompt 条件の distillation ループを追加する
- 具体的には，COCO instance のカテゴリ名を text prompt とし，teacher/student の `forward_grounding` を直接合わせる fine-tune 経路を実装する
- 損失は teacher best-query に対する mask BCE/Dice，score MSE，trunk embedding MSE，GT mask 補助損失を組み合わせる
- teacher には merged EfficientSAM3 ではなく full SAM3 checkpoint を使う
