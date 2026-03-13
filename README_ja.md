# EfficientSAM3 日本語ガイド

このファイルは、ルート [`README.md`](/home/guch1/ssd_yamaguchi/piper_ros/efficientsam3/README.md) の内容をもとにした実用向けの日本語ガイドです。原著の英語 README はそのまま残し、日本語では導入、推論、ONNX エクスポート、テキストプロンプト利用時の注意点に絞って整理しています。

## 概要

EfficientSAM3 は、SAM1 / SAM2 / SAM3 の能力を軽量バックボーンへ蒸留し、モバイル・組み込み・省 VRAM 環境でも動かしやすくした派生実装です。

- 画像バックボーンは `EfficientViT` / `RepViT` / `TinyViT`
- テキストエンコーダは `MobileCLIP-S0` / `MobileCLIP-S1` / `MobileCLIP2-L`
- 目的は「SAM3 に近い Promptable Concept Segmentation を、より小さい計算量と VRAM で動かす」こと

## できること

- 画像に対する軽量なセグメンテーション推論
- テキストプロンプトを使った物体領域の抽出
- student encoder の ONNX エクスポート
- encoder 出力を受ける decoder / text-conditioned downstream path の ONNX エクスポート

## インストール

基本の学習・推論環境:

```bash
git clone https://github.com/SimonZeng7108/efficientsam3.git
cd efficientsam3

conda create -n efficientsam3 python=3.12 -y
conda activate efficientsam3

pip install --upgrade pip

# CUDA の場合
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# ルート pyproject の依存を導入
pip install -e ".[stage1]"
```

ONNX エクスポートも行う場合は、追加で ONNX 依存を入れてください。

```bash
pip install -e ".[stage1,onnx]"
```

`README.md` の基本インストールだけでは、`onnx` / `onnxscript` / `onnxruntime` は入らない点に注意してください。

## Docker 開発環境

このリポジトリでは Docker 実行も用意されています。

```bash
bash scripts/start_dev_env.sh
docker exec -it efficientsam3-dev bash
```

コンテナ内では必要に応じて:

```bash
python3 -m pip install -e ".[dev,stage1,onnx]"
```

## 推論の基本

画像推論の入口は `build_efficientsam3_image_model` と `Sam3Processor` です。

マージ済みチェックポイントを使うときは、少なくとも次を一致させてください。

- `checkpoint_path`
- `backbone_type`
- `model_name`
- `text_encoder_type`（テキストプロンプトを使う場合）

例:

```python
from efficientsam.model_builder import build_efficientsam3_image_model
from efficientsam.sam3_image_processor import Sam3Processor

model = build_efficientsam3_image_model(
    checkpoint_path="efficient_sam3_tinyvit_21m_mobileclip_s1.pth",
    backbone_type="tinyvit",
    model_name="21m",
    text_encoder_type="MobileCLIP-S1",
    enable_inst_interactivity=False,
    eval_mode=True,
)

processor = Sam3Processor(model)
```

## ONNX エクスポートはどれを使うべきか

用途ごとに入口が違います。

### 1. まず速度と VRAM を改善したい

最初に試すべきは encoder export です。

```bash
python sam3/scripts/export_efficientsam3_onnx.py \
  --checkpoint /absolute/path/to/efficient_sam3_tinyvit_21m_mobileclip_s1.pth \
  --backbone-type tinyvit \
  --model-name 21m \
  --output /tmp/efficientsam3_encoder_tinyvit_21m.onnx \
  --dynamic-batch \
  --opset 18
```

これは student の画像 encoder を ONNX 化します。README にもある通り、ここが支配的な計算ブロックなので、まずここを切り出すのが最も現実的です。

### 2. encoder 出力の後段も分離したい

decoder neck だけを ONNX 化する場合:

```bash
python sam3/scripts/export_efficientsam3_decoder_onnx.py \
  --checkpoint /absolute/path/to/efficient_sam3_tinyvit_21m_mobileclip_s1.pth \
  --backbone-type tinyvit \
  --model-name 21m \
  --output /tmp/efficientsam3_decoder_tinyvit_21m.onnx \
  --dynamic-batch \
  --opset 18
```

このスクリプトは `image_embed` を入力に取るため、単体ではテキストプロンプト推論になりません。

### 3. テキストプロンプトで物体を取りたい

現状の ONNX 導線には 2 パターンあります。

- 可変プロンプトを使いたい
  - `onnx_encoder_server.py` を `--pytorch-checkpoint` 付きで起動する
  - encoder は ONNX Runtime、テキスト処理と grounding は PyTorch
- 固定プロンプトで良い
  - `export_efficientsam3_text_segment_onnx.py` を使う
  - プロンプト文字列は export 時にグラフへ焼き込まれる

固定プロンプト ONNX の例:

```bash
python sam3/scripts/export_efficientsam3_text_segment_onnx.py \
  --checkpoint /absolute/path/to/efficient_sam3_tinyvit_21m_mobileclip_s1.pth \
  --backbone-type tinyvit \
  --model-name 21m \
  --text-encoder-type MobileCLIP-S1 \
  --text-prompt "person" \
  --output /tmp/efficientsam3_textseg_person_tinyvit_21m.onnx \
  --dynamic-batch \
  --opset 18
```

重要:

- `export_efficientsam3_text_segment_onnx.py` は固定プロンプト専用です
- 毎回違う自然言語プロンプトを投げたいなら、現状は PyTorch 経路を併用するのが実装上の正式ルートです

## 可変テキストプロンプトの実行方法

encoder を ONNX 化しつつ、プロンプトは毎回変えたい場合:

```bash
python sam3/scripts/onnx_encoder_server.py \
  --model /tmp/efficientsam3_encoder_tinyvit_21m.onnx \
  --pytorch-checkpoint /absolute/path/to/efficient_sam3_tinyvit_21m_mobileclip_s1.pth \
  --backbone-type tinyvit \
  --model-name 21m \
  --text-encoder-type MobileCLIP-S1 \
  --host 0.0.0.0 \
  --port 18080
```

この場合:

- `POST /encode` は encoder ONNX を使う
- `POST /segment_text?prompt=person` は PyTorch モデルを使ってテキストプロンプト推論する

つまり、現状のリポジトリで「可変テキストプロンプト + 完全 ONNX」はまだ完成形ではありません。

## よくあるエラー

### `ModuleNotFoundError: No module named 'onnx'`

```bash
pip install -e ".[onnx]"
```

### `ModuleNotFoundError: No module named 'onnxscript'`

```bash
pip install -e ".[onnx]"
```

### `ModuleNotFoundError: No module named 'einops'`

```bash
pip install -e ".[stage1]"
```

### チェックポイントと指定アーキテクチャが不一致

例えば `efficient_sam3_tinyvit_21m_mobileclip_s1.pth` を使うなら、少なくとも以下を一致させてください。

```bash
--backbone-type tinyvit
--model-name 21m
--text-encoder-type MobileCLIP-S1
```

## 最終目標に対する推奨ルート

「推論速度が速い」「VRAM 使用量が小さい」「テキストプロンプトで物体検出したい」が目標なら、まずは次の順序が堅実です。

1. `tinyvit_21m + MobileCLIP-S1` か `efficientvit_b0` 系を選ぶ
2. `benchmark_inference_optimizations.py --preset vram8` で CUDA 実測を取る
3. encoder を `export_efficientsam3_onnx.py` で ONNX 化する
4. 可変テキストプロンプトが必要なら、当面は `onnx_encoder_server.py --pytorch-checkpoint ...` を使う
5. 固定クラスだけ高速化したいなら `export_efficientsam3_text_segment_onnx.py` で固定プロンプト ONNX を作る

## ベンチマーク

```bash
python sam3/scripts/benchmark_inference_optimizations.py \
  --checkpoint /absolute/path/to/efficient_sam3_tinyvit_21m_mobileclip_s1.pth \
  --backbone-type tinyvit \
  --model-name 21m \
  --preset vram8 \
  --vram-budget-gb 8
```

この結果を基準に、PyTorch eager / compile / AMP / channels_last と ONNX Runtime を比較するのがよいです。
