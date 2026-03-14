# 最終目標
SAM3を軽量化し推論速度を高速化させること．具体的には15~30FPSでの推論ができれば完璧．
推論時は都度異なるテキストプロンプトでも本家SAM3と同様の精度が出せること．

# 現状
## encoderのonnxエクスポートは成功済み．精度は良くない．
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
用途ごとに入口が違う。

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

