# EfficientSAM3 How To Use

このファイルは、現状のこのリポジトリで何をどう使えばよいかを、用途別に整理したメモです。

対象のゴール:

- まず 1 枚画像でテキストプロンプト推論を動かす
- 次に ONNX を使った経路を試す
- 最後に ROS2 + Realsense の画像トピックで速度評価する

## 0. 先に結論

最短ルートはこの順番です。

1. PyTorch で 1 枚画像のテキストプロンプト推論を通す
2. encoder を ONNX 化する
3. ONNX server 経由でテキストプロンプト推論を通す
4. ROS2 で Realsense の画像トピックから FPS と遅延を測る

## 1. ONNX とは何か

このリポジトリでの ONNX は、「モデルを別の推論ランタイムで実行しやすくするための形式」です。

重要:

- ONNX 化すると必ずモデルサイズが小さくなる、という意味ではない
- 主な狙いは推論実行の高速化や運用のしやすさ
- VRAM が減ることはあるが、常に減るとは限らない

このリポジトリでは、今まず ONNX 化して価値が高いのは画像 encoder です。
理由は、encoder が一番重い計算ブロックだからです。

## 2. 何が ONNX 化されていて、何がまだ PyTorch なのか

### `export_efficientsam3_onnx.py`

- 画像 encoder を ONNX 化する
- 今あなたが実行して成功したのはこれ
- 出力は最終マスクではなく `image_embed`

### `export_efficientsam3_decoder_onnx.py`

- encoder 出力を受ける後段を ONNX 化する
- 単体でテキストプロンプト推論になるわけではない

### `export_efficientsam3_text_segment_onnx.py`

- 固定文字列の text prompt を焼き込んだ ONNX を作る
- 例: `"person"` 固定
- 毎回違う prompt を入れたい用途には向かない

### つまり今の理解

- 可変テキストプロンプトを使いたい
  - encoder は ONNX
  - text encoder / grounding は PyTorch
- 固定 prompt でよい
  - text 側も ONNX 化できる

## 3. まず最初にやること

まずは PyTorch で 1 枚画像に対してテキストプロンプト推論を通してください。

使うスクリプト:

- [sam3/efficientsam3_examples/save_text_prompt_mask.py](/home/guch1/ssd_yamaguchi/piper_ros/efficientsam3/sam3/efficientsam3_examples/save_text_prompt_mask.py)

実行例:

```bash
python sam3/efficientsam3_examples/save_text_prompt_mask.py \
  --checkpoint /ros2_ws/efficientsam3/efficient_sam3_tinyvit_21m_mobileclip_s1.pth \
  --image /path/to/input.jpg \
  --prompt "person" \
  --output /tmp/mask.png \
  --backbone-type tinyvit \
  --model-name 21m \
  --text-encoder-type MobileCLIP-S1
```

text prompt 品質を優先して切り分けたい場合は、student text encoder を使わず、
本家 SAM3 の text encoder を残したハイブリッド構成も試してください。

```bash
python sam3/efficientsam3_examples/save_text_prompt_mask.py \
  --checkpoint /ros2_ws/efficientsam3/efficient_sam3_tinyvit_21m_mobileclip_s1.pth \
  --image /path/to/input.jpg \
  --prompt "children" \
  --output /tmp/mask.png \
  --backbone-type tinyvit \
  --model-name 21m \
  --use-teacher-text-encoder \
  --confidence-threshold 0.1
```

このモードの意味:

- visual encoder は EfficientSAM3
- text encoder は本家 SAM3
- まず text prompt 品質の劣化原因が student text encoder 側かどうかを切り分ける

これで確認すること:

- prompt に対して mask がちゃんと出るか
- どのクラス名が効きやすいか
- 精度がだいたい期待通りか

## 4. encoder を ONNX 化する

使うスクリプト:

- [sam3/scripts/export_efficientsam3_onnx.py](/home/guch1/ssd_yamaguchi/piper_ros/efficientsam3/sam3/scripts/export_efficientsam3_onnx.py)

実行例:

```bash
python sam3/scripts/export_efficientsam3_onnx.py \
  --checkpoint /ros2_ws/efficientsam3/efficient_sam3_tinyvit_21m_mobileclip_s1.pth \
  --backbone-type tinyvit \
  --model-name 21m \
  --output /tmp/efficientsam3_encoder_tinyvit_21m.onnx \
  --dynamic-batch \
  --opset 18
```

成功すると:

```bash
Exported ONNX model: /tmp/efficientsam3_encoder_tinyvit_21m.onnx
```

が出ます。

## 5. ONNX server で使う

### 5-1. encoder だけ起動する場合

これは埋め込みを返すだけです。

```bash
python sam3/scripts/onnx_encoder_server.py \
  --model /tmp/efficientsam3_encoder_tinyvit_21m.onnx \
  --host 0.0.0.0 \
  --port 18080
```

この状態で使えるのは:

- `POST /encode`

だけです。

### 5-2. 可変 text prompt で推論したい場合

これが実用上の本命です。

```bash
python sam3/scripts/onnx_encoder_server.py \
  --model /tmp/efficientsam3_encoder_tinyvit_21m.onnx \
  --pytorch-checkpoint /ros2_ws/efficientsam3/efficient_sam3_tinyvit_21m_mobileclip_s1.pth \
  --backbone-type tinyvit \
  --model-name 21m \
  --text-encoder-type MobileCLIP-S1 \
  --host 0.0.0.0 \
  --port 18080
```

この状態で使えるのは:

- `POST /encode`
- `POST /segment_text?prompt=person`

ここでの意味:

- encoder 部分は ONNX Runtime
- text prompt 推論の後段は PyTorch

text prompt 品質を優先するなら、`--text-encoder-type` を付けずに起動すると
本家 SAM3 の text encoder を使えます。

```bash
python sam3/scripts/onnx_encoder_server.py \
  --model /tmp/efficientsam3_encoder_tinyvit_21m.onnx \
  --pytorch-checkpoint /ros2_ws/efficientsam3/efficient_sam3_tinyvit_21m_mobileclip_s1.pth \
  --backbone-type tinyvit \
  --model-name 21m \
  --host 0.0.0.0 \
  --port 18080
```

### 5-3. 1 枚画像を ONNX server に投げて `mask.png` を作る

使うスクリプト:

- [sam3/scripts/request_text_mask_onnx_server.py](/home/guch1/ssd_yamaguchi/piper_ros/efficientsam3/sam3/scripts/request_text_mask_onnx_server.py)

```bash
python sam3/scripts/request_text_mask_onnx_server.py \
  --image test_image.jpg \
  --prompt "person" \
  --server http://127.0.0.1:18080 \
  --output mask.png
```

## 6. 固定 prompt を ONNX 化したい場合

例えば `"person"` だけを高速に回したいなら、固定 prompt ONNX を作れます。

```bash
python sam3/scripts/export_efficientsam3_text_segment_onnx.py \
  --checkpoint /ros2_ws/efficientsam3/efficient_sam3_tinyvit_21m_mobileclip_s1.pth \
  --backbone-type tinyvit \
  --model-name 21m \
  --text-encoder-type MobileCLIP-S1 \
  --text-prompt "person" \
  --output /tmp/efficientsam3_textseg_person_tinyvit_21m.onnx \
  --dynamic-batch \
  --opset 18
```

注意:

- この ONNX は `"person"` 固定
- 毎回別の文字列に変える用途には向かない

## 7. ROS2 + Realsense で速度評価する

使うスクリプト:

- [ros_wrappers/ros2_efficientsam3_benchmark.py](/home/guch1/ssd_yamaguchi/piper_ros/efficientsam3/ros_wrappers/ros2_efficientsam3_benchmark.py)

まずは PyTorch を基準に測ります。

```bash
python ros_wrappers/ros2_efficientsam3_benchmark.py \
  --backend pytorch \
  --checkpoint /ros2_ws/efficientsam3/efficient_sam3_tinyvit_21m_mobileclip_s1.pth \
  --backbone-type tinyvit \
  --model-name 21m \
  --text-encoder-type MobileCLIP-S1 \
  --prompt "person" \
  --input-topic /camera/color/image_raw \
  --target-fps 5 \
  --report-interval-sec 5
```

次に ONNX server 経路を測ります。

事前に server を起動:

```bash
python sam3/scripts/onnx_encoder_server.py \
  --model /tmp/efficientsam3_encoder_tinyvit_21m.onnx \
  --pytorch-checkpoint /ros2_ws/efficientsam3/efficient_sam3_tinyvit_21m_mobileclip_s1.pth \
  --backbone-type tinyvit \
  --model-name 21m \
  --text-encoder-type MobileCLIP-S1 \
  --host 0.0.0.0 \
  --port 18080
```

その上で benchmark:

```bash
python ros_wrappers/ros2_efficientsam3_benchmark.py \
  --backend onnx_server \
  --server http://127.0.0.1:18080 \
  --prompt "person" \
  --input-topic /camera/color/image_raw \
  --target-fps 5 \
  --report-interval-sec 5
```

ログで見る値:

- `input_fps`
  - カメラから入ってきた速度
- `processed_fps`
  - 実際に推論できた速度
- `avg_latency_ms`
  - 平均推論時間
- `max_latency_ms`
  - 最大推論時間

## 8. どの使い方を選ぶべきか

### 1 枚画像でまず試したい

```bash
python sam3/efficientsam3_examples/save_text_prompt_mask.py ...
```

### 本家 SAM3 と EfficientSAM3 の差を数値で見たい

```bash
python sam3/scripts/compare_text_prompt_models.py \
  --image test_image.jpg \
  --prompt "children" \
  --eff-checkpoint /ros2_ws/efficientsam3/efficient_sam3_tinyvit_21m_mobileclip_s1.pth \
  --confidence-threshold 0.1 \
  --backbone-type tinyvit \
  --model-name 21m \
  --text-encoder-type MobileCLIP-S1
```

この比較では次の 3 つを同条件で出します。

- `sam3`
- `efficientsam3_student_text`
- `efficientsam3_teacher_text`

### 可変 text prompt で少しでも高速化したい

```bash
python sam3/scripts/onnx_encoder_server.py ... --pytorch-checkpoint ...
python sam3/scripts/request_text_mask_onnx_server.py ...
```

### 固定 prompt だけでよい

```bash
python sam3/scripts/export_efficientsam3_text_segment_onnx.py ...
```

### Realsense のトピックで実測したい

```bash
python ros_wrappers/ros2_efficientsam3_benchmark.py ...
```

## 9. 現時点のおすすめ

現時点では、以下をおすすめします。

1. `save_text_prompt_mask.py` で 1 枚画像推論を確認
2. うまく取れない場合は `--use-teacher-text-encoder --confidence-threshold 0.1` で切り分け
3. encoder ONNX を export
4. `onnx_encoder_server.py --pytorch-checkpoint ...` で可変 text prompt 推論
5. `ros2_efficientsam3_benchmark.py` で PyTorch と ONNX server を比較

理由:

- 可変 text prompt が必要なら、この経路が一番わかりやすい
- encoder は重いので ONNX 化の恩恵を受けやすい
- text 側まで完全 ONNX にするのは、今の実装では固定 prompt 向け
