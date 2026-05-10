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
- ONNX Runtime に TensorRT Execution Provider を組み合わせると，GPU 向け最適化で速度が上がることがある

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

text prompt が空マスクになる場合は，まず `--confidence-threshold 0.1` あるいは
`0.05` を試してください。現状の EfficientSAM3 Stage 1 重みでは，本家 SAM3 より
score がかなり低く出ることがあります。

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
  --confidence-threshold 0.05 \
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
- 現状の full text-seg ONNX export はかなり重く，環境によっては export 自体が失敗することがある
- 主経路としては，まず encoder / decoder を分割して評価する方が現実的

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
  --confidence-threshold 0.05 \
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
  --confidence-threshold 0.05 \
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

### まず速度だけ見たい

PyTorch 単体:

```bash
python sam3/scripts/benchmark_text_prompt_single_image.py \
  --backend pytorch \
  --image test_image.jpg \
  --prompt "children" \
  --checkpoint /ros2_ws/efficientsam3/efficient_sam3_tinyvit_21m_mobileclip_s1.pth \
  --backbone-type tinyvit \
  --model-name 21m \
  --text-encoder-type MobileCLIP-S1 \
  --confidence-threshold 0.05 \
  --device cuda
```

ONNX server:

```bash
python sam3/scripts/benchmark_text_prompt_single_image.py \
  --backend onnx_server \
  --image test_image.jpg \
  --prompt "children" \
  --server http://127.0.0.1:18080 \
  --confidence-threshold 0.05
```

注意:

- 現在の `onnx_encoder_server.py` は HTTP 通信と PyTorch downstream を含む
- そのため，この経路は必ずしも PyTorch 単体より速くならない
- 現状は「encoder ONNX の動作確認」には使えるが，「最速経路」とは限らない

### 同一プロセス ONNX encoder を試す

```bash
python sam3/scripts/benchmark_text_prompt_single_image.py \
  --backend onnx_local \
  --image test_image.jpg \
  --prompt "children" \
  --encoder-onnx /tmp/efficientsam3_encoder_tinyvit_21m.onnx \
  --checkpoint /ros2_ws/efficientsam3/efficient_sam3_tinyvit_21m_mobileclip_s1.pth \
  --backbone-type tinyvit \
  --model-name 21m \
  --text-encoder-type MobileCLIP-S1 \
  --confidence-threshold 0.05 \
  --device cuda
```

注意:

- `onnxruntime-gpu` が入っていれば encoder 部分は高速化しやすい
- `onnx_local` は `scalp` を反映するよう修正済みで，`children` / `0.05` では PyTorch と同じ 6 件検出まで戻る
- CUDA EP だけだと PyTorch より遅いことがあります
- ただし TensorRT EP を有効化し，さらに FP16 を使うと，単画像ベンチでは PyTorch を明確に上回りました

比較例:

- PyTorch: `140.35 ms`, `7.13 FPS`
- `onnx_local` + CUDA EP: `171.18 ms`, `5.84 FPS`
- `onnx_local` + TensorRT EP: `133.20 ms`, `7.51 FPS`
- `onnx_local` + TensorRT EP + FP16: `117.36 ms`, `8.52 FPS`

TensorRT EP を明示して回す例:

```bash
python sam3/scripts/benchmark_text_prompt_single_image.py \
  --backend onnx_local \
  --ort-provider tensorrt \
  --trt-fp16 \
  --trt-engine-cache \
  --image test_image.jpg \
  --prompt "children" \
  --encoder-onnx /tmp/efficientsam3_encoder_tinyvit_21m.onnx \
  --checkpoint /ros2_ws/efficientsam3/efficient_sam3_tinyvit_21m_mobileclip_s1.pth \
  --backbone-type tinyvit \
  --model-name 21m \
  --text-encoder-type MobileCLIP-S1 \
  --confidence-threshold 0.05 \
  --device cuda
```

### encoder と decoder の両方を ONNX で試す

```bash
python sam3/scripts/benchmark_text_prompt_single_image.py \
  --backend onnx_split \
  --image test_image.jpg \
  --prompt "children" \
  --encoder-onnx /tmp/efficientsam3_encoder_tinyvit_21m.onnx \
  --decoder-onnx /tmp/efficientsam3_decoder_tinyvit_21m.onnx \
  --checkpoint /ros2_ws/efficientsam3/efficient_sam3_tinyvit_21m_mobileclip_s1.pth \
  --backbone-type tinyvit \
  --model-name 21m \
  --text-encoder-type MobileCLIP-S1 \
  --confidence-threshold 0.05 \
  --device cuda
```

注意:

- CUDA EP でも TensorRT EP でも，現状の単画像ベンチでは `onnx_split` は `onnx_local` より遅いです
- つまり encoder と decoder を別々の ORT call に分けると，分割コストが大きいです

### `forward_image()` 全体を 1 ONNX で試す

```bash
python sam3/scripts/export_efficientsam3_backbone_onnx.py \
  --checkpoint /ros2_ws/efficientsam3/efficient_sam3_tinyvit_21m_mobileclip_s1.pth \
  --backbone-type tinyvit \
  --model-name 21m \
  --output /tmp/efficientsam3_backbone_tinyvit_21m.onnx \
  --dynamic-batch \
  --opset 18
```

```bash
python sam3/scripts/benchmark_text_prompt_single_image.py \
  --backend onnx_backbone \
  --ort-provider tensorrt \
  --trt-fp16 \
  --trt-engine-cache \
  --image test_image.jpg \
  --prompt "children" \
  --backbone-onnx /tmp/efficientsam3_backbone_tinyvit_21m.onnx \
  --checkpoint /ros2_ws/efficientsam3/efficient_sam3_tinyvit_21m_mobileclip_s1.pth \
  --backbone-type tinyvit \
  --model-name 21m \
  --text-encoder-type MobileCLIP-S1 \
  --confidence-threshold 0.05 \
  --device cuda
```

注意:

- `onnx_backbone` は encoder と neck を 1 回の ORT call にまとめます
- ただし今回の実測では `onnx_local + TensorRT FP16` の方が速く，必ずしもこちらが最速ではありません

TensorRT ランタイム導入例:

```bash
docker exec piper-humble-dev bash -lc \
  "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
   tensorrt-libs libnvinfer10 libnvinfer-plugin10 libnvonnxparsers10"
```

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
python ros_wrappers/ros2_efficientsam3_benchmark.py \
  --backend onnx_local \
  --encoder-onnx /tmp/efficientsam3_encoder_tinyvit_21m.onnx \
  --checkpoint /ros2_ws/efficientsam3/efficient_sam3_tinyvit_21m_mobileclip_s1.pth \
  --backbone-type tinyvit \
  --model-name 21m \
  --text-encoder-type MobileCLIP-S1 \
  --prompt "children" \
  --confidence-threshold 0.05 \
  --input-topic /camera/color/image_raw \
  --target-fps 5 \
  --report-interval-sec 5
```

比較用に PyTorch も同条件で回します。

```bash
python ros_wrappers/ros2_efficientsam3_benchmark.py ...
```

`onnx_split` を ROS2 で比較したい場合:

```bash
python ros_wrappers/ros2_efficientsam3_benchmark.py \
  --backend onnx_split \
  --encoder-onnx /tmp/efficientsam3_encoder_tinyvit_21m.onnx \
  --decoder-onnx /tmp/efficientsam3_decoder_tinyvit_21m.onnx \
  --checkpoint /ros2_ws/efficientsam3/efficient_sam3_tinyvit_21m_mobileclip_s1.pth \
  --backbone-type tinyvit \
  --model-name 21m \
  --text-encoder-type MobileCLIP-S1 \
  --prompt "children" \
  --confidence-threshold 0.05 \
  --input-topic /camera/color/image_raw \
  --target-fps 5 \
  --report-interval-sec 5
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

## 11. text prompt 品質を改善したい場合

単に backbone を小さくするだけだと、`children` のような可変 text prompt 品質が落ちやすいです。
このリポジトリには geometry prompt 向けの fine-tune はありましたが、text prompt 向けの
prompt-conditioned distillation は無かったので追加しました。

追加した入口:

- [stage1_geometry_finetune/train_text_prompt_finetune.py](/home/guch1/ssd_yamaguchi/piper_ros/efficientsam3/stage1_geometry_finetune/train_text_prompt_finetune.py)
- [stage1_geometry_finetune/scripts/train_text_prompt_finetune.sh](/home/guch1/ssd_yamaguchi/piper_ros/efficientsam3/stage1_geometry_finetune/scripts/train_text_prompt_finetune.sh)

考え方:

- 軽量 student trunk は学習する
- text encoder / grounding / segmentation downstream は frozen teacher SAM3 を使う
- COCO のカテゴリ名を text prompt にして、teacher/student の `forward_grounding` を直接合わせる
- 損失は `mask distillation + score distillation + embedding distillation + GT mask 補助`

実行例:

```bash
python stage1_geometry_finetune/train_text_prompt_finetune.py \
  --data-root /path/to/coco \
  --sam3-checkpoint /path/to/full_sam3_teacher.pth \
  --stage1-checkpoint /path/to/stage1_student.pth \
  --student-backbone tiny_vit_21m \
  --output-dir output/text_prompt_finetune \
  --batch-size 4 \
  --epochs 1 \
  --num-samples 128 \
  --amp \
  --device cuda
```

注意:

- `--sam3-checkpoint` は full SAM3 teacher checkpoint を使ってください
- `efficient_sam3_*_mobileclip_s1.pth` の merged checkpoint を teacher に使うと、この経路では NaN になりました

## 12. 本家 SAM3 を品質基準として比較したい場合

本家 SAM3 の full checkpoint があるなら、同じ単画像ベンチで速度と品質を比較できます。
たとえば今回確認した `/ros2_ws/src/sam3/sam3.pt` は teacher として使えます。

```bash
python sam3/scripts/benchmark_text_prompt_single_image.py \
  --backend sam3 \
  --sam3-checkpoint /ros2_ws/src/sam3/sam3.pt \
  --image /ros2_ws/efficientsam3/test_image.jpg \
  --prompt "children" \
  --confidence-threshold 0.1 \
  --runs 10 \
  --warmup-runs 2 \
  --device cuda
```

この結果と EfficientSAM3 の `onnx_local + TensorRT FP16` を比較すると、
「本家 SAM3 に対してどれだけ速度が上がり、どれだけ score が落ちるか」を同じ条件で見られます。

## 13. 今の最有力モデル

README の model zoo を見ると、`ft` があるのは一部だけです。
現時点で速度と精度のバランスが最も良かったのは次です。

- `efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth`
- 実行構成: `onnx_local + TensorRT FP16`

実測の要点:

- 本家 SAM3 (`children`): `detections=20`, `top_score=0.9580`
- `repvit-m1.1 ft` PyTorch: `detections=16`, `8.87 FPS`
- `repvit-m1.1 ft` `onnx_local + TensorRT FP16`: `detections=15`, `9.24 FPS`

使う checkpoint:

```text
/root/.cache/huggingface/hub/models--Simon7108528--EfficientSAM3/snapshots/08f44d3bcce47ac577962a21608ee934c68b1b45/stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth
```

encoder ONNX export:

```bash
python sam3/scripts/export_efficientsam3_onnx.py \
  --checkpoint /root/.cache/huggingface/hub/models--Simon7108528--EfficientSAM3/snapshots/08f44d3bcce47ac577962a21608ee934c68b1b45/stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth \
  --backbone-type repvit \
  --model-name m1.1 \
  --output /tmp/efficientsam3_encoder_repvit_m1_1_ft.onnx \
  --dynamic-batch \
  --opset 18
```

推奨ベンチ構成:

```bash
python sam3/scripts/benchmark_text_prompt_single_image.py \
  --backend onnx_local \
  --ort-provider tensorrt \
  --trt-fp16 \
  --trt-engine-cache \
  --cache-text-prompt \
  --checkpoint /root/.cache/huggingface/hub/models--Simon7108528--EfficientSAM3/snapshots/08f44d3bcce47ac577962a21608ee934c68b1b45/stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth \
  --encoder-onnx /tmp/efficientsam3_encoder_repvit_m1_1_ft.onnx \
  --image /ros2_ws/efficientsam3/test_image.jpg \
  --prompt "children" \
  --backbone-type repvit \
  --model-name m1.1 \
  --text-encoder-type MobileCLIP-S1 \
  --confidence-threshold 0.05 \
  --device cuda
```

`--cache-text-prompt` は、同じ prompt を繰り返すときだけ有効です。
今回の実測では:

- 通常: `9.17 FPS`
- `--cache-text-prompt`: `9.58 FPS`

なので、固定 prompt に近い運用なら付けた方がよいです。

後段をさらに TensorRT に寄せる実験として、`grounding core` も ONNX 化できます。

```bash
python sam3/scripts/export_efficientsam3_decoder_onnx.py \
  --checkpoint /root/.cache/huggingface/hub/models--Simon7108528--EfficientSAM3/snapshots/08f44d3bcce47ac577962a21608ee934c68b1b45/stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth \
  --backbone-type repvit \
  --model-name m1.1 \
  --output /tmp/efficientsam3_decoder_repvit_m1_1_ft.onnx \
  --dynamic-batch \
  --opset 18
```

```bash
python sam3/scripts/export_efficientsam3_grounding_core_onnx.py \
  --checkpoint /root/.cache/huggingface/hub/models--Simon7108528--EfficientSAM3/snapshots/08f44d3bcce47ac577962a21608ee934c68b1b45/stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth \
  --backbone-type repvit \
  --model-name m1.1 \
  --text-encoder-type MobileCLIP-S1 \
  --text-prompt children \
  --output /tmp/efficientsam3_grounding_core_repvit_m1_1_ft.onnx \
  --dynamic-batch \
  --opset 18
```

```bash
python sam3/scripts/benchmark_text_prompt_single_image.py \
  --backend onnx_grounding_core \
  --ort-provider tensorrt \
  --trt-fp16 \
  --trt-engine-cache \
  --cache-text-prompt \
  --checkpoint /root/.cache/huggingface/hub/models--Simon7108528--EfficientSAM3/snapshots/08f44d3bcce47ac577962a21608ee934c68b1b45/stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth \
  --encoder-onnx /tmp/efficientsam3_encoder_repvit_m1_1_ft.onnx \
  --decoder-onnx /tmp/efficientsam3_decoder_repvit_m1_1_ft.onnx \
  --grounding-core-onnx /tmp/efficientsam3_grounding_core_repvit_m1_1_ft.onnx \
  --image /ros2_ws/efficientsam3/test_image.jpg \
  --prompt children \
  --backbone-type repvit \
  --model-name m1.1 \
  --text-encoder-type MobileCLIP-S1 \
  --confidence-threshold 0.05 \
  --device cuda
```

今回の実測では:

- `onnx_local + cache-text-prompt`: `103.02 ms`, `9.71 FPS`
- `onnx_grounding_core + cache-text-prompt`: `103.87 ms`, `9.63 FPS`

つまり、後段 ONNX/TensorRT 化は成立しますが、現状は `onnx_local` を明確には超えません。

いま一番効いたのは、最後の visual level だけを downsample して encoder に入れる方法です。

```bash
python sam3/scripts/benchmark_text_prompt_single_image.py \
  --backend onnx_local \
  --ort-provider tensorrt \
  --trt-fp16 \
  --trt-engine-cache \
  --cache-text-prompt \
  --encoder-feature-downsample 2 \
  --checkpoint /root/.cache/huggingface/hub/models--Simon7108528--EfficientSAM3/snapshots/08f44d3bcce47ac577962a21608ee934c68b1b45/stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth \
  --encoder-onnx /tmp/efficientsam3_encoder_repvit_m1_1_ft.onnx \
  --image /ros2_ws/efficientsam3/test_image.jpg \
  --prompt children \
  --backbone-type repvit \
  --model-name m1.1 \
  --text-encoder-type MobileCLIP-S1 \
  --confidence-threshold 0.05 \
  --device cuda
```

今回の実測:

- baseline: `103.02 ms`, `9.71 FPS`
- `--encoder-feature-downsample 2`: `70.02 ms`, `14.28 FPS`

出力比較:

- detections: `16 -> 15`
- best mask IoU: `0.8299`
- bbox も近い

これは layer 削減ではなく、encoder に入る image token 数そのものを減らす方法です。
いまのところ、精度を大きく崩さず 100ms を大きく割れた最有力案です。

安定性の整理:

- `encoder-feature-downsample=2`
  - 安定推奨
  - 複数ケースで `17.6-18.0 FPS`
  - IoU も高い
- `encoder-feature-downsample=3`
  - 速度重視の実験設定
  - `19 FPS` 前後まで伸びる
  - ただし人物ケースでは mask の崩れが増える

適応型も試せます。

- 実装:
  - 最終 visual level の feature map から空間勾配和 `complexity` を計算
  - `complexity >= 0.27` なら factor=2
  - それ未満なら factor=3
- 目的:
  - crowded / 細かい scene では安全側に factor=2
  - 単純な scene では速度重視で factor=3

```bash
python sam3/efficientsam3_examples/save_text_prompt_mask.py \
  --checkpoint /root/.cache/huggingface/hub/models--Simon7108528--EfficientSAM3/snapshots/08f44d3bcce47ac577962a21608ee934c68b1b45/stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth \
  --image /ros2_ws/efficientsam3/test_image.jpg \
  --prompt children \
  --output /ros2_ws/efficientsam3/adaptive_mask.png \
  --backbone-type repvit \
  --model-name m1.1 \
  --text-encoder-type MobileCLIP-S1 \
  --confidence-threshold 0.05 \
  --device cuda \
  --adaptive-encoder-feature-downsample
```

一括評価と保存:

```bash
python sam3/scripts/evaluate_feature_downsample_configs.py \
  --checkpoint /root/.cache/huggingface/hub/models--Simon7108528--EfficientSAM3/snapshots/08f44d3bcce47ac577962a21608ee934c68b1b45/stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth \
  --encoder-onnx /tmp/efficientsam3_encoder_repvit_m1_1_ft.onnx \
  --backbone-type repvit \
  --model-name m1.1 \
  --text-encoder-type MobileCLIP-S1 \
  --confidence-threshold 0.05 \
  --device cuda \
  --runs 10 \
  --warmup-runs 3 \
  --factor 1 \
  --factor 2 \
  --factor 3 \
  --adaptive \
  --adaptive-feature-threshold 0.27 \
  --case /ros2_ws/efficientsam3/test_image.jpg::children \
  --case /ros2_ws/efficientsam3/groceries.jpg::object \
  --case /ros2_ws/efficientsam3/groceries.jpg::apple \
  --output-dir /ros2_ws/efficientsam3/feature_downsample_results
```

保存物:

- `/home/guch1/ssd_yamaguchi/piper_ros/efficientsam3/feature_downsample_results/`
- `*_mask.png`: 二値マスク
- `*_overlay.png`: 元画像への重ね表示

注意:

- adaptive downsample は FPS 側の改善策です
- 現 checkpoint では `truck` や一部 video frame の `person` で 0 件になることがあり、zero-shot 一般化そのものはまだ弱いです

単画像でそのまま mask を保存したい場合も使えます。

```bash
python sam3/efficientsam3_examples/save_text_prompt_mask.py \
  --checkpoint /root/.cache/huggingface/hub/models--Simon7108528--EfficientSAM3/snapshots/08f44d3bcce47ac577962a21608ee934c68b1b45/stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth \
  --image /ros2_ws/efficientsam3/test_image.jpg \
  --prompt children \
  --output /tmp/mask_down2.png \
  --overlay-output /tmp/mask_down2_overlay.png \
  --backbone-type repvit \
  --model-name m1.1 \
  --text-encoder-type MobileCLIP-S1 \
  --confidence-threshold 0.05 \
  --device cuda \
  --encoder-feature-downsample 2
```

`overlay-output` には mask に加えて best detection の bbox を重ねた画像を保存します。

複数検出をまとめて保存したい場合:

```bash
python sam3/efficientsam3_examples/save_text_prompt_mask.py \
  --checkpoint /root/.cache/huggingface/hub/models--Simon7108528--EfficientSAM3/snapshots/08f44d3bcce47ac577962a21608ee934c68b1b45/stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth \
  --image /ros2_ws/efficientsam3/test_image.jpg \
  --prompt children \
  --output /ros2_ws/efficientsam3/test_image_children_factor2_merged_mask.png \
  --overlay-output /ros2_ws/efficientsam3/test_image_children_factor2_merged_overlay.png \
  --backbone-type repvit \
  --model-name m1.1 \
  --text-encoder-type MobileCLIP-S1 \
  --confidence-threshold 0.05 \
  --device cuda \
  --encoder-feature-downsample 2 \
  --selection-mode topk_nms \
  --max-detections 8 \
  --nms-iou-threshold 0.6
```

この設定では、複数候補を統合した union mask と、複数 bbox を重ねた overlay を保存します。

さらに runtime 実験として、query 数や層数も削れます。

```bash
python sam3/scripts/benchmark_text_prompt_single_image.py \
  --backend onnx_local \
  --ort-provider tensorrt \
  --trt-fp16 \
  --trt-engine-cache \
  --cache-text-prompt \
  --max-queries 64 \
  --max-encoder-layers 4 \
  --checkpoint /root/.cache/huggingface/hub/models--Simon7108528--EfficientSAM3/snapshots/08f44d3bcce47ac577962a21608ee934c68b1b45/stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth \
  --encoder-onnx /tmp/efficientsam3_encoder_repvit_m1_1_ft.onnx \
  --image /ros2_ws/efficientsam3/test_image.jpg \
  --prompt children \
  --backbone-type repvit \
  --model-name m1.1 \
  --text-encoder-type MobileCLIP-S1 \
  --confidence-threshold 0.03 \
  --device cuda
```

今回の実測:

- `Q=100`: `9.91 FPS`, detections `15`
- `Q=64`: `10.06 FPS`, detections `9`
- `Q=64 + encoder_layers=4 + threshold=0.03`: `11.78 FPS`, detections `7`

解釈:

- `100ms` の壁を越えるには、query 数削減だけでなく encoder 層削減が効く
- ただし encoder 層削減は品質低下を伴うので、threshold 再調整とセットで使う

## 14. 方式A/B/Cで動画的に使う

text prompt を毎フレーム回すだけでなく、SAM3 の tracking を使う比較もできます。
用途ごとにスクリプトを分けています。

- A方式: 毎フレーム text prompt で再検出
- B方式: 最初だけ text prompt で検出し、その後は tracking
- C方式: tracking しつつ、一定周期で text prompt 検出を再実行して補正

追加したスクリプト:

- [sam3/scripts/benchmark_text_prompt_per_frame.py](/home/guch1/ssd_yamaguchi/piper_ros/efficientsam3/sam3/scripts/benchmark_text_prompt_per_frame.py)
- [sam3/scripts/benchmark_text_prompt_then_track.py](/home/guch1/ssd_yamaguchi/piper_ros/efficientsam3/sam3/scripts/benchmark_text_prompt_then_track.py)
- [sam3/scripts/benchmark_text_prompt_track_with_refresh.py](/home/guch1/ssd_yamaguchi/piper_ros/efficientsam3/sam3/scripts/benchmark_text_prompt_track_with_refresh.py)

例:

```bash
python sam3/scripts/benchmark_text_prompt_per_frame.py \
  --input /tmp/sam3_bench_frames \
  --prompt "children" \
  --sam3-checkpoint /ros2_ws/src/sam3/sam3.pt \
  --max-frames 16 \
  --device cuda
```

```bash
python sam3/scripts/benchmark_text_prompt_then_track.py \
  --input /tmp/sam3_bench_frames \
  --prompt "children" \
  --sam3-checkpoint /ros2_ws/src/sam3/sam3.pt \
  --max-frames 16 \
  --device cuda
```

```bash
python sam3/scripts/benchmark_text_prompt_track_with_refresh.py \
  --input /tmp/sam3_bench_frames \
  --prompt "children" \
  --sam3-checkpoint /ros2_ws/src/sam3/sam3.pt \
  --max-frames 16 \
  --refresh-interval 4 \
  --device cuda
```

現時点の傾向:

- C方式は B方式と同等精度で、疑似動画ではわずかに速かった
- ただし full SAM3 video model ベースなので、まだ `1 FPS` 前後で遅い
- 現状の最速構成 `repvit-m1.1 ft + onnx_local + TensorRT FP16` (`9.24 FPS`) を置き換えるほどではない

つまり tracking は考え方として正しいですが、今のまま full SAM3 側へ乗ると遅すぎます。
今後は EfficientSAM3 側の軽い detector と組み合わせる方向で詰めるのが筋です。

## 16. hybrid 方式: EfficientSAM3 detect + SAM3 track

full SAM3 video をそのまま使うと遅いので、text prompt 検出だけ EfficientSAM3 に置き換え、
track だけ SAM3 tracker に任せる hybrid 方式も追加しました。

追加したスクリプト:

- [sam3/scripts/benchmark_efficientsam3_track_with_refresh.py](/home/guch1/ssd_yamaguchi/piper_ros/efficientsam3/sam3/scripts/benchmark_efficientsam3_track_with_refresh.py)

例:

```bash
python sam3/scripts/benchmark_efficientsam3_track_with_refresh.py \
  --input /tmp/sam3_bench_frames \
  --prompt "children" \
  --checkpoint /root/.cache/huggingface/hub/models--Simon7108528--EfficientSAM3/snapshots/08f44d3bcce47ac577962a21608ee934c68b1b45/stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth \
  --sam3-checkpoint /ros2_ws/src/sam3/sam3.pt \
  --backbone-type repvit \
  --model-name m1.1 \
  --text-encoder-type MobileCLIP-S1 \
  --confidence-threshold 0.05 \
  --max-frames 16 \
  --refresh-interval 4 \
  --max-detections 4 \
  --device cuda
```

今回の実測:

- `max_detections=8`, `refresh_interval=4`: `2.72 FPS`
- `max_detections=4`, `refresh_interval=4`: `3.50 FPS`
- `max_detections=4`, `refresh_interval=8`: `3.24 FPS`

解釈:

- full SAM3 video の方式C `1.12 FPS` よりはかなり良い
- ただし単画像最速の `RepViT-M1.1 ft + onnx_local + TensorRT FP16` `9.24 FPS` よりは遅い
- 今の tracker は重いので、tracking を混ぜる場合でも object 数を絞る必要がある

## 15. 16GB VRAM で fine-tune は可能か

結論だけ言うと、16GB では「軽い pilot run は可能性あり、フル設定の本学習は厳しい」です。

根拠:

- 既定設定は `1008x1008`, `batch_size=4`, `accumulation_steps=4`
- geometry fine-tune の README でも標準は 8 GPU 前提
- text prompt fine-tune は frozen teacher と student を同時に回すので、geometry fine-tune より重い

16GB でやるなら、最初は次に落としてください。

```bash
python stage1_geometry_finetune/train_text_prompt_finetune.py \
  --data-root /path/to/coco \
  --sam3-checkpoint /ros2_ws/src/sam3/sam3.pt \
  --stage1-checkpoint /root/.cache/huggingface/hub/models--Simon7108528--EfficientSAM3/snapshots/08f44d3bcce47ac577962a21608ee934c68b1b45/stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth \
  --student-backbone repvit_m1_1 \
  --output-dir output/text_prompt_finetune_pilot \
  --batch-size 1 \
  --epochs 1 \
  --num-samples 32 \
  --num-workers 2 \
  --amp \
  --device cuda
```

この pilot run が通ってから、`num_samples` や `epochs` を増やす方が安全です。

## 17. 複数検出と ROI 再精査

`children` のように 1 つの best mask だけでは不十分な場合は，複数候補を統合した上で，
必要なら軽い ROI 再精査を使います。

まずは複数検出を union mask と bbox overlay で保存します。

```bash
python sam3/efficientsam3_examples/save_text_prompt_mask.py \
  --checkpoint /root/.cache/huggingface/hub/models--Simon7108528--EfficientSAM3/snapshots/08f44d3bcce47ac577962a21608ee934c68b1b45/stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth \
  --image test_image.jpg \
  --prompt children \
  --output merged_mask.png \
  --overlay-output merged_overlay.png \
  --backbone-type repvit \
  --model-name m1.1 \
  --text-encoder-type MobileCLIP-S1 \
  --confidence-threshold 0.05 \
  --device cuda \
  --encoder-feature-downsample 2 \
  --selection-mode topk_nms \
  --max-detections 8 \
  --nms-iou-threshold 0.6
```

次に，低信頼候補だけ同じ EfficientSAM3 の geometric box prompt で切り直します。
これは text prompt で出した box を再利用するので，本家 SAM3 fallback よりかなり軽いです。

```bash
python sam3/efficientsam3_examples/save_text_prompt_mask.py \
  --checkpoint /root/.cache/huggingface/hub/models--Simon7108528--EfficientSAM3/snapshots/08f44d3bcce47ac577962a21608ee934c68b1b45/stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth \
  --image test_image.jpg \
  --prompt children \
  --output geometric_refine_mask.png \
  --overlay-output geometric_refine_overlay.png \
  --backbone-type repvit \
  --model-name m1.1 \
  --text-encoder-type MobileCLIP-S1 \
  --confidence-threshold 0.05 \
  --device cuda \
  --encoder-feature-downsample 2 \
  --selection-mode topk_nms \
  --max-detections 8 \
  --nms-iou-threshold 0.6 \
  --roi-refine-method geometric_box \
  --refine-rois \
  --refine-score-threshold 0.25 \
  --refine-fill-threshold 0.55 \
  --geometric-refine-expand-ratio 0.18 \
  --max-refine-rois 8
```

今回の比較では:

- prompt ensemble: 改善なし
- SAM3 fallback: 改善余地はあるが重すぎる
- geometric box refine: 追加コストが小さく，本命

なので，常用するならまず `geometric_box` を使ってください。

補足:

- `--refine-fill-threshold`
  - box に対して mask が薄い候補も再精査対象にします
- `--geometric-refine-expand-ratio`
  - geometric prompt の box を少し広げて，全身 coverage を狙います

今回の追加検証では，この 2 つを入れた `geometric_box v2` が
旧設定より mask 面積を少し拡大できました。

## 18. text prompt 蒸留の pilot を強くする

蒸留スクリプトには次を追加しました。

- prompt variants
  - COCO カテゴリ名だけでなく `a person`, `the person` なども混ぜる
- box 補助損失
  - GT bbox へ student の box を寄せる
- neck 学習オプション
  - trunk だけでなく lightweight neck も学習対象にできる

例:

```bash
python stage1_geometry_finetune/train_text_prompt_finetune.py \
  --data-root /path/to/coco \
  --sam3-checkpoint /ros2_ws/src/sam3/sam3.pt \
  --stage1-checkpoint /root/.cache/huggingface/hub/models--Simon7108528--EfficientSAM3/snapshots/08f44d3bcce47ac577962a21608ee934c68b1b45/stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth \
  --student-backbone repvit_m1_1 \
  --output-dir output/text_prompt_finetune_pilot \
  --batch-size 1 \
  --epochs 1 \
  --num-samples 32 \
  --num-workers 2 \
  --box-weight 0.2 \
  --train-neck \
  --amp \
  --device cuda
```

COCO 実データがまだ無いので学習自体は未実行ですが，
スクリプトとオプション追加までは確認済みです。

## 19. COCO pilot データを Docker 内で取得する

まずは pilot 用に `annotations + val2017` だけを取るのが現実的です。
`train2017` は回線次第で数時間かかるので，最初の smoke では後回しでよいです。

```bash
docker exec piper-humble-dev bash -lc '
cd /ros2_ws/efficientsam3 && \
DOWNLOAD_TRAIN=0 DOWNLOAD_VAL=1 \
bash stage1_geometry_finetune/scripts/download_coco_pilot.sh
'
```

full train まで揃えるなら:

```bash
docker exec piper-humble-dev bash -lc '
cd /ros2_ws/efficientsam3 && \
DOWNLOAD_TRAIN=1 DOWNLOAD_VAL=1 \
bash stage1_geometry_finetune/scripts/download_coco_pilot.sh
'
```

## 20. text prompt 蒸留の smoke を回す

`val2017` だけでも，teacher / student / loss / checkpoint 保存の smoke は回せます。

```bash
docker exec piper-humble-dev bash -lc '
source /ros2_ws/efficientsam3/.venv/bin/activate && \
cd /ros2_ws/efficientsam3 && \
python3 stage1_geometry_finetune/train_text_prompt_finetune.py \
  --data-root /ros2_ws/efficientsam3/data/coco_pilot \
  --sam3-checkpoint /ros2_ws/src/sam3/sam3.pt \
  --stage1-checkpoint /root/.cache/huggingface/hub/models--Simon7108528--EfficientSAM3/snapshots/08f44d3bcce47ac577962a21608ee934c68b1b45/stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth \
  --student-backbone repvit_m1_1 \
  --output-dir /ros2_ws/efficientsam3/output/text_prompt_finetune_pilot_val \
  --split val \
  --batch-size 1 \
  --epochs 1 \
  --num-samples 8 \
  --num-workers 0 \
  --box-weight 0.2 \
  --train-neck \
  --amp \
  --device cuda
'
```

今回の smoke は成功していて，

- `output/text_prompt_finetune_pilot_val/text_prompt_finetune_epoch_1.pth`
- `output/text_prompt_finetune_pilot_val/history.json`

が保存されています。

## 21. pilot checkpoint を推論用 merged 重みに戻す

pilot 学習の checkpoint は，現在は `student_trunk + student_neck` を含みます。
そのままでは既存推論スクリプトへ渡せないため，
比較したいときは merged checkpoint へ戻します。

```bash
docker exec piper-humble-dev bash -lc '
source /ros2_ws/efficientsam3/.venv/bin/activate && \
cd /ros2_ws/efficientsam3 && \
python3 stage1_geometry_finetune/convert_text_prompt_finetune.py \
  --finetune-ckpt /ros2_ws/efficientsam3/output/text_prompt_finetune_pilot_val/text_prompt_finetune_epoch_1.pth \
  --pretrained /root/.cache/huggingface/hub/models--Simon7108528--EfficientSAM3/snapshots/08f44d3bcce47ac577962a21608ee934c68b1b45/stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth \
  --output /ros2_ws/efficientsam3/output/text_prompt_finetune_pilot_val/efficient_sam3_repvit_m1.1_mobileclip_s1_ft_textprompt_pilot.pth
'
```

## 22. 現時点の結論

- `val2017` の `8 samples / 1 epoch` smoke は通った
- ただし，この超小規模 pilot では `children` の品質は改善せず，むしろ 0 detections になった
- つまり「蒸留の経路は成立した」が，「この規模で本家 SAM3 並み coverage になる」とはまだ言えない
- 本当に品質差を詰めるには `train2017` ベースの学習が必要

追加で分かったこと:

- `--train-neck` で更新した neck 重みも merged checkpoint へ戻すよう修正済み
- student forward も teacher downstream ではなく `EfficientSAM3 downstream` に整合済み
- それでも `val2017` の tiny pilot は score collapse を起こし，`confidence-threshold 0.0` でも `top score 0.0000` に近い
- したがって，現時点の failure は「保存漏れ」ではなく，「学習規模が小さすぎて text grounding が崩壊している」ことが主因と見るべき

## 23. train2017 の長時間ダウンロード

`train2017` は Docker 内でバックグラウンド起動済みです。

進捗確認:

```bash
docker exec piper-humble-dev bash -lc '
tail -n 40 /ros2_ws/efficientsam3/output/coco_train_download.log
'
```

## 24. 重要: exact trunk 初期化に修正済み

その後の切り分けで，初期の text prompt 蒸留が壊れていた主因は
custom `StudentTrunk` の初期化が baseline EfficientSAM3 と一致していなかったことでした。

現在は次の形に直しています。

- `student_model = build_efficientsam3_image_model(...)`
- `student_trunk = deepcopy(student_model.backbone.vision_backbone.trunk)`
- merged 変換も exact trunk key に対応済み

この修正により，学習前 no-op checkpoint を merged に戻したとき，
baseline と raw score が完全一致するようになりました。

## 25. 現在有効な pilot 条件

`val2017` だけでも，exact trunk 版なら改善が出ます。

```bash
docker exec piper-humble-dev bash -lc '
source /ros2_ws/efficientsam3/.venv/bin/activate && \
cd /ros2_ws/efficientsam3 && \
python3 stage1_geometry_finetune/train_text_prompt_finetune.py \
  --data-root /ros2_ws/efficientsam3/data/coco_pilot \
  --sam3-checkpoint /ros2_ws/src/sam3/sam3.pt \
  --stage1-checkpoint /root/.cache/huggingface/hub/models--Simon7108528--EfficientSAM3/snapshots/08f44d3bcce47ac577962a21608ee934c68b1b45/stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth \
  --student-backbone repvit_m1_1 \
  --student-text-encoder-type MobileCLIP-S1 \
  --output-dir /ros2_ws/efficientsam3/output/text_prompt_finetune_exact_val32 \
  --split val \
  --batch-size 1 \
  --epochs 2 \
  --num-samples 32 \
  --num-workers 0 \
  --lr 5e-6 \
  --box-weight 0.2 \
  --train-neck \
  --amp \
  --device cuda
'
```

## 26. 現在の最良 pilot 結果

`val32 / 2 epochs / lr=5e-6` では次の結果が出ています。

- `children`
  - baseline `mask_pixels=60233`
  - finetuned `mask_pixels=59716`
  - baseline `top_score=0.7889`
  - finetuned `top_score=0.6716`
- `groceries/object`
  - baseline `mask_pixels=50559`
  - finetuned `mask_pixels=46381`
  - baseline `top_score=0.4140`
  - finetuned `top_score=0.5283`

速度確認:

- `avg_latency_ms=65.75`
- `throughput_fps=15.21`

TensorRT 高速経路でも確認済み:

- `avg_latency_ms=59.34`
- `throughput_fps=16.85`
- `last_top_score=0.67209`

つまり，exact trunk 版の pilot では，
少なくとも `children` で baseline にかなり近い coverage まで戻せています。

## 27. train2017 が揃ったら最初に回すスクリプト

16GB GPU 前提の控えめな本学習寄り pilot はこれです。

```bash
docker exec piper-humble-dev bash -lc '
source /ros2_ws/efficientsam3/.venv/bin/activate && \
cd /ros2_ws/efficientsam3 && \
DATA_ROOT=/ros2_ws/efficientsam3/data/coco_pilot \
SAM3_CHECKPOINT=/ros2_ws/src/sam3/sam3.pt \
STAGE1_CHECKPOINT=/root/.cache/huggingface/hub/models--Simon7108528--EfficientSAM3/snapshots/08f44d3bcce47ac577962a21608ee934c68b1b45/stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth \
OUTPUT_DIR=/ros2_ws/efficientsam3/output/text_prompt_finetune_train_pilot \
bash stage1_geometry_finetune/scripts/train_text_prompt_finetune_16gb_pilot.sh
'
```

## 28. 現時点の汎化確認

`exact_val32` checkpoint を COCO `val2017` の別カテゴリでも確認すると，
次の傾向があります。

- `person`
  - baseline `mask_pixels=0`
  - finetuned `mask_pixels=49519`
- `car`
  - baseline `0`
  - finetuned `0`
- `dog`
  - baseline `0`
  - finetuned `0`
- `bottle`
  - baseline `0`
  - finetuned `0`

要するに，

- `children / person` 系では改善が見え始めた
- ただし語彙全体へはまだ広がっていない

このため，次段階は `train2017` を使った本学習寄り pilot が必要です。

## 29. 比較基準は本家 SAM3 に切り替え済み

今後の比較元は「蒸留前 EfficientSAM3」ではなく，
本家 SAM3 (`/ros2_ws/src/sam3/sam3.pt`) の出力です。

teacher 比較は次で回せます。

```bash
docker exec piper-humble-dev bash -lc '
source /ros2_ws/efficientsam3/.venv/bin/activate && \
cd /ros2_ws/efficientsam3 && \
python3 sam3/scripts/evaluate_text_prompt_checkpoint.py \
  --teacher-checkpoint /ros2_ws/src/sam3/sam3.pt \
  --baseline-checkpoint /root/.cache/huggingface/hub/models--Simon7108528--EfficientSAM3/snapshots/08f44d3bcce47ac577962a21608ee934c68b1b45/stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth \
  --finetuned-checkpoint /ros2_ws/efficientsam3/output/text_prompt_finetune_exact_val32/efficient_sam3_repvit_m1.1_mobileclip_s1_ft_textprompt_exact_val32.pth \
  --image /ros2_ws/efficientsam3/test_image.jpg \
  --prompt children \
  --output-dir /ros2_ws/efficientsam3/output/text_prompt_finetune_exact_val32/eval_teacher_ref \
  --backbone-type repvit \
  --model-name m1.1 \
  --text-encoder-type MobileCLIP-S1 \
  --confidence-threshold 0.05 \
  --device cuda \
  --encoder-feature-downsample 2 \
  --selection-mode topk_nms \
  --max-detections 8 \
  --nms-iou-threshold 0.6 \
  --roi-refine-method geometric_box \
  --refine-score-threshold 0.25 \
  --refine-fill-threshold 0.55 \
  --geometric-refine-expand-ratio 0.18 \
  --max-refine-rois 8
'
```

現時点の teacher 基準:

- `children`
  - baseline `iou_to_teacher=0.4516`
  - finetuned `iou_to_teacher=0.4443`
- `person`
  - baseline `iou_to_teacher=0.0000`
  - finetuned `iou_to_teacher=0.8418`

つまり，`person` 系は大きく前進した一方，
`children` の multi-person coverage はまだ本家 SAM3 を基準にすると足りません。
