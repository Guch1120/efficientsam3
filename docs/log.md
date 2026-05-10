# 実行ログ

## 2026-03-14
- `docs/AGENT.md` を ExecPlan 運用前提に更新し，複雑タスクでは `docs/plan.md` を self-contained な living document として扱う方針を明文化
- `docs/PLANS.md` を追加し，`docs/plan.md` の書式と更新規則を整理
- `docs/plan.md` を cookbook の ExecPlan 方針に合わせて再構成し，目的・現状・マイルストーン・検証戦略・進捗・発見・判断履歴・リスクを一か所に集約
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
- ただし threshold `0.05` まで下げると `efficientsam3_student_text` は detections=6, top_score=0.0844 まで回復
- `/tmp/efficientsam3_children_threshold005.png` では非ゼロ領域が存在し、完全無反応ではなく低 score 側に寄っていることを確認
- `onnx_encoder_server.py --confidence-threshold 0.05` でも `children` の mask を返せることを確認
- `/tmp/onnx_children_threshold005.png` は `bbox (372, 343, 449, 492)`, `nonzero 8803`
- 現実的な当面運用は「encoder ONNX + PyTorch downstream + low confidence threshold」
- ROS2 では `/camera/color/image_raw` が未配信で，Realsense パッケージも確認できなかった
- 代替として単画像反復ベンチ `benchmark_text_prompt_single_image.py` を追加
- GPU 単画像反復ベンチ結果
- PyTorch (`children`, threshold `0.05`): avg `132.19 ms`, `7.56 FPS`
- ONNX server (`children`, threshold `0.05`): avg `2082.72 ms`, `0.48 FPS`
- 現行 `onnx_encoder_server.py` は HTTP 往復と PyTorch downstream を含むため，単純な runtime としては PyTorch 単体より大幅に遅い
- 速度改善には，少なくとも HTTP サーバ構成を外すか，decoder/downstream の ONNX 化または同一プロセス化が必要
- コンテナ内 venv に `onnxruntime-gpu` を導入し，CUDA provider が有効であることを確認
- `onnx_local` ベンチ結果 (`children`, threshold `0.05`): avg `106.37 ms`, `9.40 FPS`
- 速度面では PyTorch 単体より改善したが，detections=0 で精度が崩れた
- `onnx_local` を threshold `0.0` で確認すると detections=200, top_score=`5.7009e-05`
- つまり同一プロセス ONNX encoder 経路では score スケールが大きく変わっており，現状の threshold をそのまま使えない
- 次の課題は score calibration か decoder/downstream の ONNX 化
- `onnx_local` の再構成を見直した結果，手実装側が `model.backbone.scalp=1` を無視して 4 段 pyramid を使っていたことを確認
- 実際の `forward_image()` は 3 段 (`288/144/72`) を返しており，same-process ONNX 経路はここが不一致だった
- `benchmark_text_prompt_single_image.py` を修正し，`onnx_local` でも `scalp` を反映するよう変更
- 修正後の `onnx_local` は `children`, threshold `0.05` で detections=6, top_score=`0.0834` まで回復し，PyTorch とほぼ同じ検出件数になった
- 修正後の速度は avg `157.14 ms`, `6.36 FPS` で，PyTorch 単体 avg `129.03 ms`, `7.75 FPS` より遅かった
- よって現状の `encoder ONNX + PyTorch downstream` same-process 経路は，精度は戻せても速度優位は出ていない
- `ros2_efficientsam3_benchmark.py` にも `onnx_local` backend を追加し，将来の Realsense 実測で HTTP を挟まない比較ができるようにした
- `export_efficientsam3_decoder_onnx.py` も `scalp` を反映するよう修正し，decoder ONNX export の再実行が成功した
- `export_efficientsam3_text_segment_onnx.py` は既定の `torch.export` では data-dependent guard により失敗した
- 同スクリプトを `dynamo=False` に倒し，固定 prompt の language buffers を `detach()` して legacy exporter へ切り替える修正を追加した
- ただし full text-seg ONNX export は最終的にコンテナ内で `exit 136` となり，現状のモデル規模では重すぎる可能性が高い
- したがって「固定 prompt まで含む full ONNX」は現時点の主経路から外し，decoder/downstream の分割 export を優先する
- `benchmark_text_prompt_single_image.py` に `onnx_split` backend を追加し，encoder ONNX と decoder ONNX を ORT，text/downstream だけを PyTorch にした経路を測定可能にした
- `onnx_split` 実測 (`children`, threshold `0.05`): avg `190.53 ms`, `5.25 FPS`, detections=6, top_score=`0.0834`
- よって encoder/decoder を ORT に分けても PyTorch 単体より速くならず，call boundary と GPU 転送のコストが支配的な可能性が高い
- `ros2_efficientsam3_benchmark.py` にも `onnx_split` backend を追加し，Realsense 実測でも同条件比較できるようにした
- ORT の `TensorrtExecutionProvider` も確認したが，`libnvinfer.so.10` 不足でロードできず，現状は `CUDAExecutionProvider` へフォールバックしている
- したがって現行コンテナでは TensorRT 最適化は未利用であり，追加の TensorRT ランタイム導入が必要
- コンテナ内に `tensorrt-libs` と `libnvinfer10` / `libnvinfer-plugin10` / `libnvonnxparsers10` を導入し，`libnvinfer.so.10` 不足を解消した
- 導入後，ORT セッションは encoder / decoder ともに `TensorrtExecutionProvider` を有効化できた
- 単画像ベンチ再測定:
- PyTorch: avg `140.35 ms`, `7.13 FPS`, detections=6
- `onnx_local` + CUDA EP: avg `171.18 ms`, `5.84 FPS`, detections=6
- `onnx_local` + TensorRT EP: avg `133.20 ms`, `7.51 FPS`, detections=6
- `onnx_split` + CUDA EP: avg `206.80 ms`, `4.84 FPS`, detections=6
- `onnx_split` + TensorRT EP: avg `161.25 ms`, `6.20 FPS`, detections=6
- 結論として，TensorRT EP は有効で，`onnx_local` では PyTorch 単体をわずかに上回った
- 一方で `onnx_split` は TensorRT を使っても PyTorch を超えず，分割境界のコストが依然として支配的だった
- `backbone.forward_image()` 全体を 1 ONNX にした `onnx_backbone` 経路も追加し，画像側をより大きい塊で TensorRT 化できるかを確認した
- `onnx_backbone` 実測:
- CUDA EP: avg `181.04 ms`, `5.52 FPS`, detections=6
- TensorRT EP: avg `142.79 ms`, `7.00 FPS`, detections=6
- さらに TensorRT で `trt_fp16_enable=True` と engine cache を有効化して再測定した
- `onnx_local` + TensorRT FP16: avg `117.36 ms`, `8.52 FPS`, detections=6
- `onnx_backbone` + TensorRT FP16: avg `131.59 ms`, `7.60 FPS`, detections=6
- `onnx_split` + TensorRT FP16: avg `138.74 ms`, `7.21 FPS`, detections=6
- 現時点の最速は `onnx_local + TensorRT FP16` であり，PyTorch avg `140.35 ms` を明確に上回った
- したがって「画像側をより大きい 1 ONNX にまとめれば最速」という仮説は今回の構成では成立せず，最適点は `encoder ONNX + TensorRT` にある
- `tinyvit_21m` について解像度 sweep を試みたが，`1008` 以外は `AssertionError: input feature has wrong size` で失敗した
- つまり現行の `tinyvit_21m` checkpoint / backbone は事実上 `1008` 固定であり，解像度を下げて FPS を稼ぐ手はこの構成では使えない
- 追加比較として `efficient_sam3_repvit-m2_3_mobileclip_s1.pth` を動作確認し，`children` で検出 8 件を維持できることを確認した
- `repvit-m2.3` の速度:
- PyTorch: avg `133.32 ms`, `7.50 FPS`, detections=8
- `onnx_local` + TensorRT FP16: avg `111.50 ms`, `8.97 FPS`, detections=8
- `repvit-m2.3` の追加 VRAM:
- PyTorch: `1760 MiB`
- `onnx_local` + TensorRT FP16: `2553 MiB`
- `tinyvit_21m` よりは少し速いが，まだ 15 FPS には届かない
- 論文調査を行い，精度維持寄りで既存コードに乗せやすい候補を `MobileSAM` と `EdgeSAM` に絞った
- `MobileSAM` は decoupled distillation で backbone 軽量化を行う一方，`EdgeSAM` は prompt-conditioned distillation を明示しており，現状の text prompt 劣化には後者の方が直接効くと判断した
- geometry prompt 側には既に EdgeSAM 風の dual-path / iterative refinement が入っているため，同じ発想を text prompt 側へ拡張する実装を追加した
- 追加ファイル:
- `stage1_geometry_finetune/data/coco_text_prompt_dataset.py`
- `stage1_geometry_finetune/text_prompt_model.py`
- `stage1_geometry_finetune/text_prompt_losses.py`
- `stage1_geometry_finetune/train_text_prompt_finetune.py`
- `stage1_geometry_finetune/scripts/train_text_prompt_finetune.sh`
- `COCOTextPromptDataset` は COCO instance annotation を 1 object = 1 text prompt sample へ展開し，画像，カテゴリ名，GT mask を返す
- `TextPromptFinetuneModel` は trainable student trunk と frozen SAM3 teacher/downstream を組み合わせ，teacher/student の `forward_grounding` を直接比較できる
- 損失は teacher best-query に対する mask BCE/Dice，score MSE，embedding MSE，GT mask 補助損失で構成した
- Docker 内で `python3 -m py_compile` と `train_text_prompt_finetune.py --help` の smoke は通過した
- merged EfficientSAM3 checkpoint を teacher に使った forward smoke では `pred_masks` / `pred_logits` が NaN になった
- これは full SAM3 teacher ではなく merged EfficientSAM3 重みを渡したためで，この fine-tune 経路では full SAM3 checkpoint が必要
- `/home/guch1/ssd_yamaguchi/piper_ros/src/sam3/sam3.pt` を確認し，3.3G の本物の SAM3 checkpoint であることを確認した
- Docker 内でも `/ros2_ws/src/sam3/sam3.pt` を `build_sam3_image_model(checkpoint_path=..., load_from_HF=False)` で正常ロードできた
- これまで明示的に checkpoint を指定しなくても SAM3 が動いていた理由は，`model_builder.py` が `checkpoint_path=None` のとき `hf_hub_download()` を自動で呼んでいたため
- `benchmark_text_prompt_single_image.py` に `--backend sam3` と `--sam3-checkpoint` を追加し，本家 SAM3 も同一条件で単画像ベンチできるようにした
- `sam3.pt` の単画像ベンチ結果 (`children`, threshold `0.1`, 10 runs):
- avg `463.51 ms`, `2.16 FPS`, detections=8, top_score=`0.9580`
- したがって，本家 SAM3 は品質基準としては有効だが，現状の単画像推論速度では実時間用途の本命にはなりにくい
- `piper-humble-dev` を再起動し，docs のルール通り `/ros2_ws/efficientsam3/.venv` 上で検証を再開した
- GPU 状態確認: `NVIDIA GeForce RTX 5060 Ti`, 利用中メモリ `597 / 16311 MiB`
- `compare_text_prompt_models.py` を `sam3.pt` と `efficient_sam3_repvit-m2_3_mobileclip_s1.pth` で再実行
- `children`, threshold `0.05` の結果:
- `sam3`: detections=20, top_score=`0.9580`
- `efficientsam3_student_text`: detections=8, top_score=`0.0914`
- `efficientsam3_teacher_text`: detections=0
- つまり，現状の EfficientSAM3 では `children` に対する score の落ち込みが大きく，本家 SAM3 の品質にはまだ届かない
- 一方で，`repvit-m2.3` + `onnx_local` + TensorRT FP16 を軽い設定で再実行した結果:
- avg `111.01 ms`, `9.01 FPS`, detections=8, top_score=`0.0872`
- よって現時点の運用判断は，「品質基準は本家 SAM3，実運用の速度本命は EfficientSAM3 + TensorRT」で変わらない
- README のモデル表を再確認し，`ft` があるのは `RepViT-M1.1`, `TinyViT-11M`, `EfficientViT-B1` であることを確認した
- 速度と精度のバランスから，最有力候補を `efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth` に設定した
- Hugging Face から `stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth` をコンテナ内へ取得
- 保存先: `/root/.cache/huggingface/hub/models--Simon7108528--EfficientSAM3/snapshots/08f44d3bcce47ac577962a21608ee934c68b1b45/stage1_all_converted/efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth`
- 同 checkpoint で encoder ONNX export に成功
- 出力: `/tmp/efficientsam3_encoder_repvit_m1_1_ft.onnx`
- 品質比較 (`children`, threshold `0.05`):
- `sam3`: detections=20, top_score=`0.9580`
- `repvit-m1.1 ft` student text: detections=16, top_score=`0.1683`
- `repvit-m1.1 ft` teacher text: detections=0
- つまり `repvit-m2.3` 非 ft より，`repvit-m1.1 ft` の方が明確に text prompt 品質が良い
- 速度比較 (`children`, threshold `0.05`, 6 runs):
- `repvit-m1.1 ft` PyTorch: avg `112.68 ms`, `8.87 FPS`, detections=16
- `repvit-m1.1 ft` `onnx_local + TensorRT FP16`: avg `108.20 ms`, `9.24 FPS`, detections=15, top_score=`0.1685`
- 現時点では `repvit-m1.1 ft + onnx_local + TensorRT FP16` が，速度と品質の両立という観点で最有力
- `sam3/scripts/benchmark_text_prompt_per_frame.py` / `benchmark_text_prompt_then_track.py` / `benchmark_text_prompt_track_with_refresh.py` を追加し，A/B/C 方式を別ファイルで比較できるようにした
- upstream SAM3 video 推論は Triton の JIT で `Python.h` を必要とするため，`piper-humble-dev` 内に `python3.12-dev` を追加した
- 疑似動画 `/tmp/sam3_bench_frames`（`test_image.jpg` の複製 4 フレーム）で smoke:
- A方式 per-frame text prompt: avg `1697.80 ms`, `0.59 FPS`
- B方式 初回検出+tracking: avg `1565.09 ms`, `0.64 FPS`
- C方式 refresh interval 2: avg `1579.24 ms`, `0.63 FPS`
- 4フレームでは初回 compile / 準備コストが大きすぎるため，9FPS 比較には不向き
- 同じ疑似動画を 16 フレームへ延長して再測定:
- B方式 初回検出+tracking: avg `921.58 ms`, `1.09 FPS`
- C方式 refresh interval 4: avg `895.04 ms`, `1.12 FPS`
- 今回の同一画像疑似動画では C方式が B方式よりわずかに速く，精度指標（`max_num_objects=6`, `top_score=0.9532`）は同等だった
- ただし full SAM3 video model ベースなので，現時点では `repvit-m1.1 ft + onnx_local + TensorRT FP16` の `9.24 FPS` より大幅に遅い
- `TinyViT-11M ft` (`efficient_sam3_tiny_vit_11m_mobileclip_s1_ft.pth`) も追加比較した
- `children`, threshold `0.05` では detections=0 で，今回の prompt では `RepViT-M1.1 ft` より悪かった
- encoder ONNX export 自体は `/tmp/efficientsam3_encoder_tiny_vit_11m_ft.onnx` として成功
- したがって，現時点の第一候補は変わらず `RepViT-M1.1 ft`
- `sam3/scripts/benchmark_efficientsam3_track_with_refresh.py` を追加し，`EfficientSAM3 detector + SAM3 tracker` の hybrid periodic refresh benchmark を実装した
- 実装では `RepViT-M1.1 ft` の text prompt 検出マスクを refresh フレームで生成し，`sam3.pt` から tracker/backbone 重みを抜いて `build_tracker(with_backbone=True)` にロードした
- `tracker` には full checkpoint の `tracker.*` に加えて `detector.backbone.*` も流し込み，`propagate_preflight=True` で区間追跡を回すようにした
- `tracker unexpected keys: 295` は主に tracker 単体では不要な full checkpoint 側の残差で，benchmark 自体は正常完了した
- hybrid benchmark (`children`, 16 frame 疑似動画, `refresh_interval=4`, `max_detections=8`):
- avg `367.53 ms`, `2.72 FPS`, avg objects `8`
- hybrid benchmark (`children`, 16 frame 疑似動画, `refresh_interval=4`, `max_detections=4`):
- avg `285.54 ms`, `3.50 FPS`, avg objects `4`
- hybrid benchmark (`children`, 16 frame 疑似動画, `refresh_interval=8`, `max_detections=4`):
- avg `308.28 ms`, `3.24 FPS`, avg objects `4`
- したがって hybrid 方式では `max_detections=4`, `refresh_interval=4` が今回の範囲で最良だった
- full SAM3 video の C方式 `1.12 FPS` よりは大きく改善したが，依然として `RepViT-M1.1 ft + onnx_local + TensorRT FP16` の `9.24 FPS` よりかなり遅い
- 今回の結論として，tracking を混ぜるなら full SAM3 detector を避ける価値はあるが，tracker 自体のコストがまだ大きい
- fine-tune の 16GB VRAM 見積もり:
- README / config 上の既定は `1008x1008`, `batch_size=4`, `accumulation_steps=4`, 8 GPU 前提で重い
- geometry fine-tune でも single GPU 例はあるが，そのままの設定では 16GB は厳しい
- text prompt fine-tune は frozen teacher + student forward/backward を同時に回すため，geometry fine-tune よりさらに重い
- 現実的には `batch_size=1`, `num_workers` 小さめ, `--amp`, `num_samples` を絞る軽量実験から始めるべき
- 16GB では「短い pilot run は可能性あり，フル設定の本学習は厳しい」という判断
- `benchmark_text_prompt_single_image.py` に `--cache-text-prompt` を追加し，同一 prompt の text encoder 出力を再利用できるようにした
- `onnx_local + TensorRT FP16` を逐次再測定:
- 通常: avg `109.11 ms`, `9.17 FPS`
- `--cache-text-prompt`: avg `104.44 ms`, `9.58 FPS`
- 同一 prompt を繰り返す運用なら，text encoder 再計算を省くことで約 4% の改善が得られた
- `torch.compile` は PyTorch / `onnx_local` の両方で逆効果だったため，主戦略から外した
- `profile_text_prompt_pipeline.py` を追加し，`onnx_local` 最速経路を段階ごとに profiling した
- `RepViT-M1.1 ft + onnx_local + TensorRT FP16 + cache-text-prompt` の内訳:
- total `116.65 ms`
- encoder ONNX `9.10 ms`
- neck decoder `17.00 ms`
- prompt encode `3.50 ms`
- transformer encoder `49.28 ms`
- transformer decoder `19.27 ms`
- segmentation heads `15.06 ms`
- postprocess `1.33 ms`
- query 数は `200`
- つまり bottleneck は image encoder ではなく，後段の `transformer encoder + decoder + segmentation head`
- `benchmark_text_prompt_single_image.py` と `profile_text_prompt_pipeline.py` に `--max-queries`, `--max-encoder-layers`, `--max-decoder-layers` を追加し，runtime で構造軽量化を試せるようにした
- query 削減だけの実測 (`cache-text-prompt` 併用):
- `Q=100`: `100.89 ms`, `9.91 FPS`, detections `15`
- `Q=64`: `99.40 ms`, `10.06 FPS`, detections `9`
- `Q=32`: `102.48 ms`, `9.76 FPS`, detections `4`
- したがって query 数削減だけでも `100 ms` を切れる
- `Q=64` の profile 内訳:
- total `107.09 ms`
- transformer encoder `48.03 ms`
- transformer decoder `12.69 ms`
- segmentation heads `13.97 ms`
- query 削減で効いたのは主に decoder 側で，encoder 側はほぼ残る
- encoder 層削減の実測:
- `Q=64 + encoder_layers=4 + threshold=0.05`: `83.91 ms`, `11.92 FPS`, detections `5`
- `Q=64 + encoder_layers=4 + threshold=0.03`: `84.86 ms`, `11.78 FPS`, detections `7`
- `Q=64 + encoder_layers=4 + threshold=0.01`: `87.20 ms`, `11.47 FPS`, detections `50`
- decoder 層削減のみ (`Q=64 + decoder_layers=4`) は `100.91 ms`, `9.91 FPS`, detections `10` で改善は限定的
- `Q=100 + encoder_layers=4` も試したが:
- threshold `0.05`: `83.32 ms`, `12.00 FPS`, detections `3`
- threshold `0.03`: `83.88 ms`, `11.92 FPS`, detections `6`
- 今回の範囲では，速度を最も押し上げたのは query 数削減よりも encoder 層削減だった
- ただし品質低下も大きいので，現時点の実験候補は `Q=64 + encoder_layers=4 + threshold=0.03`
## 2026-03-15 Prompt Cache / Grounding Core Evaluation

- 目的:
  - `1. 後段を TensorRT に寄せる`
  - `2. 同一 prompt 再利用を text embedding より先まで広げる`
- 実行環境:
  - Docker `piper-humble-dev`
  - venv `/ros2_ws/efficientsam3/.venv`
  - checkpoint `efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth`
  - prompt `children`

### 2. Prompt 再利用の確認

- profiling 結果:
  - `encoder_onnx`: 約 `9.10 ms`
  - `neck_decoder`: 約 `17.00 ms`
  - `prompt_encode`: 約 `3.50 ms`
  - `transformer_encoder`: 約 `49.28 ms`
  - `transformer_decoder`: 約 `19.27 ms`
  - `segmentation_heads`: 約 `15.06 ms`
- 読み取り:
  - 主ボトルネックは `transformer_encoder`
  - `prompt_encode` は image/geometric token に依存しており、text だけでは閉じない
  - そのため、`language_features/language_mask` 以上の大きな再利用余地は小さい
- 既存 `--cache-text-prompt` の効果:
  - `onnx_local + TensorRT FP16`
  - 通常: `109.11 ms`, `9.17 FPS`
  - cache: `104.44 ms`, `9.58 FPS`

### 1. 後段 TensorRT 化の評価

- full text-seg ONNX:
  - `export_efficientsam3_text_segment_onnx.py`
  - `exit 136` で失敗
  - geometry encoder を含む full export は現状重すぎる
- fixed-text downstream ONNX:
  - `export_efficientsam3_text_downstream_onnx.py`
  - これも `exit 136`
- grounding core ONNX:
  - 新規 `export_efficientsam3_grounding_core_onnx.py`
  - `_run_encoder/_run_decoder/_run_segmentation_heads` を ONNX 化
  - export 成功
  - TensorRT FP16 実行も成功

### Grounding Core Benchmark

- baseline:
  - backend `onnx_local`
  - `TensorRT FP16 + engine cache + cache-text-prompt`
  - `103.02 ms`, `9.71 FPS`
  - detections `15`
- new:
  - backend `onnx_grounding_core`
  - `encoder ONNX + decoder ONNX + grounding_core ONNX`
  - `TensorRT FP16 + engine cache + cache-text-prompt`
  - `103.87 ms`, `9.63 FPS`
  - detections `16`

### 結論

- `prompt` 再利用はすでに効いているが、追加余地は小さい
- 後段 ONNX/TensorRT 化は成立した
- ただし `100 ms` の壁を破るほどの差は出ない
- 現時点では `onnx_local + cache-text-prompt` が依然として最有力
- 次に速度を上げるには runtime 最適化ではなく、計算量そのものを減らす設計変更が必要

## 2026-03-15 Encoder Token Reduction Evaluation

- 仮説:
  - `transformer_encoder` の主コストは `72x72` 最終 visual level の image token
  - query 削減ではなく、encoder に入る image token 数を減らす方が本筋
- 実装:
  - `benchmark_text_prompt_single_image.py`
  - `profile_text_prompt_pipeline.py`
  - `--encoder-feature-downsample`
  - 最終 visual level のみを average pool し、positional encoding を再計算

### 実測

- baseline:
  - `onnx_local + TensorRT FP16 + cache-text-prompt`
  - `103.02 ms`, `9.71 FPS`
- `--encoder-feature-downsample 2`:
  - `70.02 ms`, `14.28 FPS`
  - detections `15`
  - top score `0.2210`

### 出力比較

- baseline:
  - detections `16`
  - top score `0.1683`
- downsample=2:
  - detections `15`
  - top score `0.2208`
- best mask IoU:
  - `0.8299`
- best bbox:
  - baseline `[375.27, 342.59, 460.29, 496.33]`
  - downsample `[359.23, 337.39, 449.65, 492.11]`

### プロファイル

- `encoder_onnx`: `11.13 ms`
- `neck_decoder`: `23.00 ms`
- `prompt_encode`: `3.09 ms`
- `transformer_encoder`: `7.67 ms`
- `transformer_decoder`: `34.65 ms`
- `segmentation_heads`: `23.87 ms`

### 解釈

- image token 数の削減は大きく効く
- `transformer_encoder` は大幅短縮した
- 代わりに decoder / seg head 比率は上がるが、全体では明確に高速化した
- 層削減よりも品質面の崩れが小さい可能性が高い

### 追加確認

- `test_image.jpg`, `children`
  - baseline `(16, 0.1683)`
  - downsample=2 `(15, 0.2208)`
- `groceries.jpg`, `object`
  - baseline `(76, 0.3872)`
  - downsample=2 `(51, 0.4140)`
- `groceries.jpg`, `apple`
  - baseline `(48, 0.3409)`
  - downsample=2 `(42, 0.4062)`
- `groceries.jpg`, `banana`
  - baseline `(0, None)`
  - downsample=2 `(0, None)`
- `groceries.jpg`, `bottle`
  - baseline `(0, None)`
  - downsample=2 `(0, None)`

### 推論スクリプト反映

- `save_text_prompt_mask.py` に `--encoder-feature-downsample` を追加
- `/tmp/mask_down2.png` への保存確認済み

### 安定性評価

一括比較スクリプト:

- `sam3/scripts/evaluate_feature_downsample_configs.py`

3 ケース, `runs=10`, `warmup=3` での比較:

- `test_image.jpg`, `children`
  - factor=1: `104.98 ms`, `9.53 FPS`
  - factor=2: `55.42 ms`, `18.04 FPS`, IoU `0.8302`
  - factor=3: `53.45 ms`, `18.71 FPS`, IoU `0.5709`
- `groceries.jpg`, `object`
  - factor=1: `106.80 ms`, `9.36 FPS`
  - factor=2: `56.74 ms`, `17.63 FPS`, IoU `0.9621`
  - factor=3: `52.20 ms`, `19.16 FPS`, IoU `0.9106`
- `groceries.jpg`, `apple`
  - factor=1: `106.24 ms`, `9.41 FPS`
  - factor=2: `56.37 ms`, `17.74 FPS`, IoU `0.9627`
  - factor=3: `52.62 ms`, `19.00 FPS`, IoU `0.9267`

判断:

- `factor=2` は 3 ケースとも安定
- `factor=3` は 20 FPS に近いが，人物ケースで崩れが大きい
- 現時点の安定推奨値は `encoder-feature-downsample=2`
- `factor=3` は速度重視の実験設定

### 適応型 downsample

- 実装:
  - 最終 visual level feature map の空間勾配和を `complexity` として計算
  - `complexity >= 0.27` なら factor=2
  - それ未満なら factor=3
- 実測:
  - `test_image.jpg`, `children`
    - complexity `0.2787`
    - adaptive は factor=2
    - `56.83 ms`, `17.60 FPS`, IoU `0.8302`
  - `groceries.jpg`, `object`
    - complexity `0.2561`
    - adaptive は factor=3
    - `53.45 ms`, `18.71 FPS`, IoU `0.9106`
  - `groceries.jpg`, `apple`
    - complexity `0.2561`
    - adaptive は factor=3
    - `53.00 ms`, `18.87 FPS`, IoU `0.9267`

### 保存結果

- 結果画像は `/home/guch1/ssd_yamaguchi/piper_ros/efficientsam3/feature_downsample_results/`
- `*_mask.png` と `*_overlay.png` を保存

### 追加の zero-shot 確認

- `truck.jpg`
  - `truck`, `vehicle`, `car`, `pickup`, `pickup truck`, `automobile`
  - すべて detections `0`
- `assets/videos/0001` の一部フレーム
  - prompt `person`
  - `0.jpg`, `50.jpg`, `100.jpg`, `150.jpg`, `200.jpg`
  - すべて detections `0`

解釈:

- adaptive downsample は runtime 改善には効く
- ただし zero-shot の語彙幅そのものは，現 checkpoint 依存の限界がまだ大きい
- FPS 安定化と zero-shot 一般化は別問題として扱うべき

### ROI 再精査の比較

`children` のように「複数検出は出るが全身 mask が甘い」ケースに対して，
ROI 再精査を 3 方式で比較した．保存先はすべて repo 直下の次のファイル．

- `roi_refine_baseline_mask.png`
- `roi_refine_baseline_overlay.png`
- `roi_refine_ensemble_mask.png`
- `roi_refine_ensemble_overlay.png`
- `roi_refine_sam3_mask.png`
- `roi_refine_sam3_overlay.png`
- `roi_refine_sam3_top2_mask.png`
- `roi_refine_sam3_top2_overlay.png`
- `roi_refine_geometric_mask.png`
- `roi_refine_geometric_overlay.png`
- `roi_refine_geometric_all_mask.png`
- `roi_refine_geometric_all_overlay.png`

共通条件:

- checkpoint: `efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth`
- prompt: `children`
- `encoder-feature-downsample=2`
- `selection-mode=topk_nms`
- `max-detections=8`

実測:

- baseline
  - `selected detections=8 / total 15`
  - `top selected score=0.2208`
  - `wall_sec=6.746`
- ROI prompt ensemble (`children, child, kid, kids, person, people`)
  - `refined detections=0`
  - `wall_sec=11.978`
- ROI SAM3 fallback, all 8 ROI
  - `refined detections=8`
  - `top selected score=0.9664`
  - `wall_sec=42.788`
- ROI SAM3 fallback, top 2 ROI
  - `refined detections=2`
  - `top selected score=0.9664`
  - `wall_sec=20.450`
- geometric box refine, top 2 ROI
  - `refined detections=2`
  - `top selected score=0.9390`
  - `wall_sec=7.079`
- geometric box refine, top 8 ROI
  - `refined detections=6`
  - `top selected score=0.9565`
  - `wall_sec=7.188`

mask 面積比較:

- baseline: `59053 px`
- prompt ensemble: `59053 px`
- SAM3 fallback top2: `60885 px`
- geometric box top2: `57918 px`
- geometric box top8: `58639 px`

判断:

- prompt ensemble は今回の `children` では改善なし
- SAM3 fallback は品質上限の確認には使えるが，速度コストが大きすぎて常用不可
- geometric box refine は同じ EfficientSAM3 の中で完結し，増分コストが小さい
- 現時点で ROI 再精査の本命は `geometric_box`

### geometric box refine の coverage 改善

`geometric_box` を「低 score だけ」ではなく「mask fill ratio が低い候補」も対象にし，
さらに geometric prompt 自体を少し広げる設定を追加した．

追加した主な引数:

- `--refine-fill-threshold`
- `--geometric-refine-expand-ratio`

検証コマンド:

- `children`
  - `encoder-feature-downsample=2`
  - `selection-mode=topk_nms`
  - `max-detections=8`
  - `roi-refine-method=geometric_box`
  - `refine-fill-threshold=0.55`
  - `geometric-refine-expand-ratio=0.18`

結果:

- `roi_refine_geometric_v2_mask.png`
- `roi_refine_geometric_v2_overlay.png`
- `refined detections=1`
- `top selected score=0.7889`
- `wall_sec=7.104`
- mask 面積 `60233 px`

比較:

- 旧 geometric box top8: `58639 px`
- 新 geometric box v2: `60233 px`

解釈:

- score と fill ratio を使った選別により，不必要な ROI 再精査を減らしつつ
  coverage はわずかに改善した
- 依然として本家 SAM3 の全身品質には届かないが，
  same-model refinement としては改善方向が確認できた

### text prompt 蒸留の pilot 強化

学習側の足場も強化した．

- `COCOTextPromptDataset`
  - prompt variants を追加
  - `gt_box_cxcywh` を返すようにした
- `TextPromptFinetuneModel`
  - `--train-neck` 相当の neck 学習を可能にした
- `compute_text_prompt_distill_loss`
  - `box_weight` 付きの box L1 補助損失を追加
- `train_text_prompt_finetune.py`
  - `--box-weight`
  - `--disable-prompt-variants`
  - `--train-neck`

確認:

- `py_compile` は通過
- `train_text_prompt_finetune.py --help` で新オプション表示を確認

未確認:

- COCO 実データが手元に無いため，pilot 学習の実行自体はまだ未実施

### COCO pilot データ取得

COCO は手元に無かったため，pilot 用の取得スクリプトを追加した．

- `stage1_geometry_finetune/scripts/download_coco_pilot.sh`

対応:

- raw COCO の標準配置をそのまま展開
- `DOWNLOAD_TRAIN=0/1`
- `DOWNLOAD_VAL=0/1`
- 壊れた zip を検出したら削除して再取得

今回の取得結果:

- `DOWNLOAD_TRAIN=0`
- `DOWNLOAD_VAL=1`
- root: `/ros2_ws/efficientsam3/data/coco_pilot`
- 使用量: `2.6G`
- 取得済み:
  - `annotations/instances_train2017.json`
  - `annotations/instances_val2017.json`
  - `val2017/`

`train2017.zip` は接続速度だと数時間級なので，今回の turn では取得していない．

### text prompt 蒸留 pilot 実行

`val2017` ベースの smoke として，最小学習を 1 回通した．

コマンド条件:

- split: `val`
- samples: `8`
- batch size: `1`
- epochs: `1`
- `train-neck`
- `box-weight=0.2`
- AMP 有効

重要な修正:

- `RepViT-M1.1 ft` checkpoint の trunk key は
  `detector.backbone.vision_backbone.trunk.model.*`
  形式だったため，student trunk へ写す key 変換を追加した

結果:

- `Loaded Stage1 weights: missing=554 unexpected=653`
- `loss_total=2.5671`
- `loss_box=0.1629`
- 保存物:
  - `output/text_prompt_finetune_pilot_val/text_prompt_finetune_epoch_1.pth`
  - `output/text_prompt_finetune_pilot_val/history.json`

解釈:

- teacher / student / COCO dataset / optimizer / checkpoint 保存まで一連で通った
- つまり 16GB 前提でも，少量サンプルの pilot distillation は成立する
- ただし，この 1 回だけでは本家 SAM3 並み coverage へ寄るとはまだ言えない
- 本当に gap が縮むかは，この checkpoint を使った学習前後比較が次の課題

### pilot checkpoint を merged 推論重みに変換

pilot 学習で得た student trunk を，そのまま既存推論スクリプトで比較できるように
merged checkpoint へ戻す変換スクリプトを追加した．

- `stage1_geometry_finetune/convert_text_prompt_finetune.py`

今回の変換:

- input:
  - `output/text_prompt_finetune_pilot_val/text_prompt_finetune_epoch_1.pth`
  - `efficient_sam3_repvit_m1.1_mobileclip_s1_ft.pth`
- output:
  - `output/text_prompt_finetune_pilot_val/efficient_sam3_repvit_m1.1_mobileclip_s1_ft_textprompt_pilot.pth`
- replaced weights:
  - `653`

### pilot 学習後の即時推論比較

同条件 `children` で baseline と pilot 版を比較した．

- baseline
  - `top selected score=0.7889`
  - mask 面積 `60233 px`
- pilot 版
  - `0 detections`
  - mask 面積 `0 px`

解釈:

- `val2017`, `8 samples`, `1 epoch` は smoke としては有効
- しかし品質改善には全く足りず，むしろ悪化した
- したがって，蒸留で本家 SAM3 に近づける可能性はあるが，
  少量 pilot だけで同等領域になるとは言えない
- 本学習寄りの評価には `train2017` が必要

### train2017 ダウンロード開始

`train2017` は回線速度上，数時間級になるため，Docker 内でバックグラウンド起動した．

- log:
  - `output/coco_train_download.log`
- 現在の状態:
  - `download: http://images.cocodataset.org/zips/train2017.zip`

この turn では完了していないが，以後はこの log を見れば進捗確認できる．

### neck 重み保存漏れの修正

pilot 学習結果が推論に反映されていなかった原因の 1 つとして，
`--train-neck` で更新した neck 重みを checkpoint 保存も merged 変換もしていなかった問題を修正した．

- `stage1_geometry_finetune/text_prompt_model.py`
  - `get_finetune_state_dict()` を追加
- `stage1_geometry_finetune/train_text_prompt_finetune.py`
  - checkpoint 保存を `student_trunk` だけでなく `student_neck` 含みに変更
- `stage1_geometry_finetune/convert_text_prompt_finetune.py`
  - `detector.backbone.vision_backbone.convs.*` も merged checkpoint に反映

修正後の convert:

- replaced weights:
  - `671`

前回の `653` より増えており，neck 側も反映されている．

### student runtime 整合の修正

さらに，本質的な不整合も見つかった．

- 旧実装:
  - student trunk を teacher SAM3 downstream に流して loss を計算
- 実際の推論:
  - student trunk を EfficientSAM3 downstream に流して実行

この学習先と実行先のズレを解消するため，
`TextPromptFinetuneModel` の student forward を `EfficientSAM3 downstream` へ切り替えた．

- student text:
  - `self.student_model.backbone.forward_text(...)`
- student grounding:
  - `self.student_model.forward_grounding(...)`

つまり，学習時も推論時も同じ student runtime を通るように修正した．

### neck 保存 + runtime 整合後の再試験

#### val32 / 2 epochs / train-neck

- output:
  - `output/text_prompt_finetune_neckfix_val32/text_prompt_finetune_epoch_2.pth`
- losses:
  - epoch0 `loss_total=2.5410`
  - epoch1 `loss_total=2.2294`
- converted:
  - `output/text_prompt_finetune_neckfix_val32/efficient_sam3_repvit_m1.1_mobileclip_s1_ft_textprompt_neckfix.pth`
  - replaced weights `671`

評価:

- `children`
  - baseline `mask_pixels=60233`
  - finetuned `mask_pixels=0`
- `groceries/object`
  - baseline `mask_pixels=50559`
  - finetuned `mask_pixels=0`

#### runtime 整合版 val8 / 1 epoch / lr=1e-5

- output:
  - `output/text_prompt_finetune_runtimealign_val8/text_prompt_finetune_epoch_1.pth`
- loss:
  - `loss_total=3.4362`
- converted:
  - `output/text_prompt_finetune_runtimealign_val8/efficient_sam3_repvit_m1.1_mobileclip_s1_ft_textprompt_runtimealign.pth`
  - replaced weights `671`

評価:

- `children`
  - baseline `mask_pixels=60233`
  - finetuned `mask_pixels=0`
- `groceries/object`
  - baseline `mask_pixels=50559`
  - finetuned `mask_pixels=0`

threshold を `0.0` まで下げた raw 確認では，

- `children`
  - total detections `200`
  - top selected score `0.0000`

だった．

解釈:

- checkpoint 自体が壊れて読めていないわけではない
- ただし tiny `val2017` pilot は score collapse を起こしており，
  実用的な text grounding 品質は完全に失われる
- 現時点では，保存漏れ修正と runtime 整合修正は正しかったが，
  `val2017` だけの超小規模 pilot は有効な品質改善実験になっていない
- 次に進めるべきなのは `train2017` を用いた本学習寄り pilot である

### 決定的な原因: custom StudentTrunk 初期化が baseline と一致していなかった

さらに no-op 変換を確認したところ，
学習を 1 step も回していないのに converted checkpoint が baseline と一致しないことが分かった．

旧 no-op 確認:

- baseline
  - `score_max=0.168258`
  - `score_mean=0.022564`
- old no-op converted
  - `score_max=0.0000017`
  - `score_mean=0.0000008`

解釈:

- これは学習崩壊ではなく，
  custom `StudentTrunk` の初期化自体が baseline EfficientSAM3 の実 trunk と一致していなかったことを示す

対策:

- `student_model = build_efficientsam3_image_model(...)` を先に作る
- `student_trunk = deepcopy(student_model.backbone.vision_backbone.trunk)` に切り替える
- trunk 出力が list なので，学習側では `[0]` を使う
- `stage1_checkpoint` の manual trunk load は不要にした

### convert prefix バグ修正

exact trunk に切り替えたあと，
`convert_text_prompt_finetune.py` 側で `trunk.model.model...` という二重 prefix ができており，
trunk 重みの大半が merged checkpoint へ戻っていなかった．

修正:

- trunk state key が `model.` で始まる場合:
  - `detector.backbone.vision_backbone.trunk.{key}`
- それ以外の legacy key:
  - `detector.backbone.vision_backbone.trunk.model.{key}`

修正後の no-op exact:

- `replaced_weights=671`
- baseline と no-op exact の raw score は完全一致
  - baseline `score_max=0.168258`, `score_mean=0.022564`
  - noop exact `score_max=0.168258`, `score_mean=0.022564`

これで，学習前の初期化と変換経路が baseline と整合した．

### exact trunk 版 pilot の結果

#### val8 / 1 epoch / lr=5e-6

- output:
  - `output/text_prompt_finetune_exact_val8/text_prompt_finetune_epoch_1.pth`
- loss:
  - `loss_total=2.3347`
- converted:
  - `output/text_prompt_finetune_exact_val8/efficient_sam3_repvit_m1.1_mobileclip_s1_ft_textprompt_exact.pth`
  - `replaced_weights=671`

評価:

- `children`
  - baseline `mask_pixels=60233`
  - finetuned `mask_pixels=11321`
  - finetuned `top_score=0.0949`
- `groceries/object`
  - baseline `mask_pixels=50559`
  - finetuned `mask_pixels=51108`
  - finetuned `top_score=0.4640`

解釈:

- 0 detections 地獄は脱出した
- exact trunk / exact convert の土台は正しい
- ただし `val8` では `children` の coverage はまだ弱い

#### val32 / 2 epochs / lr=5e-6

- output:
  - `output/text_prompt_finetune_exact_val32/text_prompt_finetune_epoch_2.pth`
- losses:
  - epoch0 `loss_total=1.5127`
  - epoch1 `loss_total=0.9731`
- converted:
  - `output/text_prompt_finetune_exact_val32/efficient_sam3_repvit_m1.1_mobileclip_s1_ft_textprompt_exact_val32.pth`
  - `replaced_weights=671`

評価:

- `children`
  - baseline `mask_pixels=60233`
  - finetuned `mask_pixels=59716`
  - baseline `top_score=0.7889`
  - finetuned `top_score=0.6716`
  - selected detections `8 / total 102`
- `groceries/object`
  - baseline `mask_pixels=50559`
  - finetuned `mask_pixels=46381`
  - baseline `top_score=0.4140`
  - finetuned `top_score=0.5283`

速度確認:

- command:
  - `benchmark_text_prompt_single_image.py --backend pytorch ... --encoder-feature-downsample 2`
- result:
  - `avg_latency_ms=65.75`
  - `throughput_fps=15.21`
  - `last_top_score=0.67157`

高速経路確認:

- command:
  - `benchmark_text_prompt_single_image.py --backend onnx_local --ort-provider tensorrt --trt-fp16 --trt-engine-cache --cache-text-prompt ... --encoder-feature-downsample 2`
- result:
  - `avg_latency_ms=59.34`
  - `throughput_fps=16.85`
  - `last_detection_count_indicator=103`
  - `last_top_score=0.67209`
  - `ort_providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`

結論:

- exact trunk + exact convert に直したことで，
  tiny pilot でも本当に品質改善が推論へ反映されるようになった
- `children` は baseline mask 面積 `60233` に対して `59716` まで回復しており，
  本家 SAM3 との差を詰める方向へ動き始めた
- 速度も PyTorch で `15.21 FPS`，TensorRT 経路で `16.85 FPS` まで維持している
- 次の本命は `train2017` を用いた本学習寄り pilot である

### COCO val2017 での汎化確認

`exact_val32` checkpoint が `children` 専用の偶然ではないかを切り分けるため，
COCO val から `person / car / dog / bottle` を抜き出して比較した．

使用画像:

- `person`: `val2017/000000425226.jpg`
- `car`: `val2017/000000508602.jpg`
- `dog`: `val2017/000000289343.jpg`
- `bottle`: `val2017/000000117914.jpg`

結果:

- `person`
  - baseline `mask_pixels=0`
  - finetuned `mask_pixels=49519`
  - finetuned `top_score=0.2883`
- `car`
  - baseline `mask_pixels=0`
  - finetuned `mask_pixels=0`
- `dog`
  - baseline `mask_pixels=0`
  - finetuned `mask_pixels=0`
- `bottle`
  - baseline `mask_pixels=0`
  - finetuned `mask_pixels=0`

解釈:

- `exact_val32` は少なくとも `person` では zero-shot を押し上げる兆候がある
- ただし `car / dog / bottle` まではまだ広がっていない
- つまり現段階では，
  - `children`, `person` 系の改善は確認
  - 語彙全体の汎化は未達
- 次の本命はやはり `train2017` を使った本学習寄り pilot

### teacher 基準への切り替え

比較基準を「蒸留前 EfficientSAM3」ではなく，
本家 SAM3 (`/ros2_ws/src/sam3/sam3.pt`) の出力へ切り替えるため，
評価スクリプトを拡張した．

- `save_text_prompt_mask.py`
  - `--use-sam3-model`
  - `--sam3-checkpoint`
- `evaluate_text_prompt_checkpoint.py`
  - `--teacher-checkpoint`
  - `iou_to_teacher`

#### teacher 基準の実測

対象:

- `test_image.jpg / children`
- `val2017/000000425226.jpg / person`

結果:

- `children`
  - teacher `mask_pixels=109970`
  - baseline `iou_to_teacher=0.4516`
  - finetuned `iou_to_teacher=0.4443`
- `person`
  - teacher `mask_pixels=49900`
  - baseline `iou_to_teacher=0.0000`
  - finetuned `iou_to_teacher=0.8418`

解釈:

- `children` は現状まだ teacher 基準では baseline と同等か，わずかに未満
- `person` は teacher 基準で大幅改善している
- したがって，今の exact-val32 蒸留は
- `children / person` 系には効き始めている
- ただし multi-person coverage を本家 SAM3 並みにするには train split 学習が必要

### train2017 pilot (2048 samples / 1 epoch) の結果

`train2017` の展開は完了した．

- 保存先:
  - `data/coco_pilot/train2017`
- 枚数:
  - `118287`
- 容量:
  - `19G`

16GB 向け pilot:

- script:
  - `stage1_geometry_finetune/scripts/train_text_prompt_finetune_16gb_pilot.sh`
- 実条件:
  - `split=train`
  - `num_samples=2048`
  - `epochs=1`
  - `batch_size=1`
  - `num_workers=2`
  - `lr=5e-6`
  - `score_weight=0.25`
  - `train-neck`
  - `amp`
- output:
  - `output/text_prompt_finetune_train_pilot/text_prompt_finetune_epoch_1.pth`
- loss:
  - `loss_total=1.3835`

所要時間:

- 約 `18分`

teacher 基準評価:

- `children`
  - teacher `mask_pixels=109970`
  - baseline `iou_to_teacher=0.4516`
  - finetuned `iou_to_teacher=0.0000`
- `person`
  - teacher `mask_pixels=49900`
  - baseline `iou_to_teacher=0.0000`
  - finetuned `iou_to_teacher=0.0000`

解釈:

- exact trunk / val32 では改善が出ていたが，
  この train pilot 設定は teacher 基準では崩壊した
- `train2017` を使えば自動的に改善するわけではない
- 現在の loss / prompt 設計のまま train split へ広げると不安定

次の打ち手:

- `score_weight` をさらに下げるか 0 にする
- train sample 数を減らして短い teacher-IoU 検証ループにする
- epoch 終了後ではなく途中 checkpoint で早めに評価する
