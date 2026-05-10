# 目的と全体像
最終目標は，SAM3 の text prompt 品質をできるだけ維持したまま軽量化し，推論速度を高速化することです．
目安としては，ROS2 上で Realsense の画像トピックを入力に 15 から 30 FPS の範囲を狙います．
同時に，都度異なる text prompt を与えても，本家 SAM3 に近い挙動でマスクを返せることを目標にします．

現時点の中心課題は，EfficientSAM3 の軽量 encoder を使うと text grounding の score が大きく落ち，本家 SAM3 と同等のしきい値では検出が成立しないことです．
したがって，今の実行計画は次の 2 系統で進めます．

- 品質側: text prompt 品質の劣化原因を切り分ける
- 速度側: encoder ONNX 化でどこまで実効速度を上げられるかを測る

## 現在判明していること
### 確認済みの事実
- encoder ONNX export は成功している
- `efficient_sam3_tinyvit_21m_mobileclip_s1.pth` は Stage 1 merged 系で，README 上も text 性能低下が想定内とされている
- 本家 SAM3 は `children` prompt で対象画像から複数人物を検出できる
- EfficientSAM3 は通常しきい値では `children` を検出できない
- ただし `confidence_threshold=0.05` まで下げると，EfficientSAM3 student text でも非ゼロマスクは返る
- `onnx_encoder_server.py` の HTTP 経路は遅く，速度改善の本命ではない
- `onnxruntime-gpu` を用いた同一プロセス `onnx_local` encoder 経路は，当初の実装では `scalp` 無視により誤った feature pyramid を作っていた
- `scalp` を反映して再構成すると，`onnx_local` の detection 件数は PyTorch と揃う
- ただし修正後の `onnx_local` は PyTorch 単体より少し遅く，現状では速度優位がない
- `onnx_split` でも detection 件数は維持できるが，さらに遅くなる
- TensorRT provider を導入すると，`onnx_local` は PyTorch をわずかに上回る
- ただし `onnx_split` は TensorRT を使ってもまだ PyTorch より遅い
- `backbone.forward_image()` を丸ごと ONNX 化した `onnx_backbone` も TensorRT では改善するが，最速ではない
- TensorRT の FP16 と engine cache を有効化すると，`onnx_local` が最速になる
- `prompt_encode` の再利用余地を調べると，`language_features/language_mask` 以上は image 依存の幾何 token が混ざるため，大きくは広げにくい
- `transformer encoder / decoder / seg head` をまとめた `grounding core` ONNX export は成功した
- ただし `encoder ONNX + decoder ONNX + grounding_core ONNX` を TensorRT FP16 で動かしても，`onnx_local` とほぼ同速で，100ms の壁は破れなかった
- `tinyvit_21m` は 1008 以外の解像度で動かず，解像度 sweep による最適化余地がない
- `repvit-m2.3` は `tinyvit_21m` より少し速いが，依然として 15 FPS には届かない
- 論文調査では `MobileSAM` の decoupled distillation と `EdgeSAM` の prompt-in-the-loop distillation が，既存の Stage1/Stage2 構成に最も接続しやすい
- 現状の text prompt 劣化には，`EdgeSAM` が指摘する「task-agnostic な encoder 蒸留だけでは不十分」という状況がそのまま当てはまる
- `/ros2_ws/src/sam3/sam3.pt` が本物の full SAM3 checkpoint として利用可能である
- 本家 SAM3 の単画像ベンチでは `children` で 8 detections / `2.16 FPS` と，品質は高いが速度は低い
- `piper-humble-dev` 上での再確認でも，本家 SAM3 は `children` に対して EfficientSAM3 より大幅に高い score を維持した
- `repvit-m2.3` + TensorRT FP16 は約 `9 FPS` まで到達するが，score は依然として本家 SAM3 よりかなり低い
- README の `ft` 重みを確認した結果，最有力候補は `RepViT-M1.1 ft` である
- `RepViT-M1.1 ft` は `repvit-m2.3` 非 ft より text prompt 品質が明確に改善しつつ，速度は同等以上だった
- upstream SAM3 video API を使った tracking benchmark（A/B/C 方式）を追加した
- 疑似動画では C方式（周期的 text prompt 補正）は B方式（初回検出+tracking）と同等精度で，速度はわずかに上回った
- ただし full SAM3 video model ベースでは 1 FPS 前後で，現行 EfficientSAM3 + TensorRT 単画像経路より大幅に遅い
- `TinyViT-11M ft` は今回の `children` prompt では detections=0 で，`RepViT-M1.1 ft` を超えなかった
- 16GB VRAM 環境では text prompt fine-tune のフル設定は厳しく，pilot run 前提に縮小条件が必要
- `EfficientSAM3 detector + SAM3 tracker` の hybrid periodic refresh benchmark を追加し，full SAM3 video よりは高速になることを確認した

### まだ未解決の点
- `encoder ONNX + PyTorch downstream` が遅い主因が ORT 呼び出し，GPU 転送，split 実行のどこか
- decoder/downstream を ONNX 化すべきか，split 実行をやめるべきか
- full text-seg ONNX export が現実的なコストで成立するか
- `grounding core` ONNX が `onnx_local` を超えない主因が，ONNX 境界なのか，prompt encode なのか，TensorRT 最適化限界なのか
- image token 数を削る設計変更で，どこまで精度を維持したまま 100ms を下回れるか
- TensorRT provider を使った場合に状況が変わるか
- 本家 SAM3 text encoder を残すハイブリッド構成で，十分な品質改善ができるか
- ROS2 実環境での FPS と遅延がどこまで出るか
- tracking 方式を EfficientSAM3 側へどう取り込むか
- tracking に periodic refresh を入れたとき，実動画でどこまで drift 補正が効くか
- hybrid tracking が 9 FPS 級の単画像最速経路へどこまで近づけるか

## マイルストーン
### M1. 品質の切り分け
- 本家 SAM3 と EfficientSAM3 を同一画像・同一 prompt で比較できる状態を維持する
- `confidence_threshold` を含む主要パラメータの影響を整理する
- teacher text / student text / visual encoder 差のどこが支配的かを切り分ける

完了条件:
- text grounding 劣化の主要因について，少なくとも次の方針が選べる
- `low threshold で運用`
- `追加 fine-tune`
- `decoder/downstream 再構成`

### M2. 速度経路の確立
- PyTorch 単体，HTTP server 経路，same-process ONNX encoder 経路を同条件で比較する
- ORT GPU の効果を定量化する
- 遅い構成は理由つきで除外する

完了条件:
- 実用候補となる runtime 構成を 1 つ以上決定できる

### M3. ROS2 ベンチマーク
- Realsense 画像トピックを入力に処理速度と平均遅延を測る
- 単画像ベンチとのギャップを記録する
- 必要なら publish 頻度や入力解像度も調整する

完了条件:
- `processed_fps` と `avg_latency_ms` を記録した比較結果が残る

## 検証戦略
速度と品質を同時に追うため，次の指標を基準にします．

- 品質:
  - detection 件数
  - top score
  - 非ゼロマスクの有無
  - bbox の妥当性
- 速度:
  - 平均推論時間
  - FPS
  - ROS2 上の平均遅延
- 実用性:
  - 可変 text prompt が使えるか
  - HTTP 通信など不要なオーバーヘッドがないか

比較対象は少なくとも次を維持します．

- 本家 SAM3
- EfficientSAM3 PyTorch
- EfficientSAM3 encoder ONNX + PyTorch downstream
- 必要なら decoder/downstream ONNX 追加経路

## 進捗
### 完了済み
- encoder export 用の import 問題を修正し，ONNX export を通した
- 単画像 text prompt 推論スクリプトを追加した
- ONNX server 経路のエラーハンドリングと device 不一致を修正した
- 単画像ベンチスクリプトを追加した
- `onnxruntime-gpu` を導入し，same-process ORT GPU 経路を測定した
- `onnx_local` の `scalp` 不整合を修正し，精度崩れの主因が手実装ミスだったことを確認した
- decoder ONNX export を `scalp` 対応に修正し，再 export を確認した
- full text-seg ONNX export の失敗要因を切り分け，legacy exporter でも重いことを確認した
- `onnx_split` を単画像ベンチに追加し，encoder/decoder を ORT に寄せても速度改善しないことを確認した
- 現行コンテナでは `libnvinfer.so.10` が無く，TensorRT provider は使えないことを確認した
- TensorRT ランタイム導入後，`TensorrtExecutionProvider` が encoder / decoder 両方で有効になることを確認した
- `backbone.forward_image()` 全体を ONNX export し，`onnx_backbone` ベンチを追加した
- TensorRT FP16 と engine cache を有効化した比較で，`onnx_local` が最速になることを確認した
- `tinyvit_21m` の解像度 sweep が実質不可能であることを確認した
- `repvit-m2.3` を追加比較し，より軽い backbone でも 15 FPS には未到達であることを確認した
- COCO category を text prompt として使う prompt-conditioned distillation 用データセットを追加した
- teacher/student の `forward_grounding` を直接比較する text prompt fine-tune モデルを追加した
- teacher best-query を基準に，mask BCE/Dice，score MSE，embedding MSE，GT mask 補助損失を組み合わせる学習スクリプトを追加した

### 進行中
- `encoder ONNX + PyTorch downstream` が PyTorch より遅い理由の切り分け
- decoder/downstream ONNX 化で split 実行オーバーヘッドをどこまで削れるかの見極め
- full text-seg ONNX を主経路から外し，分割 export を優先する設計整理
- full SAM3 teacher checkpoint を使った text prompt fine-tune の実学習検証
- full SAM3 checkpoint を品質基準として使い，EfficientSAM3 との差分を同一条件で測る
- quality/speed の二軸を明示し，「品質基準モデル」と「実運用モデル」を分けて運用判断する
- まずは `ft` がある中サイズ backbone を優先比較し，大型 backbone の非 ft より優先する
- tracking 用の A/B/C benchmark を追加し，方式Cの periodic refresh を比較可能にした
- 同一 prompt の text encoder 出力を再利用する `--cache-text-prompt` を benchmark に追加した
- runtime 軽量化の第一弾として `--max-queries`, `--max-encoder-layers`, `--max-decoder-layers` を benchmark に追加した
- image token 削減の runtime 実験として `--encoder-feature-downsample` を benchmark に追加した
- fixed prompt 向け full downstream ONNX を試したが，geometry encoder を含む export は `exit 136` で失敗した
- `grounding core` 専用 exporter を追加し，後段 TensorRT 化の成立性を確認した

### 次にやること
- full SAM3 teacher checkpoint を用意して text prompt fine-tune を短く回す
- `children` などの prompt で fine-tune 前後の検出件数と score を比較する
- 改善後に TensorRT 最速経路で再度速度確認を行う
- fine-tune が重ければ，学習は後回しにして `sam3.pt` を教師兼ベースラインにした推論比較を先に詰める
- 学習なしで進める場合は，threshold と backbone を調整しつつ EfficientSAM3 の実用域を探る
- 現時点の第一候補は `repvit-m1.1 ft + onnx_local + TensorRT FP16`
- tracking 側は full SAM3 video model では遅いため，次は EfficientSAM3 の最速経路と組み合わせられる補正方式を考える
- 16GB VRAM では full 設定 fine-tune は重いので，やるなら batch 1 / AMP / 少量サンプルの pilot run から始める
- hybrid tracking は成立したが，最良でも `3.50 FPS` なので tracker 部のさらなる削減が必要
- 単画像最速経路では，構造変更よりも prompt cache のような安全な runtime 最適化を優先する
- bottleneck profiling の結果，最大のボトルネックは image encoder ではなく transformer encoder である
- runtime 最適化だけでは約 `103 ms` が壁であり，次の改善は計算量自体を減らす設計変更が必要
- 最終 visual level の規則的 downsample は，精度を大きく崩さず `70.02 ms / 14.28 FPS` まで改善した
- 複数ケース比較では `encoder-feature-downsample=2` が安定構成で，`17.6-18.0 FPS` を維持した
- `encoder-feature-downsample=3` は `19 FPS` 前後まで伸びるが，人物ケースで IoU 低下が目立つ
- 複数候補の保存は `best 1件` をやめ，`topk_nms` による union mask と複数 bbox を標準化した
- ROI 再精査では `prompt ensemble` よりも `geometric box refine` が有望で，本家 SAM3 fallback よりはるかに軽い
- `geometric box refine` は score だけでなく fill ratio を基準にする方が安定しやすい
- text prompt 蒸留側では prompt variants と box 補助損失を入れる準備が整った
- COCO pilot (`annotations + val2017`) は取得完了し，`split=val` の最小学習 smoke まで通った

## 発見と驚き
- teacher text encoder に戻しても，現行の TinyViT-21M Stage 1 merged では本家 SAM3 相当の text grounding は回復しなかった
- `confidence_threshold=0.05` まで下げると，完全空マスクではなくなる
- ONNX server は encoder が ONNX でも HTTP 往復のせいで極端に遅い
- ORT GPU same-process 経路の精度崩れは score calibration ではなく，`scalp` 無視の手実装ミスが主因だった
- 正しい pyramid に揃えると精度は戻るが，速度は PyTorch 単体よりやや遅い
- full text-seg ONNX は exporter を調整してもなお重く，固定 prompt の完全 ONNX は想像以上にハードルが高い
- decoder まで ORT に分割しても，PyTorch 単体を超えられず，分割実行そのものの限界が見えてきた
- ORT provider 一覧に TensorRT は見えても，実際には TensorRT ライブラリ不足で使えないことがある
- TensorRT ランタイムを実際に入れると `onnx_local` は改善し，CUDA EP より明確に速くなる
- しかし分割構成の `onnx_split` は TensorRT を使ってもなお PyTorch に負ける
- 画像側をより大きい 1 ONNX にまとめた `onnx_backbone` も悪くないが，`onnx_local` を超えなかった
- 「より大きい 1 graph が常に最速」というわけではなく，TensorRT の最適化結果次第で逆転する
- 解像度を下げて改善するつもりだったが，`tinyvit_21m` は固定解像度前提でそこが使えなかった
- backbone を `repvit-m2.3` に替えても改善は限定的で，15 FPS へはまだ距離がある
- full SAM3 の tracking は精度側には有望でも，そのままでは速度改善にならない
- periodic refresh は tracking only と同等以上に動いたが，ベースモデルが重いため runtime の本命にはなりにくい
- EfficientSAM3 detector と組み合わせると full SAM3 video より改善するが，tracker 自体がまだ重く，15 FPS 目標には届かない
- 同一 prompt の text encoder 再利用は小幅だが安全に効く最適化だった
- `grounding core` まで ONNX/TensorRT に寄せても，定常時は `onnx_local` とほぼ同速で，後段 ONNX 化だけでは決定打にならなかった
- image token 削減は，query/層削減よりも本質的な改善方向である可能性が高い
- encoder 層削減は速度効果が大きいが，品質低下を伴うため threshold 再調整込みで扱う必要がある
- same-model の geometric prompt は，text prompt 由来の粗い box を軽く再利用できる
- SAM3 fallback は局所 ROI でも重く，品質上限確認用途に留めるべき

## 判断履歴
- 可変 text prompt を維持したいので，固定 prompt 専用 ONNX は主経路にしない
- 速度評価では HTTP server を主戦略から外し，same-process runtime を優先する
- 現段階では `low threshold` は暫定運用であり，根本解決ではない
- `score calibration` 単独よりも，まず same-process 再構成の正しさを揃える
- その結果を踏まえ，次の本筋は `decoder/downstream ONNX 化` の有効性見極めに移る
- fixed prompt 完全 ONNX は副次評価に留め，主戦略にはしない
- `onnx_split` も遅いので，主戦略は「分割 ONNX の延長」ではなく「call boundary を減らす構成」へ寄せる
- 当面の最有力候補は `encoder ONNX + TensorRT EP + PyTorch downstream` の `onnx_local`
- 当面の最有力候補は `encoder ONNX + TensorRT EP + FP16 + engine cache + PyTorch downstream` の `onnx_local`
- 15 FPS を本気で狙うなら，次はさらに小さい backbone か，text 側計算の削減が必要
- ただし backbone を小さくするだけでは text prompt 品質を落としやすいので，次の改善軸は text prompt 条件付き蒸留を優先する
- tracking を使うなら，「重い text grounding を毎フレーム回さない」こと自体は有効だが，full SAM3 video model の採用は速度目標と両立しない
- 直近の最速値は `repvit-m1.1 ft + onnx_local + TensorRT FP16 + cache-text-prompt` の `9.58 FPS`
- 実験用の最速値は `Q=100 + encoder_layers=4` で `12.00 FPS`，品質寄りの候補は `Q=64 + encoder_layers=4 + threshold=0.03` で `11.78 FPS`
- 精度維持を重視すると，現時点の最良 runtime は依然として `onnx_local + cache-text-prompt`
- ただし現時点の最良 speed/quality trade-off は `onnx_local + cache-text-prompt + encoder-feature-downsample=2`
- 20 FPS を狙う次段階は，`factor=3` の精度低下を抑える設計か，適応的な token 圧縮の導入になる
- ROI 再精査を使うなら，本命は `geometric_box` であり，SAM3 fallback は主経路に入れない
- 学習の本命は trunk-only のままではなく，必要なら lightweight neck までを含む pilot distillation に拡張する
- `train2017` フル取得は回線時間が長いため，まずは `val2017` smoke で学習経路の成立を優先する

## リスクと未解決事項
- Stage 1 merged checkpoint のままでは，本家 SAM3 品質の再現が難しい可能性が高い
- 手元に `ft` 系 checkpoint が無く，追加比較の幅が限られている
- ROS2 / Realsense 実機入力はまだ立ち上がっておらず，最終ベンチに未着手
- 現在の split 実行は速度優位が出ておらず，decoder/downstream を含む再設計が必要になる可能性が高い
- full text-seg ONNX は export 自体が重く，環境メモリ制約に引っかかる可能性がある
- ORT の `CUDAExecutionProvider` だけでは十分でなく，TensorRT provider の利用可否やエンジン化の手間が新たな論点になる
- 現状コンテナでは `libnvinfer.so.10` が無いため，TensorRT provider の追加検証には環境更新が必要
- TensorRT 導入コストが大きく，コンテナ容量と再現性の管理が必要
- text prompt fine-tune は full SAM3 teacher checkpoint を必要とし，teacher 重みが無いと本検証を回せない
- geometric box refine は score を大きく改善するが，mask 面積の増加は限定的で，見た目品質の改善量はまだ要検証
- 公開データ蒸留は足場を強化したが，COCO 実データでの学習効果はまだ未検証
- `val2017` smoke は通ったが，本学習に必要な `train2017` はまだ未取得
