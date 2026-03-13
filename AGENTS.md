# Repository Guidelines

## Project Structure & Module Organization
このリポジトリは軽量 SAM3 系モデルと学習補助コードをまとめています。推論系の実装は `efficientsam/`、Stage 1 蒸留は `stage1/`、幾何プロンプト微調整は `stage1_geometry_finetune/`、評価スクリプトは `eval/`、ROS ノードは `ros_wrappers/` にあります。上流 SAM3 由来の実装と補助資産は `sam3/` にまとまっているため、変更時は派生コードか上流コードかを明確に分けてください。データ取得補助は `data/`、開発用 Docker スクリプトは `scripts/`、画像や図は `images/` を使います。

## Build, Test, and Development Commands
このリポジトリでは開発作業を Docker 内で完結させます。

```bash
bash scripts/start_dev_env.sh
docker exec -it efficientsam3-dev bash
docker exec -it efficientsam3-dev bash -lc "python3 -m pip install -e '.[dev,stage1]'"
docker exec -it efficientsam3-dev bash -lc "python3 -m pytest"
docker exec -it efficientsam3-dev bash -lc "python3 sam3/scripts/benchmark_inference_optimizations.py --preset vram8"
bash scripts/stop_dev_env.sh
```

`start_dev_env.sh` は GPU 付き開発コンテナをビルドして起動します。`pytest` は追加したテストの回帰確認用です。ベンチマーク変更時は `--preset vram8` の実測結果も添えてください。

## Coding Style & Naming Conventions
Python 3.12 を前提に、4 スペースインデント、`snake_case` 関数、`PascalCase` クラス、英語の識別子を使います。整形と静的検査は `black`、`ruff`、`mypy` を基準にします。設定の詳細は主に `sam3/pyproject.toml` にあり、`black` の行長は 88 です。重みファイル、動画、生成物は `.gitignore` 対象なのでコミットしないでください。

## Testing Guidelines
現状、トップレベルの専用 `tests/` は薄いため、変更に近い単位で最小の回帰テストを追加してください。命名は `test_*.py`、関数名は `test_*` に統一します。学習や推論を触る変更では、少なくとも対象スクリプトの smoke test、必要なら `eval/` または `sam3/scripts/` の再実行結果を残してください。

## Commit & Pull Request Guidelines
履歴では `Add ...` のような短い命令形英語サマリが中心です。1 コミット 1 目的を守り、大きい重みやデータは含めません。PR には目的、変更箇所、再現コマンド、性能差分、必要ならスクリーンショットや定量結果を記載し、データセットやチェックポイントの取得前提も明記してください。
