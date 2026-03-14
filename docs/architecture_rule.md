# ルールと注意すること
## ルール
1. システムはdockerコンテナ内のpython仮想環境venvを用いて実行される．
2. コンテナ名はpiper-humble-devである．docker-compose.ymlは/home/guch1/ssd_yamaguchi/piper_ros/dockerにある．
3. **efficientsam3環境はコンテナ内パスで/ros2_ws/efficientsam3下のpython venv環境内**である．
4. **本家SAM3はコンテナ内パスで/ros2_ws/src/sam3下**にある．***動作テスト実行ファイルは/ros2_ws/src/sam3.run_sam3_groceries.py**である．ファイル内で画像パスとテキストプロンプトを記述している．

## 注意すること
1. **efficientSAM3の実行環境は仮想環境venvを利用しているので仮想環境を有効化**すること．
2. docker内でのパスとホストのパスが異なるので注意．コンテナでのワーキングディレクトリ設定やユーザ設定名が異なる．
3. コンテナ内ワーキングディレクトリは`/ros2_ws`である．また，コンテナ内ユーザはrootユーザである．
4. 依存関係や必要ライブラリは仮想環境内にインストールしているはずだが，不足があるかもしれない．