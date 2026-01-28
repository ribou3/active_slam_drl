[README.md](https://github.com/user-attachments/files/24903061/README.md)
# Active SLAM with D3QN (Double Dueling Deep Q-Network)

## 📌 プロジェクト概要
本プロジェクトは、未知環境においてロボット（TurtleBot3）が効率的に地図作成を行う「Active SLAM」を、深層強化学習（D3QN）を用いて実装・検証するものです。
ROS 2 Humble と Gazebo シミュレーション環境をベースに、自己位置推定と環境探索の最適化を目指しています。

## 💻 現在の進捗状況（2026/01/28時点）
* **環境構築**: WSL2 (Ubuntu 22.04) 上に ROS 2 Humble 環境を再構築完了。
* **依存関係**: NumPy 1.x 系へのダウングレードにより、Matplotlib および PyTorch の実行環境を安定化。
* **学習フェーズ**: `ros2_train_no_slam.py` にて、比較用データとなる「SLAMなし状態でのナビゲーション学習」を GPU (CUDA) を用いて実行中。
* **動作確認**: Gazeboのリセットサービスに一部警告が出るものの、フォールバック処理によりエピソードの継続的な回行を確認済み。

## 📂 主要ディレクトリ構成 [cite: 1, 2, 23, 24]
* `active_slam_rl/`: メインのソースコード（D3QNエージェント、報酬設計、環境定義）
* `models/`: シミュレーション用のカスタムモデル（Circuit, Maze等）
* `worlds/`: 学習用 Gazebo ワールドファイル群
* `training_output_.../`: 学習済みモデル（.pth）および学習ログ（CSV/グラフ）の出力先

## 🚀 実行方法
### 1. ワークスペースのビルド
```bash
cd ~/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

### 2. シミュレーション環境の起動
```bash
ros2 launch active_slam_rl training_env.launch.py
```

### 3. 学習の開始（No SLAM 比較用）
```bash
python3 ros2_train_no_slam.py
```

---
*Student ID: 922901*  
*Affiliation: Osaka Institute of Technology*
