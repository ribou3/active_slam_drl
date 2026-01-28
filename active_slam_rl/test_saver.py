import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class TestResultSaver:
    def __init__(self, base_dir):
        self.base_dir = base_dir

    def save_episode(self, episode, backend_data):
        """
        エピソード完了時のデータを保存するメイン関数
        backend_data: TestBackendNodeから渡されるデータオブジェクト
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(self.base_dir, f"{timestamp}_ep{episode}")
        os.makedirs(save_dir, exist_ok=True)

        # 1. メトリクス(CSV)の保存
        self._save_csv(save_dir, backend_data)

        # 2. 地図+不確実性(画像)の生成と保存
        self._save_map_image(save_dir, backend_data)

        # 3. グラフ(画像)の生成と保存
        self._save_graph_image(save_dir, backend_data)

        return save_dir

    def _save_csv(self, save_dir, data):
        csv_path = os.path.join(save_dir, "metrics.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Step', 'D_opt', 'Frontier', 'Coverage'])
            
            length = len(data.d_opt_history)
            for i in range(length):
                d = data.d_opt_history[i]
                fr = data.frontier_history[i] if i < len(data.frontier_history) else 0.0
                # Coverageは履歴があればそれを使う、なければ最新のみ
                cov = data.coverage_history[i] if i < len(data.coverage_history) else 0.0
                writer.writerow([i, d, fr, cov])

    def _save_map_image(self, save_dir, data):
        """GUIの画面キャプチャではなく、データから高品質な図を再描画して保存"""
        if data.map_data is None: return

        fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        # 地図データ展開
        w = data.map_data.info.width
        h = data.map_data.info.height
        res = data.map_data.info.resolution
        ox = data.map_data.info.origin.position.x
        oy = data.map_data.info.origin.position.y
        
        grid = np.array(data.map_data.data).reshape((h, w))
        
        # 表示用配列作成 (-1:Gray, 0:White, 100:Black)
        disp = np.full_like(grid, 127, dtype=np.uint8)
        disp[grid == 0] = 255
        disp[grid == 100] = 0
        
        ax.imshow(disp, cmap='gray', origin='lower', extent=[ox, ox+w*res, oy, oy+h*res], alpha=0.8)
        
        # 軌跡と不確実性の描画
        if data.uncertainty_map:
            ux, uy, ud = zip(*data.uncertainty_map)
            sc = ax.scatter(ux, uy, c=ud, cmap='jet', s=5, alpha=0.9)
            plt.colorbar(sc, ax=ax, label="Uncertainty (D-opt)")

        ax.set_title("Result Map & Uncertainty Trajectory", color='white')
        ax.tick_params(colors='white')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "map_final.png"))
        plt.close(fig)

    def _save_graph_image(self, save_dir, data):
        if not data.d_opt_history: return

        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
        ax.plot(data.d_opt_history, label='D-optimality', color='blue')
        if data.frontier_history:
            ax.plot(data.frontier_history, label='Frontier Reward', color='orange', alpha=0.7)
        
        ax.set_title("Metrics History")
        ax.set_xlabel("Steps")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "graph_metrics.png"))
        plt.close(fig)