import rclpy
from rclpy.executors import MultiThreadedExecutor
import threading
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd  # グラフの移動平均計算に使用
import csv           # CSV保存に使用
from datetime import datetime
import sys

from active_slam_rl.d3qn_agent import D3QNAgent
from active_slam_rl.memory import PrioritizedReplayBuffer
from active_slam_rl.turtlebot_env_no_slam import TurtleBotEnv

# ==========================================
# ハイパーパラメータ設定
# ==========================================
PARAMS = {
    'learning_rate': 0.00025,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.02,
    'exploration_episodes': 50,
    'target_update_interval': 1000,
    'memory_capacity': 20000,
    'batch_size': 64,
    'max_episodes': 2000,
    'max_steps_per_episode': 500,
    'input_dim': 100,
    'output_dim': 3,
    'model_save_dir': './training_output_no_slam'
}

# ==========================================
# グラフ作成関数 (Pandas 2.0対応 修正版)
# ==========================================
def save_paper_style_graph(csv_path, save_dir, run_id):
    """
    CSVデータを読み込み、論文のような「生データ(薄い色) + 移動平均(濃い色)」のグラフを作成して保存する
    """
    try:
        if not os.path.exists(csv_path):
            return

        df = pd.read_csv(csv_path)
        
        if len(df) < 2:
            return

        # --- 移動平均の計算 ---
        window_size = 50
        df['ma'] = df['reward'].rolling(window=window_size, min_periods=1).mean()

        plt.figure(figsize=(10, 6))

        # ★★★ 修正点: .to_numpy() を追加して型エラーを回避 ★★★
        episodes = df['episode'].to_numpy()
        rewards = df['reward'].to_numpy()
        ma_rewards = df['ma'].to_numpy()

        # 1. 生データ（報酬）のプロット
        plt.plot(episodes, rewards, color='#90EE90', alpha=0.6, linewidth=1.5, label='Raw Reward')

        # 2. 移動平均のプロット
        plt.plot(episodes, ma_rewards, color='#006400', linewidth=2.5, label=f'Moving Avg (Win={window_size})')

        # --- グラフの装飾 ---
        plt.title(f'D3QN Training Curve - {run_id}', fontsize=14)
        plt.xlabel('Episodes', fontsize=12)
        plt.ylabel('Cumulated Reward', fontsize=12)
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # 画像として保存
        plt.savefig(os.path.join(save_dir, "training_curve.png"), dpi=300)
        plt.close()

    except Exception as e:
        print(f"⚠️ Graph plotting failed: {e}")

# ==========================================
# メイン処理
# ==========================================
def main():
    rclpy.init()
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(PARAMS['model_save_dir'], run_id)
    os.makedirs(save_dir, exist_ok=True)
    
    # --- CSVファイルの初期化 ---
    csv_path = os.path.join(save_dir, "log_data.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'reward', 'epsilon', 'steps'])

    print("🔄 Initializing Environment...")
    env = TurtleBotEnv()

    # Executorを別スレッドで回す
    executor = MultiThreadedExecutor()
    executor.add_node(env)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    print("✅ ROS 2 Spinner started.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"🚀 Starting Navigation Training (No SLAM): {run_id}")
    
    agent = D3QNAgent(
        input_dim=PARAMS['input_dim'],
        output_dim=PARAMS['output_dim'],
        lr=PARAMS['learning_rate'],
        gamma=PARAMS['gamma'],
        batch_size=PARAMS['batch_size'],
        device=device
    )
    
    memory = PrioritizedReplayBuffer(PARAMS['memory_capacity'])
    
    epsilon = PARAMS['epsilon_start']
    epsilon_decay = (PARAMS['epsilon_start'] - PARAMS['epsilon_end']) / PARAMS['exploration_episodes']
    
    total_steps = 0

    try:
        for episode in range(1, PARAMS['max_episodes'] + 1):
            print(f"--- Episode {episode} Start ---")
            
            state = env.reset()
            episode_reward = 0
            
            for step in range(PARAMS['max_steps_per_episode']):
                total_steps += 1
                action = agent.select_action(state, epsilon)
                next_state, reward, done = env.step(action)
                
                with torch.no_grad():
                    s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
                    ns_t = torch.FloatTensor(next_state).unsqueeze(0).to(device)
                    q_val = agent.policy_net(s_t)[0][action]
                    target_q = reward + PARAMS['gamma'] * torch.max(agent.target_net(ns_t)) * (1 - int(done))
                    td_error = abs(target_q - q_val).item()
                
                memory.add(td_error, (state, action, reward, next_state, done))
                state = next_state
                episode_reward += reward
                
                agent.update_model(memory)
                
                if total_steps % PARAMS['target_update_interval'] == 0:
                    agent.update_target_network()
                
                if done:
                    break
            
            epsilon = max(PARAMS['epsilon_end'], epsilon - epsilon_decay)
            print(f"Ep: {episode} | Reward: {episode_reward:.2f} | Eps: {epsilon:.4f} | Steps: {step}")
            
            # --- CSV保存 ---
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode, episode_reward, epsilon, step])

            # --- モデル保存 & グラフ更新 (10エピソードごと) ---
            if episode % 10 == 0:
                torch.save(agent.policy_net.state_dict(), os.path.join(save_dir, f"d3qn_ep{episode}.pth"))
                save_paper_style_graph(csv_path, save_dir, run_id)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        print("Saving final model...")
        torch.save(agent.policy_net.state_dict(), os.path.join(save_dir, "d3qn_final.pth"))
        save_paper_style_graph(csv_path, save_dir, run_id)
        print("Training finished.")
        env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()