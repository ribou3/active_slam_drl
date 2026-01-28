import rclpy
from rclpy.executors import MultiThreadedExecutor
import threading
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import csv  # ★追加: これが必要です

from active_slam_rl.d3qn_agent import D3QNAgent
from active_slam_rl.memory import PrioritizedReplayBuffer
from active_slam_rl.turtlebot_env import TurtleBotEnv

#30→300
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
    'model_save_dir': './training_output_slam'
}

# ★追加: これがないと動きません
def save_csv(save_dir, episode, steps, total_reward, slam_reward, epsilon):
    file_path = os.path.join(save_dir, "training_log.csv")
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        # ファイルが新規ならヘッダーを書く
        if not file_exists:
            writer.writerow(['Episode', 'Steps', 'Total_Reward', 'SLAM_Reward', 'Epsilon'])
        
        writer.writerow([episode, steps, total_reward, slam_reward, epsilon])

def save_graph(rewards, slam_rewards, save_dir, run_id):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Total Reward')
    plt.plot(slam_rewards, label='SLAM Component', alpha=0.6) # SLAM報酬もグラフ化
    plt.title(f'Active SLAM Training - {run_id}')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "reward_graph.png"))
    plt.close()

def main():
    rclpy.init()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(PARAMS['model_save_dir'], run_id)
    os.makedirs(save_dir, exist_ok=True)
    
    print("🔄 Initializing Active SLAM Environment...")
    env = TurtleBotEnv()

    executor = MultiThreadedExecutor()
    executor.add_node(env)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    print("✅ ROS 2 Spinner started.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
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
    rewards_history = []
    slam_rewards_history = [] # 履歴保存用

    try:
        for episode in range(1, PARAMS['max_episodes'] + 1):
            print(f"--- Episode {episode} Start (Resetting SLAM...) ---")
            state = env.reset()
            episode_reward = 0
            episode_slam_reward = 0 # エピソードごとのSLAM報酬合計
            
            for step in range(PARAMS['max_steps_per_episode']):
                total_steps += 1
                action = agent.select_action(state, epsilon)
                
                # infoを受け取る
                next_state, reward, done, info = env.step(action)
                
                # SLAM報酬の集計 (重み付け後の値を使用)
                slam_gain = info.get("slam_reward_weighted", 0.0)
                episode_slam_reward += slam_gain

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
            
            rewards_history.append(episode_reward)
            slam_rewards_history.append(episode_slam_reward)
            
            epsilon = max(PARAMS['epsilon_end'], epsilon - epsilon_decay)
            
            # ログにSLAM報酬を表示
            print(f"Ep: {episode} | Total: {episode_reward:.2f} | SLAM: {episode_slam_reward:.2f} | Eps: {epsilon:.4f} | Steps: {step}")
            
            # ★ここで呼び出すために関数定義が必要です
            save_csv(save_dir, episode, step, episode_reward, episode_slam_reward, epsilon)
            
            if episode % 20 == 0:
                torch.save(agent.policy_net.state_dict(), os.path.join(save_dir, f"d3qn_ep{episode}.pth"))
                save_graph(rewards_history, slam_rewards_history, save_dir, run_id)

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        torch.save(agent.policy_net.state_dict(), os.path.join(save_dir, "d3qn_final.pth"))
        save_graph(rewards_history, slam_rewards_history, save_dir, run_id)
        print("Training finished.")
        env.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()