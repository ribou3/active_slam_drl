# 

import rclpy
from rclpy.executors import MultiThreadedExecutor
import threading
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import csv

from active_slam_rl.d3qn_agent import D3QNAgent
from active_slam_rl.memory import PrioritizedReplayBuffer
from active_slam_rl.turtlebot_env_frontier import TurtleBotEnvFrontier
from active_slam_rl.active_slam_reward_node import ActiveSLAMRewardNode

PARAMS = {
    'learning_rate': 0.00025,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.02,
    'exploration_episodes': 30,
    'target_update_interval': 1000,
    'memory_capacity': 20000,
    'batch_size': 64,
    'max_episodes': 2000,
    'max_steps_per_episode': 500,
    'input_dim': 100,
    'output_dim': 3,
    'model_save_dir': './training_output_frontier'
}

# CSV保存関数を拡張
def save_csv(save_dir, episode, steps, total_reward, intrinsic, d_opt, frontier, epsilon):
    file_path = os.path.join(save_dir, "training_log.csv")
    file_exists = os.path.isfile(file_path)
    with open(file_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            # ヘッダーに D_Opt と Frontier を追加
            writer.writerow(['Episode', 'Steps', 'Total_Reward', 'Intrinsic_Reward', 'D_Opt_Reward', 'Frontier_Reward', 'Epsilon'])
        writer.writerow([episode, steps, total_reward, intrinsic, d_opt, frontier, epsilon])

def save_graph(rewards, intrinsic_rewards, d_opt_rewards, frontier_rewards, save_dir, run_id):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label='Total Reward')
    plt.title('Total Reward')
    plt.xlabel('Episode')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(intrinsic_rewards, label='Total Intrinsic', color='purple', alpha=0.3)
    plt.plot(d_opt_rewards, label='D-opt', color='blue')
    plt.plot(frontier_rewards, label='Frontier', color='orange')
    plt.title('Intrinsic Rewards Breakdown')
    plt.xlabel('Episode')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "reward_breakdown.png"))
    plt.close()

def main():
    rclpy.init()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_frontier")
    save_dir = os.path.join(PARAMS['model_save_dir'], run_id)
    os.makedirs(save_dir, exist_ok=True)
    
    print("🔄 Initializing Frontier Active SLAM System (Detailed Logging)...")
    
    env = TurtleBotEnvFrontier()
    reward_node = ActiveSLAMRewardNode()

    executor = MultiThreadedExecutor()
    executor.add_node(env)
    executor.add_node(reward_node)
    
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
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
    intrinsic_history = []
    d_opt_history = []
    frontier_history = []

    try:
        for episode in range(1, PARAMS['max_episodes'] + 1):
            print(f"--- Episode {episode} Start ---")
            state = env.reset(episode=episode)
            episode_reward = 0
            episode_intrinsic = 0
            episode_d_opt = 0
            episode_frontier = 0
            
            for step in range(PARAMS['max_steps_per_episode']):
                total_steps += 1
                action = agent.select_action(state, epsilon)
                
                next_state, reward, done, info = env.step(action)
                
                # 詳細報酬の取得
                intrinsic = info.get("intrinsic_reward", 0.0)
                d_opt = info.get("d_opt_reward", 0.0)
                frontier = info.get("frontier_reward", 0.0)
                
                episode_intrinsic += intrinsic
                episode_d_opt += d_opt
                episode_frontier += frontier

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
            intrinsic_history.append(episode_intrinsic)
            d_opt_history.append(episode_d_opt)
            frontier_history.append(episode_frontier)
            
            epsilon = max(PARAMS['epsilon_end'], epsilon - epsilon_decay)
            
            print(f"Ep: {episode} | Total: {episode_reward:.2f} | Int: {episode_intrinsic:.2f} (D-opt:{episode_d_opt:.2f}, Front:{episode_frontier:.2f}) | Eps: {epsilon:.4f}")
            
            # CSV保存
            save_csv(save_dir, episode, step, episode_reward, episode_intrinsic, episode_d_opt, episode_frontier, epsilon)
            
            if episode % 20 == 0:
                torch.save(agent.policy_net.state_dict(), os.path.join(save_dir, f"d3qn_frontier_ep{episode}.pth"))
                save_graph(rewards_history, intrinsic_history, d_opt_history, frontier_history, save_dir, run_id)

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        torch.save(agent.policy_net.state_dict(), os.path.join(save_dir, "d3qn_frontier_final.pth"))
        save_graph(rewards_history, intrinsic_history, d_opt_history, frontier_history, save_dir, run_id)
        env.destroy_node()
        reward_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()