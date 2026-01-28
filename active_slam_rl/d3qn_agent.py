import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(DuelingDQN, self).__init__()
        
        # 共通層 (Feature Extraction)
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(), #
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        
        # Value Stream (状態価値 V(s))
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage Stream (行動優位性 A(s, a))
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Aggregation Layer: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

class D3QNAgent:
    def __init__(self, input_dim, output_dim, lr, gamma, batch_size, device):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.gamma = gamma       # [cite: 1] 割引率
        self.batch_size = batch_size
        self.device = device

        # ネットワークの初期化
        self.policy_net = DuelingDQN(input_dim, output_dim).to(device) # θ
        self.target_net = DuelingDQN(input_dim, output_dim).to(device) # θ-
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr) # [cite: 1] 学習率
        
        # PER用にLossを手動計算するため、nn.MSELoss()は直接使いません
        # self.loss_fn = nn.MSELoss() 

    def select_action(self, state, epsilon):
        # ε-greedy ポリシー
        if random.random() < epsilon:
            return random.randrange(self.output_dim)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.argmax().item()

    def update_model(self, memory):
        if len(memory) < self.batch_size:
            return None

        # PERからミニバッチサンプリング (weightsを受け取る)
        batch, idxs, weights = memory.sample(self.batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)

        # 重みをTensor化してデバイスへ転送 [修正箇所]
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Double DQN Logic
        # 1. 行動選択は Policy Net (θ)
        next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
        # 2. Q値評価は Target Net (θ-)
        next_q_values = self.target_net(next_states).gather(1, next_actions)
        
        # ターゲット計算: r + γ * Q(s', argmax Q(s', a; θ); θ-)
        expected_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # 現在のQ値
        curr_q_values = self.policy_net(states).gather(1, actions)

        # TD誤差計算 (優先度更新用)
        # loss計算用に勾配を切らない誤差も保持したいが、PrioritizedReplayBufferのupdateにはnumpyが必要
        diff = expected_q_values - curr_q_values
        errors = torch.abs(diff).detach().cpu().numpy()
        
        for i in range(self.batch_size):
            memory.update(idxs[i], errors[i][0])

        # --- ロス計算の修正 ---
        # 重み付きMSE (Weighted MSE): (Q_target - Q_main)^2 * weights
        loss = (weights * (diff ** 2)).mean()
        
        self.optimizer.zero_grad()
        loss.backward()

        # --- 安定化のための追加: 勾配クリッピング ---
        # 勾配が爆発して学習が壊れるのを防ぎます（D3QNでは推奨）
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)

        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        # ターゲットネットのハード更新
        self.target_net.load_state_dict(self.policy_net.state_dict())