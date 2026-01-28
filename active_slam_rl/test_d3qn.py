import torch
import torch.nn as nn
import numpy as np
import os

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

class TestD3QNAgent:
    def __init__(self, input_dim=100, output_dim=3, device=None):
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DuelingDQN(input_dim, output_dim, hidden_dim=256).to(self.device)
        self.policy_net.eval()

    def load_model(self, model_path):
        if not os.path.exists(model_path): raise FileNotFoundError(f"Model not found: {model_path}")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.policy_net.load_state_dict(checkpoint)
            print(f"✅ Model loaded: {os.path.basename(model_path)}")
        except Exception as e:
            print(f"❌ Load failed: {e}")
            raise e

    def get_action(self, state):
        with torch.no_grad():
            # 入力シェイプの確認ログ
            state_np = np.array(state)
            
            # シェイプ整形
            if state_np.ndim == 1: state_np = state_np.reshape(1, -1)
            elif state_np.ndim == 3: state_np = state_np.reshape(1, -1)
            
            state_t = torch.FloatTensor(state_np).to(self.device)
            q_values = self.policy_net(state_t)
            
            # ★★★ 診断ログ: Q値の中身を見る ★★★
            q_list = q_values.cpu().numpy()[0]
            action = q_values.argmax().item()
            
            print(f"[DEBUG AI] Q-Values: {q_list} | Selected: {action}")
            
            if q_list[0] == q_list[1] == q_list[2]:
                print("⚠️ WARNING: All Q-values are identical! (Brain Dead)")
            
            return action