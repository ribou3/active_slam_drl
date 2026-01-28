import torch
import torch.nn as nn
import numpy as np

# --- ネットワーク定義 (学習時と同じ構造・サイズ256を維持) ---
class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(DuelingDQN, self).__init__()
        
        # 共通層
        self.feature = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU()
        )
        
        # Value Stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Advantage Stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvals

# --- テスト専用エージェント ---
class TestAgent:
    def __init__(self, input_dim, output_dim, device=None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        print(f"TestAgent initialized on: {self.device}")

        # モデル構築 (学習率は不要)
        self.policy_net = DuelingDQN(input_dim, output_dim, hidden_dim=256).to(self.device)
        self.policy_net.eval() # 推論モード

    def load_model(self, model_path):
        """モデルの重みをロードする"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # 保存形式のゆらぎに対応
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.policy_net.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.policy_net.load_state_dict(checkpoint)
                
            print(f"✅ Model loaded: {model_path}")
        except Exception as e:
            print(f"❌ Load failed: {e}")
            # サイズ不一致のエラーならヒントを出す
            if "size mismatch" in str(e):
                print("ヒント: 学習時のhidden_dim(256)と合っていない可能性があります。")
            raise e

    def get_action(self, state):
        """最適行動を選択 (Greedy)"""
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_t)
            return q_values.argmax().item()