import time
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.parameter import Parameter

try:
    from active_slam_rl.turtlebot_env import TurtleBotEnv
    from active_slam_rl.turtlebot_env_frontier import TurtleBotEnvFrontier
    from active_slam_rl.turtlebot_env_no_slam import TurtleBotEnv as TurtleBotEnvNoSlam
except ImportError:
    from turtlebot_env import TurtleBotEnv
    from turtlebot_env_frontier import TurtleBotEnvFrontier
    from turtlebot_env_no_slam import TurtleBotEnv as TurtleBotEnvNoSlam

class TestEnvWrapper:
    def __init__(self, method_name):
        self.executor = MultiThreadedExecutor()
        self.node = None
        
        if "Nav Only" in method_name:
            print("Initializing No-SLAM Env...")
            self.node = TurtleBotEnvNoSlam()
        elif "Frontier" in method_name:
            print("Initializing Frontier Env...")
            self.node = TurtleBotEnvFrontier()
            
            # テスト時は判定を少し甘くする
            self.node.collision_dist = 0.12
            print("🔧 Test Mode: Collision distance set to 0.12")
        else:
            print("Initializing Standard Env...")
            self.node = TurtleBotEnv()
        
        # ★★★ 最重要修正: Sim Time を強制有効化 ★★★
        # Worldが最高速(0.0)の場合、これを入れないと制御タイミングがズレて即死します
        self.node.set_parameters([Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        
        self.executor.add_node(self.node)
        
        # センサー受信用スレッド
        self.spin_thread = threading.Thread(target=self.executor.spin, daemon=True)
        self.spin_thread.start()

    def reset(self):
        print("Env: Resetting...")
        state = self.node.reset()
        if isinstance(state, tuple): state = state[0]

        for i in range(20):
            if np.max(np.abs(state)) > 0:
                return np.array(state, dtype=np.float32).flatten()
            
            print("Env: Waiting for LIDAR data...")
            time.sleep(0.5)
            state = self.node.get_state()
            
        print("⚠️ Warning: LIDAR timed out.")
        return np.array(state, dtype=np.float32).flatten()

    def step(self, action):
        next_state, reward, done, info = self.node.step(action)
        if isinstance(next_state, tuple): next_state = next_state[0]
        return np.array(next_state, dtype=np.float32).flatten(), reward, done, info

    def close(self):
        if self.executor:
            self.executor.shutdown()
        if self.node:
            self.node.destroy_node()