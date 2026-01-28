# import rclpy
# from rclpy.node import Node
# from rclpy.duration import Duration
# from geometry_msgs.msg import Twist
# from sensor_msgs.msg import LaserScan
# from std_msgs.msg import Float32
# from gazebo_msgs.srv import SetEntityState
# from std_srvs.srv import Empty
# from rclpy.qos import QoSProfile, ReliabilityPolicy
# import numpy as np
# import time
# import subprocess

# class TurtleBotEnvFrontier(Node):
#     def __init__(self):
#         super().__init__('turtlebot_env_frontier')

#         self.action_duration = 0.1
#         self.n_observations = 100
#         self.max_range = 3.5
#         self.collision_dist = 0.15
#         self.robot_name = 'burger'

#         # Action 0: Forward, 1: Left, 2: Right
#         self.actions = [(0.2, 0.0), (0.05, 0.5), (0.05, -0.5)]

#         qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        
#         self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
#         self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile)
        
#         # 報酬ノードからのトピックを購読
#         self.reward_sub = self.create_subscription(Float32, '/intrinsic_reward', self.reward_callback, 10)

#         # Services
#         self.set_entity_state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
#         self.reset_world_client = self.create_client(Empty, '/reset_world')
#         self.unpause = self.create_client(Empty, '/unpause_physics')
#         self.pause = self.create_client(Empty, '/pause_physics')

#         self.latest_scan = None
#         self.latest_intrinsic_reward = 0.0 
#         self.services_checked = False

#         self.get_logger().info("✅ Frontier Env Initialized (With SLAM Reset)")

#     def scan_callback(self, msg):
#         self.latest_scan = msg

#     def reward_callback(self, msg):
#         self.latest_intrinsic_reward = msg.data

#     def wait_sim_time(self, sec):
#         start = self.get_clock().now()
#         duration = Duration(seconds=sec)
#         while rclpy.ok():
#             if self.get_clock().now() - start >= duration: break
#             time.sleep(0.001)

#     def _check_services_ready(self):
#         """Gazeboのサービスが利用可能かチェックする"""
#         if self.services_checked: return True
#         self.get_logger().info("⏳ Waiting for Gazebo services...")
        
#         # 5秒待ってダメならエラーを出してFalseを返す（無限待機しない）
#         if not self.unpause.wait_for_service(timeout_sec=5.0):
#             self.get_logger().error("❌ /unpause_physics service not found! Is Gazebo running?")
#             return False
            
#         # リセット用サービスのチェック
#         if not self.set_entity_state_client.wait_for_service(timeout_sec=5.0):
#             self.get_logger().warn("⚠️ /gazebo/set_entity_state not found. Will use /reset_world fallback.")
#             if not self.reset_world_client.wait_for_service(timeout_sec=5.0):
#                 self.get_logger().error("❌ Neither Teleport nor Reset World services found!")
#                 return False
        
#         self.services_checked = True
#         return True

#     def restart_slam_toolbox(self):
#         """SLAM Toolbox を強制再起動するコマンド"""
#         self.get_logger().info("💀 Killing SLAM Toolbox...")
#         subprocess.run("pkill -f slam_toolbox", shell=True)
#         time.sleep(2.0)
        
#         self.get_logger().info("🔥 Respawning SLAM Toolbox...")
#         # ログを捨ててバックグラウンドで起動
#         subprocess.Popen(
#             "ros2 launch slam_toolbox online_async_launch.py",
#             shell=True, 
#             stdout=subprocess.DEVNULL, 
#             stderr=subprocess.DEVNULL
#         )
        
#         self.get_logger().info("⏳ Waiting for SLAM to initialize...")
#         time.sleep(5.0)
#         self.get_logger().info("✅ SLAM Respawned.")

#     def step(self, action_idx):
#         self.call_service(self.unpause)

#         linear, angular = self.actions[action_idx]
#         cmd = Twist()
#         cmd.linear.x = float(linear)
#         cmd.angular.z = float(angular)
#         self.cmd_vel_pub.publish(cmd)

#         self.wait_sim_time(self.action_duration)

#         state = self.get_state()
        
#         # 報酬ノードから計算済みのIntrinsic報酬を取得
#         intrinsic_reward = self.latest_intrinsic_reward
#         self.latest_intrinsic_reward = 0.0 # リセット

#         self.call_service(self.pause)

#         done = self.check_collision(state)
        
#         # ナビゲーション報酬
#         if done:
#             r_nav = -100.0
#         else:
#             r_nav = 1.0 if angular == 0.0 else -0.05
        
#         # 衝突時はintrinsic報酬を加算しない（論文式 4.2）
#         total_reward = r_nav if done else (r_nav + intrinsic_reward)

#         info = {
#             "intrinsic_reward": intrinsic_reward,
#             "nav_reward": r_nav
#         }

#         return state, total_reward, done, info

#     def reset(self):
#         # サービスチェックに失敗したらゼロ状態を返す（学習を止めないため）
#         if not self._check_services_ready():
#             self.get_logger().error("❌ Failed to connect to Gazebo. Returning empty state.")
#             return np.zeros(self.n_observations)

#         self.call_service(self.unpause)
#         self.cmd_vel_pub.publish(Twist())
        
#         # ロボット位置のリセット
#         reset_success = False
#         try:
#             if self.set_entity_state_client.service_is_ready():
#                 req = SetEntityState.Request()
#                 req.state.name = self.robot_name
#                 req.state.pose.position.x = 0.0
#                 req.state.pose.position.y = 0.0
#                 req.state.pose.position.z = 0.01
#                 req.state.pose.orientation.w = 1.0
#                 self.call_service(self.set_entity_state_client, req)
#                 reset_success = True
#         except Exception as e:
#             self.get_logger().warn(f"Teleport failed: {e}")

#         # 失敗時はワールドリセット
#         if not reset_success:
#             self.call_service(self.reset_world_client)

#         self.call_service(self.pause)
        
#         # ★★★ ここでSLAMをリセット ★★★
#         self.restart_slam_toolbox()

#         self.call_service(self.unpause)
#         self.wait_sim_time(0.5)
#         state = self.get_state()
#         self.call_service(self.pause)
        
#         return state

#     def get_state(self):
#         if self.latest_scan is None: return np.zeros(self.n_observations)
#         ranges = np.array(self.latest_scan.ranges)
#         if len(ranges) == 0: return np.zeros(self.n_observations)
#         q = len(ranges) // 4
#         front = np.concatenate((ranges[3*q:], ranges[:q]))
#         front = np.nan_to_num(front, nan=self.max_range, posinf=self.max_range)
#         front = np.clip(front, 0.0, self.max_range)
#         if len(front) > self.n_observations:
#             idx = np.linspace(0, len(front)-1, self.n_observations)
#             obs = front[idx.astype(int)]
#         else:
#             obs = np.interp(np.linspace(0, len(front)-1, self.n_observations), np.arange(len(front)), front)
#         return obs / self.max_range

#     def check_collision(self, state):
#         if len(state) == 0: return False
#         return np.min(state) < (self.collision_dist / self.max_range)

#     def call_service(self, client, req=None):
#         if req is None: req = client.srv_type.Request()
#         future = client.call_async(req)
#         # 無限ループ防止のためタイムアウト付きで待機
#         start_wait = time.time()
#         while rclpy.ok() and not future.done():
#             if time.time() - start_wait > 5.0:
#                 self.get_logger().warn(f"Service call timed out: {client.srv_name}")
#                 return None
#             time.sleep(0.001)
#         return future.result()

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty
from rclpy.qos import QoSProfile, ReliabilityPolicy
import numpy as np
import time
import subprocess

class TurtleBotEnvFrontier(Node):
    def __init__(self):
        super().__init__('turtlebot_env_frontier')

        self.action_duration = 0.1
        self.n_observations = 100
        self.max_range = 3.5
        self.collision_dist = 0.15
        self.robot_name = 'burger'

        self.actions = [(0.2, 0.0), (0.05, 0.5), (0.05, -0.5)]

        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile)
        self.reward_sub = self.create_subscription(Float32MultiArray, '/intrinsic_reward', self.reward_callback, 10)

        # Services
        self.set_entity_state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.reset_world_client = self.create_client(Empty, '/reset_world')
        self.unpause = self.create_client(Empty, '/unpause_physics')
        self.pause = self.create_client(Empty, '/pause_physics')

        self.latest_scan = None
        self.latest_reward_data = [0.0, 0.0, 0.0, 0.0]
        self.services_checked = False

        self.get_logger().info("✅ Frontier Env Initialized (Verbose Logging Enabled)")

    def scan_callback(self, msg):
        self.latest_scan = msg

    def reward_callback(self, msg):
        self.latest_reward_data = msg.data

    def wait_sim_time(self, sec):
        start = self.get_clock().now()
        duration = Duration(seconds=sec)
        while rclpy.ok():
            if self.get_clock().now() - start >= duration: break
            time.sleep(0.001)

    def _check_services_ready(self):
        if self.services_checked: return True
        self.get_logger().info("⏳ Connecting to Gazebo services...")
        
        if not self.unpause.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("❌ /unpause_physics service not found!")
            return False
            
        if not self.set_entity_state_client.wait_for_service(timeout_sec=2.0):
            if not self.reset_world_client.wait_for_service(timeout_sec=2.0):
                return False
        
        self.services_checked = True
        return True

    def restart_slam_toolbox(self, episode=None):
        """SLAM Toolboxを強制再起動"""
        prefix = f"[Ep: {episode}] " if episode else ""
        
        self.get_logger().info(f"{prefix}💀 Killing SLAM Toolbox...", throttle_duration_sec=0)
        subprocess.run("pkill -f slam_toolbox", shell=True)
        time.sleep(2.0)
        
        self.get_logger().info(f"{prefix}🔥 Respawning SLAM Toolbox...", throttle_duration_sec=0)
        cmd = "ros2 launch slam_toolbox online_async_launch.py"
        subprocess.Popen(
            cmd, 
            shell=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        
        self.get_logger().info(f"{prefix}⏳ Waiting for SLAM initialization...", throttle_duration_sec=0)
        time.sleep(5.0) 
        self.get_logger().info(f"{prefix}✅ SLAM Respawned.", throttle_duration_sec=0)

    def reset_robot_position(self, episode=None):
        prefix = f"[Ep: {episode}] " if episode else ""
        
        if self.set_entity_state_client.service_is_ready():
            req = SetEntityState.Request()
            req.state.name = self.robot_name
            req.state.pose.position.x = 0.0
            req.state.pose.position.y = 0.0
            req.state.pose.position.z = 0.01
            req.state.pose.orientation.w = 1.0
            
            future = self.set_entity_state_client.call_async(req)
            start = time.time()
            while not future.done() and time.time() - start < 1.0:
                time.sleep(0.01)
            
            if future.done() and future.result().success:
                self.get_logger().info(f"{prefix}📍 Robot Position Reset: Success")
                return True
        
        self.get_logger().info(f"{prefix}🌍 Calling /reset_world fallback...")
        self.call_service(self.reset_world_client)
        return False

    def reset(self, episode=None):
        """
        エピソードのリセット処理
        引数 episode: 呼び出し元から現在のエピソード番号を受け取る
        """
        prefix = f"--- [Ep: {episode}] RESET --- " if episode else "--- RESET --- "
        
        if not self._check_services_ready():
            self.get_logger().error("❌ Gazebo services not ready.")
            return np.zeros(self.n_observations)

        self.get_logger().info(prefix + "Start")

        self.call_service(self.unpause)
        self.cmd_vel_pub.publish(Twist())
        
        # ロボット位置リセット
        self.reset_robot_position(episode)

        self.call_service(self.pause)
        
        # SLAM再起動 (エピソード番号付きでログ出力)
        self.restart_slam_toolbox(episode)

        self.call_service(self.unpause)
        self.wait_sim_time(1.0) 
        state = self.get_state()
        self.call_service(self.pause)
        
        self.get_logger().info(prefix + "Complete")
        
        return state

    def step(self, action_idx):
        # stepメソッド内での詳細ログは大量に出るため、重要なエラー時のみにするか
        # 必要であればデバッグ用に有効化してください。
        
        self.call_service(self.unpause)
        
        linear, angular = self.actions[action_idx]
        cmd = Twist()
        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)
        self.cmd_vel_pub.publish(cmd)

        self.wait_sim_time(self.action_duration)

        state = self.get_state()
        rewards = self.latest_reward_data
        self.latest_reward_data = [0.0, 0.0, 0.0, 0.0]
        
        intrinsic_total = rewards[0]
        r_d_opt = rewards[1]
        r_frontier = rewards[2]
        r_hit = rewards[3]

        self.call_service(self.pause)

        done = self.check_collision(state)
        
        if done:
            r_nav = -100.0
        else:
            r_nav = 1.0 if angular == 0.0 else -0.05
        
        total_reward = r_nav if done else (r_nav + intrinsic_total)

        info = {
            "intrinsic_reward": intrinsic_total,
            "d_opt_reward": r_d_opt,
            "frontier_reward": r_frontier,
            "hit_reward": r_hit,
            "nav_reward": r_nav
        }

        return state, total_reward, done, info
        
    # ... (get_state, check_collision, call_service は変更なし) ...
    def get_state(self):
        if self.latest_scan is None: return np.zeros(self.n_observations)
        ranges = np.array(self.latest_scan.ranges)
        if len(ranges) == 0: return np.zeros(self.n_observations)
        q = len(ranges) // 4
        front = np.concatenate((ranges[3*q:], ranges[:q]))
        front = np.nan_to_num(front, nan=self.max_range, posinf=self.max_range)
        front = np.clip(front, 0.0, self.max_range)
        if len(front) > self.n_observations:
            idx = np.linspace(0, len(front)-1, self.n_observations)
            obs = front[idx.astype(int)]
        else:
            obs = np.interp(np.linspace(0, len(front)-1, self.n_observations), np.arange(len(front)), front)
        return obs / self.max_range

    def check_collision(self, state):
        if len(state) == 0: return False
        return np.min(state) < (self.collision_dist / self.max_range)

    def call_service(self, client, req=None):
        if req is None: req = client.srv_type.Request()
        future = client.call_async(req)
        start_wait = time.time()
        while rclpy.ok() and not future.done():
            if time.time() - start_wait > 5.0: return None
            time.sleep(0.001)
        return future.result()