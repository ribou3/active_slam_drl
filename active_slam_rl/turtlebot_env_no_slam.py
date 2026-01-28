import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty
from rclpy.qos import QoSProfile, ReliabilityPolicy
import numpy as np
import time

class TurtleBotEnv(Node):
    def __init__(self):
        super().__init__('turtlebot_d3qn_env_no_slam')

        self.action_duration = 0.1
        self.n_observations = 100
        self.max_range = 3.5
        self.collision_dist = 0.15
        self.robot_name = 'burger'

        self.actions = [
            (0.2, 0.0),
            (0.05, 0.5),
            (0.05, -0.5)
        ]

        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile)

        # --- サービス (テレポート用とリセット用) ---
        self.set_entity_state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.reset_world_client = self.create_client(Empty, '/reset_world') # バックアップ用
        self.unpause = self.create_client(Empty, '/unpause_physics')
        self.pause = self.create_client(Empty, '/pause_physics')

        self.latest_scan = None
        self.get_logger().info("✅ Env Initialized")

    def scan_callback(self, msg):
        self.latest_scan = msg

    def wait_sim_time(self, sec):
        start = self.get_clock().now()
        duration = Duration(seconds=sec)
        while rclpy.ok():
            if self.get_clock().now() - start >= duration:
                break
            time.sleep(0.001)

    # --- 堅牢なリセット関数 ---
    def reset(self):
        # 1. 物理再開 & 停止
        self.call_service_quiet(self.unpause)
        self.cmd_vel_pub.publish(Twist())
        
        # 2. リセット試行 (テレポート優先 -> だめならワールドリセット)
        reset_success = False
        
        # プランA: テレポート (SetEntityState)
        if self.set_entity_state_client.service_is_ready():
            req = SetEntityState.Request()
            req.state.name = self.robot_name
            req.state.pose.position.x = 0.0
            req.state.pose.position.y = 0.0
            req.state.pose.position.z = 0.01
            req.state.pose.orientation.w = 1.0
            if self.call_service_quiet(self.set_entity_state_client, req):
                reset_success = True
        
        # プランB: ワールドリセット (プランAが失敗した場合)
        if not reset_success:
            if self.reset_world_client.service_is_ready():
                self.get_logger().warn("⚠️ Teleport failed. Using /reset_world instead.")
                self.call_service_quiet(self.reset_world_client)
            else:
                # どっちもダメなら、とりあえずログを出してそのまま進む（止まらないことが大事）
                self.get_logger().error("🚨 All reset services failed! Continuing from current pos.")

        # 3. 待機 & 状態取得
        self.wait_sim_time(0.5)
        state = self.get_state()

        # 4. 一時停止
        self.call_service_quiet(self.pause)
        
        return state

    def step(self, action_idx):
        self.call_service_quiet(self.unpause)

        linear, angular = self.actions[action_idx]
        cmd = Twist()
        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)
        self.cmd_vel_pub.publish(cmd)

        self.wait_sim_time(self.action_duration)
        state = self.get_state()
        
        self.call_service_quiet(self.pause)

        done = self.check_collision(state)
        reward = -100.0 if done else (1.0 if angular == 0.0 else -0.05)
        
        return state, reward, done

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

    # エラーで止まらないサービス呼び出し
    def call_service_quiet(self, client, req=None):
        if not client.service_is_ready():
            return False
        if req is None: req = client.srv_type.Request()
        
        future = client.call_async(req)
        # 裏のスレッドが処理するので、ここではただ待つ
        start = time.time()
        while not future.done():
            if time.time() - start > 2.0: # 2秒でタイムアウト
                return False
            time.sleep(0.001)
        return True