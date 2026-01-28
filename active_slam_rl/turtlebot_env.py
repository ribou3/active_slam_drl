import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32
from gazebo_msgs.srv import SetEntityState
from std_srvs.srv import Empty
from rclpy.qos import QoSProfile, ReliabilityPolicy
import numpy as np
import time
import subprocess

class TurtleBotEnv(Node):
    def __init__(self):
        super().__init__('turtlebot_d3qn_env_slam')

        self.action_duration = 0.1
        self.n_observations = 100
        self.max_range = 3.5
        self.collision_dist = 0.15
        self.robot_name = 'burger'

        self.actions = [(0.2, 0.0), (0.05, 0.5), (0.05, -0.5)]

        qos_profile = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile)
        self.reward_sub = self.create_subscription(Float32, '/uncertainty_reward', self.reward_callback, 10)

        # サービス
        self.set_entity_state_client = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        self.reset_world_client = self.create_client(Empty, '/reset_world')
        self.unpause = self.create_client(Empty, '/unpause_physics')
        self.pause = self.create_client(Empty, '/pause_physics')

        self.latest_scan = None
        self.latest_slam_reward = 0.0
        self.services_checked = False

        self.get_logger().info("✅ Env Initialized (Fixed: No 10x multiplier)")

    def scan_callback(self, msg):
        self.latest_scan = msg

    def reward_callback(self, msg):
        self.latest_slam_reward = msg.data

    def wait_sim_time(self, sec):
        start = self.get_clock().now()
        duration = Duration(seconds=sec)
        while rclpy.ok():
            if self.get_clock().now() - start >= duration: break
            time.sleep(0.001)

    def _check_services_ready(self):
        if self.services_checked: return True
        self.get_logger().info("⏳ Waiting for Gazebo services...")
        
        if not self.unpause.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("❌ /unpause_physics service not found!")
            return False
            
        if not self.set_entity_state_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("⚠️ /gazebo/set_entity_state not found. Will use /reset_world as fallback.")
            if not self.reset_world_client.wait_for_service(timeout_sec=5.0):
                self.get_logger().error("❌ Neither Teleport nor Reset World services found!")
                return False
        
        self.services_checked = True
        return True

    def restart_slam_toolbox(self):
        self.get_logger().info("💀 Killing SLAM Toolbox...")
        subprocess.run("pkill -f slam_toolbox", shell=True)
        time.sleep(2.0)
        
        self.get_logger().info("🔥 Respawning SLAM Toolbox...")
        subprocess.Popen(
            "ros2 launch slam_toolbox online_async_launch.py",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        self.get_logger().info("⏳ Waiting for SLAM to initialize...")
        time.sleep(5.0) 
        self.get_logger().info("✅ SLAM Respawned.")

    def step(self, action_idx):
        self.call_service(self.unpause)

        linear, angular = self.actions[action_idx]
        cmd = Twist()
        cmd.linear.x = float(linear)
        cmd.angular.z = float(angular)
        self.cmd_vel_pub.publish(cmd)

        self.wait_sim_time(self.action_duration)

        state = self.get_state()
        
        # 生のSLAM報酬を取得
        slam_reward = self.latest_slam_reward
        
        # ★デバッグログ: ここで何が来ているか確認！ (値がおかしければターミナルに出る)
        # self.get_logger().info(f"Incoming SLAM Reward: {slam_reward:.4f}") 

        self.latest_slam_reward = 0.0 

        self.call_service(self.pause)

        done = self.check_collision(state)
        
        nav_reward = -100.0 if done else (1.0 if angular == 0.0 else -0.05)
        
        # ★★★ 修正: 10倍を削除 ★★★
        weighted_slam_reward = slam_reward 
        total_reward = nav_reward + weighted_slam_reward

        info = {
            "slam_reward_raw": slam_reward,
            "slam_reward_weighted": weighted_slam_reward,
            "nav_reward": nav_reward
        }

        return state, total_reward, done, info

    def reset(self):
        if not self._check_services_ready():
            return np.zeros(self.n_observations)

        self.call_service(self.unpause)
        self.cmd_vel_pub.publish(Twist())
        
        reset_success = False
        if self.set_entity_state_client.service_is_ready():
            try:
                req = SetEntityState.Request()
                req.state.name = self.robot_name
                req.state.pose.position.x = 0.0
                req.state.pose.position.y = 0.0
                req.state.pose.position.z = 0.01
                req.state.pose.orientation.w = 1.0
                self.call_service(self.set_entity_state_client, req)
                reset_success = True
            except:
                pass

        if not reset_success:
            self.get_logger().warn("⚠️ Teleport failed. Using /reset_world instead.")
            self.call_service(self.reset_world_client)

        self.call_service(self.pause)
        self.restart_slam_toolbox()

        self.call_service(self.unpause)
        self.wait_sim_time(0.5)
        state = self.get_state()
        self.call_service(self.pause)
        
        return state

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
        while not future.done():
            if time.time() - start_wait > 5.0:
                self.get_logger().warn(f"Service call timed out: {client.srv_name}")
                return None
            time.sleep(0.001)
        return future.result()