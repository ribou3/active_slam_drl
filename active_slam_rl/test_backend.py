import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Float64, Float32MultiArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
import numpy as np

try:
    from active_slam_rl.test_config import TOPIC_MAP, TOPIC_ODOM, TOPIC_REWARD, TOPIC_D_OPT, USE_SIM_TIME
except ImportError:
    from test_config import TOPIC_MAP, TOPIC_ODOM, TOPIC_REWARD, TOPIC_D_OPT, USE_SIM_TIME

class TestBackendNode(Node):
    def __init__(self):
        super().__init__('test_backend_node')
        
        # データ保持用変数
        self.map_data = None
        self.trajectory = []     
        self.d_opt_history = []  
        self.frontier_history = []
        self.coverage_history = []
        self.uncertainty_map = [] # [(x, y, d_opt), ...]
        
        # Sim Time設定
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, USE_SIM_TIME)])
        
        # QoS設定 (SLAMの地図受信用)
        map_qos = QoSProfile(
            depth=1, 
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL, 
            history=HistoryPolicy.KEEP_LAST
        )
        
        # Subscribers
        self.create_subscription(OccupancyGrid, TOPIC_MAP, self.map_callback, map_qos)
        self.create_subscription(Odometry, TOPIC_ODOM, self.odom_callback, QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT))
        self.create_subscription(Float32MultiArray, TOPIC_REWARD, self.reward_callback, 10)
        self.create_subscription(Float64, TOPIC_D_OPT, self.d_opt_callback_legacy, 10)

    def map_callback(self, msg): 
        if self.map_data is None: print("✅ Backend: Map Received!")
        self.map_data = msg

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.trajectory.append((x, y))
        
        # 現在の不確実性を座標と紐付け
        if self.d_opt_history:
            current_d_opt = self.d_opt_history[-1]
            self.uncertainty_map.append((x, y, current_d_opt))

    def reward_callback(self, msg):
        # msg.data = [total, d_opt, frontier, hit]
        if len(msg.data) >= 3:
            self.d_opt_history.append(msg.data[1])
            self.frontier_history.append(msg.data[2])

    def d_opt_callback_legacy(self, msg):
        # 古いノードとの互換性用（もしreward_nodeが配列を送らない場合）
        if not self.d_opt_history:
            self.d_opt_history.append(msg.data)

    def get_coverage_ratio(self):
        if self.map_data is None: return 0.0
        data = np.array(self.map_data.data)
        known = np.sum(data != -1)
        total = len(data)
        if total == 0: return 0.0
        cov = (known / total) * 100.0
        self.coverage_history.append(cov)
        return cov
    
    def reset_metrics(self):
        self.trajectory = []
        self.d_opt_history = []
        self.frontier_history = []
        self.coverage_history = []
        self.uncertainty_map = []