# import rclpy
# from rclpy.node import Node
# from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
# from geometry_msgs.msg import PoseWithCovarianceStamped
# from nav_msgs.msg import OccupancyGrid
# from std_msgs.msg import Float32
# import numpy as np
# import math

# class ActiveSLAMRewardNode(Node):
#     def __init__(self):
#         super().__init__('active_slam_reward_node')
        
#         # --- 論文 Table 5.2 に基づくパラメータ設定 [cite: 2477] ---
#         self.alpha = 1.0       # D-opt Weight
#         self.beta = 1.0        # Frontier Weight
#         self.eta = 0.01        # D-opt Scale Factor
#         self.kappa = 0.5       # Frontier Distance Kernel
#         self.r_hit_bonus = 10.0 # Frontier Arrival Bonus
#         self.hit_threshold = 0.5 # 到達判定距離 (m) ※論文に具体的数値がないため妥当な値を設定
        
#         self.dim_l = 3.0       # 状態空間の次元 (x, y, yaw)

#         # QoS設定 (MapはBest Effortで来る可能性があるため合わせる)
#         qos_profile = QoSProfile(
#             reliability=ReliabilityPolicy.BEST_EFFORT,
#             durability=DurabilityPolicy.VOLATILE,
#             history=HistoryPolicy.KEEP_LAST,
#             depth=10
#         )

#         # Subscribers
#         self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_callback, qos_profile)
#         self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos_profile)
        
#         # Publisher
#         self.reward_pub = self.create_publisher(Float32, '/intrinsic_reward', 10)

#         # 内部状態
#         self.latest_map = None
#         self.map_info = None
#         self.frontiers = [] # Frontier座標のリスト [(x, y), ...]

#         self.get_logger().info("✅ Active SLAM Reward Node Started (D-opt + Frontier)")

#     def map_callback(self, msg):
#         """地図を受け取り、Frontier（未知と既知の境界）を検出する"""
#         self.latest_map = np.array(msg.data).reshape((msg.info.height, msg.info.width))
#         self.map_info = msg.info
#         self.detect_frontiers()

#     def detect_frontiers(self):
#         """
#         簡易的なFrontier検出
#         -1 (Unknown) と 0 (Free) が隣接しているセルを探す
#         計算負荷軽減のため、ダウンサンプリングやエッジ検出的に処理
#         """
#         if self.latest_map is None: return

#         # 処理高速化のため、numpy演算でエッジを抽出
#         # 0: Free, 100: Occupied, -1: Unknown
#         grid = self.latest_map
#         width = self.map_info.width
        
#         # 簡易実装: Unknown(-1) のセルで、上下左右に Free(0) がある場所を特定
#         # 実際には画像処理フィルタ(Sobel等)を使うと速いが、ここでは分かりやすく実装
        
#         # マスク作成
#         unknown_mask = (grid == -1)
#         free_mask = (grid == 0)
        
#         # 境界検出は重いため、一定間隔で実行するか、ロジックを簡略化する
#         # ここでは「ロボット近傍のFrontierまでの距離」が重要なので、
#         # 厳密な全探索よりもレスポンスを優先する実装にするのが一般的だが、
#         # 今回は基本的なロジックとして全探索する (Pythonだと遅い可能性あり)
        
#         # 高速化: 確率的にサンプリングしてFrontier候補を探す、またはC++ノードが推奨されるが
#         # ここではシンプルに「前回計算したFrontier」を保持しつつ更新する形を想定
#         # ※ 実装の複雑さを避けるため、pose_callback内で距離計算時に簡易探索を行う
#         pass 

#     def get_nearest_frontier_dist(self, robot_x, robot_y):
#         """ロボットから最も近いFrontierまでの距離を計算"""
#         if self.latest_map is None or self.map_info is None:
#             return 10.0 # マップがない場合は適当な遠い距離を返す

#         resolution = self.map_info.resolution
#         origin_x = self.map_info.origin.position.x
#         origin_y = self.map_info.origin.position.y
#         width = self.map_info.width
#         height = self.map_info.height

#         # ロボットのグリッド座標
#         grid_x = int((robot_x - origin_x) / resolution)
#         grid_y = int((robot_y - origin_y) / resolution)

#         # 探索範囲（全探索は重いので、ロボット周囲の一定範囲を探索）
#         search_radius = 50 # 50グリッド分（約2.5m）
#         min_x = max(0, grid_x - search_radius)
#         max_x = min(width, grid_x + search_radius)
#         min_y = max(0, grid_y - search_radius)
#         max_y = min(height, grid_y + search_radius)

#         # 部分マップ切り出し
#         sub_map = self.latest_map[min_y:max_y, min_x:max_x]
        
#         # Frontier条件: Unknown(-1) かつ 周囲8近傍に Free(0) がある
#         # ここでは簡易的に「Freeセル(0)」で、かつ「隣接に-1がある」セルをFrontierとする
#         # (ロボットが移動できるのはFreeセル側なので)
        
#         # Freeセルの座標を取得
#         free_indices = np.argwhere(sub_map == 0)
        
#         min_dist_sq = float('inf')
#         found_frontier = False

#         # Freeセルの中からFrontier（隣にUnknownがある）を探す
#         for dy, dx in free_indices:
#             # グローバル座標に戻す
#             gy = min_y + dy
#             gx = min_x + dx
            
#             # 隣接チェック (4近傍)
#             neighbors = [
#                 (gy+1, gx), (gy-1, gx), (gy, gx+1), (gy, gx-1)
#             ]
#             is_frontier = False
#             for ny, nx in neighbors:
#                 if 0 <= ny < height and 0 <= nx < width:
#                     if self.latest_map[ny, nx] == -1: # Unknown
#                         is_frontier = True
#                         break
            
#             if is_frontier:
#                 # 距離計算
#                 dist_sq = (gx - grid_x)**2 + (gy - grid_y)**2
#                 if dist_sq < min_dist_sq:
#                     min_dist_sq = dist_sq
#                     found_frontier = True

#         if found_frontier:
#             return math.sqrt(min_dist_sq) * resolution
#         else:
#             return 10.0 # 近くにFrontierなし

#     def pose_callback(self, msg):
#         """自己位置と共分散を受け取り、統合報酬を計算 """
        
#         # --- 1. D-optimality の計算 ---
#         cov_6x6 = np.array(msg.pose.covariance).reshape(6, 6)
#         indices = [0, 1, 5] # x, y, yaw
#         sigma = cov_6x6[np.ix_(indices, indices)]
        
#         d_opt_val = 1.0
#         r_d_opt = 0.0
        
#         # 数値安定化処理
#         if not np.any(np.isnan(sigma)) and not np.any(np.isinf(sigma)):
#             try:
#                 eig_vals = np.linalg.eigvals(sigma)
#                 eig_vals = np.maximum(eig_vals, 1e-9)
#                 log_sum = np.sum(np.log(eig_vals))
#                 d_opt_val = np.exp(log_sum / self.dim_l) # 式(2.13)
                
#                 # 式(4.4): tanh(eta / d_opt)
#                 if d_opt_val > 1e-9:
#                     r_d_opt = np.tanh(self.eta / d_opt_val)
#                 else:
#                     r_d_opt = 1.0
#             except:
#                 pass

#         # --- 2. Frontier 報酬の計算 ---
#         # ロボット位置
#         rx = msg.pose.pose.position.x
#         ry = msg.pose.pose.position.y
        
#         dist_frontier = self.get_nearest_frontier_dist(rx, ry)
        
#         # 式(4.5): 1 - tanh(kappa * dist)
#         r_frontier = 1.0 - np.tanh(self.kappa * dist_frontier)
        
#         # --- 3. 到達ボーナス ---
#         # 式(4.6)
#         r_hit = self.r_hit_bonus if dist_frontier < self.hit_threshold else 0.0

#         # --- 4. 統合報酬 ---
#         # 式(4.2) の intrinsic部分 (navは環境側で加算)
#         # r_intrinsic = alpha * r_d_opt + beta * r_frontier + r_hit
#         total_intrinsic_reward = (self.alpha * r_d_opt) + (self.beta * r_frontier) + r_hit
        
#         # ログ出力 (適度な頻度で)
#         # self.get_logger().info(
#         #     f"D-opt: {r_d_opt:.3f} | Frontier: {r_frontier:.3f} (Dist: {dist_frontier:.2f}) | Hit: {r_hit} | Total: {total_intrinsic_reward:.3f}",
#         #     throttle_duration_sec=1.0
#         # )

#         msg_out = Float32()
#         msg_out.data = float(total_intrinsic_reward)
#         self.reward_pub.publish(msg_out)

# def main(args=None):
#     rclpy.init(args=args)
#     node = ActiveSLAMRewardNode()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Float32MultiArray  # 配列通信用に変更
import numpy as np
import math

class ActiveSLAMRewardNode(Node):
    def __init__(self):
        super().__init__('active_slam_reward_node')
        
        # --- パラメータ (Table 5.2) ---
        self.alpha = 1.0       # D-opt Weight
        self.beta = 1.0        # Frontier Weight
        self.eta = 0.01        # D-opt Scale Factor
        self.kappa = 0.5       # Frontier Distance Kernel
        self.r_hit_bonus = 10.0 # Frontier Arrival Bonus
        self.hit_threshold = 0.5 
        
        self.dim_l = 3.0       

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, '/pose', self.pose_callback, qos_profile)
        self.map_sub = self.create_subscription(OccupancyGrid, '/map', self.map_callback, qos_profile)
        
        # 変更: 内訳を送るため MultiArray に変更
        self.reward_pub = self.create_publisher(Float32MultiArray, '/intrinsic_reward', 10)

        self.latest_map = None
        self.map_info = None

        self.get_logger().info("✅ Active SLAM Reward Node Started (Detailed Output Mode)")

    def map_callback(self, msg):
        self.latest_map = np.array(msg.data).reshape((msg.info.height, msg.info.width))
        self.map_info = msg.info

    def get_nearest_frontier_dist(self, robot_x, robot_y):
        if self.latest_map is None or self.map_info is None:
            return 10.0 

        resolution = self.map_info.resolution
        origin_x = self.map_info.origin.position.x
        origin_y = self.map_info.origin.position.y
        width = self.map_info.width
        height = self.map_info.height

        grid_x = int((robot_x - origin_x) / resolution)
        grid_y = int((robot_y - origin_y) / resolution)

        search_radius = 50 
        min_x = max(0, grid_x - search_radius)
        max_x = min(width, grid_x + search_radius)
        min_y = max(0, grid_y - search_radius)
        max_y = min(height, grid_y + search_radius)

        sub_map = self.latest_map[min_y:max_y, min_x:max_x]
        
        # 簡易Frontier探索: Free(0) かつ 近傍に Unknown(-1)
        free_indices = np.argwhere(sub_map == 0)
        
        min_dist_sq = float('inf')
        found_frontier = False

        for dy, dx in free_indices:
            gy = min_y + dy
            gx = min_x + dx
            neighbors = [(gy+1, gx), (gy-1, gx), (gy, gx+1), (gy, gx-1)]
            
            is_frontier = False
            for ny, nx in neighbors:
                if 0 <= ny < height and 0 <= nx < width:
                    if self.latest_map[ny, nx] == -1:
                        is_frontier = True
                        break
            
            if is_frontier:
                dist_sq = (gx - grid_x)**2 + (gy - grid_y)**2
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    found_frontier = True

        if found_frontier:
            return math.sqrt(min_dist_sq) * resolution
        else:
            return 10.0

    def pose_callback(self, msg):
        # 1. D-opt
        cov_6x6 = np.array(msg.pose.covariance).reshape(6, 6)
        indices = [0, 1, 5]
        sigma = cov_6x6[np.ix_(indices, indices)]
        
        r_d_opt = 0.0
        if not np.any(np.isnan(sigma)) and not np.any(np.isinf(sigma)):
            try:
                eig_vals = np.linalg.eigvals(sigma)
                eig_vals = np.maximum(eig_vals, 1e-9)
                log_sum = np.sum(np.log(eig_vals))
                d_opt_val = np.exp(log_sum / self.dim_l)
                if d_opt_val > 1e-9:
                    r_d_opt = np.tanh(self.eta / d_opt_val)
                else:
                    r_d_opt = 1.0
            except:
                pass

        # 2. Frontier
        rx = msg.pose.pose.position.x
        ry = msg.pose.pose.position.y
        dist_frontier = self.get_nearest_frontier_dist(rx, ry)
        r_frontier = 1.0 - np.tanh(self.kappa * dist_frontier)
        
        # 3. Hit Bonus
        r_hit = self.r_hit_bonus if dist_frontier < self.hit_threshold else 0.0

        # 4. Total Intrinsic
        total_intrinsic = (self.alpha * r_d_opt) + (self.beta * r_frontier) + r_hit
        
        # 変更: [合計, D-opt単体, Frontier単体, Hit単体] の配列を送る
        msg_out = Float32MultiArray()
        msg_out.data = [float(total_intrinsic), float(r_d_opt), float(r_frontier), float(r_hit)]
        self.reward_pub.publish(msg_out)

def main(args=None):
    rclpy.init(args=args)
    node = ActiveSLAMRewardNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()