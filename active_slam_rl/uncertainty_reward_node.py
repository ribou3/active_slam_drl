import rclpy
from rclpy.node import Node
# 通信品質(QoS)の設定用: SLAMからのデータを取りこぼさないために必要
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float32
import numpy as np

class UncertaintyRewardNode(Node):
    def __init__(self):
        super().__init__('uncertainty_reward_node')
        
        # ---------------------------------------------------------
        # 1. QoS (Quality of Service) の設定
        # ---------------------------------------------------------
        # SLAMノードは通常 "Best Effort" (届かなくても再送しない) でデータを送ります。
        # こちらが "Reliable" (確実性重視) で待っていると、設定不一致でデータが来ません。
        # そのため、どんな相手とも通信できる「最強の受け入れ態勢」を作ります。
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE, # 最新のデータのみ欲しい
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        # ---------------------------------------------------------
        # 2. トピックの購読 (Subscriber)
        # ---------------------------------------------------------
        # 自己位置と「共分散行列(不確かさ)」を受け取ります。
        # トピック名: '/pose' (環境によっては /amcl_pose や /slam_toolbox/pose の場合も)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/pose',
            self.pose_callback,
            qos_profile
        )
        
        # ---------------------------------------------------------
        # 3. 報酬の配信 (Publisher)
        # ---------------------------------------------------------
        # 計算した「不確かさ報酬」を学習エージェントに送ります。
        self.reward_pub = self.create_publisher(Float32, '/uncertainty_reward', 10)
        
        # ---------------------------------------------------------
        # [cite_start]4. 論文パラメータの設定 (Table 2より [cite: 230])
        # ---------------------------------------------------------
        # η (イータ): 報酬関数のスケーリング係数。論文では 0.01
        self.eta = 0.01  
        
        # l (エル): 状態空間の次元数。
        # [cite_start]2D平面移動ロボット (x, y, yaw) なので 3次元 [cite: 31]
        self.dim_l = 3.0 
        
        # 安全装置用の閾値: 共分散の合計がこれを超えたらSLAM崩壊とみなす
        self.sigma_sum_threshold = 100.0

        # 起動確認ログ
        self.get_logger().info("✅ D-opt Reward Node Started (Waiting for /pose data...)")

    def pose_callback(self, msg):
        """
        自己位置の共分散(Σ)を受け取り、D-optimalityに基づいた報酬を計算する
        """
        
        # ★ 生存確認ログ: これが出ればデータは届いています
        self.get_logger().info("📨 Message Received!", throttle_duration_sec=2.0)

        # ---------------------------------------------------------
        # [cite_start]ステップ 1: 共分散行列の整形 [cite: 209-213]
        # ---------------------------------------------------------
        # ROSのメッセージは一列のリスト(36要素)なので、6x6行列に変換
        cov_6x6 = np.array(msg.pose.covariance).reshape(6, 6)
        
        # ---------------------------------------------------------
        # [cite_start]ステップ 2: 必要な成分の抽出 (2D移動用) [cite: 483-485]
        # ---------------------------------------------------------
        # 行列のインデックス: 0:X, 1:Y, 5:Yaw (回転)
        # TurtleBotは床の上を走るのでZ軸などは無視します
        indices = [0, 1, 5]
        sigma = cov_6x6[np.ix_(indices, indices)]
        
        # ---------------------------------------------------------
        # ★ 安全装置 1: 数値異常のチェック
        # ---------------------------------------------------------
        # NaN(非数)やInf(無限大)が含まれていたら計算不可なのでリセット
        if np.any(np.isnan(sigma)) or np.any(np.isinf(sigma)):
            self.get_logger().error("❌ Math Error: Sigma contains NaN or Inf!")
            self._publish_reward(0.0)
            return

        # 共分散が大きすぎる(＝完全に迷子)場合はエラーとして処理
        sigma_sum = np.sum(np.abs(sigma))
        if sigma_sum > self.sigma_sum_threshold:
            self.get_logger().error(f"❌ Critical: Sigma sum too large ({sigma_sum:.2f})")
            self._publish_reward(0.0)
            return

        # ---------------------------------------------------------
        # [cite_start]ステップ 3: D-optimality (D最適性基準) の計算 [cite: 36-46]
        # ---------------------------------------------------------
        # 定義: D-opt = exp( 1/l * Σ log(λ_k) ) 
        # 意味: 不確かさの楕円体の体積。小さいほど自信がある。
        
        try:
            # 固有値 (λ) を計算
            eig_vals = np.linalg.eigvals(sigma)
            
            # log(0)を防ぐための数値安定化 (1e-9未満は1e-9にする)
            eig_vals = np.maximum(eig_vals, 1e-9)
            
            # 式(3)の実装
            log_sum = np.sum(np.log(eig_vals))
            d_opt = np.exp(log_sum / self.dim_l)
            
            # ---------------------------------------------------------
            # [cite_start]ステップ 4: 報酬への変換 (Intrinsic Reward) [cite: 196-201]
            # ---------------------------------------------------------
            # 式(12): R_u = tanh( η / D-opt )
            # D-opt(不確かさ)が小さいほど、報酬は 1.0 に近づく
            
            if d_opt > 1e-9:
                tanh_input = self.eta / d_opt
                intrinsic_reward = np.tanh(tanh_input)
            else:
                # 不確かさがほぼ0なら最大報酬
                intrinsic_reward = 1.0 
            
            # ---------------------------------------------------------
            # ★ 安全装置 2: 報酬値のクリッピング
            # ---------------------------------------------------------
            # tanhの数学的性質上 1.0 を超えることはないが、念のためガード
            if intrinsic_reward > 1.0:
                self.get_logger().warn(f"⚠️ Reward saturated: {intrinsic_reward}")
                intrinsic_reward = 1.0

            # ---------------------------------------------------------
            # ★ 確認用ログ (2秒に1回表示)
            # ---------------------------------------------------------
            self.get_logger().info(
                f"✅ Eigs: {np.round(eig_vals, 5)} | D-opt: {d_opt:.6f} | Reward: {intrinsic_reward:.6f}",
                throttle_duration_sec=2.0
            )
            
            self._publish_reward(intrinsic_reward)
            
        except Exception as e:
            self.get_logger().warn(f"Calculation Error: {e}")
            self._publish_reward(0.0)

    def _publish_reward(self, value):
        out_msg = Float32()
        out_msg.data = float(value)
        self.reward_pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = UncertaintyRewardNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()