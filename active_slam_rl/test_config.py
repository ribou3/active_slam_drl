import os

# --- ROS Topics ---
TOPIC_MAP = '/map'
TOPIC_ODOM = '/odom'
TOPIC_REWARD = '/intrinsic_reward'
TOPIC_D_OPT = '/d_optimality'

# --- GUI Settings ---
GUI_APPEARANCE = "Dark"  # Dark / Light
GUI_THEME = "blue"       # blue / dark-blue / green
WINDOW_SIZE = "1500x950"

# --- Simulation Settings ---
USE_SIM_TIME = True      # テスト時は必須

# --- Paths ---
# 結果保存先のルートフォルダ
SAVE_DIR_BASE = os.path.expanduser("~/ros2_ws/test_results")