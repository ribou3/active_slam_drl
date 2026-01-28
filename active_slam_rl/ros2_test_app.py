#!/usr/bin/env python3
import os
import sys
import time
import threading
import subprocess
import signal
import shutil
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg as tkagg
import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Float64

# --- パス設定 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# --- 作成したモジュールをインポート ---
try:
    from active_slam_rl.test_d3qn import TestD3QNAgent
    from active_slam_rl.test_env import TestEnvWrapper
except ImportError:
    from test_d3qn import TestD3QNAgent
    from test_env import TestEnvWrapper

# --- バックエンド: グラフ・ログ管理ノード ---
class TestBackendNode(Node):
    def __init__(self):
        super().__init__('test_backend_node')
        self.map_data = None
        self.trajectory = []
        self.d_opt_history = []
        self.create_subscription(OccupancyGrid, '/map', self.map_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(Float64, '/d_optimality', self.d_opt_callback, 10)

    def map_callback(self, msg): self.map_data = msg
    def odom_callback(self, msg): self.trajectory.append((msg.pose.pose.position.x, msg.pose.pose.position.y))
    def d_opt_callback(self, msg): self.d_opt_history.append(msg.data)
    
    def get_coverage_ratio(self):
        if self.map_data is None: return 0.0
        data = np.array(self.map_data.data)
        return (np.sum(data != -1) / len(data)) * 100.0
    
    def reset_metrics(self):
        self.trajectory = []
        self.d_opt_history = []

# --- フロントエンド: GUI ---
class TestApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Active SLAM Modular Test App")
        self.root.geometry("1200x900")

        self.is_running = False
        self.backend = None
        self.executor = None
        self.background_processes = []
        
        self.setup_ui()
        
        # ROSスレッド開始
        self.ros_thread = threading.Thread(target=self.run_ros_backend, daemon=True)
        self.ros_thread.start()

    def setup_ui(self):
        # Layout
        left_panel = ttk.Frame(self.root, width=400, padding=10)
        left_panel.pack(side="left", fill="y")
        right_panel = ttk.Frame(self.root, padding=10)
        right_panel.pack(side="right", fill="both", expand=True)

        # 1. Simulation
        frame_sim = ttk.LabelFrame(left_panel, text="1. Environment", padding=10)
        frame_sim.pack(fill="x", pady=5)
        ttk.Label(frame_sim, text="Select World:").pack(anchor="w")
        self.world_var = tk.StringVar(value="Circuit 1 (Training)")
        self.cmb_world = ttk.Combobox(frame_sim, textvariable=self.world_var, values=["Circuit 1 (Training)", "Circuit 2 (Obstacles)", "Maze (Complex)"], state="readonly")
        self.cmb_world.pack(fill="x", pady=2)
        self.btn_launch_gazebo = ttk.Button(frame_sim, text="Launch Gazebo 📺", command=self.launch_gazebo)
        self.btn_launch_gazebo.pack(fill="x", pady=2)
        self.btn_kill = ttk.Button(frame_sim, text="Kill All Processes 💀", command=self.kill_all)
        self.btn_kill.pack(fill="x", pady=2)

        # 2. Agent & Reward
        frame_agent = ttk.LabelFrame(left_panel, text="2. Agent & Reward", padding=10)
        frame_agent.pack(fill="x", pady=5)
        ttk.Label(frame_agent, text="Method:").pack(anchor="w")
        self.method_var = tk.StringVar(value="D3QN + D-opt + Frontier")
        self.cmb_method = ttk.Combobox(frame_agent, textvariable=self.method_var, values=["D3QN (Nav Only)", "D3QN + D-opt", "D3QN + D-opt + Frontier"], state="readonly")
        self.cmb_method.pack(fill="x", pady=2)
        self.btn_reward = ttk.Button(frame_agent, text="Launch Reward Node 🚀", command=self.launch_reward)
        self.btn_reward.pack(fill="x", pady=2)
        
        ttk.Button(frame_agent, text="Select Model (.pth)", command=self.browse_model).pack(fill="x", pady=5)
        self.model_path_var = tk.StringVar()
        ttk.Entry(frame_agent, textvariable=self.model_path_var).pack(fill="x")

        # 3. Control
        frame_ctrl = ttk.LabelFrame(left_panel, text="3. Control", padding=10)
        frame_ctrl.pack(fill="x", pady=5)
        self.episode_var = tk.IntVar(value=5)
        ttk.Label(frame_ctrl, text="Episodes:").pack(side="left")
        ttk.Entry(frame_ctrl, textvariable=self.episode_var, width=5).pack(side="left", padx=5)
        
        self.btn_start = ttk.Button(left_panel, text="START TEST", command=self.start_test)
        self.btn_start.pack(fill="x", pady=10)
        self.btn_stop = ttk.Button(left_panel, text="STOP", command=self.stop_test, state="disabled")
        self.btn_stop.pack(fill="x")

        # Right Panel (Logs & Vis)
        self.log_area = scrolledtext.ScrolledText(right_panel, height=12, state='disabled')
        self.log_area.pack(fill="x", pady=5)
        
        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill="both", expand=True)
        self.tab_map = ttk.Frame(self.notebook)
        self.tab_graph = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_map, text="Live Map")
        self.notebook.add(self.tab_graph, text="Metrics Graph")
        
        self.map_label = ttk.Label(self.tab_map, text="Waiting for Map...")
        self.map_label.pack(expand=True)
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.canvas_graph = tkagg.FigureCanvasTkAgg(self.fig, master=self.tab_graph)
        self.canvas_graph.get_tk_widget().pack(fill="both", expand=True)

    def log(self, msg):
        self.log_area.config(state='normal')
        self.log_area.insert(tk.END, f"> {msg}\n")
        self.log_area.see(tk.END)
        self.log_area.config(state='disabled')

    def browse_model(self):
        f = filedialog.askopenfilename(filetypes=[("Model", "*.pth *.pt"), ("All", "*.*")])
        if f: self.model_path_var.set(f)

    # --- Process Management ---
    def launch_in_term(self, cmd_list):
        cmd_str = " ".join(cmd_list)
        # 実行後にウィンドウが閉じないように exec bash を追加
        bash_cmd = f"source ~/ros2_ws/install/setup.bash; {cmd_str}; echo 'Done. Press Enter to close.'; read; exec bash"
        
        # Priority 1: Terminator (推奨)
        if shutil.which("terminator"):
            self.log(f"Opening Terminator: {cmd_list[0]}...")
            # -T: タイトル, -x: コマンド実行
            subprocess.Popen(["terminator", "-T", "ActiveSLAM Process", "-x", "bash", "-c", bash_cmd])
            
        # Priority 2: GNOME Terminal
        elif shutil.which("gnome-terminal"):
            self.log(f"Opening Gnome Terminal...")
            try:
                subprocess.Popen(["gnome-terminal", "--", "bash", "-c", bash_cmd])
            except:
                # 失敗したらバックグラウンドへ
                subprocess.Popen(cmd_list, preexec_fn=os.setsid)

        # Priority 3: XTerm (古い)
        elif shutil.which("xterm"):
            self.log(f"Opening XTerm...")
            subprocess.Popen(["xterm", "-fa", "Monospace", "-fs", "11", "-e", "bash", "-c", bash_cmd])
            
        else:
            self.log("⚠ No terminal found. Running in background.")
            subprocess.Popen(cmd_list, preexec_fn=os.setsid)

    def launch_gazebo(self):
        world_file = {"Circuit 1 (Training)": "circuit.world", "Circuit 2 (Obstacles)": "circuit2.world", "Maze (Complex)": "maze.world"}.get(self.world_var.get(), "circuit.world")
        path = f"/home/nakayama/ros2_ws/src/active_slam_rl/worlds/{world_file}"
        
        # ★★★ 修正: training_env_nogui.launch.py を呼ぶように変更 ★★★
        # gui:=false などの引数は不要になります
        self.launch_in_term(["ros2", "launch", "active_slam_rl", "training_env_no_gui.launch.py", f"world:={path}"])
    def launch_reward(self):
        node = "active_slam_reward_node" if "Nav Only" in self.method_var.get() else "uncertainty_reward_node"
        self.launch_in_term(["ros2", "run", "active_slam_rl", node])

    def kill_all(self):
        self.log("Killing all processes...")
        for p in ["gazebo", "slam_toolbox", "nav2", "active_slam_reward_node", "uncertainty_reward_node", "xterm"]:
            subprocess.run(f"pkill -f {p}", shell=True)

    # --- ROS & Testing Logic ---
    def run_ros_backend(self):
        if not rclpy.ok(): rclpy.init()
        self.executor = MultiThreadedExecutor()
        self.backend = TestBackendNode()
        self.executor.add_node(self.backend)
        try:
            self.executor.spin()
        except: pass

    def start_test(self):
        if not self.model_path_var.get():
            return messagebox.showwarning("Warn", "Select Model first")
        
        self.is_running = True
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        threading.Thread(target=self.run_test_loop, daemon=True).start()

    def stop_test(self):
        self.is_running = False

    def run_test_loop(self):
        env_wrapper = None
        try:
            # 1. エージェントのロード
            self.log("Loading Agent...")
            agent = TestD3QNAgent(100, 3)
            agent.load_model(self.model_path_var.get())
            
            # 2. 環境のセットアップ (Wrapperを使うので簡単！)
            self.log("Setting up Environment...")
            env_wrapper = TestEnvWrapper(self.method_var.get(), self.executor)
            
            # 3. エピソードループ
            for e in range(self.episode_var.get()):
                if not self.is_running: break
                self.log(f"--- Episode {e+1} ---")
                self.backend.reset_metrics()
                
                # リセット (ここでセンサ待ちが自動で走る)
                state = env_wrapper.reset()
                
                score, step, done = 0, 0, False
                while not done and step < 500 and self.is_running:
                    action = agent.get_action(state)
                    state, reward, done, _ = env_wrapper.step(action)
                    score += reward
                    step += 1
                
                sr = self.backend.get_coverage_ratio()
                self.log(f"Ep {e+1}: Reward={score:.1f}, Coverage={sr:.1f}%")
                self.root.after(100, lambda: self.update_vis(e+1))
                
        except Exception as ex:
            self.log(f"Error: {ex}")
            import traceback
            traceback.print_exc()
        finally:
            if env_wrapper: env_wrapper.close()
            self.is_running = False
            self.btn_start.config(state="normal")
            self.btn_stop.config(state="disabled")

    def update_vis(self, ep):
        # (前回と同じ描画ロジック)
        try:
            self.ax.clear()
            if self.backend.d_opt_history:
                self.ax.plot(self.backend.d_opt_history)
                self.canvas_graph.draw()
            # Map描画省略（長くなるため）
        except: pass

    def on_close(self):
        self.kill_all()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = TestApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()