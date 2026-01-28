#!/usr/bin/env python3
import os
import sys
import threading
import subprocess
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg as tkagg
import customtkinter as ctk

import rclpy
from rclpy.executors import MultiThreadedExecutor

# --- パス設定 & ローカルインポート ---
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from test_config import GUI_APPEARANCE, GUI_THEME, WINDOW_SIZE, SAVE_DIR_BASE
    from test_backend import TestBackendNode
    from test_saver import TestResultSaver
    from test_d3qn import TestD3QNAgent
    from test_env import TestEnvWrapper
except ImportError as e:
    print(f"⚠️ Import Error: {e}. Trying package imports...")
    from active_slam_rl.test_config import GUI_APPEARANCE, GUI_THEME, WINDOW_SIZE, SAVE_DIR_BASE
    from active_slam_rl.test_backend import TestBackendNode
    from active_slam_rl.test_saver import TestResultSaver
    from active_slam_rl.test_d3qn import TestD3QNAgent
    from active_slam_rl.test_env import TestEnvWrapper

# --- 設定適用 ---
ctk.set_appearance_mode(GUI_APPEARANCE)
ctk.set_default_color_theme(GUI_THEME)

class ModernApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Active SLAM Control Center (Modular V4)")
        self.geometry(WINDOW_SIZE)
        
        self.is_running = False
        self.backend = None
        self.executor = None
        self.model_path = ""
        
        # モジュール初期化
        self.saver = TestResultSaver(SAVE_DIR_BASE)
        
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.create_sidebar()
        self.create_main_area()
        
        # ROSスレッド開始
        threading.Thread(target=self.run_ros_backend, daemon=True).start()

    def create_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(10, weight=1)

        ctk.CTkLabel(self.sidebar, text="🤖 Active SLAM RL", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 10))

        ctk.CTkLabel(self.sidebar, text="1. Environment", anchor="w").grid(row=1, column=0, padx=20, pady=(10, 0))
        self.world_var = ctk.StringVar(value="Circuit 1 (Training)")
        ctk.CTkComboBox(self.sidebar, values=["Circuit 1 (Training)", "Circuit 2 (Obstacles)", "Maze (Complex)"], variable=self.world_var).grid(row=2, column=0, padx=20, pady=10)
        ctk.CTkButton(self.sidebar, text="Launch Gazebo (No GUI)", command=self.launch_gazebo, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE")).grid(row=3, column=0, padx=20, pady=10)

        ctk.CTkLabel(self.sidebar, text="2. Algorithm", anchor="w").grid(row=4, column=0, padx=20, pady=(20, 0))
        self.method_var = ctk.StringVar(value="D3QN + D-opt + Frontier")
        ctk.CTkComboBox(self.sidebar, values=["D3QN (Nav Only)", "D3QN + D-opt", "D3QN + D-opt + Frontier"], variable=self.method_var).grid(row=5, column=0, padx=20, pady=10)
        ctk.CTkButton(self.sidebar, text="Launch Reward Node", command=self.launch_reward, fg_color="transparent", border_width=2, text_color=("gray10", "#DCE4EE")).grid(row=6, column=0, padx=20, pady=10)

        ctk.CTkLabel(self.sidebar, text="3. Model", anchor="w").grid(row=7, column=0, padx=20, pady=(20, 0))
        ctk.CTkButton(self.sidebar, text="Select Model (.pth)", command=self.browse_model).grid(row=8, column=0, padx=20, pady=10)
        self.lbl_model_name = ctk.CTkLabel(self.sidebar, text="No model selected", font=ctk.CTkFont(size=10))
        self.lbl_model_name.grid(row=9, column=0, padx=20, pady=(0, 10))

        self.btn_start = ctk.CTkButton(self.sidebar, text="START TEST", command=self.start_test, fg_color="green", hover_color="darkgreen")
        self.btn_start.grid(row=11, column=0, padx=20, pady=10)
        
        self.btn_stop = ctk.CTkButton(self.sidebar, text="STOP", command=self.stop_test, fg_color="darkred", hover_color="maroon", state="disabled")
        self.btn_stop.grid(row=12, column=0, padx=20, pady=(0, 20))
        
        self.btn_save = ctk.CTkButton(self.sidebar, text="💾 Save Now", command=self.save_data_manual, fg_color="#D35B58", hover_color="#C74D49")
        self.btn_save.grid(row=13, column=0, padx=20, pady=10)

        self.btn_kill = ctk.CTkButton(self.sidebar, text="💀 Kill All", command=self.kill_all, fg_color="gray", hover_color="gray30")
        self.btn_kill.grid(row=14, column=0, padx=20, pady=20)

    def create_main_area(self):
        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_frame.grid_rowconfigure(0, weight=3)
        self.main_frame.grid_rowconfigure(1, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(1, weight=1)

        # 左上: 地図
        self.map_frame = ctk.CTkFrame(self.main_frame)
        self.map_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        self.fig_map, self.ax_map = plt.subplots(figsize=(5, 4), dpi=100)
        self.format_plot(self.fig_map, self.ax_map, "Live Map & Uncertainty")
        self.canvas_map = tkagg.FigureCanvasTkAgg(self.fig_map, master=self.map_frame)
        self.canvas_map.get_tk_widget().pack(fill="both", expand=True)

        # 右上: グラフ
        self.graph_frame = ctk.CTkFrame(self.main_frame)
        self.graph_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.fig_graph, self.ax_graph = plt.subplots(figsize=(5, 4), dpi=100)
        self.format_plot(self.fig_graph, self.ax_graph, "D-Optimality History")
        self.canvas_graph = tkagg.FigureCanvasTkAgg(self.fig_graph, master=self.graph_frame)
        self.canvas_graph.get_tk_widget().pack(fill="both", expand=True)

        # 下段: ログ
        self.log_textbox = ctk.CTkTextbox(self.main_frame, height=150)
        self.log_textbox.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=5, pady=10)
        self.log_textbox.insert("0.0", "System Ready.\n")

    def format_plot(self, fig, ax, title):
        fig.patch.set_facecolor('#2b2b2b')
        ax.set_facecolor('#2b2b2b')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        ax.set_title(title)

    def log(self, msg):
        self.log_textbox.insert(tk.END, f"> {msg}\n")
        self.log_textbox.see(tk.END)

    def browse_model(self):
        f = filedialog.askopenfilename(filetypes=[("Model", "*.pth *.pt"), ("All", "*.*")])
        if f:
            self.model_path = f
            self.lbl_model_name.configure(text=os.path.basename(f))
            self.log(f"Model selected: {os.path.basename(f)}")

    def run_in_term(self, cmd_list):
        cmd_str = " ".join(cmd_list)
        bash = f"source ~/ros2_ws/install/setup.bash; {cmd_str}; echo 'Done.'; read; exec bash"
        if shutil.which("terminator"):
            subprocess.Popen(["terminator", "-T", "ActiveSLAM", "-x", "bash", "-c", bash])
        else:
            self.log("Terminator not found. Running in background.")
            subprocess.Popen(cmd_list, preexec_fn=os.setsid)

    def launch_gazebo(self):
        w_file = {"Circuit 1 (Training)": "circuit.world", "Circuit 2 (Obstacles)": "circuit2.world", "Maze (Complex)": "maze.world"}.get(self.world_var.get(), "circuit.world")
        path = f"/home/nakayama/ros2_ws/src/active_slam_rl/worlds/{w_file}"
        self.log(f"Launching Gazebo: {w_file}")
        self.run_in_term(["ros2", "launch", "active_slam_rl", "training_env_no_gui.launch.py", f"world:={path}"])

    def launch_reward(self):
        method = self.method_var.get()
        if "Nav Only" in method: return messagebox.showinfo("Info", "Nav Only uses internal rewards.")
        node = "active_slam_reward_node" if "Frontier" in method else "uncertainty_reward_node"
        self.log(f"Launching Reward Node: {node}")
        self.run_in_term(["ros2", "run", "active_slam_rl", node])

    def kill_all(self):
        self.log("Stopping all processes...")
        for p in ["gazebo", "slam_toolbox", "nav2", "active_slam_reward_node", "uncertainty_reward_node", "terminator"]:
            subprocess.run(f"pkill -f {p}", shell=True)

    def run_ros_backend(self):
        if not rclpy.ok(): rclpy.init()
        self.executor = MultiThreadedExecutor()
        self.backend = TestBackendNode()
        self.executor.add_node(self.backend)
        try: self.executor.spin()
        except: pass

    def start_test(self):
        if not self.model_path: return messagebox.showwarning("Warning", "Select Model first.")
        self.is_running = True
        self.btn_start.configure(state="disabled")
        self.btn_stop.configure(state="normal")
        threading.Thread(target=self.run_test_loop, daemon=True).start()

    def stop_test(self):
        self.is_running = False

    def run_test_loop(self):
        env_wrapper = None
        try:
            self.log("Initializing Agent & Env...")
            agent = TestD3QNAgent(100, 3)
            agent.load_model(self.model_path)
            env_wrapper = TestEnvWrapper(self.method_var.get())
            
            for e in range(5):
                if not self.is_running: break
                self.log(f"--- Episode {e+1} Start ---")
                self.backend.reset_metrics()
                state = env_wrapper.reset()
                score, step, done = 0, 0, False
                
                while not done and step < 10000 and self.is_running:
                    action = agent.get_action(state)
                    state, reward, done, _ = env_wrapper.step(action)
                    score += reward
                    step += 1
                    if step % 50 == 0: self.after(10, self.update_visuals)
                
                cov = self.backend.get_coverage_ratio()
                reason = "Collision" if done else "Max Steps"
                self.log(f"Ep {e+1} End: Steps={step}, R={score:.1f}, Cov={cov:.1f}%, {reason}")
                
                # ★ 保存機能をSaverクラスに委譲
                saved_path = self.saver.save_episode(e+1, self.backend)
                self.log(f"💾 Saved to: {os.path.basename(saved_path)}")
                
                self.after(10, self.update_visuals)
                
        except Exception as e:
            self.log(f"Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if env_wrapper: env_wrapper.close()
            self.is_running = False
            self.btn_start.configure(state="normal")
            self.btn_stop.configure(state="disabled")
            self.log("Test Finished.")

    def update_visuals(self):
        try:
            # Map & Uncertainty
            self.ax_map.clear()
            if self.backend.map_data:
                w = self.backend.map_data.info.width
                h = self.backend.map_data.info.height
                data = np.array(self.backend.map_data.data).reshape((h, w))
                disp = np.full_like(data, 127, dtype=np.uint8)
                disp[data == 0] = 255
                disp[data == 100] = 0
                self.ax_map.imshow(disp, cmap='gray', origin='lower', alpha=0.6)

            if self.backend.uncertainty_map:
                ux, uy, ud = zip(*self.backend.uncertainty_map)
                self.ax_map.scatter(ux, uy, c=ud, cmap='jet', s=3, alpha=0.8)
            
            self.ax_map.set_title("Live Map", color='white')
            self.canvas_map.draw()

            # Graph
            self.ax_graph.clear()
            if self.backend.d_opt_history:
                self.ax_graph.plot(self.backend.d_opt_history, color='#00ffcc', label='D-opt')
                self.ax_graph.legend()
                self.ax_graph.grid(True, linestyle='--', alpha=0.3)
            self.ax_graph.set_title("Metrics", color='white')
            self.canvas_graph.draw()
        except: pass

    def save_data_manual(self):
        saved_path = self.saver.save_episode("manual", self.backend)
        self.log(f"💾 Manual Save: {saved_path}")

    def on_close(self):
        self.kill_all()
        self.destroy()

if __name__ == "__main__":
    app = ModernApp()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()