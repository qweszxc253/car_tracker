import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry # Replace with your actual F1TENTH control msg if different
from std_msgs.msg import String
import threading
import subprocess
import signal
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk

RECORDINGS_DIR = os.path.join("recordings", "rosbag")

# Topics captured into the rosbag on every recording. Edit this list to add/remove topics.
TOPICS_TO_RECORD = [
    '/pc/aruco_pose',  # Vision pose from main_node
    '/odom',           # F1TENTH odometry
    '/scan',           # LiDAR scan
]

class DataPlaneNode(Node):
    def __init__(self):
        super().__init__('data_plane_controller')
        
        # Latest states for the UI to read
        self.latest_cart_pose = "Waiting for Vision..."
        self.latest_car_speed = "Waiting for F1TENTH..."
        
        # Subscriptions
        self.create_subscription(
            PoseStamped, 
            '/pc/aruco_pose', 
            self.vision_callback, 
            10
        )
        self.create_subscription(
            Odometry,
            '/odom', # Replace with your F1TENTH high-freq topic
            self.control_callback,
            10
        )

        # Command channel to tell main_node (possibly on a remote host) when to
        # start/stop its own video+CSV recording. std_msgs/String carries a shared
        # session ID so the rosbag folder and main_node's files pair up.
        self.record_cmd_pub = self.create_publisher(String, '/recording/command', 10)

    def send_record_command(self, text):
        msg = String()
        msg.data = text
        self.record_cmd_pub.publish(msg)

    def vision_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        self.latest_cart_pose = f"X: {x:.2f}m | Y: {y:.2f}m"

    def control_callback(self, msg):
        # Example extracting linear velocity
        speed = msg.twist.twist.linear.x
        self.latest_car_speed = f"{speed:.2f} m/s"


class DataPlaneUI:
    def __init__(self, root, ros_node):
        self.root = root
        self.ros_node = ros_node
        self.root.title("F1TENTH Data Plane Controller")
        self.root.geometry("400x250")
        
        self.bag_process = None
        
        # UI Elements
        self.status_label = ttk.Label(root, text="System Ready", font=("Helvetica", 14, "bold"))
        self.status_label.pack(pady=10)
        
        self.vision_label = ttk.Label(root, text="Vision: --")
        self.vision_label.pack(pady=5)
        
        self.control_label = ttk.Label(root, text="F1TENTH: --")
        self.control_label.pack(pady=5)
        
        self.record_btn = ttk.Button(root, text="START RECORDING", command=self.toggle_recording)
        self.record_btn.pack(pady=20)
        
        # Start UI Update Loop
        self.update_ui()

    def update_ui(self):
        """Pulls latest data from the ROS node safely without blocking."""
        self.vision_label.config(text=f"Vision: {self.ros_node.latest_cart_pose}")
        self.control_label.config(text=f"F1TENTH: {self.ros_node.latest_car_speed}")
        
        # Re-run this method every 100ms (10Hz UI refresh rate)
        self.root.after(100, self.update_ui)

    def toggle_recording(self):
        if self.bag_process is None:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        os.makedirs(RECORDINGS_DIR, exist_ok=True)
        # One session ID shared by the rosbag folder and main_node's video/CSV so they pair up.
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        bag_name = os.path.join(RECORDINGS_DIR, session_id)

        cmd = ['ros2', 'bag', 'record', '-o', bag_name] + TOPICS_TO_RECORD

        # Launch detached subprocess
        self.bag_process = subprocess.Popen(cmd, preexec_fn=os.setsid)

        # Signal main_node (possibly remote) to begin its own recording under the same ID.
        self.ros_node.send_record_command(f"START:{session_id}")

        self.record_btn.config(text="STOP RECORDING")
        self.status_label.config(text="RECORDING ACTIVE", foreground="red")
        self.ros_node.get_logger().info(f"Spawned rosbag2 subprocess. Session ID: {session_id}")

    def stop_recording(self):
        if self.bag_process:
            # Safely send SIGINT to the process group to close the SQLite DB cleanly
            os.killpg(os.getpgid(self.bag_process.pid), signal.SIGINT)
            self.bag_process = None

            # Tell main_node to finalize and flush its video/CSV.
            self.ros_node.send_record_command("STOP")

            self.record_btn.config(text="START RECORDING")
            self.status_label.config(text="System Ready", foreground="black")
            self.ros_node.get_logger().info("Terminated rosbag2 subprocess.")

def main():
    rclpy.init()
    
    # 1. Initialize the ROS Node
    data_node = DataPlaneNode()
    
    # 2. Spin ROS in a background thread to prevent UI freezing
    spin_thread = threading.Thread(target=rclpy.spin, args=(data_node,), daemon=True)
    spin_thread.start()
    
    # 3. Launch the UI on the main thread
    root = tk.Tk()
    app = DataPlaneUI(root, data_node)
    
    # Protocol to handle hitting the 'X' on the window
    def on_closing():
        if app.bag_process is not None:
            app.stop_recording()
        rclpy.shutdown()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == '__main__':
    main()
