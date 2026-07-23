import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
import threading
import pyrealsense2 as rs
import numpy as np
import cv2
import math
import time

from aruco_new import ArucoDetector
from cart_tracker import CartTrackerPnP
from recorder import SessionRecorder

def yaw_to_quaternion(yaw_degrees):
    """
    Converts a Z-axis rotation in degrees to a ROS 2 compliant Quaternion.
    """
    yaw_rad = math.radians(yaw_degrees)
    return {
        'x': 0.0,
        'y': 0.0,
        'z': math.sin(yaw_rad / 2.0),
        'w': math.cos(yaw_rad / 2.0)
    }

class VisionNode(Node):
    def __init__(self):
        super().__init__('offboard_vision_node')
        
        # Unconditionally publish to a defined topic
        self.pose_pub = self.create_publisher(PoseStamped, '/pc/aruco_pose', 10)
        self.get_logger().info("Vision Node Initialized. Publishing to /pc/aruco_pose")

        # Recording command channel from the data-plane controller. The callback runs on
        # the rclpy.spin thread, so it only stashes a pending command; the capture loop
        # (which owns the cv2.VideoWriter and knows the frame shape) acts on it.
        self._cmd_lock = threading.Lock()
        self.pending_cmd = None
        self.create_subscription(String, '/recording/command', self.record_cmd_callback, 10)

    def record_cmd_callback(self, msg):
        """Parse 'START:<session_id>' / 'STOP' and hand it to the capture loop."""
        data = msg.data.strip()
        if data.startswith("START"):
            _, _, session_id = data.partition(":")
            cmd = ("START", session_id.strip() or None)
        elif data.startswith("STOP"):
            cmd = ("STOP", None)
        else:
            self.get_logger().warn(f"Ignoring unknown recording command: {data!r}")
            return

        with self._cmd_lock:
            self.pending_cmd = cmd

    def take_pending_cmd(self):
        """Atomically return and clear the pending recording command (None if none)."""
        with self._cmd_lock:
            cmd = self.pending_cmd
            self.pending_cmd = None
        return cmd

    def publish_pose(self, timestamp_msg, position, angle_deg):
        """
        Constructs and publishes the standardized PoseStamped message.
        """
        msg = PoseStamped()
        
        # 1. Apply the exact timestamp from frame acquisition
        msg.header.stamp = timestamp_msg
        msg.header.frame_id = "map" # Or "camera_link", depending on your TF tree
        
        # 2. Populate Position (convert to meters if your scale is in inches, ROS uses meters)
        # Assuming your current math outputs inches, multiply by 0.0254. 
        # If it's already meters, remove the multiplier.
        msg.pose.position.x = float(position[0] * 0.0254) 
        msg.pose.position.y = float(position[1] * 0.0254)
        msg.pose.position.z = float(position[2] * 0.0254)
        
        # 3. Populate Orientation (Quaternion)
        quat = yaw_to_quaternion(angle_deg)
        msg.pose.orientation.x = quat['x']
        msg.pose.orientation.y = quat['y']
        msg.pose.orientation.z = quat['z']
        msg.pose.orientation.w = quat['w']
        
        self.pose_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    vision_node = VisionNode()

    # Launch ROS 2 executor in a background daemon thread
    # This allows rclpy to process network events without blocking OpenCV
    spin_thread = threading.Thread(target=rclpy.spin, args=(vision_node,), daemon=True)
    spin_thread.start()

    # --- Hardware Initialization ---
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Pragmatic Choice: Disabling depth stream if unused saves USB bandwidth and CPU cycles
    if_depth = False 
    if if_depth:
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    align = rs.align(rs.stream.color)
    vision_node.get_logger().info("Starting RealSense Stream...")
    profile = pipeline.start(config)

    # --- Algorithm Initialization ---
    aruco_system = ArucoDetector(target_ids=[0,1,2,3,4])
    tracker_pnp = CartTrackerPnP(anchor_ids=[3,2,4,1], cart_id=0)

    # Video + CSV recorder, driven remotely via /recording/command.
    recorder = SessionRecorder()

    display_offset = (4 + 7/16, 3 + 15/16)

    try:
        while True:
            # 1. Acquire Frame
            frames = pipeline.wait_for_frames()
            
            # 2. STAMP IMMEDIATELY. Do not wait for processing to finish.
            # This mitigates latency jitter in your post-processing pipeline.
            frame_timestamp = vision_node.get_clock().now().to_msg()
            
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame() if if_depth else None
            color_frame = aligned_frames.get_color_frame()
            
            if not color_frame:
                continue
            
            color_image = np.asanyarray(color_frame.get_data())
            color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
            
            # 3. Vision Processing Pipeline
            ids, centers, corners = aruco_system.detect_markers_subpixel(color_image)
            status_pnp, position_pnp, angle_pnp = tracker_pnp.update(
                ids, centers, corners, depth_frame, color_intrin
            )
            
            # 4. Network Publish
            if position_pnp is not None and angle_pnp is not None:
                vision_node.publish_pose(frame_timestamp, position_pnp, angle_pnp)

                # Debug output (Optional, can be removed for pure headless performance)
                x, y, z = position_pnp
                text = f"X={(x-display_offset[0]):.2f} Y={(y-display_offset[1]):.2f} Z={z:.2f} A={angle_pnp:.1f}"
                cv2.putText(color_image, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # 4b. Handle remote start/stop commands, then record the raw frame + telemetry.
            cmd = vision_node.take_pending_cmd()
            if cmd is not None:
                action, session_id = cmd
                if action == "START" and not recorder.is_recording:
                    recorder.start(color_image.shape, fps=30.0, session_id=session_id)
                    vision_node.get_logger().info(f"Recording started (session {session_id}).")
                elif action == "STOP" and recorder.is_recording:
                    recorder.stop()
                    vision_node.get_logger().info("Recording stopped.")

            if recorder.is_recording:
                recorder.record_frame(color_image, position_pnp, angle_pnp)

            cv2.putText(color_image, f"PNP Status: {status_pnp}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.imshow('RealSense Tracker (ROS 2 Node)', color_image)
            
            # 5. GIL Yield (CRITICAL)
            # This 1ms wait forces Python to release the Global Interpreter Lock, 
            # allowing the background spin_thread to actually transmit the DDS packets.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        vision_node.get_logger().info("Keyboard Interrupt caught. Shutting down...")
    finally:
        # Flush any in-progress recording so a session left running is finalized cleanly.
        if recorder.is_recording:
            recorder.stop()
        pipeline.stop()
        cv2.destroyAllWindows()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)

if __name__ == '__main__':
    main()
