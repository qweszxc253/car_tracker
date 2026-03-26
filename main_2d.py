import pyrealsense2 as rs
import numpy as np
import cv2
from aruco2 import ArucoDetector
from cart_tracker import CartTracker, CartTrackerPnP

# 1. Setup RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

# 2. Align Object (CRITICAL for matching RGB pixels to Depth)
align_to = rs.stream.color
align = rs.align(align_to)

print("Starting Stream...")
profile = pipeline.start(config)

# 3. Initialize Detectors
aruco_system = ArucoDetector(target_ids=[0,1,2,3,4]) # 0=Floor, 1=Cart
tracker = CartTracker(ref_ids=[3,2,4], cart_id=0)
tracker_pnp = CartTrackerPnP(anchor_ids=[3,2,4,1], cart_id=0)

try:
    while True:
        frames = pipeline.wait_for_frames()
        
        # Align depth frame to color frame
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not depth_frame or not color_frame: continue
        
        # Convert to numpy
        color_image = np.asanyarray(color_frame.get_data())
        
        # 4. Detect ArUco Markers
        # NOW returns corners as well
        ids, centers, corners = aruco_system.detect_markers(color_image)
        
        # 5. Get Intrinsics (Geometry of the camera)
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        
        # 6. Update Tracker Logic
        status, position = tracker.update(ids, centers, corners, depth_frame, depth_intrin)
        status_pnp, position_pnp = tracker_pnp.update(ids, centers, corners, depth_frame, depth_intrin)
        
        # 7. Visualization
        cv2.putText(color_image, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(color_image, f"PNP Status: {status_pnp}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        if position:
            x, y, z = position
            text = f"Cart Pos: X={x:.3f} Y={y:.3f} Z={z:.3f}"
            cv2.putText(color_image, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print(text)

        if position_pnp:
            x, y, z = position_pnp
            text = f"Cart Pos (PNP): X={x:.3f} Y={y:.3f} Z={z:.3f}"
            cv2.putText(color_image, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            print(text)

        cv2.imshow('RealSense Tracker', color_image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()