import time
import pyrealsense2 as rs
import numpy as np
import cv2
from aruco_new import ArucoDetector
from cart_tracker import CartTrackerPnP
from recorder import SessionRecorder

# 1. Setup RealSense
pipeline = rs.pipeline()
config = rs.config()

if_depth = False

if if_depth:
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

# 2. Align Object
align_to = rs.stream.color
align = rs.align(align_to)

print("Starting Stream...")
profile = pipeline.start(config)

# 3. Initialize Detectors & Recorder
aruco_system = ArucoDetector(target_ids=[0,1,2,3,4]) 
tracker_pnp = CartTrackerPnP(anchor_ids=[3,2,4,1], cart_id=0)
recorder = SessionRecorder()

display_offset = (4+7/16, 3+15/16)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        
        depth_frame = aligned_frames.get_depth_frame() if if_depth else None
        color_frame = aligned_frames.get_color_frame()
        
        if not color_frame:
            continue
        
        color_image = np.asanyarray(color_frame.get_data())
        ids, centers, corners = aruco_system.detect_markers_subpixel(color_image)
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        
        # Update Tracker
        status_pnp, position_pnp, angle_pnp = tracker_pnp.update(ids, centers, corners, depth_frame, color_intrin)
        
        # Core Visualization
        cv2.putText(color_image, f"PNP Status: {status_pnp}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        
        if position_pnp:
            x, y, z = position_pnp
            text = f"Cart Pos (PNP): X={(x-display_offset[0]):.2f} Y={(y-display_offset[1]):.2f} Z={z:.2f}; Angle={angle_pnp:.1f} deg"
            cv2.putText(color_image, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            # print(text)
        else:
            # print out misses for debug
            time_stamp = time.time()
            print("missed"+str(time_stamp)) 
        
        # Recording UI and I/O Trigger
        if recorder.is_recording:
            cv2.circle(color_image, (610, 30), 10, (0, 0, 255), -1)
            cv2.putText(color_image, "REC", (560, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            
        recorder.record_frame(color_image, position_pnp, angle_pnp)
        
        # save image for experiment
        # cv2.imwrite("test.png", color_image)
        # break
        
        cv2.imshow('RealSense Tracker', color_image)
        
        # # Keyboard Controller
        key = cv2.waitKey(1) & 0xFF
        # key = 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r'):
            if not recorder.is_recording:
                recorder.start(frame_shape=color_image.shape, fps=30.0)
            else:
                recorder.stop()
                
finally:
    recorder.cleanup()
    pipeline.stop()
    cv2.destroyAllWindows()