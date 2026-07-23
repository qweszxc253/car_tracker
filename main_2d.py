import pyrealsense2 as rs
import numpy as np
import cv2
import time
import os
import csv
from aruco2 import ArucoDetector
from cart_tracker import CartTrackerPnP

# set up directories for data collection
DATA_DIR = "recordings/data"
VIDEO_DIR = "recordings/videos"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# 1. Setup RealSense
pipeline = rs.pipeline()
config = rs.config()

if_depth = False

if if_depth:
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
# config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
# make high res color only stream




# 2. Align Object (CRITICAL for matching RGB pixels to Depth)
align_to = rs.stream.color
align = rs.align(align_to)

print("Starting Stream...")
profile = pipeline.start(config)

# 3. Initialize Detectors
aruco_system = ArucoDetector(target_ids=[0,1,2,3,4]) # 0=Floor, 1=Cart
# tracker = CartTracker(ref_ids=[3,2,4], cart_id=0)
tracker_pnp = CartTrackerPnP(anchor_ids=[3,2,4,1], cart_id=0)

# set up recording vars and buffer
is_recording = False
data_buffer = []
BUFFER_LIMIT = 300
start_time_str = ""
temp_csv_path = ""
temp_video_path = ""

# Persistent file objects
video_writer = None
csv_file = None
csv_writer = None
frames_since_flush = 0


display_offset = (4+7/16, 3+15/16)


try:
    while True:
        frames = pipeline.wait_for_frames()
        
        # Align depth frame to color frame
        aligned_frames = align.process(frames)
        
        if if_depth:
            depth_frame = aligned_frames.get_depth_frame()
        else:
            depth_frame = None # dummy object
        color_frame = aligned_frames.get_color_frame()
        
        if not color_frame:
            continue
        
        # Convert to numpy
        color_image = np.asanyarray(color_frame.get_data())
        
        # 4. Detect ArUco Markers
        # NOW returns corners as well
        # ids, centers, corners = aruco_system.detect_markers(color_image)
        
        # CHANGE: now subpixel detection, not integer rounding
        ids, centers, corners = aruco_system.detect_markers_subpixel(color_image)
        
        # 5. Get Intrinsics (Geometry of the camera)
        # depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        
        # 5. Get Intrinsics (Geometry of the camera)
        # CRITICAL: Pull intrinsics from the COLOR frame, because depth is aligned to color
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        
        # 6. Update Tracker Logic
        status_pnp, position_pnp, angle_pnp = tracker_pnp.update(ids, centers, corners, depth_frame, color_intrin)
        
        # example, print focal length
        # print(f"Focal Length: fx={depth_intrin.fx}, fy={depth_intrin.fy}")
        # quit()
        
        # 6. Update Tracker Logic
        # status, position = tracker.update(ids, centers, corners, depth_frame, depth_intrin)
        # status_pnp, position_pnp, angle_pnp = tracker_pnp.update(ids, centers, corners, depth_frame, depth_intrin)
        
        # 7. Visualization
        # cv2.putText(color_image, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(color_image, f"PNP Status: {status_pnp}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
        # if position:
        #     x, y, z = position
        #     text = f"Cart Pos: X={x:.3f} Y={y:.3f} Z={z:.3f}"
        #     cv2.putText(color_image, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #     print(text)

        if position_pnp:
            x, y, z = position_pnp
            text = f"Cart Pos (PNP): X={(x-display_offset[0]):.2f} Y={(y-display_offset[1]):.2f} Z={z:.2f}; Angle={angle_pnp:.1f} deg"
            cv2.putText(color_image, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            print(text)
            
            """add write to file if recording"""
            if is_recording and csv_writer is not None:
                current_time = time.time()
                csv_writer.writerow([current_time, x, y, z, angle_pnp])
                
                # count frames
                frames_since_flush += 1
                if frames_since_flush >= 60: # flush 
                    csv_file.flush()
                    os.fsync(csv_file.fileno())
                    frames_since_flush = 0
        
        # added recording indicator
        # UI Recording Indicator
        if is_recording:
            cv2.circle(color_image, (610, 30), 10, (0, 0, 255), -1)
            cv2.putText(color_image, "REC", (560, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            if video_writer is not None:
                video_writer.write(color_image)

        cv2.imshow('RealSense Tracker', color_image)
        
        
        """key press"""
        key = cv2.waitKey(1) & 0xFF
        if key ==ord('q'):
            break
        elif key == ord('r'):
            if not is_recording:
                # START RECORDING
                is_recording = True
                start_time_str = time.strftime("%Y%m%d_%H%M%S")
                print(f"\n--- RECORDING STARTED: {start_time_str} ---")
                
                temp_csv_path = os.path.join(DATA_DIR, "temp_data.csv")
                temp_video_path = os.path.join(VIDEO_DIR, "temp_video.avi")
                
                # Open CSV file and keep it open
                csv_file = open(temp_csv_path, 'w', newline='')
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(['timestamp', 'x', 'y', 'z', 'angle_deg'])
                frames_since_flush = 0
                
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(temp_video_path, fourcc, 30.0, (640, 480))
            else:
                # STOP RECORDING
                is_recording = False
                end_time_str = time.strftime("%Y%m%d_%H%M%S")
                print(f"\n--- RECORDING STOPPED: {end_time_str} ---")
                
                # Close files safely
                if csv_file is not None:
                    csv_file.close()
                    csv_file = None
                    csv_writer = None
                
                if video_writer is not None:
                    video_writer.release()
                    video_writer = None
                    
                # Rename temp files
                final_csv_path = os.path.join(DATA_DIR, f"data-{start_time_str}-{end_time_str}.csv")
                final_video_path = os.path.join(VIDEO_DIR, f"video-{start_time_str}-{end_time_str}.avi")
                
                os.rename(temp_csv_path, final_csv_path)
                os.rename(temp_video_path, final_video_path)
                
                print(f"Exported: {final_csv_path}")
                print(f"Exported: {final_video_path}\n")
                
finally:
    # Failsafe cleanup
    if csv_file is not None:
        csv_file.close()
    if video_writer is not None:
        video_writer.release()
    pipeline.stop()
    cv2.destroyAllWindows()