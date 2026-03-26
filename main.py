"""# Install the Intel RealSense SDK wrapper
pip install pyrealsense2

# Install OpenCV
pip install opencv-python

# Install NumPy (often a dependency, but good to be explicit)
pip install numpy"""


"""
git clone https://github.com/IntelRealSense/librealsense.git

# Go into the new folder
cd librealsense

# Copy the permission rules file
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/

# Reload the system rules so it takes effect
sudo udevadm control --reload-rules && sudo udevadm trigger"""


""" for usb attach WSL:
winget install --interactive --exact dorssel.usbipd-win
> usbipd list
> usbipd attach --wsl --busid 1-6
> usbipd bind --busid 1-6"""

"""sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/"""
import pyrealsense2 as rs
import numpy as np
import cv2
from aruco import ArucoDetector

# --- 1. Initialize Pipeline and Config ---
# Configure depth and color streams

# save as check_rs.py and run: python3 check_rs.py

ctx = rs.context()

for i, d in enumerate(ctx.devices):
    print(i, d.get_info(rs.camera_info.name), d.get_info(rs.camera_info.serial_number))
    
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()

# Try to find a supported stream profile
# We want 640x480 resolution for both color and depth
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)

# --- 2. Start Streaming ---
print("Starting stream...")
pipeline.start(config)

# --- 3. Create a colorizer for depth data ---
# This helps visualize the depth data
colorizer = rs.colorizer()
detector = ArucoDetector()

try:
    while True:
        # --- 4. Wait for a coherent pair of frames: depth and color ---
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # if not depth_frame or not color_frame:
        #     continue
        window_size = 300
        if depth_frame:
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = np.asanyarray(colorizer.colorize(depth_frame).get_data())
            cv2.namedWindow('Depth',window_size) 
            cv2.imshow('Depth', depth_colormap)

        if color_frame:
            color_image = np.asanyarray(color_frame.get_data())
                    # --- 7. Display in OpenCV ---
            cv2.namedWindow('RealSense', window_size )
            cv2.imshow('RealSense', color_image)
            
            ids, centers = detector.detect_markers(color_image)
            
            # --- 5. Convert frames to NumPy arrays ---
            # This is the bridge to OpenCV
        

        # --- 6. Process images ---
        # Apply color map to depth image (for visualization)

        # Note: RealSense SDK gives color frames in BGR format by default
        # (when using rs.format.bgr8), so no cvtColor is needed.
        # If you were using rs.format.rgb8, you'd need:
        # color_image_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        # Stack images horizontally
        # images = np.hstack((color_image, depth_colormap))
        




        
        
        
        
        # Exit on 'q' key press
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break

finally:
    # --- 8. Stop Streaming ---
    print("Stopping stream...")
    pipeline.stop()