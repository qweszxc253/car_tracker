import pyrealsense2 as rs
import numpy as np
import cv2

from aruco import ArucoDetector
# 1. Setup the pipeline
pipeline = rs.pipeline()
config = rs.config()

# 2. Enable the Infrared Stream (Left camera is usually Index 1)
# Format L8 is 8-bit unrectified, Y8 is 8-bit rectified grayscale. Y8 is usually best for CV.
config.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
# config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 3. Start streaming
profile = pipeline.start(config)


# --- CRITICAL STEP: TURN OFF THE LASER EMITTER ---
device = profile.get_device()
depth_sensor = device.first_depth_sensor()
# Get the depth sensor (which controls the IR cameras too)
depth_sensor = profile.get_device().first_depth_sensor()

# 1. Disable Auto Exposure (Essential)
if depth_sensor.supports(rs.option.enable_auto_exposure):
    depth_sensor.set_option(rs.option.enable_auto_exposure, 0) 

# 2. Set Low Exposure (Physics fix for motion blur)
# Value is in microseconds. Try 1000 (1ms) to 5000 (5ms).
# Default is often 33000 (33ms) which causes massive blur.
if depth_sensor.supports(rs.option.exposure):
    depth_sensor.set_option(rs.option.exposure, 3000) 

# 3. Increase Gain (Compensate for darkness)
# Value range is usually 16-248. Start high if image is too dark.
if depth_sensor.supports(rs.option.gain):
    depth_sensor.set_option(rs.option.gain, 248)
    


if depth_sensor.supports(rs.option.emitter_enabled):
    # 0 = Off (Clean image for ArUco), 1 = On (Dots for better depth)
    depth_sensor.set_option(rs.option.emitter_enabled, 1) 
    
if depth_sensor.supports(rs.option.laser_power):
    depth_sensor.set_option(rs.option.laser_power, 360)
# -------------------------------------------------
detector = ArucoDetector()

try:
    while True:
        frames = pipeline.wait_for_frames()
        ir_frame = frames.get_infrared_frame(1) # Get Left IR
        
        if not ir_frame:
            continue
            
        # Convert to numpy array (It is already grayscale!)
        ir_image = np.asanyarray(ir_frame.get_data())
        
        # Run ArUco detection directly on this ir_image
        # It will be sharp and blur-free!
        ids, centers = detector.detect_markers(ir_image)

        cv2.imshow("IR Stream (Global Shutter)", ir_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()