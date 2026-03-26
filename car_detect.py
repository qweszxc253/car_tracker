import cv2
import cv2.aruco as aruco
import numpy as np
import time
import pyrealsense2 as rs
mean = np.array([[ 11.84773451],
       [189.57628242],
       [202.92571685]])
sd= np.array([[ 1.98283479],
       [29.94064935],
       [16.40455596]])
sd_mod = np.array([[ 1.98283479],
       [29.94064935*2.3],
       [16.40455596*2.3]])
def find_orange_lidar(color_image):
    # 1. Convert to HSV
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    
    # 246 534
    # 2. Define "Orange" range (OpenCV Hue is 0-179)
    # Orange is typically Hue 10-25. 
    # You might need to tune these values!
    lower_orange = mean - 2*sd_mod
    upper_orange = mean + 2*sd_mod

    # 3. Create a Mask (Binary Image)
    # White pixels = Orange, Black = Everything else
    mask = cv2.inRange(hsv, lower_orange, upper_orange)
    cv2.imshow("Orange Mask", mask)


    # 4. Clean up noise (Morphological Operations)
    # This removes tiny accidental orange specs
    kernel = np.ones((3, 3), np.uint8)

    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    kernel2 = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel2, iterations=1)


    # 5. Find Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour (assuming the car is the biggest orange thing)
        c = max(contours, key=cv2.contourArea)
        
        # Get the Minimum Enclosing Circle
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        # Optional: Filter by size to avoid false positives
        if True: 
            # Draw for visualization
            cv2.circle(color_image, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(color_image, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.imshow("Orange Car Detection", color_image)
            
            return (int(x), int(y)), radius

    return None, None