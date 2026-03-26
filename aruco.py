import cv2
import cv2.aruco as aruco
import numpy as np
import time

def vector_angle(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
        return 0
    dot_product = np.dot(v1, v2)
    cos_angle = dot_product / (norm_v1 * norm_v2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)

def detect_marker_in_roi(full_image, predicted_region, factor=1.5):
    h = full_image.shape[0]
    w = full_image.shape[1]
    x1_pred, y1_pred, x2_pred, y2_pred = predicted_region
    
    box_w = x2_pred - x1_pred
    box_h = y2_pred - y1_pred
    
    # Expand the box
    x1 = int(max(0, x1_pred - box_w * (factor - 1) / 2))
    y1 = int(max(0, y1_pred - box_h * (factor - 1) / 2))
    x2 = int(min(w, x2_pred + box_w * (factor - 1) / 2))
    y2 = int(min(h, y2_pred + box_h * (factor - 1) / 2))
    
    if x2 <= x1 or y2 <= y1:
        return None, 0, 0
    
    roi_img = full_image[y1:y2, x1:x2]
    if roi_img.size == 0:
        return None, 0, 0
        
    return roi_img, x1, y1

class ArucoDetector():
    def __init__(self, dictionary=aruco.DICT_4X4_50):
        self.aruco_dict = aruco.getPredefinedDictionary(dictionary)
        self.parameters = aruco.DetectorParameters()
        self.parameters.minMarkerPerimeterRate = 0.05

        
        # Optimized parameters for ROI
        self.ROI_parameters = aruco.DetectorParameters()
        self.ROI_parameters.minMarkerPerimeterRate = 0.005
        self.ROI_parameters.adaptiveThreshWinSizeMin = 3
        self.ROI_parameters.adaptiveThreshWinSizeMax = 20
        self.ROI_parameters.adaptiveThreshWinSizeStep = 1
        self.ROI_parameters.perspectiveRemovePixelPerCell = 4
        self.ROI_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX 
        
        
        """ to deal with motion blur"""
        self.parameters.polygonalApproxAccuracyRate = 0.1 #0.03
        
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)
        self.ROI_detector = aruco.ArucoDetector(self.aruco_dict, self.ROI_parameters)
        
        self.prev_ids = None
        self.prev_centers = None
        self.prev_region = None 
        
        # --- CONFIG: ONLY TRACK THESE IDS ---
        self.TARGET_IDS = [0, 1] 

    def detect_markers(self, image):
        current_ids = []
        current_corners = []
        found_targets = set() # To track if we found 0 AND 1
        
        # --- PHASE 1: ROI Search ---
        # Try to find markers using history
        if self.prev_region is not None:
            for region in self.prev_region:
                roi_image, offset_x, offset_y = detect_marker_in_roi(image, region, factor=3.0)
                
                if roi_image is None: continue
                
                roi_corners, roi_ids, _ = self.ROI_detector.detectMarkers(roi_image)
                
                if roi_ids is not None:
                    for i in range(len(roi_ids)):
                        mid = roi_ids[i][0]
                        # FILTER: Only accept Target IDs
                        if mid in self.TARGET_IDS:
                            # Offset back to global
                            c = roi_corners[i]
                            c += np.array([offset_x, offset_y])
                            
                            current_corners.append(c)
                            current_ids.append(mid)
                            found_targets.add(mid)

        # --- PHASE 2: Check Completeness & Full Scan ---
        # Logic: If we didn't find BOTH 0 and 1 in ROI, force a full scan
        # to check if the missing one is elsewhere in the image.
        missing_targets = False
        for target in self.TARGET_IDS:
            if target not in found_targets:
                missing_targets = True
                print(time.time(),"Missing target: " + str(target))
                break
        
        if missing_targets:
            # ROI failed to find everyone. Wipe results and do a Full Scan.
            # (We wipe to avoid duplicates if Full Scan finds the same ones again)
            current_ids = []
            current_corners = []
            found_targets = set()
            
            full_corners, full_ids, _ = self.detector.detectMarkers(image)
            
            if full_ids is not None:
                for i in range(len(full_ids)):
                    mid = full_ids[i][0]
                    # FILTER: Only accept Target IDs
                    if mid in self.TARGET_IDS:
                        current_corners.append(full_corners[i])
                        current_ids.append(mid)

        # --- PHASE 3: Process & Return ---
        if len(current_ids) > 0:
            centers = []
            regions = []
            final_ids = []
            
            for i in range(len(current_ids)):
                marker_id = current_ids[i]
                marker_corners = current_corners[i][0]
                
                # -- Geometric Filters (Angle/Shape) --
                top_left = marker_corners[0]
                top_right = marker_corners[1]
                bottom_right = marker_corners[2]
                bottom_left = marker_corners[3]
                
                # vector1 = top_right - top_left
                # vector2 = bottom_left - top_left
                
                # angle = vector_angle(vector1, vector2)
                # if angle > np.radians(170) or angle < np.radians(10): continue 
                
                # len1 = np.linalg.norm(vector1)
                # len2 = np.linalg.norm(vector2)
                # if len1 == 0 or len2 == 0: continue
                
                # ratio = len1 / len2
                # if ratio > 3.0 or ratio < 0.33: continue 
                
                # if len1 < 10 or len2 < 10: continue 

                # -- Save Valid Marker --
                center_x = int(np.mean(marker_corners[:, 0]))
                center_y = int(np.mean(marker_corners[:, 1]))
                centers.append((center_x, center_y))
                final_ids.append(marker_id)
                
                # Calculate Box for next ROI
                min_x = int(np.min(marker_corners[:, 0]))
                max_x = int(np.max(marker_corners[:, 0]))
                min_y = int(np.min(marker_corners[:, 1]))
                max_y = int(np.max(marker_corners[:, 1]))
                regions.append((min_x, min_y, max_x, max_y))

                # Draw
                cv2.polylines(image, [marker_corners.astype(int)], True, (0, 255, 0), 2)
                cv2.circle(image, (center_x, center_y), 4, (0, 0, 255), -1)
                cv2.putText(image, f"ID {marker_id}", (min_x, min_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                

            if len(final_ids) > 0:
                self.prev_ids = final_ids
                self.prev_centers = centers
                self.prev_region = regions
                cv2.imshow("Detected Markers", image)
                return np.array(final_ids), centers
            else:
                self.prev_ids = None; self.prev_region = None
                return None, None
        else:
            self.prev_ids = None; self.prev_region = None
            return None, None