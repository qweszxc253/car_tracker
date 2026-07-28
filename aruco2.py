import cv2
import cv2.aruco as aruco
import numpy as np
import time

def detect_marker_in_roi(full_image, predicted_region, factor=1.5):
    h, w = full_image.shape[:2]
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
    return roi_img, x1, y1

class ArucoDetector():
    def __init__(self, dictionary=aruco.DICT_4X4_50, target_ids=[0, 1]):
        # Dynamically detect OpenCV API Version (Pre-4.7 legacy vs 4.7+)
        self.is_legacy = not hasattr(aruco, 'ArucoDetector')

        # 1. Dictionary Fetching
        if hasattr(aruco, 'getPredefinedDictionary'):
            self.aruco_dict = aruco.getPredefinedDictionary(dictionary)
        else:
            self.aruco_dict = aruco.Dictionary_get(dictionary)

        # 2. Parameters Initialization
        if self.is_legacy:
            self.parameters = aruco.DetectorParameters_create()
            self.ROI_parameters = aruco.DetectorParameters_create()
        else:
            self.parameters = aruco.DetectorParameters()
            self.ROI_parameters = aruco.DetectorParameters()

        self.parameters.minMarkerPerimeterRate = 0.05
        self.parameters.polygonalApproxAccuracyRate = 0.05

        # Optimized parameters for ROI
        self.ROI_parameters.minMarkerPerimeterRate = 0.05
        self.ROI_parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        # 3. Detector Instantiation (Only for modern API)
        if not self.is_legacy:
            self.detector = aruco.ArucoDetector(self.aruco_dict, self.parameters)
            self.ROI_detector = aruco.ArucoDetector(self.aruco_dict, self.ROI_parameters)

        self.prev_ids = None
        self.prev_region = None
        self.TARGET_IDS = target_ids

    def _detect_markers_internal(self, image, is_roi=False):
        """Version-agnostic detection wrapper to handle legacy ROS2 OpenCV injections."""
        if self.is_legacy:
            params = self.ROI_parameters if is_roi else self.parameters
            return aruco.detectMarkers(image, self.aruco_dict, parameters=params)
        else:
            detector = self.ROI_detector if is_roi else self.detector
            return detector.detectMarkers(image)

    def detect_markers(self, image):
        """
        Returns: 
            ids (np.array), 
            centers (list of tuples), 
            corners (list of np.array 4x2)
        """
        current_ids = []
        current_corners = []
        found_targets = set() 
        
        # --- PHASE 1: ROI Search ---
        if self.prev_region is not None:
            for i, region in enumerate(self.prev_region):
                roi_image, offset_x, offset_y = detect_marker_in_roi(image, region, factor=2.5)
                
                if roi_image is None or roi_image.size == 0: continue
                
                roi_corners, roi_ids, _ = self._detect_markers_internal(roi_image, is_roi=True)
                
                if roi_ids is not None:
                    for k in range(len(roi_ids)):
                        mid = roi_ids[k][0]
                        if mid in self.TARGET_IDS:
                            # Transform corners back to global coordinates
                            c = roi_corners[k][0] + np.array([offset_x, offset_y])
                            current_corners.append(c)
                            current_ids.append(mid)
                            found_targets.add(mid)

        # --- PHASE 2: Check Completeness ---
        missing_targets = any(t not in found_targets for t in self.TARGET_IDS)
        
        if missing_targets:
            # Full Scan
            current_ids = []
            current_corners = []
            full_corners, full_ids, _ = self._detect_markers_internal(image, is_roi=False)
            
            if full_ids is not None:
                for i in range(len(full_ids)):
                    mid = full_ids[i][0]
                    if mid in self.TARGET_IDS:
                        current_corners.append(full_corners[i][0])
                        current_ids.append(mid)

        # --- PHASE 3: Process & Return ---
        if len(current_ids) > 0:
            centers = []
            regions = []
            final_ids = []
            final_corners = []
            
            for i in range(len(current_ids)):
                marker_id = current_ids[i]
                marker_corners = current_corners[i] # Shape (4, 2)
                
                # Center
                cx = int(np.mean(marker_corners[:, 0]))
                cy = int(np.mean(marker_corners[:, 1]))
                
                # ROI Box
                min_x = int(np.min(marker_corners[:, 0]))
                max_x = int(np.max(marker_corners[:, 0]))
                min_y = int(np.min(marker_corners[:, 1]))
                max_y = int(np.max(marker_corners[:, 1]))
                
                centers.append((cx, cy))
                regions.append((min_x, min_y, max_x, max_y))
                final_ids.append(marker_id)
                final_corners.append(marker_corners)

                # Draw
                cv2.polylines(image, [marker_corners.astype(int)], True, (0, 255, 0), 2)
                cv2.circle(image, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(image, f"ID {marker_id}", (min_x, min_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            self.prev_ids = final_ids
            self.prev_region = regions
            return np.array(final_ids), centers, final_corners
        else:
            self.prev_ids = None; self.prev_region = None
            return np.array([]), [], []
        
        
        
    def detect_markers_subpixel(self, image):
        """
        Returns: 
            ids (np.array), 
            centers (list of tuples), 
            corners (list of np.array 4x2)
        """
        current_ids = []
        current_corners = []
        found_targets = set() 
        
        # --- PHASE 1: ROI Search ---
        if self.prev_region is not None:
            for i, region in enumerate(self.prev_region):
                roi_image, offset_x, offset_y = detect_marker_in_roi(image, region, factor=2.5)
                
                if roi_image is None or roi_image.size == 0: continue
                
                roi_corners, roi_ids, _ = self._detect_markers_internal(roi_image, is_roi=True)
                
                if roi_ids is not None:
                    for k in range(len(roi_ids)):
                        mid = roi_ids[k][0]
                        if mid in self.TARGET_IDS:
                            # Transform corners back to global coordinates
                            c = roi_corners[k][0] + np.array([offset_x, offset_y])
                            current_corners.append(c)
                            current_ids.append(mid)
                            found_targets.add(mid)

        # --- PHASE 2: Check Completeness ---
        missing_targets = any(t not in found_targets for t in self.TARGET_IDS)
        
        if missing_targets:
            # Full Scan
            current_ids = []
            current_corners = []
            full_corners, full_ids, _ = self._detect_markers_internal(image, is_roi=False)
            
            if full_ids is not None:
                for i in range(len(full_ids)):
                    mid = full_ids[i][0]
                    if mid in self.TARGET_IDS:
                        current_corners.append(full_corners[i][0])
                        current_ids.append(mid)

        # --- PHASE 3: Process & Return ---
        if len(current_ids) > 0:
            centers = []
            regions = []
            final_ids = []
            final_corners = []
            
            for i in range(len(current_ids)):
                marker_id = current_ids[i]
                marker_corners = current_corners[i] # Shape (4, 2)
                
                # Center
                cx = np.mean(marker_corners[:, 0])
                cy = np.mean(marker_corners[:, 1])
                
                # ROI Box
                min_x = int(np.min(marker_corners[:, 0]))
                max_x = int(np.max(marker_corners[:, 0]))
                min_y = int(np.min(marker_corners[:, 1]))
                max_y = int(np.max(marker_corners[:, 1]))
                
                centers.append((cx, cy))
                regions.append((min_x, min_y, max_x, max_y))
                final_ids.append(marker_id)
                final_corners.append(marker_corners)

                # Draw
                cv2.polylines(image, [marker_corners.astype(int)], True, (0, 255, 0), 2)
                cv2.circle(image, (cx.astype(int), cy.astype(int)), 4, (0, 0, 255), -1)
                cv2.putText(image, f"ID {marker_id}", (min_x, min_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            self.prev_ids = final_ids
            self.prev_region = regions
            return np.array(final_ids), centers, final_corners
        else:
            self.prev_ids = None; self.prev_region = None
            return np.array([]), [], []