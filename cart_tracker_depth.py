import pyrealsense2 as rs
import numpy as np
import cv2


"""attempt to solve depth inaccuracy issue by adding depth inference using IR depth map,
Still doesnt work, depth in transform coord system has a persistent offset
Need to inquire more about the problem"""

class CartTrackerPnP:
    def __init__(self, anchor_ids=[3, 2, 4, 1], cart_id=0, use_corners=False):
        self.anchor_ids = anchor_ids
        self.cart_id = cart_id
        
        # TOGGLE: Set to False to ignore marker sizes and use pure center points
        self.use_corners = use_corners
        
        self.cart_height = -6.125 #-6.75  
        
        self.alpha = 0.01
        # This is completely ignored if use_corners is False.
        # Keep it here for the day you standardize your printed markers.
        # self.marker_size = 5.75  
        
        # --- NEW: INFRARED HEIGHT INFERENCE EMA ---
        self.depth_alpha = 0.01             # Temporal smoothing for height
        self.depth_shrink_factor = 0.5    # Shrink poly to avoid black ink edge noise
        self.rs_depth_scale = 0.001         # Default RealSense mm to meters scale
        self.ema_cart_height_world = None   # Rolling average of True World Z-Height
        
        # The exact, measured 3D World Coordinates of your floor anchors
        x_length = 108
        y_length = 48
        self.anchor_map = {
            1: np.array([0.0, 0.0, 0.0]),       # Top-Left
            3: np.array([x_length, 0.0, 0.0]),     # Top-Right
            4: np.array([0.0, y_length, 0.0]),        # Bottom-Left
            2: np.array([x_length, y_length, 0.0]),       # Bottom-Right
            5: np.array([x_length/3, y_length/2, 0.0]),    
            6: np.array([2*x_length/3, y_length/2, 0.0])
        }

        # State Machine Cache: {anchor_id: 2D_pixel_data}
        self.cached_anchors = {}

    def _get_marker_corners_3d(self, center, size):
        """Generates 3D coordinates for 4 corners. Only called if use_corners=True."""
        cx, cy, cz = center
        hs = size / 2.0
        return np.array([
            [cx - hs, cy + hs, cz],
            [cx + hs, cy + hs, cz],
            [cx + hs, cy - hs, cz],
            [cx - hs, cy - hs, cz]
        ], dtype=np.float32)
        
    def _deproject_pixel_to_point(self, K, rvec, tvec, intrin, pixel_pos, depth):
        
        """deprojection using camera pose and intrinsics, given a fixed height"""
                # 6. Extract the Camera Matrix
        R, _ = cv2.Rodrigues(rvec)
        R_inv = R.T
        C_cam = -R_inv @ tvec
        C_cam = C_cam.flatten()

        # 7. Locate the Car
        cx, cy = pixel_pos
            
        # Deproject pixel back into the real world
        
        pixel_homo = np.array([cx, cy, 1.0], dtype=np.float32)
        K_inv = np.linalg.inv(K)
        # ray_cam = K_inv @ pixel_homo
        
        # rs2_deproject_pixel_to_point automatically applies K_inv AND reverses lens distortion.
        # Passing a depth of 1.0 gives you the exact normalized ray vector in camera space.
        ray_cam = np.array(rs.rs2_deproject_pixel_to_point(intrin, [cx, cy], 1.0), dtype=np.float32)
        
        ray_world = R_inv @ ray_cam
        ray_world = ray_world / np.linalg.norm(ray_world)
        
        if abs(ray_world[2]) < 1e-6:
            return None
            
        # Intersect with the Z = 6 plane
        t_intersect = (depth - C_cam[2]) / ray_world[2]
        
        car_x = C_cam[0] + t_intersect * ray_world[0]
        car_y = C_cam[1] + t_intersect * ray_world[1]
        
        return (car_x, car_y, depth)
    def _deproject_pixel_to_point_batch(self, K, rvec, tvec, intrin, pixel_pos, depth):
        """
        Batch deprojection using camera pose and intrinsics, given a fixed height.
        pixel_pos can be a single (2,) point or an (N, 2) array of points.
        """
        # Force input into an (N, 2) array for unified batch processing
        pixel_pos = np.atleast_2d(pixel_pos)
        N = pixel_pos.shape[0]

        # 1. Extract the Camera Matrix (Only computed once per batch)
        R, _ = cv2.Rodrigues(rvec)
        R_inv = R.T
        C_cam = -R_inv @ tvec
        C_cam = C_cam.flatten() # Shape: (3,)

        # 2. Get Camera Rays
        # Note: The RealSense SDK does not natively vectorize this function.
        # We use a list comprehension here, but format it to (3, N) to vectorize everything else.
        rays_cam = [rs.rs2_deproject_pixel_to_point(intrin, [p[0], p[1]], 1.0) for p in pixel_pos]
        rays_cam = np.array(rays_cam, dtype=np.float32).T 

        # 3. Transform to World Coordinates
        rays_world = R_inv @ rays_cam  # Matrix math: (3, 3) @ (3, N) -> (3, N)
        
        # Normalize all rays simultaneously along the column axis
        rays_world = rays_world / np.linalg.norm(rays_world, axis=0)

        # 4. Intersect with the Z plane
        # Pre-allocate output array with NaNs to cleanly flag invalid intersections
        results = np.full((N, 3), np.nan)
        
        # Mask out any rays running parallel to the ground (prevents division by zero)
        valid_mask = np.abs(rays_world[2, :]) >= 1e-6
        
        if np.any(valid_mask):
            # Fully vectorized intersection math
            t_intersect = (depth - C_cam[2]) / rays_world[2, valid_mask]
            
            results[valid_mask, 0] = C_cam[0] + t_intersect * rays_world[0, valid_mask]
            results[valid_mask, 1] = C_cam[1] + t_intersect * rays_world[1, valid_mask]
            results[valid_mask, 2] = depth

        # 5. Return Formatting
        # Preserve backward compatibility: return a tuple if passed a single point
        if N == 1:
            if np.isnan(results[0, 0]):
                return None
            return tuple(results[0])
        
        # Otherwise, return the full (N, 3) array
        return results
    
    """New added helper functions for height active processing"""
    def _get_shrunk_polygon(self, corners):
        """Pulls the 4 corners inward to avoid sampling background floor noise."""
        corners = np.array(corners, dtype=np.float32)
        centroid = np.mean(corners, axis=0)
        shrunk_corners = centroid + (corners - centroid) * (1.0 - self.depth_shrink_factor)
        return shrunk_corners.astype(np.int32)

    def _get_spatial_median(self, depth_array, corners):
        """Vectorized extraction of valid IR depth pixels inside the marker."""
        shrunk_corners = self._get_shrunk_polygon(corners)
        mask = np.zeros(depth_array.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [shrunk_corners], 255)
        
        extracted_depths = depth_array[mask == 255]
        valid_depths = extracted_depths[extracted_depths > 0]
        
        if len(valid_depths) == 0:
            return None
        return np.median(valid_depths)

    def _infer_world_height(self, depth_frame, cart_corners, cart_center, intrin, rvec, tvec):
        """
        Calculates pure Z-height in the World Coordinate Frame.
        Extracts IR median -> Deprojects to Camera 3D -> Transforms to World 3D -> EMAs World Z.
        """
        depth_array = np.asanyarray(depth_frame.get_data())
        raw_median = self._get_spatial_median(depth_array, cart_corners)
        
        if raw_median is None:
            return self.ema_cart_height_world
            
        # 1. Scale RealSense raw units to Inches (to match your PnP coordinate space)
        # 1 mm = 0.0393701 inches
        dist_inches = float(raw_median * self.rs_depth_scale * 39.3701)

        # 2. Deproject to Local Camera 3D Space (X_cam, Y_cam, Z_cam)
        cx, cy = cart_center
        # Because we pass depth in inches, the output pt_cam is natively in inches
        pt_cam = np.array(
            rs.rs2_deproject_pixel_to_point(intrin, [int(cx), int(cy)], dist_inches), 
            dtype=np.float32
        ).reshape(3, 1)

        # 3. Transform from Camera Space to World Space
        R, _ = cv2.Rodrigues(rvec)
        R_inv = R.T
        
        # Equation: P_world = R_inv * (P_cam - tvec)
        pt_world = R_inv @ (pt_cam - tvec)
        
        # 4. Extract purely the World Z component (Height off the floor)
        current_z_world = float(pt_world[2][0])

        # 5. Apply Temporal EMA Filter
        if self.ema_cart_height_world is None:
            self.ema_cart_height_world = current_z_world
        else:
            self.ema_cart_height_world = (self.depth_alpha * current_z_world) + ((1.0 - self.depth_alpha) * self.ema_cart_height_world)
            
        return self.ema_cart_height_world
    
    
    """
    
    Main update function
    new: added active height inference using IR depth map with ema filter
    
    """

    def update(self, ids, centers, corners, depth_frame, intrin):
        
        """add part to also process angle"""

        # 1. Build Camera Matrix
        K = np.array([
            [intrin.fx, 0, intrin.ppx],
            [0, intrin.fy, intrin.ppy],
            [0, 0, 1]
        ], dtype=np.float32)
        dist_coeffs = np.array(intrin.coeffs, dtype=np.float32)

        # Flatten arrays safely
        ids_list = ids.flatten().tolist() if ids is not None and len(ids) > 0 else []

        # 2. Update the Anchor Cache seamlessly
        # for i, m_id in enumerate(ids_list):
        #     if m_id in self.anchor_ids and m_id in self.anchor_map:
        #         if self.use_corners:
        #             # Cache the 4x2 array of corners
        #             self.cached_anchors[m_id] = np.ascontiguousarray(corners[i], dtype=np.float32).reshape(4, 2)
        #         else:
        #             # Cache just the single (x, y) center coordinate
        #             self.cached_anchors[m_id] = np.array(centers[i], dtype=np.float32)
        # 2. Update the Anchor Cache seamlessly with EMA
        for i, m_id in enumerate(ids_list):
            if m_id in self.anchor_ids and m_id in self.anchor_map:
                
                # Extract the raw current measurement
                if self.use_corners:
                    current_meas = np.ascontiguousarray(corners[i], dtype=np.float32).reshape(4, 2)
                else:
                    current_meas = np.array(centers[i], dtype=np.float32)

                # Apply EMA filter if anchor exists, otherwise initialize
                if m_id in self.cached_anchors:
                    self.cached_anchors[m_id] = (self.alpha * current_meas) + ((1.0 - self.alpha) * self.cached_anchors[m_id])
                else:
                    self.cached_anchors[m_id] = current_meas

        # 3. Enforce the Calibration Rule
        # We MUST have all 4 anchors cached at least once to solve PnP securely
        if len(self.cached_anchors) < len(self.anchor_ids):
            return f"CALIBRATING... ({len(self.cached_anchors)}/{len(self.anchor_ids)} anchors found)", None, None

        # 4. Assemble the PnP Data cleanly from the Cache
        obj_points = []
        img_points = []
        
        for m_id, img_data in self.cached_anchors.items():
            if self.use_corners:
                obj_pts_3d = self._get_marker_corners_3d(self.anchor_map[m_id], self.marker_size)
                obj_points.extend(obj_pts_3d)
                img_points.extend(img_data)
            else:
                obj_points.append(self.anchor_map[m_id])
                img_points.append(img_data)

        obj_points = np.array(obj_points, dtype=np.float32)
        img_points = np.array(img_points, dtype=np.float32)

        # 5. Calculate Camera Pose
        success, rvec, tvec = cv2.solvePnP(
            obj_points, img_points, K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE
        )
        
        if not success:
            return "PNP FAILED", None, None
        
        # get pixel pos and deproject using helper function
        if self.cart_id in ids_list:
            cart_idx = ids_list.index(self.cart_id)
            cx, cy = centers[cart_idx]
            cart_corners = corners[cart_idx]
            
            # ---New:  EXECUTE HEIGHT INFERENCE ---
            dynamic_cart_height = self._infer_world_height(
                depth_frame, cart_corners, (cx, cy), intrin, rvec, tvec
            )
            
            # Fallback if IR hardware returned entirely 0s
            active_height = dynamic_cart_height if dynamic_cart_height is not None else self.fallback_cart_height

            # Angle Calculation
            corner1, corner2, corner3, corner4 = cart_corners[0], cart_corners[1], cart_corners[2], cart_corners[3]
            cart_corners_batch = np.array([corner1, corner2, corner3, corner4])
            
            corner_pos = self._deproject_pixel_to_point_batch(K, rvec, tvec, intrin, cart_corners_batch, active_height)
            vec = corner_pos[0] - corner_pos[1]
            angle_rad = np.arctan2(vec[1], vec[0])
            angle_deg = np.degrees(angle_rad)
            
            # Final X/Y Raycast using the dynamic IR Z-plane
            car_pos = self._deproject_pixel_to_point(K, rvec, tvec, intrin, (cx, cy), active_height)
        
            
            
            if car_pos is not None and angle_deg is not None:
                return "PNP TRACKING", car_pos, angle_deg
            else:
                return "FAILED", None, None
        

            
        return "SEARCHING FOR CAR", None, None
   


