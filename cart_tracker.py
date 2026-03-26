import pyrealsense2 as rs
import numpy as np
import cv2

class CartTracker:
    def __init__(self, ref_ids=[3,4,2], cart_id=0):
        self.ref_ids = ref_ids   # Floor Anchor
        self.ref_pos = [None, None, None]
        self.scale = [108, 49.5] # x, y scale in inches
        self.cart_id = cart_id # Car
        
        # Calibration Config
        self.calib_free = True  
        
        """Just use 2d mapping without considering height, see how accurate"""
        self.is_calibrated = False
        self.calibration_buffer = [] 
        self.required_calib_frames = 20
        
        # Coordinate Transform (World -> Reference)
        self.world_to_ref_transform = np.eye(4)

    def update(self, ids, centers, corners, depth_frame, depth_intrin):
        """
        Main logic loop.
        """
        ids_list = list(ids) if ids is not None else []
        if self.calib_free:
            if any(pos is None for pos in self.ref_pos):
                for i, ref_id in enumerate(self.ref_ids): # 'i' is the correct slot (0, 1, or 2)
                    if ref_id in ids_list:
                        idx = ids_list.index(ref_id) # 'idx' is where the camera saw it
                        self.ref_pos[i] = np.array(centers[idx]) # Assign using 'i', NOT 'idx'
            if all(pos is not None for pos in self.ref_pos):
            
                # here we can calculate the car pos
                if self.cart_id in ids_list:
                    idx = ids_list.index(self.cart_id)
                    cx, cy = centers[idx]
                    car_pos = np.array([cx, cy])
                    # construct 2d coordinate space using three ref points, index 0 being the origin, 1 being x axis, 2 being y axis
                    
                    # Vector from origin to x-axis reference point
                    vec_x = self.ref_pos[1] - self.ref_pos[0]
                    # Vector from origin to y-axis reference point
                    vec_y = self.ref_pos[2] - self.ref_pos[0]
                    # calculate 2d transform
                    transform_matrix = np.array([vec_x, vec_y]).T # 2x2 matrix
                    # Invert the transform matrix to get world to reference
                    inv_transform = np.linalg.pinv(transform_matrix)

                    
                    # Calculate car position in reference frame
                    car_pos_in_ref = np.zeros(2)
                    car_pos_in_ref = inv_transform @ (car_pos - self.ref_pos[0])

                    car_pos_in_ref[0] *= self.scale[0]  # Scale X
                    
                    car_pos_in_ref[1] *= self.scale[1]  # Scale Y
                    return "CALIB-FREE MODE", (car_pos_in_ref[0], car_pos_in_ref[1], 0)
                    
        
        return "Failed", None       
                    
                    
            


        # # --- PHASE 1: CALIBRATION (ID 0) ---
        # if not self.is_calibrated:
        #     if self.ref_id in ids_list:
        #         idx = ids_list.index(self.ref_id)
        #         ref_corners = corners[idx] 
                
        #         # Collect points from a LARGE area around the marker
        #         points_3d = self._collect_marker_points(ref_corners, depth_frame, depth_intrin)
                
        #         if len(points_3d) > 10:
        #             self.calibration_buffer.extend(points_3d)
        #             print(f"[Calib] +{len(points_3d)} pts. Total: {len(self.calibration_buffer)}")
                
        #         # Check if we have enough data to solve the plane
        #         if len(self.calibration_buffer) > 400: # Increased threshold for robust fit
        #             self._calibrate_floor(ref_corners, depth_frame, depth_intrin)
        #             return "CALIBRATED (ID 0)", None
                
        #         return f"CALIBRATING... {len(self.calibration_buffer)} pts", None
        #     else:
        #         return "WAITING FOR FLOOR (ID 0)", None

        # # --- PHASE 2: TRACKING (ID 1) ---
        # else:
        #     if self.cart_id in ids_list:
        #         idx = ids_list.index(self.cart_id)
        #         cx, cy = centers[idx]
                
        #         # Use robust median distance for the car too
        #         dist = self._get_robust_distance(depth_frame, int(cx), int(cy))
                
        #         if dist > 0:
        #             # Deproject
        #             cam_pt = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], dist)
                    
        #             # Transform
        #             vec_homo = np.array([cam_pt[0], cam_pt[1], cam_pt[2], 1.0])
        #             world_pt = self.world_to_ref_transform @ vec_homo
                    
        #             return "TRACKING ID 1", (world_pt[0], world_pt[1], world_pt[2])
        #         else:
        #             return "ID 1 INVALID DEPTH", None
            
        #     return "SEARCHING FOR CAR (ID 1)", None

    def _collect_marker_points(self, corners, depth_frame, intrin):
        """
        Samples points from a region +/- 100% larger than the marker itself.
        Strictly masks out 0-depth pixels.
        """
        points = []
        
        # Get Bounding Box of the marker
        min_x, min_y = np.min(corners, axis=0)
        max_x, max_y = np.max(corners, axis=0)
        
        w = max_x - min_x
        h = max_y - min_y
        
        # --- EXPAND REGION by 100% (User Request) ---
        # We subtract 'w' from left and add 'w' to right (same for height)
        start_x = int(max(0, min_x - w))
        start_y = int(max(0, min_y - h))
        end_x   = int(min(640, max_x + w)) # Assuming 640 width, clamp strictly
        end_y   = int(min(480, max_y + h)) # Assuming 480 height, clamp strictly
        
        # Generate a sampling grid (e.g., 10x10 grid over the expanded area)
        # This is faster than iterating every pixel
        sample_xs = np.linspace(start_x, end_x - 1, num=15, dtype=int)
        sample_ys = np.linspace(start_y, end_y - 1, num=15, dtype=int)
        
        for x in sample_xs:
            for y in sample_ys:
                # 1. Get Depth (Meters)
                d = depth_frame.get_distance(int(x), int(y))
                
                # 2. MASK: Strictly ignore 0 or crazy far values
                if 0.1 < d < 4.0: 
                    # 3. Deproject
                    pt = rs.rs2_deproject_pixel_to_point(intrin, [x, y], d)
                    points.append(pt)
                    
        return points

    def _get_robust_distance(self, depth_frame, x, y):
        """ Median filter for the target object center """
        vals = []
        # Check 5x5 area
        for i in range(-2, 3):
            for j in range(-2, 3):
                # Bounds check
                if 0 <= x+i < 640 and 0 <= y+j < 480:
                    d = depth_frame.get_distance(x+i, y+j)
                    if d > 0: vals.append(d)
        
        if not vals: return 0
        return np.median(vals)

    def _calibrate_floor(self, last_corners, depth_frame, intrin):
        print("--- COMPUTING FLOOR PLANE ---")
        cloud = np.array(self.calibration_buffer)
        
        # 1. Origin: Centroid of collected points
        origin = np.mean(cloud, axis=0)
        
        # 2. Z-Axis: Plane Normal (RANSAC)
        z_axis = self._ransac_plane_normal(cloud)
        
        # 3. X-Axis: Derived from Marker Corners (Visual Direction)
        c0, c1 = last_corners[0], last_corners[1] # TL -> TR
        
        # Robust depth for corners
        d0 = self._get_robust_distance(depth_frame, int(c0[0]), int(c0[1]))
        d1 = self._get_robust_distance(depth_frame, int(c1[0]), int(c1[1]))
        
        if d0 == 0 or d1 == 0:
            # Fallback: Use the cloud's principal component if corners are bad
            print("Warning: Corners have 0 depth. Using PCA for X direction.")
            # PCA on cloud roughly aligns with spread if scanned appropriately, 
            # but usually it's better to just hold previous valid frame. 
            # For now, we assume X aligns with camera X roughly.
            x_raw = np.array([1, 0, 0])
        else:
            p0 = np.array(rs.rs2_deproject_pixel_to_point(intrin, c0, d0))
            p1 = np.array(rs.rs2_deproject_pixel_to_point(intrin, c1, d1))
            x_raw = p1 - p0

        # 4. Orthogonalize X against Z
        x_proj = x_raw - np.dot(x_raw, z_axis) * z_axis
        norm_x = np.linalg.norm(x_proj)
        if norm_x == 0: x_axis = np.array([1, 0, 0])
        else: x_axis = x_proj / norm_x
        
        # 5. Y-Axis
        y_axis = np.cross(z_axis, x_axis)
        
        # 6. Build Matrix
        rot = np.column_stack((x_axis, y_axis, z_axis))
        cam_to_ref = np.eye(4)
        cam_to_ref[:3, :3] = rot
        cam_to_ref[:3, 3] = origin
        
        self.world_to_ref_transform = np.linalg.inv(cam_to_ref)
        self.is_calibrated = True
        print("Done. Transform Matrix:\n", self.world_to_ref_transform)

    def _ransac_plane_normal(self, points, iter=100, thresh=0.02):
        best_inliers = -1
        best_normal = np.array([0, -1, 0]) # Default 'down/up'
        n = len(points)
        
        for _ in range(iter):
            idx = np.random.choice(n, 3, replace=False)
            p1, p2, p3 = points[idx]
            
            normal = np.cross(p2 - p1, p3 - p1)
            norm_mag = np.linalg.norm(normal)
            if norm_mag < 1e-6: continue
            normal /= norm_mag
            
            # Plane eq: dot(normal, x - p1) = 0
            # Dist = abs(dot(normal, points - p1))
            diffs = points - p1
            dists = np.abs(np.dot(diffs, normal))
            inliers = np.sum(dists < thresh)
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_normal = normal
                
        return best_normal
    

# class CartTrackerPnP:
#     def __init__(self, anchor_ids=[3, 4, 2, 1], cart_id=0):
#         self.anchor_ids = anchor_ids
#         self.cart_id = cart_id
        
#         # Established height of the car marker from the floor
#         self.cart_height = 6.0  
        
#         # The physical size of your printed markers (MUST match the units of your coordinates)
#         self.marker_size = 5.75  
        
#         # 3D World Coordinates for the Centers of the Floor Anchors
#         self.anchor_map = {
#             3: np.array([0.0, 0.0, 0.0]),         # Origin
#             2: np.array([108.0, 0.0, 0.0]),       # X-axis
#             4: np.array([0.0, 49.5, 0.0]),        # Y-axis
#             1: np.array([108.0, 49.5, 0.0])       # The new 4th marker
#         }

#     def _get_marker_corners_3d(self, center, size):
#         """Generates the 3D world coordinates for the 4 corners of a flat floor marker."""
#         cx, cy, cz = center
#         hs = size / 2.0
#         # Standard ArUco corner order: Top-Left, Top-Right, Bottom-Right, Bottom-Left
#         # Assuming markers are lying flat on the floor (Z is constant across corners)
#         return np.array([
#             [cx - hs, cy + hs, cz],
#             [cx + hs, cy + hs, cz],
#             [cx + hs, cy - hs, cz],
#             [cx - hs, cy - hs, cz]
#         ], dtype=np.float32)

#     def update(self, ids, centers, corners, depth_frame, intrin):
#         if ids is None or len(ids) == 0:
#             return "NO MARKERS", None
            
#         # Flatten IDs to a standard list to solve random indexing
#         ids_list = ids.flatten().tolist()
        
#         # 1. Build Camera Matrix from RealSense Intrinsics
#         K = np.array([
#             [intrin.fx, 0, intrin.ppx],
#             [0, intrin.fy, intrin.ppy],
#             [0, 0, 1]
#         ], dtype=np.float32)
#         dist_coeffs = np.array(intrin.coeffs, dtype=np.float32)

#         # 2. Match incoming random-order markers to World Coordinates
#         obj_points = []
#         img_points = []
        
#         for i, m_id in enumerate(ids_list):
#             if m_id in self.anchor_ids and m_id in self.anchor_map:
#                 # Get the 4 physical 3D corners of this specific marker
#                 obj_pts_3d = self._get_marker_corners_3d(self.anchor_map[m_id], self.marker_size)
#                 obj_points.extend(obj_pts_3d)
                
#                 # Get the corresponding 4 2D pixel corners from the camera
#                 img_pts_2d = np.ascontiguousarray(corners[i], dtype=np.float32).reshape(4, 2)
#                 img_points.extend(img_pts_2d)

#         # We need at least 4 points (1 full marker) to run PnP, but more is vastly better
#         if len(obj_points) < 4:
#             return "WAITING FOR FLOOR ANCHORS", None

#         obj_points = np.array(obj_points, dtype=np.float32)
#         img_points = np.array(img_points, dtype=np.float32)

#         # 3. Solve PnP for Global Camera Pose
#         # SOLVEPNP_ITERATIVE is robust when given many points (like our 16 corners)
#         success, rvec, tvec = cv2.solvePnP(
#             obj_points, img_points, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
#         )
        
#         if not success:
#             return "PNP FAILED", None

#         # 4. Extract Camera Position in the World
#         R, _ = cv2.Rodrigues(rvec)
#         R_inv = R.T
#         C_cam = -R_inv @ tvec
#         C_cam = C_cam.flatten()

#         # 5. Track the Car via Ray-Plane Intersection
#         if self.cart_id in ids_list:
#             cart_idx = ids_list.index(self.cart_id)
#             cx, cy = centers[cart_idx] 
            
#             # Convert 2D car pixel to a 3D ray in camera space
#             pixel_homo = np.array([cx, cy, 1.0], dtype=np.float32)
#             K_inv = np.linalg.inv(K)
#             ray_cam = K_inv @ pixel_homo
            
#             # Rotate the ray into the world coordinate space
#             ray_world = R_inv @ ray_cam
#             ray_world = ray_world / np.linalg.norm(ray_world)
            
#             # Prevent division by zero if looking perfectly parallel to the floor
#             if abs(ray_world[2]) < 1e-6:
#                 return "FAILED: Ray parallel to floor", None
                
#             # Intersect ray with the plane Z = 6
#             t_intersect = (self.cart_height - C_cam[2]) / ray_world[2]
            
#             # Calculate final X, Y coordinates
#             car_x = C_cam[0] + t_intersect * ray_world[0]
#             car_y = C_cam[1] + t_intersect * ray_world[1]
            
#             return "PNP TRACKING", (car_x, car_y, self.cart_height)
            
#         return "SEARCHING FOR CAR", None



class CartTrackerPnP:
    def __init__(self, anchor_ids=[3, 2, 4, 1], cart_id=0, use_corners=False):
        self.anchor_ids = anchor_ids
        self.cart_id = cart_id
        
        # TOGGLE: Set to False to ignore marker sizes and use pure center points
        self.use_corners = use_corners
        
        self.cart_height = -6.75  
        
        # This is completely ignored if use_corners is False.
        # Keep it here for the day you standardize your printed markers.
        self.marker_size = 5.75  
        
        # The exact, measured 3D World Coordinates of your floor anchors
        x_length = 103.75
        y_length = 45.5
        self.anchor_map = {
            3: np.array([0.0, 0.0, 0.0]),       # Top-Left
            2: np.array([x_length, 0.0, 0.0]),     # Top-Right
            4: np.array([0.0, y_length, 0.0]),        # Bottom-Left
            1: np.array([x_length, y_length, 0.0])       # Bottom-Right
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

    def update(self, ids, centers, corners, depth_frame, intrin):
        
        """add part to also process angle"""
        # cart_id = self.cart_id
        # cart_angle = 0  # Initialize cart angle (to be updated with actual angle data)
        # cart_index = ids.flatten().tolist().index(cart_id) if ids is not None and cart_id in ids.flatten() else -1
        # if cart_index != -1:
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
        for i, m_id in enumerate(ids_list):
            if m_id in self.anchor_ids and m_id in self.anchor_map:
                if self.use_corners:
                    # Cache the 4x2 array of corners
                    self.cached_anchors[m_id] = np.ascontiguousarray(corners[i], dtype=np.float32).reshape(4, 2)
                else:
                    # Cache just the single (x, y) center coordinate
                    self.cached_anchors[m_id] = np.array(centers[i], dtype=np.float32)

        # 3. Enforce the Calibration Rule
        # We MUST have all 4 anchors cached at least once to solve PnP securely
        if len(self.cached_anchors) < len(self.anchor_ids):
            return f"CALIBRATING... ({len(self.cached_anchors)}/{len(self.anchor_ids)} anchors found)", None

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
            obj_points, img_points, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return "PNP FAILED", None

        # 6. Extract the Camera Matrix
        R, _ = cv2.Rodrigues(rvec)
        R_inv = R.T
        C_cam = -R_inv @ tvec
        C_cam = C_cam.flatten()

        # 7. Locate the Car
        if self.cart_id in ids_list:
            cart_idx = ids_list.index(self.cart_id)
            cx, cy = centers[cart_idx]
            
            # Deproject pixel back into the real world
            
            pixel_homo = np.array([cx, cy, 1.0], dtype=np.float32)
            K_inv = np.linalg.inv(K)
            ray_cam = K_inv @ pixel_homo
            
            # rs2_deproject_pixel_to_point automatically applies K_inv AND reverses lens distortion.
            # Passing a depth of 1.0 gives you the exact normalized ray vector in camera space.
            ray_cam = np.array(rs.rs2_deproject_pixel_to_point(intrin, [cx, cy], 1.0), dtype=np.float32)
            
            ray_world = R_inv @ ray_cam
            ray_world = ray_world / np.linalg.norm(ray_world)
            
            if abs(ray_world[2]) < 1e-6:
                return "FAILED: Ray parallel to floor", None
                
            # Intersect with the Z = 6 plane
            t_intersect = (self.cart_height - C_cam[2]) / ray_world[2]
            
            car_x = C_cam[0] + t_intersect * ray_world[0]
            car_y = C_cam[1] + t_intersect * ray_world[1]
            
            return "PNP TRACKING", (car_x, car_y, self.cart_height)
            
        return "SEARCHING FOR CAR", None