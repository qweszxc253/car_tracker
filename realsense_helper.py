import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt


"""contain code for 3d visualization of depth map"""
class RealSenseManager:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # EMA temporal average param
        self.alpha = 0.01
        
        # Enable depth and infrared streams
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.infrared, 1, width, height, rs.format.y8, fps)
        
        # Start streaming
        self.profile = self.pipeline.start(self.config)
        
        # Get intrinsics for deprojection
        depth_stream = self.profile.get_stream(rs.stream.depth)
        self.intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
        
        # Depth scale is needed to convert raw depth to meters
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        
        # Initialize the filter
        # Mode 0: Fill with value from left
        # Mode 1: Fill with farthest value from neighbor
        # Mode 2: Fill with nearest value from neighbor (Good for "expansion")
        self.hole_filling = rs.hole_filling_filter(1)
        
        # grid
        self.grid_u, self.grid_v = np.meshgrid(np.arange(width), np.arange(height))
        
        # temperal average of environment point cloud for background (maybe use exp decay)
        self.depth_hist_avg= None
        
        
        # # --- THE FILTER CHAIN ---
        # # 1. Decimation: Reduces resolution (e.g. by 2) to reduce noise and fill small holes naturally.
        # #    (Optional: Remove if you need full 640x480 resolution)
        # self.decimation = rs.decimation_filter() 
        # self.decimation.set_option(rs.option.filter_magnitude, 2)

        # # 2. Spatial Filter: The heavy lifter. Smooths data and fills holes with interpolation.
        # self.spatial = rs.spatial_filter()
        # self.spatial.set_option(rs.option.filter_magnitude, 2)
        # self.spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
        # self.spatial.set_option(rs.option.filter_smooth_delta, 20)
        # # 0=Disabled, 1=2 pixel radius, 2=4 pixel radius, ... 5=Unlimited
        # self.spatial.set_option(rs.option.holes_fill, 3) 

        # # 3. Temporal Filter: Uses past frames to fill missing data. 
        # #    CRITICAL for your static ceiling camera.
        # self.temporal = rs.temporal_filter()
        
        # # 4. Hole Filling: The final "Hail Mary" to plug any remaining single-pixel gaps.
        # self.hole_filling = rs.hole_filling_filter(1) # Mode 1: Farthest from around

    # def get_frames(self):
    #     """Returns synchronized infrared and depth frames."""
    #     frames = self.pipeline.wait_for_frames()
    #     infra_frame = frames.get_infrared_frame(1) # Left IR
    #     depth_frame = frames.get_depth_frame()
        
    #     if not infra_frame or not depth_frame:
    #         return None, None
    #     # # Apply hole filling filter to depth frame
    #     # filtered = self.spatial.process(depth_frame)
    #     # filtered = self.temporal.process(filtered)
    #     # filtered = self.hole_filling.process(filtered)
    #     # depth_frame = filtered
        
    #     """can be a seperate function for cumulative average"""
    #     # compute historical average
    #     if self.depth_hist_avg is None:
    #         self.depth_hist_avg = np.asanyarray(depth_frame.get_data())
    #     else:
    #         alpha = 0.1  # smoothing factor for exponential moving average
    #         self.depth_hist_avg = alpha * np.asanyarray(depth_frame.get_data()) + (1 - alpha) * self.depth_hist_avg
            
    #     return infra_frame, depth_frame
    
    # def get_frames(self):
    #     """Returns synchronized infrared and depth frames with Zero-Skipping EMA."""
    #     frames = self.pipeline.wait_for_frames()
    #     infra_frame = frames.get_infrared_frame(1) # Left IR
    #     depth_frame = frames.get_depth_frame()
        
    #     if not infra_frame or not depth_frame:
    #         return None, None
            
    #     # 1. Extract raw hardware data and cast to float32 for mathematical precision
    #     current_depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        
    #     if self.depth_hist_avg is None:
    #         self.depth_hist_avg = current_depth.copy()
    #     else:
    #         alpha = self.alpha  # EMA smoothing factor
            
    #         # 2. Build Boolean Masks (O(1) C-level execution)
    #         valid_mask = current_depth > 0
    #         uninitialized_mask = self.depth_hist_avg == 0
            
    #         # Condition A: Pixel was previously dead (0), but is now valid. 
    #         # Action: Hard overwrite. Do not average with a dead zero.
    #         init_mask = valid_mask & uninitialized_mask
    #         self.depth_hist_avg[init_mask] = current_depth[init_mask]
            
    #         # Condition B: Pixel is valid in both current frame and history.
    #         # Action: Apply EMA.
    #         ema_mask = valid_mask & ~uninitialized_mask
    #         self.depth_hist_avg[ema_mask] = (alpha * current_depth[ema_mask]) + ((1.0 - alpha) * self.depth_hist_avg[ema_mask])
            
    #         # Condition C: Pixel is currently 0. 
    #         # Action: Do nothing. The history matrix naturally retains its last known good value.
            
    #     return infra_frame, depth_frame
    def get_frames(self):
        """Returns synchronized infrared and depth frames with a Bandpass Zero-Skipping EMA."""
        frames = self.pipeline.wait_for_frames()
        infra_frame = frames.get_infrared_frame(1) # Left IR
        depth_frame = frames.get_depth_frame()
        
        if not infra_frame or not depth_frame:
            return None, None
            
        # 1. Extract raw hardware data and cast to float32 for mathematical precision
        current_depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        
        # 2. Define the absolute physical ceiling limit
        # 100 inches * 25.4 mm/inch = 2540 raw depth units
        MAX_DEPTH_UNITS = 2540
        
        if self.depth_hist_avg is None:
            self.depth_hist_avg = current_depth.copy()
        else:
            alpha = 0.1  # EMA smoothing factor
            
            # 3. Build the Bandpass Boolean Mask (O(1) C-level execution)
            # Strictly isolates pixels that are NOT dead (0) and NOT hardware hallucinations (>2540)
            valid_mask = (current_depth > 0) & (current_depth <= MAX_DEPTH_UNITS)
            uninitialized_mask = self.depth_hist_avg == 0
            
            # Condition A: Pixel was previously dead, but is now within valid physical bounds. 
            init_mask = valid_mask & uninitialized_mask
            self.depth_hist_avg[init_mask] = current_depth[init_mask]
            
            # Condition B: Pixel is within physical bounds in both current frame and history.
            ema_mask = valid_mask & ~uninitialized_mask
            self.depth_hist_avg[ema_mask] = (alpha * current_depth[ema_mask]) + ((1.0 - alpha) * self.depth_hist_avg[ema_mask])
            
            # Condition C: Pixel is currently 0 OR > 100 inches. 
            # Action: Do nothing. The matrix ignores the anomaly and retains the last physically valid coordinate.
            
        return infra_frame, depth_frame

    def pixel_to_3d(self, depth_frame, u, v):
        """Converts a 2D pixel (u,v) to 3D coordinates (x,y,z) in meters."""
        # Get depth at specific pixel
        depth_frame = depth_frame.as_depth_frame()  # Ensure it's a depth frame
        depth = depth_frame.get_distance(u, v)
        if depth == 0:  # Invalid depth
            return None
            
        # Deproject pixel to 3D point
        point_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], depth)
        return point_3d # returns [x, y, z]
    
    def get_point_cloud(self, depth_frame):
        """
        Converts depth map to a (N, 3) NumPy array of coordinates in meters.
        Explicitly removes all depth-0 singularities.
        

        """
        depth_frame = depth_frame.as_depth_frame()  # Ensure it's a depth frame
        depth_array = np.asanyarray(depth_frame.get_data())
        
        # Mask out 'singularities' (0 depth = no signal)
        mask = depth_array > 0  #dont need this if we have hole filling
        
        # Apply mask and convert units to meters
        z = depth_array[mask] * 0.001 
        u = self.grid_u[mask]
        v = self.grid_v[mask]
        
        # Pinhole Camera Model: X = (u-cx)*Z/fx, Y = (v-cy)*Z/fy
        x = (u - self.intrinsics.ppx) * z / self.intrinsics.fx
        y = (v - self.intrinsics.ppy) * z / self.intrinsics.fy
        
        # Resulting shape: (num_valid_points, 3)
        cloud = np.stack((x, y, z), axis=-1)
        return np.stack((x, y, z), axis=-1)
    # def get_point_cloud_avg(self):
        
    #     """
    #     Converts depth map to a (N, 3) NumPy array of coordinates in meters.
    #     Explicitly removes all depth-0 singularities.
        

    #     """
    #     # directly use running avg np array
    #     depth_array = self.depth_hist_avg
        
    #     mask = depth_array > 0  #dont need this if we have hole filling
        
    #     # Apply mask and convert units to meters
    #     z = depth_array[mask] * 0.001 
    #     u = self.grid_u[mask]
    #     v = self.grid_v[mask]
        
    #     # Pinhole Camera Model: X = (u-cx)*Z/fx, Y = (v-cy)*Z/fy
    #     x = (u - self.intrinsics.ppx) * z / self.intrinsics.fx
    #     y = (v - self.intrinsics.ppy) * z / self.intrinsics.fy
        
    #     # Resulting shape: (num_valid_points, 3)
        
    #     return np.stack((x, y, z), axis=-1)
    
    def get_point_cloud_avg(self):
        """Converts float32 depth map to a (N, 3) NumPy array of coordinates."""
        # Use the floating-point history buffer directly
        depth_array = self.depth_hist_avg
        
        # Mask out any pixels that are STILL strictly 0 (never initialized)
        mask = depth_array > 0  
        
        z = depth_array[mask] * 0.001  # Convert mm to meters
        u = self.grid_u[mask]
        v = self.grid_v[mask]
        
        x = (u - self.intrinsics.ppx) * z / self.intrinsics.fx
        y = (v - self.intrinsics.ppy) * z / self.intrinsics.fy
        
        return np.stack((x, y, z), axis=-1)

    def stop(self):
        self.pipeline.stop()
        
class PointCloudVisualizer:
    def __init__(self):
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(window_name="RealSense Live Stream", width=960, height=720)
        
        # Create a placeholder point cloud
        self.pcd = o3d.geometry.PointCloud()
        self.is_first_frame = True
        
        # Add a coordinate frame for reference (Red=X, Green=Y, Blue=Z)
        self.vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3))

    def update(self, points):
        if points is None or len(points) == 0: return

        # Update the points in the existing geometry
        self.pcd.points = o3d.utility.Vector3dVector(points)

        # Optional: Colorize by depth (Z) for better visibility
        # Colors: Blue (close) -> Red (far)
        z = points[:, 2]
        # Normalize Z roughly between 0m and 3m
        z_norm = np.clip(z / 3.0, 0, 1)
        colors = plt.get_cmap("jet")(z_norm)[:, :3]  # Requires matplotlib for colormap
        # OR simple grayscale if you don't want plt dependency:
        # colors = np.stack((z_norm, z_norm, z_norm), axis=-1) 
        
        self.pcd.colors = o3d.utility.Vector3dVector(colors)

        if self.is_first_frame:
            self.vis.add_geometry(self.pcd)
            # Reset view to look at the points properly
            self.vis.reset_view_point(True)
            self.is_first_frame = False
        else:
            # Tell Open3D the geometry data has changed
            self.vis.update_geometry(self.pcd)

        # Process GUI events and render
        self.vis.poll_events()
        self.vis.update_renderer()

    def close(self):
        self.vis.destroy_window()

# --- Main Logic / Usage Example ---
if __name__ == "__main__":
    camera = RealSenseManager()
    visualizer = PointCloudVisualizer()
    
    try:
        while True:
            ir_frame, d_frame = camera.get_frames()
            if ir_frame is None: continue

            # Convert IR frame to numpy for OpenCV (Aruco detection would happen here)
            ir_image = np.asanyarray(ir_frame.get_data())

            # Example: Deproject the center pixel of the IR image to 3D
            h, w = ir_image.shape
            center_3d = camera.pixel_to_3d(d_frame, w//2, h//2)

            # # generate 3d point cloud for visualization or further processing
            # point_cloud = camera.get_point_cloud(d_frame)

            # # visualize using open3d 
            # visualizer.update(point_cloud)

            # or alternatively, use running average
            pc = camera.get_point_cloud_avg()
            visualizer.update(pc)

            # cv2.imshow('IR Stream', ir_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.stop()
        cv2.destroyAllWindows()