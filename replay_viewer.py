import rerun as rr
import rerun.blueprint as rrb
import pandas as pd
import cv2
import numpy as np

"""Class for replay recordings using the rerun sdk
usage:
    python replay_viewer.py --id 20260622_120000
    python replay_viewer.py --latest 3
    python replay_viewer.py --latest
    python replay_viewer.py --csv recordings/data/data-20260416_160923-20260416_160926.csv --vid recordings/videos/20260416_160923-20260416_160926.mp4
    
"""
class TelemetryViewer:
    def __init__(self, csv_path, video_path):
        self.csv_path = csv_path
        self.video_path = video_path

    def run(self):
        # Configure a default UI layout: 3D Map on the left, Video on the right
        # blueprint = rrb.Blueprint(
        #     rrb.Horizontal(
        #         rrb.Spatial3DView(origin="arena", name="Cart Telemetry"),
        #         rrb.Spatial2DView(origin="camera", name="RealSense Feed")
        #     )
        # )
        # Configure a UI layout: 3D Map, Video, and a vertical stack of Time-Series plots
        blueprint = rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="arena", name="Cart Telemetry"),
                rrb.Spatial2DView(origin="camera", name="RealSense Feed"),
                rrb.Vertical(
                    rrb.TimeSeriesView(origin="plots/x", name="X Position (in)"),
                    rrb.TimeSeriesView(origin="plots/y", name="Y Position (in)"),
                    rrb.TimeSeriesView(origin="plots/angle", name="Angle (deg)")
                )
            )
        )
        
        # Initialize Rerun and spawn the viewer application
        rr.init("f110_tracker", spawn=True, default_blueprint=blueprint)
        
        # Override the 1GB default (e.g., force a 4GB or 8GB limit)
        rr.spawn(memory_limit="4GB")

        # --- PHASE 1: Process and Log Telemetry Data ---
        # Inside your replay_viewer.py Phase 1:
        print(f"Loading telemetry from {self.csv_path}...")
        df = pd.read_csv(self.csv_path)
        
        if df.empty:
            print("Error: CSV is empty!")
            return

        # OPTIONAL: Interpolate missing data to smooth out ArUco tracking drops
        # limit=5 prevents it from blindly guessing if tracking is lost for too long
        df[['x', 'y', 'z', 'angle_deg']] = df[['x', 'y', 'z', 'angle_deg']].interpolate(method='linear', limit=5)

        start_time = df['timestamp'].iloc[0]

        for _, row in df.iterrows():
            # If the row is STILL NaN (meaning tracking was lost for a long time), skip logging it.
            # Rerun will automatically hold the cart at the last known valid position.
            if pd.isna(row['x']):
                continue
                
            t = row['timestamp'] - start_time
            rr.set_time("time", duration=t)
            # ... log to rerun ...
            
            # Log the cart as a 3D Box. 
            # Dimensions are approximated in inches to roughly match a 1/10 scale chassis.
            rr.log(
                "arena/cart",
                rr.Boxes3D(
                    half_sizes=[[9.0, 6.0, 3.5]], 
                    centers=[[row['x'], row['y'], row['z']-3.5]],
                    rotations=rr.RotationAxisAngle(axis=[0, 0, 1], degrees=row['angle_deg']),
                    colors=[[0, 255, 150]]
                )
            )

            # Log the coordinate point to draw a continuous trajectory trail
            rr.log(
                "arena/cart/trajectory",
                rr.Points3D([row['x'], row['y'], row['z']], colors=[255, 0, 0], radii=0.5)
            )
            
            """added logging for time serie plots"""
            rr.log("plots/x", rr.Scalars(row['x']))
            rr.log("plots/y", rr.Scalars(row['y']))
            rr.log("plots/angle", rr.Scalars(row['angle_deg']))

        # --- PHASE 2: Process and Log Video Frames ---
        print(f"Loading video from {self.video_path}...")
        cap = cv2.VideoCapture(self.video_path)
        
        # Pull FPS from the video metadata, fallback to 30.0 if missing
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 30.0  

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate absolute time for this frame based on a constant FPS.
            # Because both feeds start exactly when 'is_recording' flips to True, t=0 is synced.
            t = frame_idx * (1.0 / fps)
            rr.set_time("time", duration=t)
            
            # OpenCV uses BGR, Rerun expects standard RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            rr.log("camera/rgb", rr.Image(frame_rgb))
            frame_idx += 1

        cap.release()
        print("Data loaded. Interacting with Rerun viewer...")


def find_files_by_id(target_id):
        """Locates the paired CSV and AVI files using a substring match on the ID."""
        csv_pattern = os.path.join(DATA_DIR, f"*{target_id}*.csv")
        vid_pattern = os.path.join(VIDEO_DIR, f"*{target_id}*.avi")
        
        csv_files = glob.glob(csv_pattern)
        vid_files = glob.glob(vid_pattern)
        
        if not csv_files or not vid_files:
            print(f"Error: Could not find both CSV and Video matching ID: '{target_id}'")
            sys.exit(1)
            
        return csv_files[0], vid_files[0]

def find_latest_files(n=1):
        """Locates the N-th most recent completed recording."""
        # The timestamp naming convention (YYYYMMDD_HHMMSS) means alphabetical sort = chronological sort
        csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "data-*.csv")))
        
        # Filter out temp files safely
        csv_files = [f for f in csv_files if "temp" not in f]

        if not csv_files:
            print("Error: No recordings found in the data directory.")
            sys.exit(1)
            
        if len(csv_files) < n:
            print(f"Error: Requested latest #{n}, but only {len(csv_files)} recordings exist.")
            sys.exit(1)
            
        # Select the target CSV (index -1 is the most recent)
        target_csv = csv_files[-n]
        
        # Extract the exact ID string to enforce a strict match with the video file
        base_name = os.path.basename(target_csv)
        target_id = base_name.replace("data-", "").replace(".csv", "")
        
        return find_files_by_id(target_id)
if __name__ == "__main__":
    import argparse
    import os
    import glob
    import sys

    # Define standard directories
    DATA_DIR = "recordings/data"
    VIDEO_DIR = "recordings/videos"

    

    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Visualize cart telemetry and video via Rerun.")
    
    # Create a mutually exclusive group so the user picks exactly one mode of operation
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--id", type=str, help="Load by timestamp ID (e.g., 20260622_120000)")
    group.add_argument("--latest", type=int, nargs='?', const=1, help="Load the N-th most recent recording (default: 1)")
    group.add_argument("--csv", type=str, help="Direct path to CSV (requires --vid)")
    
    parser.add_argument("--vid", type=str, help="Direct path to Video (required if using --csv)")
    
    args = parser.parse_args()

    # --- Resolve File Paths ---
    csv_path = None
    vid_path = None

    if args.id:
        print(f"Searching for recording ID: {args.id}...")
        csv_path, vid_path = find_files_by_id(args.id)
    elif args.latest is not None:
        print(f"Searching for the #{args.latest} most recent recording...")
        csv_path, vid_path = find_latest_files(args.latest)
    elif args.csv:
        if not args.vid:
            print("Error: --vid is required when using --csv.")
            sys.exit(1)
        csv_path = args.csv
        vid_path = args.vid

    # --- Execution ---
    viewer = TelemetryViewer(csv_path, vid_path)
    viewer.run()