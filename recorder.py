import os
import time
import csv
import cv2

class SessionRecorder:
    def __init__(self, data_dir="recordings/data", video_dir="recordings/videos", flush_interval=60):
        self.data_dir = data_dir
        self.video_dir = video_dir
        self.flush_interval = flush_interval
        
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.video_dir, exist_ok=True)
        
        self.is_recording = False
        self.start_time_str = ""
        self.temp_csv_path = ""
        self.temp_video_path = ""
        
        self.csv_file = None
        self.csv_writer = None
        self.video_writer = None
        self.frames_since_flush = 0
        
        self.record_null_frames = True  # Always record a row, even if tracking fails

    def start(self, frame_shape, fps=30.0, session_id=None):
        if self.is_recording:
            return

        self.is_recording = True
        # Adopt the caller-supplied session ID (shared with the rosbag folder for pairing);
        # fall back to a local timestamp when recording standalone.
        self.start_time_str = session_id if session_id else time.strftime("%Y%m%d_%H%M%S")
        print(f"\n--- RECORDING STARTED: {self.start_time_str} ---")
        
        self.temp_csv_path = os.path.join(self.data_dir, "temp_data.csv")
        self.temp_video_path = os.path.join(self.video_dir, "temp_video.avi")
        
        # Initialize CSV
        self.csv_file = open(self.temp_csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp', 'x', 'y', 'z', 'angle_deg'])
        self.frames_since_flush = 0
        
        # Initialize Video Writer (dynamically adopting to stream resolution)
        height, width = frame_shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(self.temp_video_path, fourcc, fps, (width, height))

    # def record_frame(self, frame, position=None, angle=None):
    #     if not self.is_recording:
    #         return
            
    #     # Write Video
    #     if self.video_writer is not None:
    #         self.video_writer.write(frame)
            
    #     # Write Telemetry
    #     if self.csv_writer is not None and position is not None:
    #         current_time = time.time()
    #         x, y, z = position
    #         self.csv_writer.writerow([current_time, x, y, z, angle if angle is not None else ""])
            
    #         self.frames_since_flush += 1
    #         if self.frames_since_flush >= self.flush_interval:
    #             self.csv_file.flush()
    #             os.fsync(self.csv_file.fileno())
    #             self.frames_since_flush = 0
                
    def record_frame(self, frame, position=None, angle=None):
        if not self.is_recording:
            return
            
        # Write Video
        if self.video_writer is not None:
            self.video_writer.write(frame)
            
        # Write Telemetry (Always write a row, even if tracking failed)
        if self.csv_writer is not None:
            current_time = time.time()
            if position is not None:
                x, y, z = position
                self.csv_writer.writerow([current_time, x, y, z, angle])
            elif self.record_null_frames:
                # Log the timestamp, but leave telemetry blank
                self.csv_writer.writerow([current_time, "", "", "", ""])
            
            self.frames_since_flush += 1
            if self.frames_since_flush >= self.flush_interval:
                self.csv_file.flush()
                os.fsync(self.csv_file.fileno())
                self.frames_since_flush = 0

    def stop(self):
        if not self.is_recording:
            return
            
        self.is_recording = False
        end_time_str = time.strftime("%Y%m%d_%H%M%S")
        print(f"\n--- RECORDING STOPPED: {end_time_str} ---")
        
        self.cleanup()
        
        # Commit temporary files to final persistent storage
        final_csv_path = os.path.join(self.data_dir, f"data-{self.start_time_str}-{end_time_str}.csv")
        final_video_path = os.path.join(self.video_dir, f"video-{self.start_time_str}-{end_time_str}.avi")
        
        os.rename(self.temp_csv_path, final_csv_path)
        os.rename(self.temp_video_path, final_video_path)
        
        print(f"Exported: {final_csv_path}")
        print(f"Exported: {final_video_path}\n")

    def cleanup(self):
        """Failsafe method for closing buffers safely on crash or exit."""
        if self.csv_file is not None:
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None
            
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None