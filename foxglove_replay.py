#!/usr/bin/env python3
"""Replay a recorded rosbag in Foxglove Studio.

Unlike replay_viewer.py (which uses Rerun and only understands the custom CSV
format), this plays the *raw rosbag* -- /scan, /odom, /pc/aruco_pose, etc. --
over the Foxglove WebSocket bridge. Because Foxglove understands ROS 2 messages
natively, every standard topic visualizes with no per-message glue code.

Usage:
    python foxglove_replay.py --latest
    python foxglove_replay.py --latest 3
    python foxglove_replay.py --id 20260721_191313
    python foxglove_replay.py --path recordings/rosbag/20260721_191313

Then in Foxglove Studio:
    Open connection -> Foxglove WebSocket -> ws://localhost:8765

Requires (run inside a sourced ROS 2 environment):
    sudo apt install ros-$ROS_DISTRO-foxglove-bridge
"""
import os
import sys
import glob
import shutil
import signal
import argparse
import subprocess

# Where start_recording() in data_plane.py writes bags (folder name == session ID).
ROSBAG_DIR = os.path.join("recordings", "rosbag")
DEFAULT_PORT = 8765


def is_bag_dir(path):
    """A rosbag2 recording is a directory containing a metadata.yaml."""
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "metadata.yaml"))


def find_bag_by_id(target_id):
    """Locate a bag directory by substring match on the session ID."""
    matches = [p for p in glob.glob(os.path.join(ROSBAG_DIR, f"*{target_id}*")) if is_bag_dir(p)]
    if not matches:
        sys.exit(f"Error: no rosbag matching id '{target_id}' in {ROSBAG_DIR}/")
    return matches[0]


def find_latest_bag(n=1):
    """Return the N-th most recent bag. Timestamp names sort chronologically."""
    bags = sorted(p for p in glob.glob(os.path.join(ROSBAG_DIR, "*")) if is_bag_dir(p))
    if not bags:
        sys.exit(f"Error: no rosbags found in {ROSBAG_DIR}/")
    if len(bags) < n:
        sys.exit(f"Error: requested latest #{n}, but only {len(bags)} recording(s) exist.")
    return bags[-n]


def ensure_environment():
    """Verify ros2 is on PATH and foxglove_bridge is installed; exit with hints if not."""
    if shutil.which("ros2") is None:
        sys.exit(
            "Error: 'ros2' not found. Source your ROS 2 environment first, e.g.\n"
            "    source /opt/ros/$ROS_DISTRO/setup.bash"
        )

    # `ros2 pkg prefix` returns non-zero if the package is not installed.
    found = subprocess.run(
        ["ros2", "pkg", "prefix", "foxglove_bridge"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not found:
        distro = os.environ.get("ROS_DISTRO", "$ROS_DISTRO")
        sys.exit(
            "Error: foxglove_bridge is not installed. Install it with:\n"
            f"    sudo apt install ros-{distro}-foxglove-bridge"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Play a recorded rosbag over the Foxglove WebSocket bridge.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--id", type=str, help="Load by session ID (e.g., 20260721_191313)")
    group.add_argument("--latest", type=int, nargs="?", const=1,
                       help="Load the N-th most recent bag (default: 1)")
    group.add_argument("--path", type=str, help="Direct path to a bag directory")

    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"Foxglove bridge WebSocket port (default: {DEFAULT_PORT})")
    parser.add_argument("--rate", type=float, default=1.0,
                        help="Playback rate multiplier (default: 1.0)")
    parser.add_argument("--no-loop", action="store_true",
                        help="Play once instead of looping")
    args = parser.parse_args()

    # --- Resolve which bag to play ---
    if args.id:
        bag_path = find_bag_by_id(args.id)
    elif args.latest is not None:
        bag_path = find_latest_bag(args.latest)
    else:
        bag_path = args.path
        if not is_bag_dir(bag_path):
            sys.exit(f"Error: '{bag_path}' is not a rosbag directory (no metadata.yaml).")

    ensure_environment()

    print(f"Bag:    {bag_path}")
    print(f"Bridge: ws://localhost:{args.port}")
    print("Open Foxglove Studio -> Open connection -> Foxglove WebSocket -> "
          f"ws://localhost:{args.port}\n")

    # --- Launch the bridge, then play the bag against it ---
    bridge = subprocess.Popen(
        ["ros2", "run", "foxglove_bridge", "foxglove_bridge",
         "--ros-args", "-p", f"port:={args.port}"]
    )

    play_cmd = ["ros2", "bag", "play", bag_path, "--rate", str(args.rate)]
    if not args.no_loop:
        play_cmd.append("--loop")

    try:
        # Blocks until playback finishes (or Ctrl+C for a looping replay).
        subprocess.run(play_cmd)
    except KeyboardInterrupt:
        print("\nStopping playback...")
    finally:
        # Bring the bridge down cleanly so the port is freed for the next run.
        bridge.send_signal(signal.SIGINT)
        try:
            bridge.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            bridge.kill()


if __name__ == "__main__":
    main()
