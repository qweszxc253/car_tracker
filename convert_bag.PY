import os
import sys
import glob
import csv
import math
import argparse
import shutil
import rosbag2_py
from rclpy.serialization import serialize_message
from geometry_msgs.msg import PoseStamped

# Conversion Constants
INCHES_TO_METERS = 0.0254
DEG_TO_RAD = math.pi / 180.0
LOCALIZATION_DIR = 'recordings/data'
BAG_DIR = 'recordings/rosbag'
DEFAULT_TOPIC = '/localization/pose'

def euler_to_quaternion(roll, pitch, yaw):
    """
    Converts Euler angles to quaternion.
    For a flat X-Y plane, roll and pitch are 0.0.
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw

def find_csv_by_id(target_id):
    """Locate a recording CSV by substring match on its start datetime / id."""
    pattern = os.path.join(LOCALIZATION_DIR, f"*{target_id}*.csv")
    matches = [f for f in glob.glob(pattern) if "temp" not in os.path.basename(f)]
    if not matches:
        sys.exit(f"Error: no CSV matching id '{target_id}' in {LOCALIZATION_DIR}/")
    if len(matches) > 1:
        print(f"Warning: {len(matches)} CSVs match '{target_id}'; using {os.path.basename(matches[0])}")
    return matches[0]

def find_latest_csv(n=1):
    """Return the N-th most recent recording CSV. Timestamp names sort chronologically."""
    csvs = sorted(glob.glob(os.path.join(LOCALIZATION_DIR, "data-*.csv")))
    csvs = [f for f in csvs if "temp" not in os.path.basename(f)]
    if not csvs:
        sys.exit(f"Error: no recordings found in {LOCALIZATION_DIR}/")
    if len(csvs) < n:
        sys.exit(f"Error: requested latest #{n}, but only {len(csvs)} recording(s) exist.")
    return csvs[-n]

def convert(csv_file, bag_path, topic_name, force=False):
    # rosbag2 refuses to write into an existing directory.
    if os.path.exists(bag_path):
        if force:
            shutil.rmtree(bag_path)
        else:
            sys.exit(f"Error: output bag '{bag_path}' already exists. Use --force to overwrite.")

    # Initialize ROS 2 Bag Writer
    writer = rosbag2_py.SequentialWriter()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    writer.open(storage_options, converter_options)

    # Register the topic metadata
    topic_info = rosbag2_py.TopicMetadata(
        name=topic_name,
        type='geometry_msgs/msg/PoseStamped',
        serialization_format='cdr'
    )
    writer.create_topic(topic_info)

    print(f"Reading from {csv_file} and writing to {bag_path}...")

    rows_written = 0
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)

        for row in reader:
            # Skip rows where tracking was lost (SessionRecorder writes blank telemetry).
            if not row.get('x'):
                continue

            # Parse raw CSV data
            timestamp = float(row['timestamp'])
            x_in = float(row['x'])
            y_in = float(row['y'])
            z_in = float(row['z'])
            yaw_deg = float(row['angle_deg'])

            # Convert to ROS 2 Standard Units (Meters, Radians)
            x_m = x_in * INCHES_TO_METERS
            y_m = y_in * INCHES_TO_METERS
            z_m = z_in * INCHES_TO_METERS
            yaw_rad = yaw_deg * DEG_TO_RAD

            # Construct PoseStamped Message
            msg = PoseStamped()
            msg.header.frame_id = 'map'

            # Convert float timestamp to ROS 2 sec/nanosec
            msg.header.stamp.sec = int(timestamp)
            msg.header.stamp.nanosec = int((timestamp - int(timestamp)) * 1e9)

            # Assign Translation
            msg.pose.position.x = x_m
            msg.pose.position.y = y_m
            msg.pose.position.z = z_m

            # Assign Rotation (Quaternion from Yaw)
            qx, qy, qz, qw = euler_to_quaternion(0.0, 0.0, yaw_rad)
            msg.pose.orientation.x = qx
            msg.pose.orientation.y = qy
            msg.pose.orientation.z = qz
            msg.pose.orientation.w = qw

            # Calculate bag write timestamp in nanoseconds
            bag_timestamp_ns = int(timestamp * 1e9)

            # Serialize and write to bag
            writer.write(
                topic_name,
                serialize_message(msg),
                bag_timestamp_ns
            )
            rows_written += 1

    print(f"Bag generation complete. {rows_written} poses written.")

def main():
    parser = argparse.ArgumentParser(
        description="Convert a recorded localization CSV into a ROS 2 bag.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--id", type=str,
                       help="Locate the CSV by start datetime / id (e.g., 20260721_191313)")
    group.add_argument("--latest", type=int, nargs="?", const=1,
                       help="Convert the N-th most recent CSV (default: 1)")
    group.add_argument("--csv", type=str, help="Direct path to a CSV file")

    parser.add_argument("--topic", type=str, default=DEFAULT_TOPIC,
                        help=f"Output topic name (default: {DEFAULT_TOPIC})")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite the output bag if it already exists")
    args = parser.parse_args()

    # Resolve which CSV to convert
    if args.id:
        csv_file = find_csv_by_id(args.id)
    elif args.latest is not None:
        csv_file = find_latest_csv(args.latest)
    else:
        csv_file = args.csv
        if not os.path.isfile(csv_file):
            sys.exit(f"Error: CSV file not found: {csv_file}")

    # Output bag named after the CSV (e.g. data-<start>-<end>), so it never collides
    # with the session-id-named bags recorded live in recordings/rosbag/.
    csv_stem = os.path.splitext(os.path.basename(csv_file))[0]
    bag_path = os.path.join(BAG_DIR, csv_stem)

    convert(csv_file, bag_path, args.topic, force=args.force)

if __name__ == '__main__':
    main()