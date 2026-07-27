import csv
import math
import rosbag2_py
from rclpy.serialization import serialize_message
from geometry_msgs.msg import PoseStamped

# Conversion Constants
INCHES_TO_METERS = 0.0254
DEG_TO_RAD = math.pi / 180.0

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

def main():
    # File configurations
    csv_file = 'localization.csv'
    bag_path = 'localization_bag' # Output directory for the db3 file
    topic_name = '/localization/pose'

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

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
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

    print("Bag generation complete.")

if __name__ == '__main__':
    main()