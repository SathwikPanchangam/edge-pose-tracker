import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Resolve dynamic path to the installed YAML parameter file
    pkg_share = get_package_share_directory('edge_pose_tracker')
    param_file = os.path.join(pkg_share, 'config', 'params.yaml')

    # Heartbeat Publisher node with parameters injected
    talker_node = Node(
        package='edge_pose_tracker',
        executable='talker',
        name='heartbeat_publisher',
        output='screen',
        parameters=[param_file],
        emulate_tty=True
    )

    # Heartbeat Subscriber node
    listener_node = Node(
        package='edge_pose_tracker',
        executable='listener',
        name='heartbeat_subscriber',
        output='screen',
        emulate_tty=True
    )

    return LaunchDescription([
        talker_node,
        listener_node
    ])