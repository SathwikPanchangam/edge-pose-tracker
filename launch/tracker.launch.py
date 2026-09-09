import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('edge_pose_tracker')
    param_file = os.path.join(pkg_share, 'config', 'params.yaml')

    camera_node = Node(
        package='edge_pose_tracker',
        executable='camera_publisher',
        name='camera_publisher',
        output='screen',
        parameters=[param_file],
        emulate_tty=True
    )

    pose_node = Node(
        package='edge_pose_tracker',
        executable='pose_estimator',
        name='pose_estimator',
        output='screen',
        parameters=[param_file],
        emulate_tty=True
    )

    filter_node = Node(
        package='edge_pose_tracker',
        executable='pose_filter',
        name='pose_filter',
        output='screen',
        parameters=[param_file],
        emulate_tty=True
    )

    return LaunchDescription([
        camera_node,
        pose_node,
        filter_node
    ])