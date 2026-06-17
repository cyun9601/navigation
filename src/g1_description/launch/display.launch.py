"""Standalone URDF visualization (robot_state_publisher + joint_state_publisher GUI + RViz).

Useful for inspecting the G1 model and the added base/sensor frames in isolation:
    ros2 launch g1_description display.launch.py
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("g1_description")
    urdf_path = os.path.join(pkg, "urdf", "g1_body.urdf")
    with open(urdf_path, "r") as f:
        robot_description = f.read()

    return LaunchDescription([
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        ),
        Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            output="screen",
        ),
        Node(
            package="rviz2",
            executable="rviz2",
            output="screen",
            arguments=["-d", os.path.join(pkg, "..", "g1_nav2_bringup", "rviz", "g1_nav.rviz")]
            if os.path.exists(os.path.join(pkg, "..", "g1_nav2_bringup", "rviz", "g1_nav.rviz"))
            else [],
        ),
    ])
