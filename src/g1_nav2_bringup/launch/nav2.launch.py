"""nav2 + slam_toolbox + RViz for the G1 (assumes sim is already running).

Provides map->odom (slam_toolbox) and the full nav2 stack reading /scan and the
self-filtered RGBD cloud. Uses wall-clock time (the sim publishes no /clock).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_share = get_package_share_directory("g1_nav2_bringup")
    nav2_bringup_share = get_package_share_directory("nav2_bringup")

    params_file = os.path.join(bringup_share, "config", "nav2_params.yaml")
    slam_params = os.path.join(bringup_share, "config", "slam_toolbox.yaml")
    rviz_config = os.path.join(bringup_share, "rviz", "g1_nav.rviz")

    use_rviz = LaunchConfiguration("use_rviz")

    return LaunchDescription([
        DeclareLaunchArgument("use_rviz", default_value="true"),

        # ----- SLAM (map + map->odom) -----
        Node(
            package="slam_toolbox",
            executable="async_slam_toolbox_node",
            name="slam_toolbox",
            output="screen",
            parameters=[slam_params, {"use_sim_time": False}],
        ),

        # ----- nav2 stack -----
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_share, "launch", "navigation_launch.py")),
            launch_arguments={
                "use_sim_time": "false",
                "params_file": params_file,
                "autostart": "true",
            }.items(),
        ),

        # ----- RViz -----
        Node(
            condition=IfCondition(use_rviz),
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", rviz_config],
        ),
    ])
