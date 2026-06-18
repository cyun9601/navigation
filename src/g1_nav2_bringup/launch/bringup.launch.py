"""One-shot bringup: MuJoCo G1 sim + sensors + nav2 + SLAM + RViz.

    ros2 launch g1_nav2_bringup bringup.launch.py

This entry point uses the sim's built-in geometric self-filter (the same
URDF-inflation / point-in-shape method as robot_body_filter, but with no ROS2
build needed) and pops up a live window comparing that filter's output against
the MuJoCo ground-truth self-filter.

Args:
    use_viewer:=true|false        MuJoCo passive viewer (default true)
    use_rviz:=true|false          RViz (default true)
    use_filter_viz:=true|false    live ground-truth-vs-filter window (default true)
    use_filter_metrics:=true|false  headless held-filter scorer (default true)
    payload:=box|sphere|cylinder|pole|board|lshape   carried object SHAPE to test
                                  the prior-free held-object filter (default box)
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
    sim_share = get_package_share_directory("g1_mujoco_sim")
    bringup_share = get_package_share_directory("g1_nav2_bringup")

    use_viewer = LaunchConfiguration("use_viewer")
    use_rviz = LaunchConfiguration("use_rviz")
    use_filter_viz = LaunchConfiguration("use_filter_viz")
    use_filter_metrics = LaunchConfiguration("use_filter_metrics")
    hold_object = LaunchConfiguration("hold_object")
    filter_held_object = LaunchConfiguration("filter_held_object")
    held_filter_mode = LaunchConfiguration("held_filter_mode")
    payload = LaunchConfiguration("payload")

    return LaunchDescription([
        DeclareLaunchArgument("use_viewer", default_value="true"),
        DeclareLaunchArgument("use_rviz", default_value="true"),
        DeclareLaunchArgument("use_filter_viz", default_value="true",
                              description="Live ground-truth-vs-self-filter window"),
        DeclareLaunchArgument("use_filter_metrics", default_value="true",
                              description="Headless held-object filter scorer "
                                          "(payload leak / over-removal per object)"),
        DeclareLaunchArgument("hold_object", default_value="true",
                              description="Robot carries a payload box in its hands"),
        DeclareLaunchArgument("filter_held_object", default_value="true",
                              description="Remove the carried payload from the clouds "
                                          "(false -> robot sees its own payload as an obstacle)"),
        DeclareLaunchArgument("held_filter_mode", default_value="connected",
                              description="connected (prior-free, size-invariant, default) | "
                                          "carry_volume | online (needs motion) | shape"),
        DeclareLaunchArgument("payload", default_value="box",
                              description="Carried object SHAPE for testing the prior-free "
                                          "held-object filter: box | sphere | cylinder | pole | "
                                          "board | lshape (or 'scene' to keep the XML geom)"),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(sim_share, "launch", "sim.launch.py")),
            launch_arguments={
                "use_viewer": use_viewer,
                # bringup self-filters with the sim's built-in geometric filter.
                "use_body_filter": "false",
                "hold_object": hold_object,
                "filter_held_object": filter_held_object,
                "held_filter_mode": held_filter_mode,
                "payload": payload,
            }.items(),
        ),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(bringup_share, "launch", "nav2.launch.py")),
            launch_arguments={"use_rviz": use_rviz}.items(),
        ),

        # Live window: ground-truth self-filter vs the built-in geometric filter.
        Node(
            condition=IfCondition(use_filter_viz),
            package="g1_mujoco_sim",
            executable="self_filter_viz_node",
            name="self_filter_viz",
            output="screen",
        ),

        # Headless scorer: payload leak / over-removal per carried object, so the
        # prior-free held-object filter can be compared across payload shapes.
        Node(
            condition=IfCondition(use_filter_metrics),
            package="g1_mujoco_sim",
            executable="held_filter_metrics_node",
            name="held_filter_metrics",
            output="screen",
            parameters=[{"payload": payload}],
        ),
    ])
