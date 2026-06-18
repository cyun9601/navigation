"""Bring up the MuJoCo G1 simulation + sensor pipeline (no nav2).

Launches:
  * robot_state_publisher   (G1 URDF -> TF for body + sensor frames)
  * g1_sim_node             (MuJoCo: planar base, LiDAR, RGBD, odom/TF/cmd_vel)
  * robot_body_filter x2    (optional self-filter; use_body_filter:=true)
  * pointcloud_to_laserscan (filtered LiDAR cloud -> /scan)

Self-filter selection:
  use_body_filter:=false (default) -> sim's built-in self-filter
        (/lidar/points_self_filtered, /camera/points_self_filtered)
  use_body_filter:=true            -> robot_body_filter nodes
        (/lidar/points_filtered,     /camera/points_filtered)

Held-object testing:
  payload:=box|sphere|cylinder|pole|board|lshape  -> the SHAPE the robot carries
        (default box; 'scene' keeps the geom in g1_nav_scene.xml). The prior-free
        held-object filter is never told the shape; this just swaps what it must
        discover from the LiDAR/RGBD returns.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    sim_share = get_package_share_directory("g1_mujoco_sim")
    desc_share = get_package_share_directory("g1_description")
    bringup_share = get_package_share_directory("g1_nav2_bringup")

    scene_xml = os.path.join(sim_share, "worlds", "g1_nav_scene.xml")
    urdf_path = os.path.join(desc_share, "urdf", "g1_body.urdf")
    with open(urdf_path, "r") as f:
        robot_description = f.read()

    use_viewer = LaunchConfiguration("use_viewer")
    use_body_filter = LaunchConfiguration("use_body_filter")
    hold_object = LaunchConfiguration("hold_object")
    filter_held_object = LaunchConfiguration("filter_held_object")
    held_filter_mode = LaunchConfiguration("held_filter_mode")
    payload = LaunchConfiguration("payload")
    body_vibration = LaunchConfiguration("body_vibration")

    # When robot_body_filter is on, scan comes from its output; otherwise the
    # sim's internal self-filtered cloud.
    lidar_filtered = PythonExpression([
        "'/lidar/points_filtered' if '", use_body_filter,
        "' == 'true' else '/lidar/points_self_filtered'"])

    rbf_lidar_yaml = os.path.join(bringup_share, "config", "robot_body_filter_lidar.yaml")
    rbf_camera_yaml = os.path.join(bringup_share, "config", "robot_body_filter_camera.yaml")
    p2s_yaml = os.path.join(bringup_share, "config", "pointcloud_to_laserscan.yaml")

    return LaunchDescription([
        DeclareLaunchArgument("use_viewer", default_value="true",
                              description="Launch the MuJoCo passive viewer"),
        DeclareLaunchArgument("use_body_filter", default_value="false",
                              description="Use robot_body_filter instead of the sim's built-in self-filter"),
        DeclareLaunchArgument("hold_object", default_value="true",
                              description="Robot carries a payload box in its hands"),
        DeclareLaunchArgument("filter_held_object", default_value="true",
                              description="Remove the carried payload from the sensor clouds "
                                          "(false -> payload is seen as an obstacle and the robot freezes)"),
        DeclareLaunchArgument("held_filter_mode", default_value="connected",
                              description="How the payload is filtered: connected (prior-free, "
                                          "size-invariant, region-grow from the hand, default) | "
                                          "carry_volume (fixed hand region) | online (voxel "
                                          "estimate, needs motion) | shape (known primitive)"),
        DeclareLaunchArgument("payload", default_value="box",
                              description="Which payload SHAPE the robot carries (for testing the "
                                          "prior-free filter against varied objects): "
                                          "box | sphere | cylinder | pole | board | lshape "
                                          "(or 'scene' to keep the geom in the scene XML)"),
        DeclareLaunchArgument("body_vibration", default_value="true",
                              description="Inject walking-induced sensor shake (bob/sway/surge + "
                                          "roll/pitch + jitter) into the physical LiDAR/camera poses "
                                          "while keeping /odom and TF smooth -- tests real-world "
                                          "robustness to point-cloud/TF misregistration"),

        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            output="screen",
            parameters=[{"robot_description": robot_description}],
        ),

        Node(
            package="g1_mujoco_sim",
            executable="g1_sim_node",
            name="g1_mujoco_sim",
            output="screen",
            parameters=[{
                "scene_xml": scene_xml,
                "use_viewer": use_viewer,
                "hold_object": hold_object,
                "filter_held_object": filter_held_object,
                "held_filter_mode": held_filter_mode,
                "held_object_preset": payload,
                "body_vibration": body_vibration,
            }],
        ),

        # ----- robot_body_filter (optional) -----
        GroupAction(
            condition=IfCondition(use_body_filter),
            actions=[
                Node(
                    package="robot_body_filter",
                    executable="test_chain_nodelet",  # standalone node entry
                    name="robot_body_filter_lidar",
                    output="screen",
                    parameters=[rbf_lidar_yaml, {"robot_description": robot_description}],
                    remappings=[("~/input", "/lidar/points_raw"),
                                ("~/output", "/lidar/points_filtered")],
                ),
                Node(
                    package="robot_body_filter",
                    executable="test_chain_nodelet",
                    name="robot_body_filter_camera",
                    output="screen",
                    parameters=[rbf_camera_yaml, {"robot_description": robot_description}],
                    remappings=[("~/input", "/camera/depth/points"),
                                ("~/output", "/camera/points_filtered")],
                ),
            ],
        ),

        # ----- pointcloud_to_laserscan: filtered LiDAR cloud -> /scan -----
        Node(
            package="pointcloud_to_laserscan",
            executable="pointcloud_to_laserscan_node",
            name="pointcloud_to_laserscan",
            output="screen",
            parameters=[p2s_yaml],
            remappings=[("cloud_in", lidar_filtered),
                        ("scan", "/scan")],
        ),
    ])
