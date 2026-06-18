#!/usr/bin/env python3
"""MuJoCo <-> ROS2 bridge for navigating a Unitree G1 with nav2.

Design (see plan): the G1 is driven as a *planar mobile base*. We do NOT solve
bipedal locomotion. Instead the robot is held in a fixed standing pose and its
floating-base qpos (x, y, yaw) is integrated from /cmd_vel and written directly,
then `mj_forward` updates kinematics (no dynamics -> never falls).

Sensors:
  * 3D LiDAR  : `mj_multiRay` from a torso mount just below the camera -> PointCloud2 (lidar_link)
  * RGBD cam  : MuJoCo Renderer (rgb + depth) -> Image + PointCloud2 (optical)

Self-filter: raw clouds (including hits on the robot's own body) are always
published for an external `robot_body_filter`. As a built-in fallback, the node
can also publish *self-filtered* clouds by dropping points whose MuJoCo geom/
segmentation id belongs to a robot body.

Published:  /odom, /joint_states, TF odom->base_link,
            /lidar/points_raw (+ /lidar/points_self_filtered),
            /camera/color/image_raw, /camera/depth/image_raw, /camera/camera_info,
            /camera/depth/points (+ /camera/points_self_filtered)
Subscribed: /cmd_vel
"""
import math
import threading

import numpy as np
import mujoco

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState, PointCloud2, PointField, Image, CameraInfo
from tf2_ros import TransformBroadcaster


# ----------------------------- math helpers --------------------------------
def quat_mul(a, b):
    """Hamilton product, quaternions as (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def quat_yaw(yaw):
    return np.array([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])


def quat_pitch(pitch):
    return np.array([math.cos(pitch / 2.0), 0.0, math.sin(pitch / 2.0), 0.0])


def quat_roll(roll):
    return np.array([math.cos(roll / 2.0), math.sin(roll / 2.0), 0.0, 0.0])


def mat2quat(R):
    """3x3 rotation matrix -> quaternion (w, x, y, z)."""
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


def quat2mat(q):
    """quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    R = np.empty(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, dtype=float))
    return R.reshape(3, 3)


# ----------------------------- the node ------------------------------------
class G1SimNode(Node):
    def __init__(self):
        super().__init__("g1_mujoco_sim")

        # ---- parameters ----
        self.declare_parameter("scene_xml", "")
        self.declare_parameter("rate", 50.0)                 # main update rate [Hz]
        self.declare_parameter("use_viewer", True)
        self.declare_parameter("standing_height", 0.793)     # pelvis z when standing

        # base / frames
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("cmd_vel_timeout", 0.5)       # stop if no cmd_vel [s]

        # lidar (offsets MUST match g1_description URDF lidar_joint/mid360_joint)
        # The LiDAR is a child of the real robot link `lidar_parent` (torso_link,
        # the body that carries the LiDAR on the actual Unitree G1), offset by
        # lidar_offset in that link's frame. This is the official Unitree G1
        # `mid360_link` mount (world z ~1.25 m on the head/neck) -- the real
        # Livox Mid-360 position, not a tall virtual mast. Rays that graze the
        # robot's own shoulders/arms are dropped by the self-filter.
        self.declare_parameter("lidar_frame", "lidar_link")
        self.declare_parameter("lidar_parent", "torso_link")
        self.declare_parameter("lidar_offset", [0.0002835, 0.00003, 0.40618])
        self.declare_parameter("lidar_rate", 10.0)
        self.declare_parameter("lidar_channels", 16)         # vertical rings
        self.declare_parameter("lidar_h_samples", 180)       # horizontal samples (2 deg)
        # Wide downward FOV so the LiDAR sees the robot's own shoulders/arms/torso
        # (this is what makes the self-filter necessary and observable).
        self.declare_parameter("lidar_vfov_deg", [-85.0, 12.0])
        self.declare_parameter("lidar_range_max", 20.0)
        self.declare_parameter("lidar_range_min", 0.10)

        # camera (offsets MUST match g1_description URDF)
        self.declare_parameter("enable_camera", True)
        self.declare_parameter("camera_frame", "camera_depth_optical_frame")
        self.declare_parameter("camera_offset", [0.10, 0.0, 1.25])
        self.declare_parameter("camera_pitch_deg", 47.6)     # downward; official Unitree G1 D435 mount (rpy pitch 0.8307767 rad)
        self.declare_parameter("camera_fovy_deg", 58.0)
        self.declare_parameter("camera_width", 320)
        self.declare_parameter("camera_height", 240)
        self.declare_parameter("camera_rate", 5.0)
        self.declare_parameter("camera_range_max", 8.0)
        self.declare_parameter("camera_stride", 2)           # pixel decimation
        self.declare_parameter("show_camera_window", True)   # pop up the RGB view
        self.declare_parameter("camera_window_scale", 2.0)   # upscale the preview

        # body vibration (walking-induced sensor shake).
        # A real walking humanoid does NOT hold its torso perfectly level: it bobs
        # up/down, sways side-to-side, surges fore/aft and rocks in roll/pitch at
        # the gait cadence, plus high-frequency mounting jitter. We inject this into
        # the PHYSICAL sensor poses only (the LiDAR raycast origin, mounted on
        # torso_link via FK, and the camera render pose), while /odom and the
        # published TF stay on the smooth planar pose -- exactly the real situation
        # where the body shakes but the state estimator reports a clean base_link.
        # The nav stack therefore sees clouds that DON'T register perfectly to the
        # TF it trusts (point smearing, costmap "breathing", localization jitter) --
        # the failure mode that decides real-world viability. Amplitude scales with
        # commanded speed (gated to ~0 when standing) and ramps in/out smoothly.
        self.declare_parameter("body_vibration", True)        # enable the shake
        self.declare_parameter("vib_stride_freq", 0.9)        # gait cadence [strides/s] (~1.8 steps/s)
        self.declare_parameter("vib_ref_speed", 0.5)          # speed [m/s] at which amplitudes are nominal
        self.declare_parameter("vib_max_scale", 1.5)          # cap on the speed-amplitude factor
        self.declare_parameter("vib_turn_weight", 0.3)        # |wz| -> equivalent walking speed [m per rad/s]
        self.declare_parameter("vib_ramp", 0.08)              # gait-intensity low-pass (per tick, 0..1)
        # linear amplitudes (peak displacement at vib_ref_speed) [m]
        self.declare_parameter("vib_bob", 0.015)              # vertical bob (2x stride freq)
        self.declare_parameter("vib_sway", 0.012)             # lateral sway (1x stride freq)
        self.declare_parameter("vib_surge", 0.008)            # fore/aft surge (2x stride freq)
        # micro angular amplitudes (peak, at vib_ref_speed) [deg] -- dominate the
        # apparent shake at range (1 deg ~ 17 cm of point error at 10 m)
        self.declare_parameter("vib_roll", 1.2)               # roll rock (1x stride freq)
        self.declare_parameter("vib_pitch", 1.0)              # pitch rock (2x stride freq)
        # high-frequency random jitter (gaussian std, scaled by gait intensity)
        self.declare_parameter("vib_noise_lin", 0.004)        # translational [m]
        self.declare_parameter("vib_noise_ang", 0.3)          # angular [deg]

        # self filter (built-in, geometric body-model filter)
        # Realistic point-in-shape containment, modelled on robot_self_filter /
        # robot_body_filter: drop points that fall inside the robot's own
        # COLLISION geometry placed at the current FK link poses. Uses only what
        # a real robot has (URDF collision shapes + joint encoders -> TF); it
        # does NOT use which object a ray/pixel hit. Each link collision geom is
        # tested with its exact primitive (sphere/cylinder/box); collision
        # meshes use their oriented bounding box (MuJoCo geom_size half-extents).
        # scale (multiplicative) + margin (additive padding) inflate each shape,
        # mirroring self_see_default_scale / self_see_default_padding.
        self.declare_parameter("publish_self_filtered", True)
        self.declare_parameter("self_filter_margin", 0.04)   # additive padding [m]
        self.declare_parameter("self_filter_scale", 1.0)     # multiplicative scale

        # held object (attached payload). When the robot grasps something, that
        # object is NOT in the URDF, so the sensors hit it and nav2 treats it as
        # an obstacle right in front of the robot. Three filter modes, from most
        # to least prior knowledge (held_filter_mode):
        #   "shape"        -- known primitive rigidly fixed to the grasp link (a
        #                     grasp-database / MoveIt AttachedCollisionObject).
        #                     Requires knowing the object shape in advance.
        #   "carry_volume" -- PRIOR-FREE. Drop everything inside a fixed region
        #                     attached to the grasp link (the hand's reach). No
        #                     object model at all; over-removes a bit near the hand.
        #   "connected"    -- PRIOR-FREE (default). Remove the point-cloud region
        #                     that is spatially CONNECTED to the gripper: voxelise
        #                     the non-body returns, seed at the hand, region-grow
        #                     the connected component, drop it. The held object
        #                     touches the hand so it is one contiguous component
        #                     of ANY size/shape (a broom is removed end to end),
        #                     with no fixed dead-zone; a separate obstacle is a
        #                     different component and survives -- even when the
        #                     robot is stationary (connectivity is per-frame
        #                     geometry, not temporal persistence).
        #   "online"       -- PRIOR-FREE. Estimate the payload's occupied volume on
        #                     the fly: per sensor, transform points into the GRASP
        #                     frame, keep those in the carry region, and accumulate
        #                     a persistence voxel model -- but ONLY while the base
        #                     is actually moving (a rigid payload is stationary in
        #                     the grasp frame while the world sweeps past; a STILL
        #                     robot cannot tell its payload from a static obstacle
        #                     at the hand, so learning is frozen and it falls back
        #                     to carry_volume). Strictly additive: it can only
        #                     remove confirmed voxels INSIDE the carry region, never
        #                     more than carry_volume. Recovers the free space
        #                     carry_volume wastes, for ANY shape, LiDAR+depth.
        # Default is connected: size-invariant, prior-free, works while stationary.
        # carry_volume is a simpler fixed dead-zone; online refines while moving;
        # shape is best when a grasp-DB primitive exists. All use only the grasp-
        # link FK pose (own encoders -> TF) + raw returns, so they transfer to
        # hardware; none consults a simulator oracle.
        self.declare_parameter("hold_object", True)          # robot carries a payload
        self.declare_parameter("filter_held_object", True)   # remove it from the clouds
        self.declare_parameter("held_filter_mode", "connected")  # connected | carry_volume | online | shape
        # which payload SHAPE the robot carries, for testing the filter against a
        # variety of objects without hand-editing the scene. A preset name
        # (box | sphere | cylinder | pole | board | lshape; see held_objects.py)
        # rewrites the held_object body's geom(s) at load; "scene" keeps the geom
        # already in the XML. The prior-free filter modes never read this -- it
        # only sets what the robot physically carries (i.e. what the sensors hit).
        self.declare_parameter("held_object_preset", "box")
        self.declare_parameter("held_object_parent", "right_wrist_yaw_link")
        self.declare_parameter("held_object_pos", [0.17, 0.149, -0.04])   # nominal in-hand grasp point in parent frame [m]
        self.declare_parameter("held_object_quat", [1.0, 0.0, 0.0, 0.0])  # grasp orientation in parent frame (w,x,y,z); "shape" mode only
        self.declare_parameter("held_object_margin", 0.05)   # additive padding for the payload filter [m]
        # connectivity params (connected mode):
        self.declare_parameter("held_connect_voxel", 0.05)   # connectivity voxel size ~ link tolerance [m]
        self.declare_parameter("held_connect_seed", 0.15)    # seed radius around the gripper [m]
        self.declare_parameter("held_connect_max_reach", 1.2)  # candidate gate: ignore returns beyond this from grip [m]
        self.declare_parameter("held_connect_max_voxels", 800)  # growth budget: bigger component -> flood -> fail safe to carry_volume
        self.declare_parameter("held_connect_min_z", 0.10)   # ignore returns below this (world z) so the floor can't bridge [m]
        self.declare_parameter("held_connect_ttl", 12)       # frames a hull voxel survives unseen (cross-sensor fusion / release)
        # prior-free estimator params (carry_volume / online):
        self.declare_parameter("held_carry_radius", 0.32)    # carry-region radius around the grasp point [m]
        self.declare_parameter("held_voxel", 0.03)           # online voxel size [m]
        self.declare_parameter("held_min_obs", 2)            # online: voxel hits to count as payload
        self.declare_parameter("held_move_eps", 0.04)        # online: base sweep [m] needed to learn (motion gate)
        # publish the ground-truth self points (.../points_self_gt) from the
        # MuJoCo geom/segmentation ids, for the filter-comparison view.
        self.declare_parameter("publish_truth", True)

        p = self.get_parameter
        self.scene_xml = p("scene_xml").value
        self.rate = float(p("rate").value)
        self.dt = 1.0 / self.rate
        self.use_viewer = bool(p("use_viewer").value)
        self.stand_h = float(p("standing_height").value)
        self.odom_frame = p("odom_frame").value
        self.base_frame = p("base_frame").value
        self.cmd_timeout = float(p("cmd_vel_timeout").value)

        self.lidar_frame = p("lidar_frame").value
        self.lidar_parent = p("lidar_parent").value
        self.lidar_off = np.array(p("lidar_offset").value, dtype=float)
        self.lidar_channels = int(p("lidar_channels").value)
        self.lidar_h = int(p("lidar_h_samples").value)
        vfov = p("lidar_vfov_deg").value
        self.lidar_vfov = (math.radians(vfov[0]), math.radians(vfov[1]))
        self.lidar_rmax = float(p("lidar_range_max").value)
        self.lidar_rmin = float(p("lidar_range_min").value)
        self.lidar_period = max(1, round(self.rate / float(p("lidar_rate").value)))

        self.enable_camera = bool(p("enable_camera").value)
        self.camera_frame = p("camera_frame").value
        self.camera_off = np.array(p("camera_offset").value, dtype=float)
        self.camera_pitch = math.radians(float(p("camera_pitch_deg").value))
        self.camera_fovy = float(p("camera_fovy_deg").value)
        self.cam_w = int(p("camera_width").value)
        self.cam_h = int(p("camera_height").value)
        self.cam_rmax = float(p("camera_range_max").value)
        self.cam_stride = int(p("camera_stride").value)
        self.camera_period = max(1, round(self.rate / float(p("camera_rate").value)))
        self.show_camera_window = bool(p("show_camera_window").value)
        self.cam_window_scale = float(p("camera_window_scale").value)

        # body vibration (walking-induced sensor shake)
        self.vib_enable = bool(p("body_vibration").value)
        self.vib_freq = float(p("vib_stride_freq").value)
        self.vib_ref_speed = float(p("vib_ref_speed").value)
        self.vib_max_scale = float(p("vib_max_scale").value)
        self.vib_turn_w = float(p("vib_turn_weight").value)
        self.vib_ramp = float(p("vib_ramp").value)
        self.vib_bob = float(p("vib_bob").value)
        self.vib_sway = float(p("vib_sway").value)
        self.vib_surge = float(p("vib_surge").value)
        self.vib_roll = math.radians(float(p("vib_roll").value))
        self.vib_pitch = math.radians(float(p("vib_pitch").value))
        self.vib_noise_lin = float(p("vib_noise_lin").value)
        self.vib_noise_ang = math.radians(float(p("vib_noise_ang").value))
        # fixed inter-axis phase offsets so the components don't peak together
        self.vib_ph_bob = math.pi / 2.0
        self.vib_ph_surge = 0.0
        self.vib_ph_pitch = math.pi / 2.0

        self.publish_self_filtered = bool(p("publish_self_filtered").value)
        self.publish_truth = bool(p("publish_truth").value)
        self.self_margin = float(p("self_filter_margin").value)
        self.self_scale = float(p("self_filter_scale").value)

        self.hold_object = bool(p("hold_object").value)
        self.filter_held_object = bool(p("filter_held_object").value)
        self.held_mode = str(p("held_filter_mode").value)
        self.held_preset = str(p("held_object_preset").value)
        self.held_parent = p("held_object_parent").value
        self.held_pos = np.array(p("held_object_pos").value, dtype=float)
        self.held_quat = np.array(p("held_object_quat").value, dtype=float)
        self.held_margin = float(p("held_object_margin").value)
        self.held_connect_voxel = float(p("held_connect_voxel").value)
        self.held_connect_seed = float(p("held_connect_seed").value)
        self.held_connect_max_reach = float(p("held_connect_max_reach").value)
        self.held_connect_max_voxels = int(p("held_connect_max_voxels").value)
        self.held_connect_min_z = float(p("held_connect_min_z").value)
        self.held_connect_ttl = int(p("held_connect_ttl").value)
        self.held_carry_radius = float(p("held_carry_radius").value)
        self.held_voxel = float(p("held_voxel").value)
        self.held_min_obs = int(p("held_min_obs").value)
        self.held_move_eps = float(p("held_move_eps").value)

        if not self.scene_xml:
            raise RuntimeError("parameter 'scene_xml' is required")

        # ---- load model ----
        # Optionally swap the carried payload's shape (held_object_preset): the
        # held_object body's geom(s) are replaced via MuJoCo's model editor, so we
        # can test the prior-free filter against many shapes without editing the
        # XML. "scene" keeps whatever geom is already in g1_nav_scene.xml.
        from g1_mujoco_sim import held_objects
        self.get_logger().info(f"loading MuJoCo scene: {self.scene_xml}")
        if held_objects.is_scene_keyword(self.held_preset):
            self.model = mujoco.MjModel.from_xml_path(self.scene_xml)
        else:
            try:
                self.model = held_objects.build_model(self.scene_xml, self.held_preset)
                self.get_logger().info(
                    f"payload preset '{self.held_preset}': "
                    f"{held_objects.PRESETS[self.held_preset]['desc']}")
            except Exception as e:
                self.get_logger().error(
                    f"payload preset '{self.held_preset}' failed ({e}); "
                    f"using scene XML as-is")
                self.model = mujoco.MjModel.from_xml_path(self.scene_xml)
        self.data = mujoco.MjData(self.model)

        self._resolve_ids()
        self._precompute_lidar_dirs()

        # ---- robot state ----
        self.base_x = 0.0
        self.base_y = 0.0
        self.base_yaw = 0.0
        self.cmd = Twist()
        self.last_cmd_time = self.get_clock().now()
        self.lock = threading.Lock()

        # body-vibration state: gait phase accumulator + low-passed intensity
        # (0 while standing, so the initial pose below is the clean nominal one)
        self.vib_phase = 0.0
        self.vib_amp = 0.0

        # initialise pose
        self._apply_kinematics()

        # ---- renderer (camera) ----
        self.renderer = None
        self.cv2 = None
        self.cam_window = "G1 camera (rgbd_cam)"
        if self.enable_camera:
            try:
                self.renderer = mujoco.Renderer(self.model, height=self.cam_h, width=self.cam_w)
                self._compute_intrinsics()
                self.get_logger().info("camera renderer initialised")
            except Exception as e:  # headless / no GL
                self.get_logger().warn(f"camera renderer unavailable, disabling camera: {e}")
                self.enable_camera = False

        # ---- live camera preview window (separate from the 3D viewer) ----
        if self.enable_camera and self.show_camera_window:
            try:
                import cv2
                self.cv2 = cv2
                cv2.namedWindow(self.cam_window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(self.cam_window,
                                 int(self.cam_w * self.cam_window_scale),
                                 int(self.cam_h * self.cam_window_scale))
                self.get_logger().info("camera preview window opened")
            except Exception as e:
                self.get_logger().warn(f"camera window unavailable: {e}")
                self.cv2 = None

        # ---- viewer ----
        self.viewer = None
        if self.use_viewer:
            try:
                import mujoco.viewer as mjv
                self.viewer = mjv.launch_passive(self.model, self.data)
                self.get_logger().info("MuJoCo viewer launched")
            except Exception as e:
                self.get_logger().warn(f"viewer unavailable: {e}")
                self.viewer = None

        # ---- ROS interfaces ----
        sensor_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST, depth=5)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.pub_odom = self.create_publisher(Odometry, "odom", 10)
        self.pub_joints = self.create_publisher(JointState, "joint_states", 10)
        self.pub_lidar_raw = self.create_publisher(PointCloud2, "lidar/points_raw", sensor_qos)
        self.pub_lidar_filt = self.create_publisher(PointCloud2, "lidar/points_self_filtered", sensor_qos)
        if self.publish_truth:
            self.pub_lidar_self_gt = self.create_publisher(PointCloud2, "lidar/points_self_gt", sensor_qos)
            # payload-only ground truth (the returns that hit the held object),
            # so the metrics node can score held-object filtering on its own.
            if self.held_body is not None:
                self.pub_lidar_held_gt = self.create_publisher(PointCloud2, "lidar/points_held_gt", sensor_qos)
        if self.enable_camera:
            self.pub_rgb = self.create_publisher(Image, "camera/color/image_raw", sensor_qos)
            self.pub_depth = self.create_publisher(Image, "camera/depth/image_raw", sensor_qos)
            self.pub_caminfo = self.create_publisher(CameraInfo, "camera/camera_info", sensor_qos)
            self.pub_cam_raw = self.create_publisher(PointCloud2, "camera/depth/points", sensor_qos)
            self.pub_cam_filt = self.create_publisher(PointCloud2, "camera/points_self_filtered", sensor_qos)
            if self.publish_truth:
                self.pub_cam_self_gt = self.create_publisher(PointCloud2, "camera/points_self_gt", sensor_qos)
                if self.held_body is not None:
                    self.pub_cam_held_gt = self.create_publisher(PointCloud2, "camera/points_held_gt", sensor_qos)

        self.sub_cmd = self.create_subscription(Twist, "cmd_vel", self._cmd_cb, 10)

        self.tick = 0
        self.timer = self.create_timer(self.dt, self._update)
        self.get_logger().info("G1 MuJoCo sim node ready.")

    # --------------------------------------------------------------------- #
    def _body_id(self, name):
        """Body id by name, or None if the scene does not contain it."""
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return i if i >= 0 else None

    def _geom_id(self, name):
        """Geom id by name, or None if the scene does not contain it."""
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
        return i if i >= 0 else None

    # --------------------------------------------------------------------- #
    def _resolve_ids(self):
        m = self.model
        # floating base qpos start
        self.base_qadr = m.jnt_qposadr[m.joint("floating_base_joint").id]

        # actuated (hinge) joints -> for joint_states
        self.joint_names, self.joint_qadr = [], []
        for j in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
            if name is None or name == "floating_base_joint":
                continue
            self.joint_names.append(name)
            self.joint_qadr.append(m.jnt_qposadr[j])
        self.joint_qadr = np.array(self.joint_qadr, dtype=int)

        # mocap bodies
        self.lidar_body = m.body("lidar_mount").id
        self.lidar_mocap = m.body_mocapid[self.lidar_body]
        self.camera_body = m.body("camera_mount").id
        self.camera_mocap = m.body_mocapid[self.camera_body]
        self.lidar_site = m.site("lidar_site").id

        # held object (attached payload), if present in the scene. It is a mocap
        # body like the sensor mounts; the node rewrites its pose every tick.
        self.held_body = self._body_id("held_object")
        self.held_geom = self._geom_id("held_object_geom")
        if self.held_body is not None and self.held_geom is None:
            self.get_logger().warn("'held_object' body has no 'held_object_geom'; "
                                   "disabling held-object handling")
            self.held_body = None
        if self.held_body is not None:
            self.held_mocap = m.body_mocapid[self.held_body]
            self.held_parent_id = m.body(self.held_parent).id
            # grasp orientation in the parent frame (params, w,x,y,z) as a matrix
            qn = self.held_quat / (np.linalg.norm(self.held_quat) or 1.0)
            R = np.empty(9)
            mujoco.mju_quat2Mat(R, qn)
            self.held_offR = R.reshape(3, 3)
            # "shape" mode only: the known primitive (a grasp database / MoveIt
            # attached collision object). The prior-free modes never read this --
            # they discover the volume from the returns. Reading the scene geom
            # here just plays the role of the grasp DB for the "shape" demo.
            t = int(m.geom_type[self.held_geom])
            sz = m.geom_size[self.held_geom].astype(float)
            self.held_shape = {int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
                               int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder"}.get(t, "box")
            self.held_size = sz
            # ALL geoms of the payload body (a preset may be multi-geom, e.g. an
            # L-shape). Used to mark payload returns as ground truth; the prior-
            # free filters still discover the volume from the returns themselves.
            self.held_geoms = np.array(
                [g for g in range(m.ngeom) if m.geom_bodyid[g] == self.held_body],
                dtype=np.int32)
            if self.held_mode == "shape" and self.held_geoms.size > 1:
                self.get_logger().warn(
                    "held_filter_mode='shape' models only the primary geom; this "
                    f"payload has {self.held_geoms.size} geoms -- use a prior-free "
                    "mode (connected/carry_volume/online) for multi-part objects")
        else:
            self.held_geoms = np.zeros(0, dtype=np.int32)
        self.has_held = self.hold_object and self.held_body is not None

        # online estimator: one persistence voxel model PER SENSOR (the LiDAR and
        # camera have different rates/viewpoints, so they must not decay each
        # other's voxels). _held_reset() (re)initialises them.
        self._HELD_INC = 1               # reinforce a voxel seen this update
        self._HELD_DEC = 1               # decay a voxel not seen this update
        self._HELD_CAP = 10              # max persistence count
        self._HELD_BOOT = 2              # motion-confirmed updates before the model is trusted
        self._VOX_NEIGH = np.array(      # 26-neighbourhood (+self) for dilation
            [[dx, dy, dz] for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)],
            dtype=np.int64)
        # int64 voxel-key packing. Only carry-region points (always within
        # held_carry_radius of the grasp point) are ever packed, so indices stay
        # tiny and can never reach the +/- _VOX_OFF aliasing range.
        self._VOX_OFF = 1 << 14
        self._VOX_M = 1 << 15            # base = 2*OFF; 32768^3 < 2^63, no overflow
        self.grasp_t = np.zeros(3)       # grasp-link world pose, set each tick
        self.grasp_R = np.eye(3)
        self._held_reset()

        # real robot link the LiDAR is mounted on
        self.lidar_parent_id = m.body(self.lidar_parent).id

        # camera id for rendering
        self.cam_id = m.camera("rgbd_cam").id

        # robot bodies (everything except world + the mocap mounts) for self
        # filter. The held object is a mocap body too, and is deliberately NOT a
        # robot body: it must NOT be caught by the body self-filter (it is not in
        # the URDF). It is removed only by the separate attached-object filter
        # below, so that with filter_held_object:=false it stays an obstacle.
        self.robot_bodies = set()
        skip = {self.lidar_body, self.camera_body}
        if self.held_body is not None:
            skip.add(self.held_body)
        for b in range(1, m.nbody):
            if b in skip:
                continue
            self.robot_bodies.add(b)

        # Self-filter body model. robot_self_filter / robot_body_filter use the
        # <collision> geometry, but here the SENSORS strike the VISUAL meshes
        # (the LiDAR raycast and the camera render both hit visual geoms), and
        # the G1 collision model does not fully cover them -- e.g. the wrist/hand
        # collision shape is smaller than the rendered rubber-hand mesh, and the
        # waist links have no collision geom at all. Filtering against collision
        # geoms alone therefore leaks those visual surfaces, which then get
        # marked as obstacles right in front of the robot. So the body model is
        # ALL robot geoms (collision + visual) -- exactly the surfaces the sensors
        # can observe. Poses come from FK (d.geom_xpos/xmat = TF from joint
        # encoders on a real robot); sizes are the known URDF geometry. Spheres
        # are tested vectorized; every other shape is tested in its local frame.
        # (On real hardware you would instead filter against the collision model
        # or a dedicated self-filter mesh; here visual == what is measured.)
        GT = mujoco.mjtGeom
        self._GT_BOX = GT.mjGEOM_BOX
        self._GT_MESH = GT.mjGEOM_MESH
        self._GT_CYLINDER = GT.mjGEOM_CYLINDER
        self._GT_CAPSULE = GT.mjGEOM_CAPSULE
        self._GT_ELLIPSOID = GT.mjGEOM_ELLIPSOID
        coll = np.array(
            [g for g in range(m.ngeom) if m.geom_bodyid[g] in self.robot_bodies],
            dtype=np.int32)
        is_sphere = m.geom_type[coll] == GT.mjGEOM_SPHERE
        self.self_sphere_ids = coll[is_sphere]
        self.self_sphere_r = m.geom_size[self.self_sphere_ids, 0].astype(np.float64)
        # non-sphere collision geoms, tested per-geom in their local frame
        self.self_local_ids = coll[~is_sphere].tolist()
        # ground-truth self lookup: True for every geom that belongs to the robot
        # body. The self-filter logic deliberately never uses this (it works from
        # URDF shapes only); it is the reference the filter is judged against and
        # is published on .../points_self_gt for the live filter-comparison view.
        self.geom_is_self = np.zeros(m.ngeom, dtype=bool)
        self.geom_is_self[coll] = True
        # The payload is "self" (should be removed) for the ground-truth viz only
        # when we are actually filtering it; otherwise it is a legitimate obstacle.
        if self.has_held and self.filter_held_object:
            self.geom_is_self[self.held_geoms] = True
        # payload-only ground truth (always, regardless of the filter flag): which
        # geoms ARE the held object, for the held-object metrics / .../points_held_gt.
        self.geom_is_held = np.zeros(m.ngeom, dtype=bool)
        if self.held_geoms.size:
            self.geom_is_held[self.held_geoms] = True
        self.get_logger().info(
            f"self-filter: {coll.size} collision geoms "
            f"({self.self_sphere_ids.size} spheres + {len(self.self_local_ids)} other)")
        if self.held_body is not None:
            if self.held_mode == "shape":
                extra = f"shape={self.held_shape} size={np.round(self.held_size[:3], 3).tolist()} "
            elif self.held_mode == "connected":
                extra = (f"connect_voxel={self.held_connect_voxel} seed={self.held_connect_seed} "
                         f"max_reach={self.held_connect_max_reach} ")
            else:
                extra = f"carry_r={self.held_carry_radius} voxel={self.held_voxel} "
            self.get_logger().info(
                f"held object: hold={self.has_held} filter={self.filter_held_object} "
                f"mode={self.held_mode} {extra}"
                f"on '{self.held_parent}' grasp_pt={np.round(self.held_pos, 3).tolist()} "
                f"(prior-free unless mode=shape)")

        # geomgroup mask: include groups 0,1; exclude group 2 (sensor markers)
        self.geomgroup = np.array([1, 1, 0, 1, 1, 1], dtype=np.uint8)

    def _precompute_lidar_dirs(self):
        """Unit ray directions in the LiDAR local frame (x fwd, y left, z up)."""
        el = np.linspace(self.lidar_vfov[0], self.lidar_vfov[1], self.lidar_channels)
        az = np.linspace(-math.pi, math.pi, self.lidar_h, endpoint=False)
        AZ, EL = np.meshgrid(az, el)          # (channels, h)
        ce = np.cos(EL)
        self.dir_local = np.stack([
            (ce * np.cos(AZ)).ravel(),
            (ce * np.sin(AZ)).ravel(),
            np.sin(EL).ravel(),
        ], axis=1).astype(np.float64)         # (nray, 3)
        self.nray = self.dir_local.shape[0]
        self._ray_geomid = np.zeros(self.nray, dtype=np.int32)
        self._ray_dist = np.zeros(self.nray, dtype=np.float64)
        self.get_logger().info(
            f"LiDAR: {self.lidar_channels} rings x {self.lidar_h} = {self.nray} rays")

    def _compute_intrinsics(self):
        fovy = math.radians(self.camera_fovy)
        self.fy = self.cam_h / (2.0 * math.tan(fovy / 2.0))
        self.fx = self.fy  # square pixels in MuJoCo
        self.cx = (self.cam_w - 1) / 2.0
        self.cy = (self.cam_h - 1) / 2.0

    # --------------------------------------------------------------------- #
    def _cmd_cb(self, msg: Twist):
        with self.lock:
            self.cmd = msg
            self.last_cmd_time = self.get_clock().now()

    # --------------------------------------------------------------------- #
    def _step_vibration(self, vx, vy, wz):
        """Advance the gait phase and the speed-driven shake intensity.

        Cadence is held ~constant; the amplitude scales with how fast the robot
        is commanded to move (translation + turning-in-place), low-passed so the
        shake ramps in/out smoothly instead of snapping when /cmd_vel jumps.
        """
        if not self.vib_enable:
            return
        self.vib_phase += 2.0 * math.pi * self.vib_freq * self.dt
        if self.vib_phase > 2.0 * math.pi:
            self.vib_phase -= 2.0 * math.pi
        move = math.hypot(vx, vy) + self.vib_turn_w * abs(wz)
        target = 0.0
        if self.vib_ref_speed > 0.0:
            target = min(move / self.vib_ref_speed, self.vib_max_scale)
        self.vib_amp += (target - self.vib_amp) * self.vib_ramp

    def _vibration_offsets(self):
        """Body-frame shake at the current phase/intensity.

        Returns (dx, dy, dz, droll, dpitch): fore/aft, lateral, vertical [m] and
        roll/pitch [rad]. Lateral sway and roll oscillate once per stride; vertical
        bob, fore/aft surge and pitch rock twice per stride (once per step). High-
        frequency gaussian jitter is added on top. Everything scales with the gait
        intensity, so a standing robot returns zeros (stable sensors).
        """
        g = self.vib_amp
        if not self.vib_enable or g <= 1e-4:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        p = self.vib_phase
        dy = g * self.vib_sway * math.sin(p)
        droll = g * self.vib_roll * math.sin(p)
        dz = g * self.vib_bob * math.sin(2.0 * p + self.vib_ph_bob)
        dx = g * self.vib_surge * math.sin(2.0 * p + self.vib_ph_surge)
        dpitch = g * self.vib_pitch * math.sin(2.0 * p + self.vib_ph_pitch)
        nl = g * self.vib_noise_lin
        na = g * self.vib_noise_ang
        if nl > 0.0:
            dx += np.random.normal(0.0, nl)
            dy += np.random.normal(0.0, nl)
            dz += np.random.normal(0.0, nl)
        if na > 0.0:
            droll += np.random.normal(0.0, na)
            dpitch += np.random.normal(0.0, na)
        return dx, dy, dz, droll, dpitch

    # --------------------------------------------------------------------- #
    def _apply_kinematics(self):
        """Write base pose + mocap sensor poses, then run forward kinematics."""
        d, m = self.data, self.model
        # Walking-induced sensor shake: a small body-frame perturbation
        # (bob/sway/surge + roll/pitch + jitter) added to the PHYSICAL pose that
        # drives the sensors. /odom and TF (published from base_x/y/yaw) stay
        # clean, so the clouds are realistically inconsistent with the TF. dx, dy
        # are in the body frame; dz is vertical; droll/dpitch tilt the body.
        dx, dy, dz, droll, dpitch = self._vibration_offsets()
        c, s = math.cos(self.base_yaw), math.sin(self.base_yaw)
        dxw = c * dx - s * dy                  # body-frame planar offset -> world
        dyw = s * dx + c * dy
        # body orientation = nominal yaw, then the gait roll/pitch rock. Reused for
        # both the floating-base qpos (LiDAR, mounted via FK) and the camera.
        qb = quat_mul(quat_yaw(self.base_yaw),
                      quat_mul(quat_roll(droll), quat_pitch(dpitch)))
        R_body = quat2mat(qb)

        # floating base qpos = [x, y, z, qw, qx, qy, qz]
        a = self.base_qadr
        d.qpos[a + 0] = self.base_x + dxw
        d.qpos[a + 1] = self.base_y + dyw
        d.qpos[a + 2] = self.stand_h + dz
        d.qpos[a + 3:a + 7] = qb
        # standing pose for all actuated joints = 0 (natural G1 stance)
        # (qpos already 0-initialised; leave as is)

        # First kinematics pass: gives the real link poses (incl. the LiDAR's
        # parent link, torso_link) so we can mount the LiDAR on it.
        # Only positions are needed for raycasting/rendering (no dynamics), so
        # kinematics-only is much cheaper than a full mj_forward.
        mujoco.mj_kinematics(m, d)

        # LiDAR is rigidly mounted on the real parent link: world = parent o offset.
        # Store origin + rotation analytically so the raycast needs no extra
        # kinematics pass (cheap, runs every tick).
        pp = d.xpos[self.lidar_parent_id].copy()
        pR = d.xmat[self.lidar_parent_id].reshape(3, 3).copy()
        self.lidar_R = pR                      # ray directions rotate with the link
        self.lidar_origin = pp + pR @ self.lidar_off
        d.mocap_pos[self.lidar_mocap] = self.lidar_origin
        d.mocap_quat[self.lidar_mocap] = mat2quat(pR)

        # held object: rigidly attached to the grasp link. Its world pose is the
        # grasp link's FK pose composed with the fixed grasp offset -- exactly the
        # transform a real robot has (link TF from joint encoders o known grasp).
        # The SAME pose feeds both the rendered/raycast geom (mocap) and the
        # attached-object self-filter, so the removed volume matches what the
        # sensors see, with no simulator oracle.
        if self.held_body is not None:
            if self.has_held:
                hp = d.xpos[self.held_parent_id].copy()
                hR = d.xmat[self.held_parent_id].reshape(3, 3).copy()
                # grasp-link world pose -- the frame the prior-free filter works
                # in (a real robot gets this from joint encoders -> TF).
                self.grasp_t = hp
                self.grasp_R = hR
                self.held_c = hp + hR @ self.held_pos
                self.held_R = hR @ self.held_offR
                d.mocap_pos[self.held_mocap] = self.held_c
                d.mocap_quat[self.held_mocap] = mat2quat(self.held_R)
            else:
                # not carrying anything: park the payload far underground / out of
                # every sensor's view so it neither renders nor is raycast, and
                # clear any online model so the next grasp starts clean (release).
                d.mocap_pos[self.held_mocap] = [0.0, 0.0, -10.0]
                if self._held_models["lidar"]["obj_keys"].size or \
                        self._held_models["camera"]["obj_keys"].size:
                    self._held_reset()

        # camera mount: base_link-relative, orientation = body frame then pitch-down
        # about Y. Mounted on the SAME perturbed body pose as the LiDAR (base_link
        # origin at world z=0, raised by the bob dz), so both sensors shake together.
        body_origin = np.array([self.base_x + dxw, self.base_y + dyw, dz])
        d.mocap_pos[self.camera_mocap] = body_origin + R_body @ self.camera_off
        d.mocap_quat[self.camera_mocap] = quat_mul(qb, quat_pitch(self.camera_pitch))

    # --------------------------------------------------------------------- #
    def _update(self):
        if not rclpy.ok():
            return
        now = self.get_clock().now()
        # integrate base from cmd_vel (planar)
        with self.lock:
            cmd = self.cmd
            age = (now - self.last_cmd_time).nanoseconds * 1e-9
        if age > self.cmd_timeout:
            vx = vy = wz = 0.0
        else:
            vx, vy, wz = cmd.linear.x, cmd.linear.y, cmd.angular.z

        c, s = math.cos(self.base_yaw), math.sin(self.base_yaw)
        self.base_x += (vx * c - vy * s) * self.dt
        self.base_y += (vx * s + vy * c) * self.dt
        self.base_yaw = math.atan2(math.sin(self.base_yaw + wz * self.dt),
                                   math.cos(self.base_yaw + wz * self.dt))

        self._step_vibration(vx, vy, wz)
        self._apply_kinematics()

        stamp = now.to_msg()
        self._publish_odom_tf(stamp, vx, vy, wz)
        self._publish_joint_states(stamp)

        # Offset camera vs lidar so the two heavy callbacks rarely land on the
        # same tick (keeps odom/TF cadence smooth in the single-threaded executor).
        if self.tick % self.lidar_period == 0:
            self._publish_lidar(stamp)
        elif self.enable_camera and (self.tick % self.camera_period == 2):
            self._publish_camera(stamp)

        if self.viewer is not None:
            if self.viewer.is_running():
                self.viewer.sync()
            else:
                self.get_logger().info("viewer closed; shutting down")
                rclpy.shutdown()

        self.tick += 1

    # --------------------------------------------------------------------- #
    def _publish_odom_tf(self, stamp, vx, vy, wz):
        qb = quat_yaw(self.base_yaw)
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = self.odom_frame
        t.child_frame_id = self.base_frame
        t.transform.translation.x = self.base_x
        t.transform.translation.y = self.base_y
        t.transform.translation.z = 0.0
        t.transform.rotation.w = qb[0]
        t.transform.rotation.x = qb[1]
        t.transform.rotation.y = qb[2]
        t.transform.rotation.z = qb[3]
        self.tf_broadcaster.sendTransform(t)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.base_x
        odom.pose.pose.position.y = self.base_y
        odom.pose.pose.orientation.w = qb[0]
        odom.pose.pose.orientation.z = qb[3]
        odom.twist.twist.linear.x = vx
        odom.twist.twist.linear.y = vy
        odom.twist.twist.angular.z = wz
        self.pub_odom.publish(odom)

    def _publish_joint_states(self, stamp):
        js = JointState()
        js.header.stamp = stamp
        js.name = self.joint_names
        js.position = [float(self.data.qpos[a]) for a in self.joint_qadr]
        self.pub_joints.publish(js)

    # --------------------------------------------------------------------- #
    def _publish_lidar(self, stamp):
        d, m = self.data, self.model
        origin = self.lidar_origin
        # rotate local ray directions into world by the LiDAR mount (parent link)
        # rotation: world_dir = R @ dir_local  ->  world = dir_local @ R.T
        dl = self.dir_local
        world = dl @ self.lidar_R.T

        vec = world.reshape(-1).copy()
        self._ray_geomid.fill(-1)
        mujoco.mj_multiRay(m, d, origin, vec, self.geomgroup, 1, self.lidar_body,
                           self._ray_geomid, self._ray_dist, None,
                           self.nray, self.lidar_rmax)

        dist = self._ray_dist
        # A return exists where the range reading is within [rmin, rmax]; this is
        # the same information a real LiDAR gives (a valid distance), not the
        # identity of what was hit.
        hit = (self._ray_geomid >= 0) & (dist >= self.lidar_rmin) & (dist <= self.lidar_rmax)
        # local points (in lidar frame) = dist * dir_local
        pts = (dl[hit] * dist[hit, None]).astype(np.float32)

        header_stamp = stamp
        self.pub_lidar_raw.publish(self._cloud(header_stamp, self.lidar_frame, pts))
        if self.publish_truth:
            # ground truth: which hit points struck a robot-body geom. Aligned
            # with `pts` (same hit mask), so coordinates match the raw cloud.
            hit_geoms = self._ray_geomid[hit]
            gt_self = self.geom_is_self[hit_geoms]
            self.pub_lidar_self_gt.publish(
                self._cloud(header_stamp, self.lidar_frame, pts[gt_self]))
            if self.held_body is not None:
                gt_held = self.geom_is_held[hit_geoms]
                self.pub_lidar_held_gt.publish(
                    self._cloud(header_stamp, self.lidar_frame, pts[gt_held]))
        if self.publish_self_filtered:
            # world points = origin + R @ local; test against the body model.
            world_pts = self.lidar_origin + pts.astype(np.float64) @ self.lidar_R.T
            keep = self._self_filter_keep(world_pts, "lidar")
            self.pub_lidar_filt.publish(self._cloud(header_stamp, self.lidar_frame, pts[keep]))

    # --------------------------------------------------------------------- #
    def _publish_camera(self, stamp):
        # Apply the camera mocap pose to kinematics so the rendered view tracks
        # the base (done here, only on camera ticks, to keep the main loop light).
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)
        r = self.renderer
        # RGB
        r.disable_depth_rendering()
        r.update_scene(self.data, camera=self.cam_id)
        rgb = r.render()
        self.pub_rgb.publish(self._image(stamp, self.camera_frame, rgb, "rgb8"))
        if self.cv2 is not None:
            # cv2 wants BGR; renderer gives RGB
            self.cv2.imshow(self.cam_window, rgb[:, :, ::-1])
            self.cv2.waitKey(1)

        # Depth
        r.enable_depth_rendering()
        r.update_scene(self.data, camera=self.cam_id)
        depth = r.render().astype(np.float32)
        r.disable_depth_rendering()
        self.pub_depth.publish(self._image(stamp, self.camera_frame, depth, "32FC1"))
        self.pub_caminfo.publish(self._caminfo(stamp))

        # back-project to optical-frame points (x right, y down, z forward)
        st = self.cam_stride
        dsub = depth[::st, ::st]
        h, w = dsub.shape
        us = (np.arange(0, self.cam_w, st)[:w] - self.cx)
        vs = (np.arange(0, self.cam_h, st)[:h] - self.cy)
        UU, VV = np.meshgrid(us, vs)
        Z = dsub
        valid = (Z > 0.05) & (Z < self.cam_rmax)
        X = (UU * Z / self.fx)
        Y = (VV * Z / self.fy)
        pts = np.stack([X[valid], Y[valid], Z[valid]], axis=1).astype(np.float32)
        self.pub_cam_raw.publish(self._cloud(stamp, self.camera_frame, pts))

        if self.publish_truth:
            # ground truth via segmentation: render the geom id at each pixel and
            # decimate/mask exactly like the depth cloud, so it aligns with `pts`.
            r.enable_segmentation_rendering()
            r.update_scene(self.data, camera=self.cam_id)
            seg = r.render()[:, :, 0].astype(np.int32)
            r.disable_segmentation_rendering()
            seg_sub = seg[::st, ::st][valid]
            gt_self = np.zeros(pts.shape[0], dtype=bool)
            hit_pix = seg_sub >= 0                       # -1 == background (no geom)
            gt_self[hit_pix] = self.geom_is_self[seg_sub[hit_pix]]
            self.pub_cam_self_gt.publish(self._cloud(stamp, self.camera_frame, pts[gt_self]))
            if self.held_body is not None:
                gt_held = np.zeros(pts.shape[0], dtype=bool)
                gt_held[hit_pix] = self.geom_is_held[seg_sub[hit_pix]]
                self.pub_cam_held_gt.publish(self._cloud(stamp, self.camera_frame, pts[gt_held]))

        if self.publish_self_filtered:
            # optical (x right, y down, z fwd) -> MuJoCo cam (x right, y up, z back)
            cam_pos = self.data.cam_xpos[self.cam_id]
            cam_R = self.data.cam_xmat[self.cam_id].reshape(3, 3)
            p_cam = pts.astype(np.float64) * np.array([1.0, -1.0, -1.0])
            world_pts = cam_pos + p_cam @ cam_R.T
            keep = self._self_filter_keep(world_pts, "camera")
            self.pub_cam_filt.publish(self._cloud(stamp, self.camera_frame, pts[keep]))

    # --------------------------------------------------------------------- #
    def _self_filter_keep(self, world_pts, sensor="lidar"):
        """Geometric self-filter: True for points to KEEP (not on the robot).

        Point-in-shape containment against the robot's collision geometry at the
        current FK poses (the robot_self_filter / robot_body_filter approach):
        a point is the robot's own body if it lies inside any collision shape,
        inflated by ``self_scale`` (multiplicative) and ``self_margin`` (additive
        padding). Each shape is tested exactly -- sphere by centre distance;
        cylinder/capsule/box/ellipsoid in the geom's local frame; collision
        meshes by their oriented bounding box (geom_size half-extents). Uses only
        URDF geometry + joint-derived link poses, so it transfers to a real
        robot; it never consults which object a ray/pixel actually struck.
        """
        n = world_pts.shape[0]
        if n == 0:
            return np.ones(0, dtype=bool)
        P = world_pts
        sc, pad = self.self_scale, self.self_margin
        inside = np.zeros(n, dtype=bool)

        # spheres: frame-independent, vectorized over all sphere geoms at once
        if self.self_sphere_ids.size:
            C = self.data.geom_xpos[self.self_sphere_ids]     # (G,3)
            r = self.self_sphere_r * sc + pad                 # (G,)
            d2 = ((P[:, None, :] - C[None, :, :]) ** 2).sum(2)
            inside |= (d2 <= (r * r)[None, :]).any(axis=1)

        # every other collision shape: transform points into the geom's local
        # frame (world -> local = (p - c) @ R, R columns = geom axes) and test.
        for g in self.self_local_ids:
            c = self.data.geom_xpos[g]
            R = self.data.geom_xmat[g].reshape(3, 3)
            loc = (P - c) @ R
            t = self.model.geom_type[g]
            sz = self.model.geom_size[g]
            if t == self._GT_BOX or t == self._GT_MESH:
                half = sz * sc + pad
                inside |= (np.abs(loc) <= half).all(axis=1)
            elif t == self._GT_CYLINDER:
                r = sz[0] * sc + pad
                hl = sz[1] * sc + pad
                inside |= (np.abs(loc[:, 2]) <= hl) & ((loc[:, :2] ** 2).sum(1) <= r * r)
            elif t == self._GT_CAPSULE:
                r = sz[0] * sc + pad
                hl = sz[1] * sc
                zc = np.clip(loc[:, 2], -hl, hl)
                inside |= (loc[:, 0] ** 2 + loc[:, 1] ** 2
                           + (loc[:, 2] - zc) ** 2) <= r * r
            elif t == self._GT_ELLIPSOID:
                half = sz * sc + pad
                inside |= ((loc / half) ** 2).sum(1) <= 1.0
            else:  # unknown type -> conservative bounding sphere
                r = self.model.geom_rbound[g] * sc + pad
                inside |= (loc ** 2).sum(1) <= r * r

        # attached payload: an extra carried object, removed by a separate filter
        # (connectivity to the gripper, known shape, conservative carry volume, or
        # a prior-free online voxel estimate) -- see _held_inside.
        inside |= self._held_inside(P, sensor, inside)
        return ~inside

    # --------------------------------------------------------------------- #
    def _held_inside(self, P, sensor, body_inside):
        """Mask (True = drop) of points that belong to the carried payload.

        ``held_filter_mode`` picks how much prior knowledge is used:

          "connected"    -- PRIOR-FREE: drop the point-cloud region spatially
                            connected to the gripper (region-grow). Size-invariant
                            and works while stationary. (Needs body_inside to know
                            which points are the robot vs candidates.)
          "shape"        -- a known primitive (grasp DB / attached collision
                            object). Needs the object shape in advance.
          "carry_volume" -- PRIOR-FREE: drop the whole carry region around the
                            hand. No object model; over-removes near the hand.
          "online"       -- PRIOR-FREE: drop points whose grasp-frame voxel is in
                            a per-sensor persistence model learned ONLY while the
                            base moves. Strictly additive.

        Uses only the grasp-link FK pose + the raw returns -- transfers to a real
        robot and never consults a simulator oracle.
        """
        n = P.shape[0]
        if n == 0 or not (self.has_held and self.filter_held_object):
            return np.zeros(n, dtype=bool)

        if self.held_mode == "connected":
            return self._held_connected(P, body_inside)

        Pg = (P - self.grasp_t) @ self.grasp_R          # world -> grasp frame

        if self.held_mode == "shape":
            loc = (Pg - self.held_pos) @ self.held_offR
            sz, pad = self.held_size, self.held_margin
            if self.held_shape == "sphere":
                r = sz[0] + pad
                return (loc ** 2).sum(1) <= r * r
            if self.held_shape == "cylinder":
                r, hl = sz[0] + pad, sz[1] + pad
                return (np.abs(loc[:, 2]) <= hl) & ((loc[:, :2] ** 2).sum(1) <= r * r)
            half = sz[:3] + pad
            return (np.abs(loc) <= half).all(axis=1)

        # prior-free: a carry region attached to the hand bounds what can ever be
        # called "held" (own kinematics, not the object). Centred on the nominal
        # grasp point in the grasp frame.
        dc = Pg - self.held_pos
        in_region = (dc * dc).sum(1) <= self.held_carry_radius ** 2

        if self.held_mode == "carry_volume":
            return in_region
        return self._held_online(Pg, in_region, sensor)

    # --------------------------------------------------------------------- #
    def _held_online(self, Pg, in_region, sensor):
        """Prior-free online payload estimate (per sensor, motion-gated).

        Online may only LEARN while the base actually sweeps relative to the
        world: a rigid payload is stationary in the grasp frame while the world
        moves through it, but a STILL robot cannot distinguish its payload from a
        static obstacle at the hand. So below a motion threshold the model is
        frozen (no voxel is baked from a static obstacle) and, until a model is
        motion-confirmed, removal falls back to the full carry region. Removal is
        strictly additive: only confirmed voxels INSIDE the carry region are
        dropped -- never more than carry_volume, never anything outside the gate.
        """
        st = self._held_models[sensor]
        moved = True
        if st["last_t"] is not None:
            dtrans = float(np.linalg.norm(self.grasp_t - st["last_t"]))
            dR = st["last_R"].T @ self.grasp_R
            ang = math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1.0) * 0.5)))
            moved = (dtrans + ang * self.held_carry_radius) > self.held_move_eps
        if moved:                                       # learn only when sweeping
            self._held_model_update(st, Pg[in_region])
            st["last_t"] = self.grasp_t.copy()
            st["last_R"] = self.grasp_R.copy()

        if not (st["obs_frames"] > self._HELD_BOOT and st["obj_keys"].size):
            return in_region                            # no trusted model -> carry volume

        # tight: drop only in-region points whose voxel is a confirmed payload
        # voxel. Query in-region points only -- they are within carry_radius, so
        # their voxel indices stay tiny (no int64 key aliasing) and it is cheap.
        remove = np.zeros(Pg.shape[0], dtype=bool)
        idx = np.where(in_region)[0]
        if idx.size:
            hit = np.isin(self._vox_pack(self._vox_ijk(Pg[idx])), st["obj_keys"])
            remove[idx[hit]] = True
        return remove

    def _held_reset(self):
        """(Re)initialise the held-object models (also call on release)."""
        self._held_models = {s: {"count": {}, "obj_keys": np.empty(0, np.int64),
                                 "obs_frames": 0, "last_t": None, "last_R": None}
                             for s in ("lidar", "camera")}
        # connected mode: shared GRASP-FRAME voxel hull {packed key -> last tick}.
        # Built by connectivity on each cloud and shared across sensors so the
        # dense camera's hull also clears the sparse LiDAR; the TTL clears it on
        # release. Grasp-frame keys are stable as the base moves.
        self._held_hull = {}

    # --------------------------------------------------------------------- #
    def _held_connected(self, P, body_inside):
        """Prior-free, size-invariant payload removal by CONNECTIVITY to the hand.

        The held object physically touches the gripper, so its returns form one
        spatially-contiguous component with the hand. Each cloud is voxelised in
        the GRASP frame; we seed at the gripper, region-grow the 26-connected
        component, and add it to a shared grasp-frame voxel **hull**. Points
        (this cloud) whose grasp-frame voxel is in the hull are dropped. This
        removes an object of ANY size/shape end-to-end with no fixed dead-zone,
        and -- connectivity being a per-frame geometric property, not temporal
        persistence -- it works while the robot is stationary; a separate obstacle
        is a different component and is kept.

        The hull is shared across sensors and kept for ``held_connect_ttl`` frames:
        the **dense depth camera** builds a complete hull that then also clears the
        **sparse LiDAR** (whose own single scan is too thin to stay connected),
        and the TTL clears the hull on release. Grasp-frame keys are stable as the
        base moves, so the hull tracks the hand.

        Safeguards: candidates exclude the robot body (growth can't run through it)
        and floor returns below ``held_connect_min_z`` (the ground can't bridge the
        object to the world); everything is capped at ``held_connect_max_reach``
        from the grip, so a stray bridge can only ever delete a bounded blob.
        """
        n = P.shape[0]
        v = self.held_connect_voxel
        reach2 = self.held_connect_max_reach ** 2

        # bounded carry-volume fail-safe (used when connectivity is untrustworthy:
        # empty seed, or a flood that exceeds the voxel budget). Removes only the
        # fixed sphere at the hand -- never floods a wall/table.
        def carry_fallback():
            Pg_all = (P - self.grasp_t) @ self.grasp_R
            return ((Pg_all - self.held_pos) ** 2).sum(1) <= self.held_carry_radius ** 2

        # candidates = non-body, non-floor returns within reach of the grip,
        # expressed in the grasp frame (where the hull lives).
        cand = (~body_inside) & (P[:, 2] >= self.held_connect_min_z)
        ci = np.where(cand)[0]
        if ci.size:
            Pg = (P[ci] - self.grasp_t) @ self.grasp_R
            keep = (Pg ** 2).sum(1) <= reach2
            ci, Pg = ci[keep], Pg[keep]
        if ci.size == 0:
            return carry_fallback()

        # voxelise in the grasp frame; SEED ON THE CONTACT POINT (held_pos), not
        # the bare link origin, so the seed reaches the object for any grasp
        # offset, and guarantee at least the nearest candidate voxel.
        ijk = np.floor(Pg / v).astype(np.int64)
        keys = [(int(a), int(b), int(c)) for a, b, c in ijk]
        occupied = set(keys)
        dseed = ((Pg - self.held_pos) ** 2).sum(1)
        seed = {k for k, near in zip(keys, dseed <= self.held_connect_seed ** 2) if near}
        seed &= occupied
        if not seed:
            seed = {keys[int(np.argmin(dseed))]}

        # region-grow the 26-connected component, but ABORT to the bounded
        # fallback if it floods past a plausible held-object voxel budget -- a
        # table/shelf/wall/person at near-contact is one component with the grip
        # and would otherwise be silently carved out of the costmap.
        comp, stack, neigh = set(), list(seed), self._VOX_NEIGH.tolist()
        budget = self.held_connect_max_voxels
        while stack:
            k = stack.pop()
            if k in comp:
                continue
            comp.add(k)
            if len(comp) > budget:                          # flood -> fail safe
                return carry_fallback()
            kx, ky, kz = k
            for dx, dy, dz in neigh:
                nk = (kx + dx, ky + dy, kz + dz)
                if nk in occupied and nk not in comp:
                    stack.append(nk)

        for k in comp:                                      # refresh hull (grasp frame)
            self._held_hull[k] = self.tick
        # expire stale hull voxels (release / cross-sensor TTL).
        self._held_hull = {k: t for k, t in self._held_hull.items()
                           if self.tick - t <= self.held_connect_ttl}

        # drop candidate points whose grasp-frame voxel is in the (dilated) hull.
        held = np.zeros(n, dtype=bool)
        if self._held_hull:
            o, M = self._VOX_OFF, self._VOX_M
            hull = self._vox_dilate(np.fromiter(
                ((kx + o) + (ky + o) * M + (kz + o) * (M * M)
                 for kx, ky, kz in self._held_hull), dtype=np.int64, count=-1))
            q = self._vox_pack(np.floor(Pg / v).astype(np.int64))
            held[ci] = np.isin(q, hull)
        return held

    # --------------------------------------------------------------------- #
    def _vox_ijk(self, Pg):
        """Grasp-frame points -> integer voxel indices (N,3)."""
        return np.floor(Pg / self.held_voxel).astype(np.int64)

    def _vox_pack(self, ijk):
        """Voxel indices -> packed int64 keys (mixed-radix, all axes >= 0)."""
        o, M = self._VOX_OFF, self._VOX_M
        return (ijk[:, 0] + o) + (ijk[:, 1] + o) * M + (ijk[:, 2] + o) * (M * M)

    def _vox_unpack(self, keys):
        """Packed int64 keys -> voxel indices (K,3)."""
        o, M = self._VOX_OFF, self._VOX_M
        iz, rem = np.divmod(keys, M * M)
        iy, ix = np.divmod(rem, M)
        return np.stack([ix - o, iy - o, iz - o], axis=1)

    def _held_model_update(self, st, cand):
        """Update one sensor's grasp-frame persistence voxel model from candidates.

        Voxels seen this update are reinforced (+1, capped); voxels not seen decay
        (-1) and are dropped at 0. A rigidly-held object is seen every (moving)
        update so it climbs above ``held_min_obs``; transient world points that
        drift through the carry region never persist. The object-voxel set is
        dilated by one voxel for margin. Called only when the base is moving.
        """
        st["obs_frames"] += 1
        cnt = st["count"]
        ck = (np.unique(self._vox_pack(self._vox_ijk(cand)))
              if cand.shape[0] else np.empty(0, dtype=np.int64))
        seen = set(ck.tolist())
        for k in list(cnt.keys()):                    # decay voxels not seen now
            if k not in seen:
                v = cnt[k] - self._HELD_DEC
                if v <= 0:
                    del cnt[k]
                else:
                    cnt[k] = v
        for k in seen:                                # reinforce voxels seen now
            cnt[k] = min(self._HELD_CAP, cnt.get(k, 0) + self._HELD_INC)
        obj = np.fromiter((k for k, v in cnt.items() if v >= self.held_min_obs),
                          dtype=np.int64, count=-1)
        st["obj_keys"] = self._vox_dilate(obj)

    def _vox_dilate(self, keys):
        """Add the 26-neighbourhood to a set of packed voxel keys (1-voxel margin)."""
        if keys.size == 0:
            return keys
        ijk = self._vox_unpack(keys)                  # (K,3)
        d = (ijk[:, None, :] + self._VOX_NEIGH[None, :, :]).reshape(-1, 3)
        return np.unique(self._vox_pack(d))

    # --------------------------------------------------------------------- #
    @staticmethod
    def _cloud(stamp, frame, pts):
        msg = PointCloud2()
        msg.header.stamp = stamp
        msg.header.frame_id = frame
        n = int(pts.shape[0])
        msg.height = 1
        msg.width = n
        msg.fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * n
        msg.is_dense = True
        msg.data = np.ascontiguousarray(pts, dtype=np.float32).tobytes()
        return msg

    @staticmethod
    def _image(stamp, frame, arr, encoding):
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = frame
        msg.height = arr.shape[0]
        msg.width = arr.shape[1]
        msg.encoding = encoding
        msg.is_bigendian = 0
        if encoding == "32FC1":
            msg.step = msg.width * 4
            msg.data = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
        else:
            msg.step = msg.width * 3
            msg.data = np.ascontiguousarray(arr, dtype=np.uint8).tobytes()
        return msg

    def _caminfo(self, stamp):
        ci = CameraInfo()
        ci.header.stamp = stamp
        ci.header.frame_id = self.camera_frame
        ci.width = self.cam_w
        ci.height = self.cam_h
        ci.distortion_model = "plumb_bob"
        ci.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        ci.k = [self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0]
        ci.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ci.p = [self.fx, 0.0, self.cx, 0.0, 0.0, self.fy, self.cy, 0.0, 0.0, 0.0, 1.0, 0.0]
        return ci


def main():
    rclpy.init()
    node = G1SimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.viewer is not None:
            try:
                node.viewer.close()
            except Exception:
                pass
        if node.cv2 is not None:
            try:
                node.cv2.destroyAllWindows()
            except Exception:
                pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
