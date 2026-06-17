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


def mat2quat(R):
    """3x3 rotation matrix -> quaternion (w, x, y, z)."""
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


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

        self.publish_self_filtered = bool(p("publish_self_filtered").value)
        self.publish_truth = bool(p("publish_truth").value)
        self.self_margin = float(p("self_filter_margin").value)
        self.self_scale = float(p("self_filter_scale").value)

        if not self.scene_xml:
            raise RuntimeError("parameter 'scene_xml' is required")

        # ---- load model ----
        self.get_logger().info(f"loading MuJoCo scene: {self.scene_xml}")
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
        if self.enable_camera:
            self.pub_rgb = self.create_publisher(Image, "camera/color/image_raw", sensor_qos)
            self.pub_depth = self.create_publisher(Image, "camera/depth/image_raw", sensor_qos)
            self.pub_caminfo = self.create_publisher(CameraInfo, "camera/camera_info", sensor_qos)
            self.pub_cam_raw = self.create_publisher(PointCloud2, "camera/depth/points", sensor_qos)
            self.pub_cam_filt = self.create_publisher(PointCloud2, "camera/points_self_filtered", sensor_qos)
            if self.publish_truth:
                self.pub_cam_self_gt = self.create_publisher(PointCloud2, "camera/points_self_gt", sensor_qos)

        self.sub_cmd = self.create_subscription(Twist, "cmd_vel", self._cmd_cb, 10)

        self.tick = 0
        self.timer = self.create_timer(self.dt, self._update)
        self.get_logger().info("G1 MuJoCo sim node ready.")

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

        # real robot link the LiDAR is mounted on
        self.lidar_parent_id = m.body(self.lidar_parent).id

        # camera id for rendering
        self.cam_id = m.camera("rgbd_cam").id

        # robot bodies (everything except world + the two mocap mounts) for self filter
        self.robot_bodies = set()
        for b in range(1, m.nbody):
            if b in (self.lidar_body, self.camera_body):
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
        self.get_logger().info(
            f"self-filter: {coll.size} collision geoms "
            f"({self.self_sphere_ids.size} spheres + {len(self.self_local_ids)} other)")

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
    def _apply_kinematics(self):
        """Write base pose + mocap sensor poses, then run forward kinematics."""
        d, m = self.data, self.model
        # floating base qpos = [x, y, z, qw, qx, qy, qz]
        a = self.base_qadr
        d.qpos[a + 0] = self.base_x
        d.qpos[a + 1] = self.base_y
        d.qpos[a + 2] = self.stand_h
        qb = quat_yaw(self.base_yaw)
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

        # camera mount: base-relative, orientation = yaw then pitch-down about Y
        c, s = math.cos(self.base_yaw), math.sin(self.base_yaw)
        cx_ = self.base_x + c * self.camera_off[0] - s * self.camera_off[1]
        cy_ = self.base_y + s * self.camera_off[0] + c * self.camera_off[1]
        d.mocap_pos[self.camera_mocap] = [cx_, cy_, self.camera_off[2]]
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
            gt_self = self.geom_is_self[self._ray_geomid[hit]]
            self.pub_lidar_self_gt.publish(
                self._cloud(header_stamp, self.lidar_frame, pts[gt_self]))
        if self.publish_self_filtered:
            # world points = origin + R @ local; test against the body model.
            world_pts = self.lidar_origin + pts.astype(np.float64) @ self.lidar_R.T
            keep = self._self_filter_keep(world_pts)
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

        if self.publish_self_filtered:
            # optical (x right, y down, z fwd) -> MuJoCo cam (x right, y up, z back)
            cam_pos = self.data.cam_xpos[self.cam_id]
            cam_R = self.data.cam_xmat[self.cam_id].reshape(3, 3)
            p_cam = pts.astype(np.float64) * np.array([1.0, -1.0, -1.0])
            world_pts = cam_pos + p_cam @ cam_R.T
            keep = self._self_filter_keep(world_pts)
            self.pub_cam_filt.publish(self._cloud(stamp, self.camera_frame, pts[keep]))

    # --------------------------------------------------------------------- #
    def _self_filter_keep(self, world_pts):
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
        return ~inside

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
