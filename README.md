# g1_nav — Unitree G1 obstacle-avoidance navigation in MuJoCo with nav2

Simulate a Unitree G1 as a **planar mobile base** in MuJoCo, expose **3D LiDAR +
RGBD** sensors to ROS 2 with a **self-filter** (the robot does not see its own
body as an obstacle), and drive it with **nav2** for obstacle-avoidance
navigation. SLAM (`slam_toolbox`) builds the map online.

```
nav2 ──/cmd_vel──▶ g1_sim_node (MuJoCo, planar base)
                       │  raycast (mj_multiRay)        render (depth)
                       ├─ /lidar/points_raw ───────────┐
                       ├─ /lidar/points_self_filtered ──┤ self-filter (URDF body model + FK)
                       ├─ /camera/depth/points          │
                       ├─ /camera/points_self_filtered ─┘
                       ├─ /odom + TF odom→base_link
                       └─ /joint_states → robot_state_publisher → body/sensor TF
   /lidar/points_self_filtered ─▶ pointcloud_to_laserscan ─▶ /scan ─▶ nav2 + slam_toolbox
   /camera/points_self_filtered ─────────────────────────────────────▶ nav2 voxel layer
```

## Packages
- **g1_description** — G1 29-DoF URDF + meshes, plus a navigation `base_link`,
  `base_footprint`, `lidar_link`, and `camera_link`/`camera_depth_optical_frame`.
- **g1_mujoco_sim** — `g1_sim_node`, the MuJoCo↔ROS2 bridge, and the MuJoCo scene
  (`worlds/g1_nav_scene.xml`: flat ground + box/cylinder obstacles + walls).
- **g1_nav2_bringup** — nav2 params, slam_toolbox config, pointcloud_to_laserscan
  config, robot_body_filter configs (optional), launch files, RViz config.

## Build
```bash
cd ~/g1_nav
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Run
Everything (sim + sensors + self-filter + SLAM + nav2 + RViz):
```bash
ros2 launch g1_nav2_bringup bringup.launch.py
```
Then in RViz use **2D Goal Pose** to send a navigation goal. The G1 plans and
drives to it while avoiding obstacles.

Launch args: `use_viewer:=true|false` (MuJoCo window), `use_rviz:=true|false`,
`use_body_filter:=true|false` (see Self-filter below).

Sim + sensors only (no nav2), e.g. for teleop / sensor inspection:
```bash
ros2 launch g1_mujoco_sim sim.launch.py
ros2 run teleop_twist_keyboard teleop_twist_keyboard   # publishes /cmd_vel
```

## Locomotion model
The G1 is **not** walking. It is held in a fixed standing pose and its floating
base `(x, y, yaw)` is integrated from `/cmd_vel`; kinematics are updated with
`mj_kinematics` (no dynamics → it never falls). This isolates nav2 obstacle
avoidance from the (separate, hard) bipedal-control problem. To add a real
walking controller later, feed `/cmd_vel` to it and disable the planar
integrator in `g1_sim_node`.

## Sensors
- **3D LiDAR** (Livox-Mid-360-style): mounted on the **real `torso_link`** (the
  body that carries the LiDAR on the actual Unitree G1) via `lidar_parent` +
  `lidar_offset` = `torso_link + (0.0003, 0, 0.406)` → world ≈ 1.25 m, matching
  the official Unitree G1 `mid360_link` mount on the head/neck. Simulated by
  `mj_multiRay`, 16 rings × 180 az over a wide downward FOV `[-85°, +12°]` so it
  also sees the robot's own shoulders/arms/torso → `/lidar/points_raw`. The
  ray origin/orientation track the torso link's actual pose each tick.
- **RGBD** (RealSense-D455-style on the head, pitched down): MuJoCo renderer
  produces `/camera/color/image_raw`, `/camera/depth/image_raw`,
  `/camera/camera_info`, and `/camera/depth/points`.

All key sizes/offsets are `g1_sim_node` ROS params. **If you change a sensor
offset, change it in BOTH `g1_sim_node` params and the `g1_description` URDF** —
they must agree (the sim computes sensor world pose from the same offsets that
the URDF/robot_state_publisher use for TF).

## Self-filter (the part that makes the robot not detect itself)
Default = **built-in point-in-shape body-model self-filter** in `g1_sim_node`,
modelled on the standard ROS packages `robot_self_filter` / `robot_body_filter`.
It uses only what a real robot actually has — the robot's own **collision**
geometry (URDF) and its current joint configuration — so the method transfers
directly to hardware:

- The body model is **all robot geoms** (collision + visual), each placed at its
  current **forward-kinematics pose** — the same link poses TF publishes from
  `/joint_states` on a real robot. `robot_self_filter` uses `<collision>` shapes
  only, but here the sensors strike the **visual** meshes (the LiDAR raycast and
  the camera render both hit visual geometry) and the G1 collision model does not
  fully cover them — the wrist/hand collision shape is smaller than the rendered
  rubber-hand mesh, and the waist links have no collision geom — so collision-only
  filtering leaks those surfaces and marks them as obstacles in front of the
  robot. Filtering against the full visual geometry is exactly what the sensors
  can observe. (On real hardware you would instead use the collision model or a
  dedicated self-filter mesh.)
- A sensor point is dropped if it falls **inside any** collision shape, tested
  with that shape's exact primitive: sphere by centre distance; cylinder /
  capsule / box / ellipsoid in the geom's local frame; collision **meshes** by
  their oriented bounding box (tight `geom_size` half-extents, not a loose
  bounding sphere). Shapes are inflated by `self_filter_scale` (multiplicative)
  and `self_filter_margin` (additive padding, default 0.04 m) — the analogues of
  `self_see_default_scale` / `self_see_default_padding`.
- LiDAR points are transformed to the world frame from the mount pose; depth
  points from the camera's optical→world pose, then tested the same way.

It deliberately does **not** use which MuJoCo geom a ray hit or per-pixel
segmentation ids — that ground-truth is a simulator-only oracle with no real
sensor equivalent. Output: `/lidar/points_self_filtered`,
`/camera/points_self_filtered`. These feed `/scan` and the nav2 costmaps.

(Validated against the simulator's geom-id ground truth: in the standing pose
the containment filter removes exactly the same LiDAR self-points as the oracle
— 195/195, no environment points over-removed — at ≈2 ms/scan.)

**Not implemented (referenced extensions):** `robot_body_filter` also adds a
*shadow test* — removing veiling points behind the body along the sensor ray;
our single-hit raycast never produces such points in sim, so containment
suffices. For depth cameras the render-based filters (`rgbd_self_filter`,
`realtime_urdf_filter`) instead rasterise the URDF from the camera view and cut
pixels at/behind the rendered self-depth; that is the real-sensor equivalent of
the segmentation oracle we removed, and a GPU-dependent alternative to the
containment test used here.

Why it matters: the LiDAR's downward rings hit the robot's own torso/arms
(~400–1500 points), all within the `/scan` height band. **Without the filter
those points mark the robot's own footprint as a lethal obstacle and it cannot
move.** With it, the body is removed and navigation works (verified: a goal at
(4.5, 2.2) succeeded, threading through the obstacle field).

Optional standard ROS self-filter (`robot_body_filter`, URDF-collision based):
```bash
cd ~/g1_nav/src && git clone https://github.com/peci1/robot_body_filter.git
cd ~/g1_nav && colcon build && source install/setup.bash
ros2 launch g1_nav2_bringup bringup.launch.py use_body_filter:=true
```
This routes `/lidar/points_raw`→`/lidar/points_filtered` and
`/camera/depth/points`→`/camera/points_filtered` through `robot_body_filter`
instead of the built-in filter. (Configs in `g1_nav2_bringup/config/`. Not built
here — offline; the built-in filter is the verified default.)

## Verified
- All sensor topics publish; `/lidar/points_raw` (≈2340 pts) vs
  `/lidar/points_self_filtered` (≈1725 pts) → ≈615 self-points removed.
- `/cmd_vel` drives the planar base; `/odom` + TF (`odom→base_link`,
  `torso_link→lidar_link` = (0.0003,0,0.406)) correct.
- Full stack: SLAM `/map`, `map→odom`, costmaps, all nav2 lifecycle nodes
  `active`; a `NavigateToPose` goal **SUCCEEDED** avoiding obstacles.

## Tuning notes
- LiDAR raycast against the detailed robot meshes is the main per-scan cost
  (~32 ms for 2880 rays). Increase `lidar_h_samples`/`lidar_channels` for density
  at the cost of rate, or lower them for speed. Current rates ≈ odom 30 Hz,
  scan/lidar ≈ 6 Hz, camera ≈ 3 Hz.
- Obstacles live in `g1_mujoco_sim/worlds/g1_nav_scene.xml` — edit freely.
