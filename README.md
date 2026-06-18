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

## Holding an object (payload self-filter — works for unknown shapes)
When the robot grasps something, that object is **not** in the URDF, so the
LiDAR/camera strike it and nav2 marks it as an obstacle ~0.3 m in front of the
robot — the robot **freezes while carrying a load**. The filter removes it.
You do **not** need to know the object's shape in advance. There is a spectrum
from full prior knowledge to none, selected by `held_filter_mode`:

| mode | prior knowledge | how | trade-off |
|------|-----------------|-----|-----------|
| `connected` (**default**) | **none** | remove the cloud region spatially **connected to the gripper** (region-grow), shared across sensors as a grasp-frame voxel hull | **size-invariant** (a broom is removed end-to-end, no dead-zone) and works **while stationary**; needs a dense-enough cloud (the depth camera) |
| `carry_volume` | **none** | drop everything in a fixed sphere attached to the hand | trivial & robust, but over-removes a fixed dead-zone and **breaks per object size** |
| `online` | **none** | estimate the payload's occupied voxels in the grasp frame, learning only while the base moves | recovers free space, any shape — but needs motion and can leak when still |
| `shape` | object shape known (grasp DB) | a primitive rigidly fixed to the grasp link — MoveIt *attached collision object* / `robot_body_filter` | tightest & motion-independent, but needs a model per object |

**Why `connected` is the default.** A fixed sphere (`carry_volume`) is the thing
you flagged — its behaviour depends entirely on the object: too small and a
broom's far end is left as a phantom obstacle; too big and it blinds a useless
dead-zone. Connectivity sidesteps size completely: the held object physically
touches the hand, so its returns form one contiguous component with the gripper.
We voxelise the non-body returns in the **grasp frame** (`FK(grasp link)` from
joint encoders → TF, same on hardware), seed at the hand, region-grow the
26-connected component, and drop it — **any size/shape, no model, no training**,
and because connectivity is a *per-frame geometric* property it works **while the
robot is still** (no motion needed, unlike `online`). A separate obstacle is a
different component and survives.

> **Two catches we found (adversarial review), and the fixes.**
> 1. **Sparse LiDAR fragments the object.** A single 16-ring scan is too sparse at
>    close range (~5–7 cm between rings), so the object breaks into disconnected
>    voxels and leaks. Fix: `connected` builds the component into a **shared
>    grasp-frame voxel hull**, **fused across sensors** and kept for
>    `held_connect_ttl` frames — the **dense depth camera** forms a complete hull
>    that then also clears the sparse **LiDAR** (the cloud feeding `/scan`); TTL
>    clears it on release. → *LiDAR-only, no camera, `connected` is unreliable —
>    use `carry_volume`.*
> 2. **Unbounded growth could carve out a real surface.** A table/wall/person at
>    near-contact is one connected component with the gripper, so naive region-grow
>    would flood and delete it — a hole **bigger and more dangerous** than the
>    sphere. Fix: growth is **capped by a voxel budget** (`held_connect_max_voxels`)
>    and **fails safe to `carry_volume`** (a bounded sphere) the moment a component
>    floods past a plausible payload size; the seed is centered on the **contact
>    point** (not the bare link origin) with a guaranteed nearest-voxel seed; floor
>    returns (`held_connect_min_z`) and the robot body are excluded so they can't
>    bridge. Net: `connected` is **never worse than `carry_volume`** in the
>    dangerous direction — verified, a 2 m wall touching the box drops from 2874 →
>    309 points removed (just the bounded sphere). (A SAM / segmentation model —
>    idea #2 — is the camera-only, GPU-heavy alternative noted under *Self-filter*;
>    connectivity gets the same per-frame segmentation on the camera with no net.)

The demo carries a box held in both hands (`held_object` body in
`g1_nav_scene.xml`, posed each tick by `g1_sim_node` to track
`right_wrist_yaw_link`). Toggle to see the problem, the fix, and the modes:
```bash
# default: prior-free connectivity removes the payload (any size) -> robot navigates
ros2 launch g1_nav2_bringup bringup.launch.py
# payload NOT removed -> robot sees its own load as an obstacle and won't move
ros2 launch g1_nav2_bringup bringup.launch.py filter_held_object:=false
# other modes
ros2 launch g1_nav2_bringup bringup.launch.py held_filter_mode:=carry_volume
ros2 launch g1_nav2_bringup bringup.launch.py held_filter_mode:=online   # needs motion
ros2 launch g1_nav2_bringup bringup.launch.py held_filter_mode:=shape    # known primitive
# carry nothing at all
ros2 launch g1_nav2_bringup bringup.launch.py hold_object:=false
```
Params (`g1_sim_node`): `held_object_parent` (grasp link), `held_object_pos`
(in-hand grasp point), connectivity: `held_connect_voxel`, `held_connect_seed`,
`held_connect_max_reach`, `held_connect_max_voxels` (flood budget),
`held_connect_min_z`, `held_connect_ttl`; sphere: `held_carry_radius`; online:
`held_voxel`/`held_min_obs`/`held_move_eps`. The `held_object_geom` in the scene
only sets what the robot *physically carries* — in the prior-free modes the
filter never reads it.

**Validated** against the simulator geom-id ground truth, for box / cylinder /
sphere / a long 0.8 m plank the filter was **never told about**, while the robot
is **stationary**:

| scenario | mode | payload removed | leaked | real world over-removed |
|----------|------|-----------------|--------|-------------------------|
| camera, box / plank | `connected` | 100 % | 0 | ~0 |
| **LiDAR via camera-built hull**, box / plank | `connected` | **100 %** (75/75, 44/44) | **0** | **0** |
| separated obstacle inside old 0.32 m sphere | `connected` | — | — | obstacle **KEPT** (25/25) ✅ |
| **2 m wall touching the box** (flood test) | `connected` | (box removed) | — | wall **309/2948** only (fail-safe to sphere; was 2874 unbounded) ✅ |
| LiDAR-only, no camera | `connected` | partial (fragments) | leaks\* | 0 |
| long plank | `carry_volume` | only the near 0.32 m | **far end leaks** | 0 |

Cost ≈ 5 ms/scan (LiDAR), ≈45 ms (camera, 19 k pts, dominated by the existing body
containment stage; connectivity itself is a small BFS bounded by the voxel
budget). \*The sparse-LiDAR leak is the safe direction (over-cautious, never
blinds the robot to a real obstacle); it is why the camera hull is fused in, and
why `carry_volume` is the recommended fallback when no depth camera is available.

**Known limitations / future work** (from the design reviews): connectivity needs
a dense cloud (camera or motion-accumulated LiDAR); when a payload genuinely
touches a wall the bounded fail-safe removes only the sphere, so the wall stays an
obstacle (correct) but the payload's far part leaks until separated; the
`held_connect_max_voxels` budget trades off "largest carriable object" vs "wall
flood" — a wardrobe-sized load would trip the fail-safe; ground handling is a flat
world-z cut (a RANSAC ground plane / height-above-base would be robust on ramps
and would let low payload parts be removed); capsule carry gates / two-hand grasp
frames for elongated bimanual loads; per-sensor voxel size (coarser for LiDAR);
the `online` mode's time-based decay and world-stationarity test.

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
