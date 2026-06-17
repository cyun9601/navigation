#!/usr/bin/env python3
"""Visualize what the LiDAR and the camera each mark as a costmap obstacle.

Headless diagnostic: loads the MuJoCo scene at the standing pose, reproduces the
exact sim sensor pipeline + the node's self-filter (g1_sim_node._self_filter_keep),
then applies the SAME thresholds nav2 uses (see config):

  LiDAR  -> /scan  : base_link z in [min_height, max_height], 2D range >= range_min
  Camera -> voxel  : world z in [min_obstacle_height, max_obstacle_height],
                     sensor range <= obstacle_max_range

It also re-checks each *kept* (obstacle) point against the simulator's
ground-truth (LiDAR geom id / camera segmentation id) to flag any SELF point
(e.g. an arm) that leaked through the filter into the obstacle set.

Run:
  MUJOCO_GL=egl python3 scripts/viz_obstacles.py
Outputs: /tmp/obstacle_viz.png  (+ stats to stdout)
"""
import math
import os
import importlib.util

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE = os.path.join(HERE, "src/g1_mujoco_sim/worlds/g1_nav_scene.xml")
NODE = os.path.join(HERE, "src/g1_mujoco_sim/g1_mujoco_sim/g1_sim_node.py")

import mujoco

# --- nav2 / p2s thresholds (keep in sync with the config files) ---
SCAN_Z_MIN, SCAN_Z_MAX = 0.10, 1.80      # pointcloud_to_laserscan min/max_height
SCAN_RANGE_MIN = 0.20                      # p2s range_min (2D)
CAM_Z_MIN, CAM_Z_MAX = 0.05, 2.0          # voxel min/max_obstacle_height
CAM_OBS_RANGE = 6.0                        # voxel obstacle_max_range
ROBOT_RADIUS, INFLATION = 0.30, 0.55      # costmap footprint / inflation

# --- sim params (must match g1_sim_node defaults) ---
STAND_H = 0.793
LIDAR_PARENT = "torso_link"
LIDAR_OFF = np.array([0.0002835, 0.00003, 0.40618])
LIDAR_VFOV = (math.radians(-85.0), math.radians(12.0))
LIDAR_CH, LIDAR_H = 16, 180
LIDAR_RMIN, LIDAR_RMAX = 0.10, 20.0
CAM_OFF = np.array([0.10, 0.0, 1.25])
CAM_PITCH = math.radians(47.6)
CAM_W, CAM_H, CAM_FOVY = 320, 240, 58.0
CAM_STRIDE, CAM_RMAX = 2, 8.0


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw])


def quat_pitch(p):
    return np.array([math.cos(p/2), 0.0, math.sin(p/2), 0.0])


def load_filter(model, data):
    """Build a bare G1SimNode carrying just the self-filter state, so we call
    the node's REAL _self_filter_keep / _resolve_ids (no ROS spin)."""
    spec = importlib.util.spec_from_file_location("g1mod", NODE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    G = mod.G1SimNode
    f = object.__new__(G)
    f.model, f.data = model, data
    lb = model.body("lidar_mount").id
    cb = model.body("camera_mount").id
    f.robot_bodies = set(b for b in range(1, model.nbody) if b not in (lb, cb))
    f.lidar_body, f.camera_body = lb, cb
    f.lidar_parent = LIDAR_PARENT

    class _L:  # silent logger
        def info(self, *a, **k):
            pass
    f.get_logger = lambda: _L()
    G._resolve_ids(f)
    f.self_scale, f.self_margin = 1.0, 0.04
    return f, G


def main():
    m = mujoco.MjModel.from_xml_path(SCENE)
    d = mujoco.MjData(m)
    flt, G = load_filter(m, d)

    # ground-truth: is geom g part of the robot body?
    geom_is_self = np.zeros(m.ngeom, bool)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in flt.robot_bodies:
            geom_is_self[g] = True

    # standing pose at origin
    bq = m.jnt_qposadr[m.joint("floating_base_joint").id]
    d.qpos[bq+2] = STAND_H
    d.qpos[bq+3] = 1.0
    mujoco.mj_kinematics(m, d)

    # ---------------- LiDAR ----------------
    lp = m.body(LIDAR_PARENT).id
    pp = d.xpos[lp].copy()
    pR = d.xmat[lp].reshape(3, 3).copy()
    origin = pp + pR @ LIDAR_OFF
    el = np.linspace(*LIDAR_VFOV, LIDAR_CH)
    az = np.linspace(-math.pi, math.pi, LIDAR_H, endpoint=False)
    AZ, EL = np.meshgrid(az, el)
    ce = np.cos(EL)
    dl = np.stack([(ce*np.cos(AZ)).ravel(), (ce*np.sin(AZ)).ravel(),
                   np.sin(EL).ravel()], 1)
    nray = dl.shape[0]
    gid = np.zeros(nray, np.int32)
    dist = np.zeros(nray)
    mujoco.mj_multiRay(m, d, origin, (dl @ pR.T).reshape(-1).copy(),
                       np.array([1, 1, 0, 1, 1, 1], np.uint8), 1, flt.lidar_body,
                       gid, dist, None, nray, LIDAR_RMAX)
    hit = (gid >= 0) & (dist >= LIDAR_RMIN) & (dist <= LIDAR_RMAX)
    lpts = (dl[hit] * dist[hit, None])
    lworld = origin + lpts @ pR.T              # world == base_link (base at origin, z=0)
    lself_gt = geom_is_self[gid[hit]]          # ground truth self
    lkeep = flt._self_filter_keep(lworld)      # node's real filter

    # ---------------- Camera ----------------
    cam_b = m.body("camera_mount").id
    cmoc = m.body_mocapid[cam_b]
    d.mocap_pos[cmoc] = CAM_OFF
    d.mocap_quat[cmoc] = quat_mul(np.array([1., 0, 0, 0]), quat_pitch(CAM_PITCH))
    mujoco.mj_kinematics(m, d)
    mujoco.mj_camlight(m, d)
    cam_id = m.camera("rgbd_cam").id
    r = mujoco.Renderer(m, height=CAM_H, width=CAM_W)
    fy = CAM_H / (2*math.tan(math.radians(CAM_FOVY)/2))
    fx = fy
    cx, cy = (CAM_W-1)/2, (CAM_H-1)/2
    r.enable_depth_rendering()
    r.update_scene(d, camera=cam_id)
    depth = r.render().astype(np.float32)
    r.disable_depth_rendering()
    st = CAM_STRIDE
    ds = depth[::st, ::st]
    h, w = ds.shape
    us = (np.arange(0, CAM_W, st)[:w] - cx)
    vs = (np.arange(0, CAM_H, st)[:h] - cy)
    UU, VV = np.meshgrid(us, vs)
    Z = ds
    valid = (Z > 0.05) & (Z < CAM_RMAX)
    cpts = np.stack([(UU*Z/fx)[valid], (VV*Z/fy)[valid], Z[valid]], 1).astype(np.float64)
    cam_pos = d.cam_xpos[cam_id].copy()
    cam_R = d.cam_xmat[cam_id].reshape(3, 3).copy()
    cworld = cam_pos + (cpts * np.array([1., -1., -1.])) @ cam_R.T
    # ground truth via segmentation
    r.enable_segmentation_rendering()
    r.update_scene(d, camera=cam_id)
    seg = r.render()[:, :, 0].astype(np.int32)
    r.disable_segmentation_rendering()
    segv = seg[::st, ::st][valid]
    cself_gt = np.zeros(cpts.shape[0], bool)
    good = segv >= 0
    cself_gt[good] = geom_is_self[segv[good]]
    ckeep = flt._self_filter_keep(cworld)
    crange = np.linalg.norm(cworld - cam_pos, axis=1)

    # ---------------- apply nav2 bands ----------------
    lr2d = np.hypot(lworld[:, 0], lworld[:, 1])
    l_band = (lworld[:, 2] >= SCAN_Z_MIN) & (lworld[:, 2] <= SCAN_Z_MAX) & (lr2d >= SCAN_RANGE_MIN)
    l_obs = lkeep & l_band                       # LiDAR -> scan obstacle
    c_band = (cworld[:, 2] >= CAM_Z_MIN) & (cworld[:, 2] <= CAM_Z_MAX) & (crange <= CAM_OBS_RANGE)
    c_obs = ckeep & c_band                       # camera -> voxel obstacle

    # leaks: obstacle points that are actually the robot's own body
    l_leak = l_obs & lself_gt
    c_leak = c_obs & cself_gt

    print("================= OBSTACLE / SELF-LEAK STATS =================")
    print(f"LiDAR : raw_hits={hit.sum()}  self(GT)={lself_gt.sum()}  "
          f"removed_by_filter={(~lkeep).sum()}  -> scan_obstacles={l_obs.sum()}")
    print(f"        SELF LEAK into scan band = {l_leak.sum()} pts "
          f"{'  <-- arm/body marked as obstacle!' if l_leak.sum() else '(none)'}")
    if l_leak.sum():
        lk = lworld[l_leak]
        print(f"          leak xy-range x[{lk[:,0].min():.2f},{lk[:,0].max():.2f}] "
              f"y[{lk[:,1].min():.2f},{lk[:,1].max():.2f}] z[{lk[:,2].min():.2f},{lk[:,2].max():.2f}]")
    print(f"Camera: raw_pts={cpts.shape[0]}  self(GT)={cself_gt.sum()}  "
          f"removed_by_filter={(~ckeep).sum()}  -> voxel_obstacles={c_obs.sum()}")
    print(f"        SELF LEAK into voxel band = {c_leak.sum()} pts "
          f"{'  <-- arm/body marked as obstacle!' if c_leak.sum() else '(none)'}")
    if c_leak.sum():
        ck = cworld[c_leak]
        print(f"          leak xy-range x[{ck[:,0].min():.2f},{ck[:,0].max():.2f}] "
              f"y[{ck[:,1].min():.2f},{ck[:,1].max():.2f}] z[{ck[:,2].min():.2f},{ck[:,2].max():.2f}]")
    # how many obstacles fall inside footprint+inflation in front (blocks forward)
    def near_front(wp, obs):
        rr = np.hypot(wp[obs, 0], wp[obs, 1])
        return int(((rr <= ROBOT_RADIUS + INFLATION) & (wp[obs, 0] > -0.1)).sum())
    print(f"obstacles within (radius+inflation={ROBOT_RADIUS+INFLATION:.2f}m) & front: "
          f"lidar={near_front(lworld,l_obs)}  camera={near_front(cworld,c_obs)}")
    print("=============================================================")

    # ---------------- plots ----------------
    fig, ax = plt.subplots(2, 2, figsize=(15, 14))

    def draw_footprint(a):
        a.add_patch(Circle((0, 0), ROBOT_RADIUS, fill=False, ec="k", lw=2, label="footprint 0.30"))
        a.add_patch(Circle((0, 0), ROBOT_RADIUS+INFLATION, fill=False, ec="orange",
                           ls="--", lw=1.5, label="inflation 0.85"))
        a.plot(0, 0, "k+", ms=14)

    # LiDAR top-down
    a = ax[0, 0]
    rem = ~lkeep
    a.scatter(lworld[rem, 0], lworld[rem, 1], s=4, c="lightgray", label="removed (self-filtered)")
    a.scatter(lworld[l_obs, 0], lworld[l_obs, 1], s=6, c="tab:red", label="OBSTACLE -> /scan")
    if l_leak.sum():
        a.scatter(lworld[l_leak, 0], lworld[l_leak, 1], s=40, c="magenta",
                  marker="x", label="SELF LEAK")
    draw_footprint(a)
    a.set_title(f"LiDAR top-down (XY, base frame)  obstacles={l_obs.sum()}")
    a.set_xlabel("x fwd [m]")
    a.set_ylabel("y left [m]")
    a.set_xlim(-3, 5)
    a.set_ylim(-4, 4)
    a.set_aspect("equal")
    a.legend(loc="upper right", fontsize=8)
    a.grid(alpha=0.3)

    # LiDAR side
    a = ax[0, 1]
    a.scatter(lworld[rem, 0], lworld[rem, 2], s=4, c="lightgray")
    a.scatter(lworld[l_obs, 0], lworld[l_obs, 2], s=6, c="tab:red")
    if l_leak.sum():
        a.scatter(lworld[l_leak, 0], lworld[l_leak, 2], s=40, c="magenta", marker="x")
    a.axhline(SCAN_Z_MIN, color="b", ls=":", label=f"scan z band [{SCAN_Z_MIN},{SCAN_Z_MAX}]")
    a.axhline(SCAN_Z_MAX, color="b", ls=":")
    a.set_title("LiDAR side (XZ)")
    a.set_xlabel("x fwd [m]")
    a.set_ylabel("z up [m]")
    a.set_xlim(-3, 5)
    a.set_ylim(0, 2.2)
    a.legend(loc="upper right", fontsize=8)
    a.grid(alpha=0.3)

    # Camera top-down
    a = ax[1, 0]
    rem = ~ckeep
    a.scatter(cworld[rem, 0], cworld[rem, 1], s=2, c="lightgray", label="removed (self-filtered)")
    a.scatter(cworld[c_obs, 0], cworld[c_obs, 1], s=3, c="tab:green", label="OBSTACLE -> voxel")
    if c_leak.sum():
        a.scatter(cworld[c_leak, 0], cworld[c_leak, 1], s=40, c="magenta",
                  marker="x", label="SELF LEAK")
    draw_footprint(a)
    a.set_title(f"Camera top-down (XY)  obstacles={c_obs.sum()}  (NOTE: no ground removal)")
    a.set_xlabel("x fwd [m]")
    a.set_ylabel("y left [m]")
    a.set_xlim(-1, 5)
    a.set_ylim(-3, 3)
    a.set_aspect("equal")
    a.legend(loc="upper right", fontsize=8)
    a.grid(alpha=0.3)

    # Camera side
    a = ax[1, 1]
    a.scatter(cworld[rem, 0], cworld[rem, 2], s=2, c="lightgray")
    a.scatter(cworld[c_obs, 0], cworld[c_obs, 2], s=3, c="tab:green")
    if c_leak.sum():
        a.scatter(cworld[c_leak, 0], cworld[c_leak, 2], s=40, c="magenta", marker="x")
    a.axhline(CAM_Z_MIN, color="b", ls=":", label=f"voxel z band [{CAM_Z_MIN},{CAM_Z_MAX}]")
    a.axhline(CAM_Z_MAX, color="b", ls=":")
    a.set_title("Camera side (XZ)")
    a.set_xlabel("x fwd [m]")
    a.set_ylabel("z up [m]")
    a.set_xlim(-1, 5)
    a.set_ylim(0, 2.2)
    a.legend(loc="upper right", fontsize=8)
    a.grid(alpha=0.3)

    fig.suptitle("What each sensor marks as a costmap obstacle (standing pose, self-filter ON)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    out = "/tmp/obstacle_viz.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
