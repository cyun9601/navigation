#!/usr/bin/env python3
"""Live comparison window: ground-truth self-filtering vs the logic self-filter.

Pops up a matplotlib window that, for each sensor, classifies every *raw* point
by comparing two things published by the running pipeline:

  * ground truth  -- is this point the robot's own body?  Taken from MuJoCo geom
    / segmentation ids, published by g1_sim_node on  .../points_self_gt.
  * the logic filter's output (.../points_self_filtered by default) -- did the
    running geometric self-filter keep this point?  Override via the
    lidar_filtered_topic / camera_filtered_topic params to point at any other
    filter's output (e.g. robot_body_filter's .../points_filtered).

Per raw point (matched across the three clouds by coordinate, all are in the
same sensor frame so coordinates are preserved):

  green   kept   & not body  ->  correct keep   (real obstacle / world)
  gray    removed & body      ->  correct removal
  red     kept   & body       ->  LEAK           (robot body left in as obstacle)
  purple  removed & not body  ->  over-removal   (real point wrongly deleted)

So red + purple together are exactly where the logic filter disagrees with the
ground-truth filter. Counts are shown in each subplot title.

Run (usually launched by bringup.launch.py with use_filter_viz:=true):
  ros2 run g1_mujoco_sim self_filter_viz_node
"""
import threading

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import message_filters
from sensor_msgs.msg import PointCloud2


def pc2_xyz(msg):
    """Extract an (N,3) float32 array of xyz from a PointCloud2 (any layout)."""
    n = msg.width * msg.height
    if n == 0:
        return np.zeros((0, 3), np.float32)
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8).reshape(n, msg.point_step)
    off = {f.name: f.offset for f in msg.fields}

    def col(name):
        o = off[name]
        return raw[:, o:o + 4].copy().view(np.float32).reshape(-1)

    return np.stack([col("x"), col("y"), col("z")], axis=1)


def _keyset(xyz, q=1.0e4):
    """Set of quantized integer-coordinate tuples (0.1 mm grid)."""
    if xyz.shape[0] == 0:
        return set()
    return set(map(tuple, np.round(xyz * q).astype(np.int64)))


def _member(xyz, keyset, q=1.0e4):
    """Boolean mask: is each row of xyz present in keyset?"""
    if xyz.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    k = np.round(xyz * q).astype(np.int64)
    return np.fromiter((tuple(r) in keyset for r in k), dtype=bool, count=k.shape[0])


class FilterCompareViz(Node):
    """Synchronizes raw / ground-truth-self / filtered clouds and classifies them."""

    def __init__(self):
        super().__init__("self_filter_viz")
        self.declare_parameter("enable_camera", True)
        # output of the filter under test; defaults to the built-in self-filter.
        self.declare_parameter("lidar_filtered_topic", "/lidar/points_self_filtered")
        self.declare_parameter("camera_filtered_topic", "/camera/points_self_filtered")
        enable_camera = bool(self.get_parameter("enable_camera").value)
        lidar_filt = self.get_parameter("lidar_filtered_topic").value
        camera_filt = self.get_parameter("camera_filtered_topic").value

        self.lock = threading.Lock()
        self.snap = {}      # sensor name -> classified arrays + counts
        self._syncs = []    # keep ApproximateTimeSynchronizer refs alive

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)

        self.sensors = [
            ("lidar", "/lidar/points_raw", "/lidar/points_self_gt", lidar_filt),
        ]
        if enable_camera:
            self.sensors.append(
                ("camera", "/camera/depth/points", "/camera/points_self_gt",
                 camera_filt))

        for name, raw_t, gt_t, filt_t in self.sensors:
            subs = [message_filters.Subscriber(self, PointCloud2, t, qos_profile=qos)
                    for t in (raw_t, gt_t, filt_t)]
            sync = message_filters.ApproximateTimeSynchronizer(
                subs, queue_size=10, slop=0.2)
            sync.registerCallback(
                lambda r, g, f, n=name: self._on_clouds(n, r, g, f))
            self._syncs.append(sync)

        self.get_logger().info(
            "self_filter_viz: comparing ground-truth self vs logic self-filter "
            f"for {[(s[0], s[3]) for s in self.sensors]}")

    def _on_clouds(self, name, raw_msg, gt_msg, filt_msg):
        raw = pc2_xyz(raw_msg)
        gt = pc2_xyz(gt_msg)
        filt = pc2_xyz(filt_msg)

        is_self = _member(raw, _keyset(gt))     # ground truth: point is robot body
        is_kept = _member(raw, _keyset(filt))   # robot_body_filter kept this point

        leak = is_self & is_kept                 # body left in   -> red
        ok_rm = is_self & ~is_kept               # body removed    -> gray
        ok_keep = ~is_self & is_kept             # world kept      -> green
        over = ~is_self & ~is_kept               # world removed   -> purple

        with self.lock:
            self.snap[name] = {
                "raw": raw,
                "leak": leak, "ok_rm": ok_rm, "ok_keep": ok_keep, "over": over,
                "n_raw": int(raw.shape[0]), "n_self": int(is_self.sum()),
                "n_leak": int(leak.sum()), "n_over": int(over.sum()),
            }


def _run_gui(node):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    names = [s[0] for s in node.sensors]
    nrows = len(names)
    fig, axes = plt.subplots(nrows, 2, figsize=(13, 6 * nrows), squeeze=False)
    fig.suptitle("Ground-truth self-filter  vs  logic self-filter   "
                 "(red = LEAK, purple = over-removal)", fontsize=13)

    def update(_frame):
        with node.lock:
            snap = {k: dict(v) for k, v in node.snap.items()}
        for row, name in enumerate(names):
            ax_xy, ax_xz = axes[row][0], axes[row][1]
            ax_xy.clear()
            ax_xz.clear()
            s = snap.get(name)
            if s is None:
                ax_xy.set_title(f"{name}: waiting for data...")
                ax_xy.text(0.5, 0.5, "no synchronized clouds yet",
                           ha="center", va="center", transform=ax_xy.transAxes)
                continue
            raw = s["raw"]
            layers = [
                ("ok_rm", "lightgray", 3, "removed & body (ok)"),
                ("ok_keep", "tab:green", 4, "kept & world (ok)"),
                ("over", "tab:purple", 14, "OVER-REMOVED (world deleted)"),
                ("leak", "tab:red", 18, "LEAK (body kept as obstacle)"),
            ]
            for key, color, size, label in layers:
                m = s[key]
                if not m.any():
                    continue
                p = raw[m]
                ax_xy.scatter(p[:, 0], p[:, 1], s=size, c=color, label=label)
                ax_xz.scatter(p[:, 0], p[:, 2], s=size, c=color)
            ax_xy.set_title(
                f"{name}  raw={s['n_raw']}  body(GT)={s['n_self']}  "
                f"LEAK={s['n_leak']}  over-rm={s['n_over']}")
            ax_xy.set_xlabel("x [m]")
            ax_xy.set_ylabel("y [m]")
            ax_xy.set_aspect("equal")
            ax_xy.grid(alpha=0.3)
            ax_xy.legend(loc="upper right", fontsize=7)
            ax_xz.set_title(f"{name}  (side: x vs z)")
            ax_xz.set_xlabel("x [m]")
            ax_xz.set_ylabel("z [m]")
            ax_xz.grid(alpha=0.3)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

    # keep a ref so the animation isn't garbage-collected
    node._ani = FuncAnimation(fig, update, interval=300, cache_frame_data=False)
    plt.show()


def main():
    rclpy.init()
    node = FilterCompareViz()
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    try:
        _run_gui(node)
    except Exception as e:  # no display, backend issue, etc.
        node.get_logger().error(f"GUI unavailable ({e}); spinning headless.")
        spin.join()
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
