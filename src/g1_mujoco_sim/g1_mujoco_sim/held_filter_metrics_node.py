#!/usr/bin/env python3
"""Quantitative scorer for the held-object (payload) self-filter.

While `self_filter_viz_node` shows the filter result as a coloured scatter, this
node turns it into numbers, so you can compare how well the prior-free filter
handles different carried shapes (`payload:=box|sphere|cylinder|pole|board|
lshape`). Headless -- no GUI -- so it also runs over SSH / in CI.

For each sensor it synchronises four clouds the sim publishes (all in the sensor
frame, so points match by coordinate):

  raw       /<sensor>/points_raw            every return
  self_gt   /<sensor>/points_self_gt        returns on the robot body OR payload
  held_gt   /<sensor>/points_held_gt        returns on the PAYLOAD only
  filtered  /<sensor>/points_self_filtered  what the filter kept (the obstacles)

and reports, accumulated since start (after a short warm-up):

  payload leak   = payload returns still KEPT / payload returns total
                   -> the carried object left in as a phantom obstacle. Want 0.
  over-removal   = world returns REMOVED / world returns total
                   -> real obstacles wrongly deleted by the filter. Want ~0.
  body leak      = body returns still kept / body returns total (the body
                   self-filter; reported for context).

A run PASSes when payload-leak <= max_payload_leak and over-removal <=
max_over_removal (both fractions, set as params).

Run (usually launched by bringup.launch.py with use_filter_metrics:=true):
  ros2 run g1_mujoco_sim held_filter_metrics_node --ros-args -p payload:=pole
"""
import threading
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import message_filters
from sensor_msgs.msg import PointCloud2

from g1_mujoco_sim.self_filter_viz_node import pc2_xyz, _keyset, _member


def _pct(num, den):
    """num/den as a percentage, or None when there is nothing to divide."""
    return (100.0 * num / den) if den > 0 else None


def _fmt_pct(p):
    return "  n/a" if p is None else f"{p:5.2f}%"


class HeldFilterMetrics(Node):
    """Synchronises raw / self-GT / payload-GT / filtered clouds and scores the
    payload filter (leak + over-removal), accumulated per sensor."""

    def __init__(self):
        super().__init__("held_filter_metrics")
        self.declare_parameter("enable_camera", True)
        self.declare_parameter("payload", "")            # label only (the preset)
        self.declare_parameter("lidar_filtered_topic", "/lidar/points_self_filtered")
        self.declare_parameter("camera_filtered_topic", "/camera/points_self_filtered")
        self.declare_parameter("report_period", 3.0)     # seconds between reports
        # Skip the first N frames per sensor before counting (clears the brief boot
        # transient while the cross-sensor hull settles).
        self.declare_parameter("warmup_frames", 8)
        # Score over a SLIDING WINDOW of the most recent frames, not a cumulative
        # average since start: a cumulative mean never forgets an early transient
        # spike, so it reports a steady filter as permanently degraded. The window
        # tracks current steady-state behaviour (and recovers if conditions change).
        self.declare_parameter("window_frames", 40)
        self.declare_parameter("max_payload_leak", 0.05)  # PASS threshold (fraction)
        self.declare_parameter("max_over_removal", 0.02)  # PASS threshold (fraction)

        gp = self.get_parameter
        enable_camera = bool(gp("enable_camera").value)
        self.payload = str(gp("payload").value) or "(scene)"
        lidar_filt = gp("lidar_filtered_topic").value
        camera_filt = gp("camera_filtered_topic").value
        self.warmup = int(gp("warmup_frames").value)
        self.window = int(gp("window_frames").value)
        self.max_leak = float(gp("max_payload_leak").value)
        self.max_over = float(gp("max_over_removal").value)

        self.lock = threading.Lock()
        self._syncs = []
        self.acc = {}   # sensor -> running totals

        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=10)

        # (name, raw, self_gt, held_gt, filtered)
        self.sensors = [
            ("lidar", "/lidar/points_raw", "/lidar/points_self_gt",
             "/lidar/points_held_gt", lidar_filt),
        ]
        if enable_camera:
            self.sensors.append(
                ("camera", "/camera/depth/points", "/camera/points_self_gt",
                 "/camera/points_held_gt", camera_filt))

        for name, raw_t, self_t, held_t, filt_t in self.sensors:
            self.acc[name] = {"frames": 0, "buf": deque(maxlen=self.window)}
            subs = [message_filters.Subscriber(self, PointCloud2, t, qos_profile=qos)
                    for t in (raw_t, self_t, held_t, filt_t)]
            sync = message_filters.ApproximateTimeSynchronizer(
                subs, queue_size=10, slop=0.2)
            sync.registerCallback(
                lambda r, s, h, f, n=name: self._on_clouds(n, r, s, h, f))
            self._syncs.append(sync)

        self.timer = self.create_timer(float(gp("report_period").value), self._report)
        self.get_logger().info(
            f"held_filter_metrics: scoring payload='{self.payload}' over "
            f"{[s[0] for s in self.sensors]} "
            f"(PASS: payload-leak <= {self.max_leak:.0%}, "
            f"over-removal <= {self.max_over:.0%})")

    def _on_clouds(self, name, raw_msg, self_msg, held_msg, filt_msg):
        raw = pc2_xyz(raw_msg)
        is_self = _member(raw, _keyset(pc2_xyz(self_msg)))   # robot body OR payload
        is_held = _member(raw, _keyset(pc2_xyz(held_msg)))   # payload only
        is_kept = _member(raw, _keyset(pc2_xyz(filt_msg)))   # survived the filter
        is_body = is_self & ~is_held                          # robot body only
        is_world = ~is_self                                   # real environment

        frame = dict(raw=int(raw.shape[0]),
                     payload_total=int(is_held.sum()),
                     payload_leak=int((is_held & is_kept).sum()),
                     body_total=int(is_body.sum()),
                     body_leak=int((is_body & is_kept).sum()),
                     world_total=int(is_world.sum()),
                     over=int((is_world & ~is_kept).sum()))
        a = self.acc[name]
        with self.lock:
            a["frames"] += 1
            if a["frames"] <= self.warmup:
                return   # let the pipeline settle before counting
            a["buf"].append(frame)   # sliding window (deque drops the oldest)

    def _report(self):
        with self.lock:
            snap = {k: (list(v["buf"])) for k, v in self.acc.items()}
        lines = [f"===== held-filter metrics   payload='{self.payload}'   "
                 f"(last {self.window} frames) ====="]
        for name, *_ in self.sensors:
            buf = snap[name]
            if not buf:
                lines.append(f" {name:<7} waiting for synchronized clouds...")
                continue
            tot = {k: sum(f[k] for f in buf) for k in buf[0]}
            n = len(buf)
            leak_p = _pct(tot["payload_leak"], tot["payload_total"])
            over_p = _pct(tot["over"], tot["world_total"])
            body_p = _pct(tot["body_leak"], tot["body_total"])
            # PASS needs payload data; a missing metric (object out of FOV) is n/a.
            ok = ((leak_p is None or leak_p <= self.max_leak * 100.0) and
                  (over_p is None or over_p <= self.max_over * 100.0) and
                  leak_p is not None)
            verdict = "PASS" if ok else ("n/a " if leak_p is None else "FAIL")
            lines.append(
                f" {name:<7} frames={n:<4} avg_raw={tot['raw'] // n}")
            lines.append(
                f"   payload : seen={tot['payload_total']:<6} "
                f"left-in={tot['payload_leak']:<5} -> {_fmt_pct(leak_p)}  "
                f"[want <= {self.max_leak * 100:.2f}%]")
            lines.append(
                f"   over-rm : world={tot['world_total']:<6} "
                f"removed={tot['over']:<5} -> {_fmt_pct(over_p)}  "
                f"[want <= {self.max_over * 100:.2f}%]")
            lines.append(
                f"   body    : seen={tot['body_total']:<6} "
                f"left-in={tot['body_leak']:<5} -> {_fmt_pct(body_p)}")
            lines.append(f"   verdict : {verdict}")
        self.get_logger().info("\n".join(lines))


def main():
    rclpy.init()
    node = HeldFilterMetrics()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
