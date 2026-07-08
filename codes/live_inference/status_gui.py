#!/usr/bin/env python3
"""Simple CV2 status dashboard for the live model interface.

Same wiring as run.py (subscribe joint/rgb/depth, publish /delta_action) but
with a small OpenCV window that shows, at a glance:
  - whether the MODEL is loaded,
  - whether each of the three input TOPICS is being received (and at what rate),
  - whether the action TOPIC is being published,
  - a scrolling LOG of the current stage.

It reuses LiveInterface unchanged -- it only subclasses it to record per-topic
receive times and a publish counter for the display; the inference path is
identical to run.py. ROS spins in a background thread; cv2 imshow/waitKey run
on the main thread (they must).

Usage:
    source /opt/ros/jazzy/setup.bash
    conda run -n robomimic_env python codes/live_inference/status_gui.py

Same flags as run.py (--config / --ckpt / --*-topic / --depth-max / --rate /
--device). Press 'q' or Esc in the window to quit.
"""
import os
import time
import argparse
import threading
from collections import deque

import numpy as np
import cv2

import rclpy

import config as config_mod
import sensors
from ros_node import LiveInterface


# =====================================================================
# Shared state between the ROS thread and the GUI thread
# =====================================================================
class LogBus:
    """Thread-safe rolling log + a one-line 'current stage' string."""

    def __init__(self, maxlen=200):
        self.lines = deque(maxlen=maxlen)
        self.stage = "starting"
        self._lock = threading.Lock()

    def log(self, msg, stage=None):
        stamp = time.strftime("%H:%M:%S")
        with self._lock:
            self.lines.append(f"{stamp}  {msg}")
            if stage is not None:
                self.stage = stage
        print(f"[{stamp}] {msg}", flush=True)

    def snapshot(self):
        with self._lock:
            return list(self.lines), self.stage


# =====================================================================
# LiveInterface + lightweight instrumentation for the dashboard
# =====================================================================
class MonitoredInterface(LiveInterface):
    """LiveInterface that also records receive/publish activity for the GUI."""

    def __init__(self, cfg, bus):
        self.bus = bus
        self.last_recv = {"joint": None, "rgb": None, "depth": None}
        self.recv_hist = {"joint": deque(maxlen=60),
                          "rgb": deque(maxlen=60),
                          "depth": deque(maxlen=60)}
        self.pub_count = 0
        self.last_pub_t = None
        self.last_action = None
        super().__init__(cfg)          # loads model + creates subs/publisher
        self._wrap_publisher()

    def _mark(self, key):
        now = time.time()
        self.last_recv[key] = now
        self.recv_hist[key].append(now)

    def rate(self, key):
        """Approximate incoming Hz over the last ~2 s."""
        now = time.time()
        recent = [t for t in self.recv_hist[key] if now - t <= 2.0]
        return len(recent) / 2.0 if recent else 0.0

    # override the 3 subscription callbacks: record activity, then behave as usual
    def on_joint(self, msg):
        super().on_joint(msg)
        self._mark("joint")

    def on_rgb(self, msg):
        super().on_rgb(msg)
        self._mark("rgb")

    def on_depth(self, msg):
        super().on_depth(msg)
        self._mark("depth")

    def _wrap_publisher(self):
        orig = self.pub.publish

        def wrapped(msg):
            self.pub_count += 1
            self.last_pub_t = time.time()
            self.last_action = list(msg.data)
            return orig(msg)

        self.pub.publish = wrapped


# =====================================================================
# Drawing helpers
# =====================================================================
GREEN = (80, 220, 80)
YELLOW = (60, 200, 240)
RED = (70, 70, 235)
GREY = (150, 150, 150)
WHITE = (235, 235, 235)


def _text(img, s, org, color=WHITE, scale=0.5, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _dot(img, center, color):
    cv2.circle(img, center, 7, color, -1, cv2.LINE_AA)


def _status_from_age(age, warm_period):
    """Green if fresh, yellow if stale, red if never seen."""
    if age is None:
        return RED, "WAITING"
    if age <= warm_period:
        return GREEN, "OK"
    return YELLOW, f"STALE {age:.1f}s"


def _thumb(img, dst_xywh, frame, label):
    """Blit a small BGR preview into the canvas at dst (x, y, w, h)."""
    x, y, w, h = dst_xywh
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), 1)
    _text(img, label, (x, y - 6), GREY, 0.42)
    if frame is None:
        _text(img, "no frame", (x + w // 2 - 34, y + h // 2), GREY, 0.45)
        return
    fr = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    img[y:y + h, x:x + w] = fr


def render(node, bus, cfg, model_ready):
    W, H = 860, 660
    img = np.full((H, W, 3), 24, np.uint8)

    _text(img, "Live Inference -- Status Monitor", (20, 34), WHITE, 0.72, 2)

    lines, stage = bus.snapshot()
    _text(img, f"stage: {stage}", (20, 60), YELLOW, 0.55)

    y = 100
    # ----- model -----
    mcolor = GREEN if model_ready else YELLOW
    _dot(img, (28, y - 5), mcolor)
    if model_ready and node is not None:
        info = node.runner.info
        _text(img, f"MODEL loaded  ({os.path.basename(cfg['ckpt'])})", (46, y), WHITE, 0.52)
        _text(img, f"To={info['To']} Ta={info['Ta']} Tp={info['Tp']} "
                   f"ac_dim={info['ac_dim']}  rgb={node.runner.rgb_hw} depth={node.runner.depth_hw}",
              (46, y + 20), GREY, 0.42)
    else:
        _text(img, "MODEL loading ...", (46, y), WHITE, 0.52)
    y += 52

    now = time.time()
    warm = max(1.5, 2.0 / cfg["rate"] * 3)  # a topic is "fresh" if seen within this window

    # ----- three subscriptions -----
    subs = [
        ("joint", "SUB  joint", cfg["joint_topic"]),
        ("rgb", "SUB  rgb", cfg["rgb_topic"]),
        ("depth", "SUB  depth", cfg["depth_topic"]),
    ]
    for key, label, topic in subs:
        last = None if node is None else node.last_recv[key]
        age = None if last is None else now - last
        color, state = _status_from_age(age, warm)
        hz = 0.0 if node is None else node.rate(key)
        _dot(img, (28, y - 5), color)
        _text(img, f"{label}", (46, y), WHITE, 0.52)
        _text(img, topic, (150, y), GREY, 0.44)
        _text(img, f"{state}  {hz:4.1f} Hz", (620, y), color, 0.5)
        y += 34

    # ----- publisher -----
    pub_count = 0 if node is None else node.pub_count
    last_pub = None if node is None else node.last_pub_t
    pub_age = None if last_pub is None else now - last_pub
    if pub_count == 0:
        pcolor, pstate = RED, "not publishing yet"
    elif pub_age is not None and pub_age <= warm:
        pcolor, pstate = GREEN, "PUBLISHING"
    else:
        pcolor, pstate = YELLOW, f"idle {pub_age:.1f}s"
    _dot(img, (28, y - 5), pcolor)
    _text(img, "PUB  action", (46, y), WHITE, 0.52)
    _text(img, cfg["action_topic"], (150, y), GREY, 0.44)
    _text(img, f"{pstate}  #{pub_count}", (620, y), pcolor, 0.5)
    y += 24
    if node is not None and node.last_action is not None:
        act = ", ".join(f"{v:+.3f}" for v in node.last_action)
        _text(img, f"last delta = [{act}]", (46, y + 8), WHITE, 0.44)
    y += 34

    # ----- log panel -----
    cv2.line(img, (20, y), (W - 20, y), (60, 60, 60), 1)
    y += 20
    _text(img, "log", (20, y), GREY, 0.5)
    y += 22
    for ln in list(lines)[-14:]:
        _text(img, ln, (24, y), (200, 200, 200), 0.42)
        y += 18

    # ----- thumbnails (confirm data is really flowing) -----
    rgb_prev = depth_prev = None
    if node is not None:
        try:
            if node.rgb_bytes is not None:
                rgb = sensors.decode_rgb(node.rgb_bytes, (150, 200))  # (h, w)
                rgb_prev = rgb[..., ::-1].copy()  # RGB -> BGR for cv2
        except Exception:
            rgb_prev = None
        try:
            if node.depth_raw is not None:
                d = node.depth_raw.astype(np.float32)
                d = np.clip(d, 0, cfg["depth_max"]) / cfg["depth_max"]
                d8 = (d * 255).astype(np.uint8)
                depth_prev = cv2.applyColorMap(cv2.resize(d8, (200, 150)), cv2.COLORMAP_JET)
        except Exception:
            depth_prev = None
    _thumb(img, (640, 300, 200, 150), rgb_prev, "rgb in")
    _thumb(img, (640, 480, 200, 150), depth_prev, "depth in")

    return img


# =====================================================================
# ROS thread: init, load model, spin
# =====================================================================
def ros_thread(cfg, bus, holder, stop_evt):
    try:
        bus.log("initializing rclpy", stage="init rclpy")
        rclpy.init()
        bus.log(f"loading model: {os.path.basename(cfg['ckpt'])}", stage="loading model")
        node = MonitoredInterface(cfg, bus)
        holder["node"] = node
        holder["ready"] = True
        bus.log("model ready; subscriptions + publisher created", stage="running")
        bus.log(f"waiting for topics: {cfg['joint_topic']}, {cfg['rgb_topic']}, {cfg['depth_topic']}")
        while rclpy.ok() and not stop_evt.is_set():
            try:
                rclpy.spin_once(node, timeout_sec=0.1)
            except Exception:
                # a publish/callback error thrown *because* we're shutting down
                # (Ctrl+C invalidates the context mid-tick) is not a real fault
                if not rclpy.ok() or stop_evt.is_set():
                    break
                raise
        bus.log("ROS stopped", stage="stopping")
    except Exception as e:  # a genuine failure (bad ckpt, etc.) -- show it in the GUI
        bus.log(f"ERROR: {e}", stage="error")
        holder["error"] = str(e)
    finally:
        node = holder.get("node")
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# =====================================================================
def main():
    p = argparse.ArgumentParser(description="CV2 status dashboard for the live model interface")
    p.add_argument("--config", default=config_mod.DEFAULT_CONFIG)
    p.add_argument("--ckpt", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--depth-max", type=float, default=None)
    p.add_argument("--joint-topic", default=None)
    p.add_argument("--rgb-topic", default=None)
    p.add_argument("--depth-topic", default=None)
    p.add_argument("--action-topic", default=None)
    p.add_argument("--rate", type=float, default=None)
    args = p.parse_args()

    cfg = config_mod.load_config(args.config)
    for key in ("ckpt", "device", "joint_topic", "rgb_topic", "depth_topic", "action_topic"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    if args.depth_max is not None:
        cfg["depth_max"] = args.depth_max
    if args.rate is not None:
        cfg["rate"] = args.rate

    bus = LogBus()
    holder = {"node": None, "ready": False, "error": None}
    stop_evt = threading.Event()
    t = threading.Thread(target=ros_thread, args=(cfg, bus, holder, stop_evt), daemon=True)
    t.start()

    win = "live_inference status"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            frame = render(holder.get("node"), bus, cfg, holder.get("ready", False))
            cv2.imshow(win, frame)
            k = cv2.waitKey(50) & 0xFF
            if k in (ord("q"), 27):  # q / Esc
                break
            # window closed by the X button
            if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
            # ROS thread ended (Ctrl+C, rclpy shutdown, or a load/spin error):
            # don't leave the GUI spinning on stale data -- exit too.
            if not t.is_alive():
                lines, _ = bus.snapshot()
                # give the user a moment to read a fatal error before closing
                if holder.get("error"):
                    for _ in range(60):
                        cv2.imshow(win, render(holder.get("node"), bus, cfg,
                                                holder.get("ready", False)))
                        if (cv2.waitKey(50) & 0xFF) in (ord("q"), 27):
                            break
                break
    except KeyboardInterrupt:
        pass  # Ctrl+C in the terminal -- fall through to clean shutdown below
    finally:
        bus.log("shutting down", stage="stopping")
        stop_evt.set()
        t.join(timeout=3.0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
