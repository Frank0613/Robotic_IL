#!/usr/bin/env python3
"""ROS2 dataset replay + action collector (Python 3.12 + rclpy).

Plays an exported demo (.npz) onto the obs topics one frame at a time in
LOCK-STEP: publish frame i, wait for the policy's action on /policy/action,
record it, then publish frame i+1. This guarantees clean 1:1 alignment and
absorbs the slow DDPM inference without dropping frames.

At the end it saves the predicted actions to a .npz. The comparison against the
dataset ground truth is done offline by score_ros_run.py (no ROS needed).
"""
import time
import argparse
import threading

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray, Int32
from builtin_interfaces.msg import Time


def np_to_image(arr, encoding, stamp):
    """Encode a numpy array as sensor_msgs/Image without cv_bridge (no cv2/Qt dep)."""
    msg = Image()
    msg.header.stamp = stamp
    msg.height = int(arr.shape[0])
    msg.width = int(arr.shape[1])
    msg.encoding = encoding
    msg.is_bigendian = 0
    if encoding == "rgb8":
        a = np.ascontiguousarray(arr, dtype=np.uint8)
        msg.step = msg.width * 3
    elif encoding == "32FC1":
        a = np.ascontiguousarray(arr, dtype=np.float32)
        msg.step = msg.width * 4
    else:
        raise ValueError(f"unsupported encoding {encoding!r}")
    msg.data = a.tobytes()
    return msg


class ReplayEval(Node):
    def __init__(self, npz_path, out_path, timeout, max_frames):
        super().__init__("replay_eval")
        self.data = np.load(npz_path, allow_pickle=True)
        self.demo_name = str(self.data["demo_name"])
        self.actions = self.data["actions"].astype(np.float32)
        self.T = self.actions.shape[0]
        if max_frames > 0:
            self.T = min(self.T, max_frames)
        self.out_path = out_path
        self.timeout = timeout

        self.rgb_pub = self.create_publisher(Image, "/eval/rgb", 10)
        self.depth_pub = self.create_publisher(Image, "/eval/depth", 10)
        self.js_pub = self.create_publisher(JointState, "/eval/joint_states", 10)
        self.reset_pub = self.create_publisher(Int32, "/eval/reset", 10)
        self.create_subscription(Float64MultiArray, "/policy/action", self.on_action, 50)

        self._ev = threading.Event()
        self._last = None
        self.preds = np.zeros((self.T, self.actions.shape[1]), dtype=np.float32)

    def on_action(self, msg):
        self._last = np.asarray(msg.data, dtype=np.float32)
        self._ev.set()

    def _wait_for_bridge(self, timeout=30.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if (self.rgb_pub.get_subscription_count() > 0
                    and self.depth_pub.get_subscription_count() > 0
                    and self.js_pub.get_subscription_count() > 0):
                return True
            time.sleep(0.2)
        return False

    def run(self):
        self.get_logger().info(f"demo {self.demo_name}: replaying {self.T} frames")
        if not self._wait_for_bridge():
            self.get_logger().error("policy_bridge never subscribed to obs topics; is it running?")
            return
        # tell the bridge (and through it the model_server) to start a new episode
        self.reset_pub.publish(Int32(data=int(self.T)))
        time.sleep(0.5)

        for i in range(self.T):
            stamp = Time(sec=int(i), nanosec=0)  # identical stamp across the 3 obs msgs

            rgb = np_to_image(self.data["rgb"][i], "rgb8", stamp)
            d = self.data["depth"][i]
            if d.ndim == 3:
                d = d[..., 0]
            depth = np_to_image(d, "32FC1", stamp)
            js = JointState()
            js.header.stamp = stamp
            js.name = [f"joint_{k}" for k in range(6)]
            js.position = [float(x) for x in self.data["joint_pos"][i][:6]]
            js.velocity = [float(x) for x in self.data["joint_vel"][i][:6]]

            self._ev.clear()
            self.rgb_pub.publish(rgb)
            self.depth_pub.publish(depth)
            self.js_pub.publish(js)

            if not self._ev.wait(self.timeout):
                self.get_logger().error(f"timeout waiting for action at frame {i}")
                self.T = i  # truncate to what we got
                break
            self.preds[i] = self._last
            if i % 10 == 0 or i == self.T - 1:
                self.get_logger().info(f"{i + 1}/{self.T}")

        np.savez(self.out_path, demo_name=self.demo_name, preds=self.preds[:self.T])
        self.get_logger().info(f"saved {self.T} predictions -> {self.out_path}")


def main():
    ap = argparse.ArgumentParser(description="ROS2 dataset replay + action collector")
    ap.add_argument("--npz", default="/tmp/ros_eval_demo.npz", help="exported demo .npz")
    ap.add_argument("--out", default="/tmp/ros_eval_preds.npz")
    ap.add_argument("--timeout", type=float, default=30.0, help="per-frame action timeout (s)")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = full demo; >0 limits frames")
    args = ap.parse_args()

    rclpy.init()
    node = ReplayEval(args.npz, args.out, args.timeout, args.max_frames)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    spin_thread = threading.Thread(target=ex.spin, daemon=True)
    spin_thread.start()
    try:
        node.run()
    finally:
        ex.shutdown()
        spin_thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
