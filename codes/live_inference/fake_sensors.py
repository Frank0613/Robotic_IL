#!/usr/bin/env python3
"""Fake robot -- publishes the 3 input topics so the live interface can be
tested end-to-end WITHOUT a real arm.

It publishes random JointState / RGB / depth on the same topic names the live
interface subscribes to (read from config.json, so they always match), using
the same QoS as the real sensors (rgb/depth BEST_EFFORT, joint RELIABLE). It
also subscribes to the action topic and logs it, so you can confirm the loop
closes. Pair it with run.py or status_gui.py in another terminal.

Usage:
    source /opt/ros/jazzy/setup.bash
    conda run -n robomimic_env python codes/live_inference/fake_sensors.py
"""
import argparse

import numpy as np
import cv2

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Float32MultiArray

import config as config_mod


class FakeSensors(Node):
    def __init__(self, cfg, hz):
        super().__init__("fake_sensors")
        best = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT,
                          history=HistoryPolicy.KEEP_LAST)
        self.pj = self.create_publisher(JointState, cfg["joint_topic"], 10)
        self.pr = self.create_publisher(CompressedImage, cfg["rgb_topic"], best)
        self.pd = self.create_publisher(Image, cfg["depth_topic"], best)
        self.create_subscription(Float32MultiArray, cfg["action_topic"], self.on_action, 10)
        self.n = 0
        self.got_action = 0
        self.create_timer(1.0 / hz, self.tick)
        self.get_logger().info(
            f"faking joint={cfg['joint_topic']} rgb={cfg['rgb_topic']} "
            f"depth={cfg['depth_topic']} at {hz}Hz; listening on {cfg['action_topic']}")

    def on_action(self, msg):
        self.got_action += 1
        if self.got_action <= 3 or self.got_action % 20 == 0:
            vals = ", ".join(f"{v:+.3f}" for v in msg.data)
            self.get_logger().info(f"<- {self.got_action} actions received, last=[{vals}]")

    def tick(self):
        self.n += 1
        t = self.n * 0.1

        js = JointState()
        js.position = [float(0.1 * np.sin(t + i)) for i in range(6)]
        js.velocity = [0.0] * 6
        self.pj.publish(js)

        bgr = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
        ci = CompressedImage()
        ci.format = "jpeg"
        ci.data = cv2.imencode(".jpg", bgr)[1].tobytes()
        self.pr.publish(ci)

        depth = (np.random.rand(480, 640) * 800).astype(np.uint16)
        di = Image()
        di.height, di.width = 480, 640
        di.encoding = "16UC1"
        di.is_bigendian = 0
        di.step = 640 * 2
        di.data = depth.tobytes()
        self.pd.publish(di)

        if self.n % 20 == 0:
            self.get_logger().info(f"-> published {self.n} frames of joint/rgb/depth")


def main():
    p = argparse.ArgumentParser(description="Fake sensor publisher for testing the live interface")
    p.add_argument("--config", default=config_mod.DEFAULT_CONFIG)
    p.add_argument("--hz", type=float, default=10.0, help="publish rate")
    args = p.parse_args()
    cfg = config_mod.load_config(args.config)

    rclpy.init()
    node = FakeSensors(cfg, args.hz)
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
