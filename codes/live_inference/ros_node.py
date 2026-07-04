"""The rclpy Node: wires ROS subscriptions/publisher to PolicyRunner."""
import time

import numpy as np

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage, Image, JointState
from std_msgs.msg import Float32MultiArray

import sensors
from policy import PolicyRunner


class LiveInterface(Node):
    def __init__(self, cfg):
        super().__init__("live_interface")

        self.get_logger().info(f"loading {cfg['ckpt']} ...")
        self.runner = PolicyRunner(cfg["ckpt"], device=cfg.get("device"), depth_max=cfg["depth_max"])
        info = self.runner.info
        self.get_logger().info(
            f"model ready: To={info['To']} Ta={info['Ta']} Tp={info['Tp']} "
            f"obs_keys={info['obs_keys']} ac_dim={info['ac_dim']} "
            f"rgb_hw={self.runner.rgb_hw} depth_hw={self.runner.depth_hw} "
            f"depth_max={cfg['depth_max']}")

        self.joint_pos = None
        self.joint_vel = None
        self.rgb_bytes = None
        self.depth_raw = None
        self._last_warn_t = 0.0
        self._frame_count = 0

        # Camera nodes publish BEST_EFFORT (sensor-data QoS); a RELIABLE
        # subscriber silently receives nothing from them. joint_state stays
        # RELIABLE. Mirrors ros_subscriber.py in RealRobot_DataCollector.
        sensor_qos = QoSProfile(
            depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)

        self.pub = self.create_publisher(Float32MultiArray, cfg["action_topic"], 10)
        self.create_subscription(JointState, cfg["joint_topic"], self.on_joint, 10)
        self.create_subscription(CompressedImage, cfg["rgb_topic"], self.on_rgb, sensor_qos)
        self.create_subscription(Image, cfg["depth_topic"], self.on_depth, sensor_qos)
        self.timer = self.create_timer(1.0 / cfg["rate"], self.on_tick)

        self.get_logger().info(
            f"live_interface ready: joint={cfg['joint_topic']} rgb={cfg['rgb_topic']} "
            f"depth={cfg['depth_topic']} action={cfg['action_topic']} rate={cfg['rate']}Hz")

    # ---------- subscriptions: just cache the latest of each ----------
    def on_joint(self, msg):
        pos = np.asarray(msg.position, dtype=np.float32)
        # real driver may not populate velocity; zero-fill like ros_subscriber.py does
        vel = (np.asarray(msg.velocity, dtype=np.float32)
               if len(msg.velocity) == len(msg.position) else np.zeros_like(pos))
        self.joint_pos = pos[:6]
        self.joint_vel = vel[:6]

    def on_rgb(self, msg):
        self.rgb_bytes = bytes(msg.data)

    def on_depth(self, msg):
        try:
            self.depth_raw = sensors.depth_image_to_np(msg)
        except ValueError as e:
            self.get_logger().error(str(e))

    # ---------- inference tick ----------
    def on_tick(self):
        if self.joint_pos is None or self.rgb_bytes is None or self.depth_raw is None:
            now = time.time()
            if now - self._last_warn_t > 5.0:
                self.get_logger().warn("waiting for obs (joint/rgb/depth)...")
                self._last_warn_t = now
            return

        try:
            action, latency_ms = self.runner.step(
                self.joint_pos, self.joint_vel, self.rgb_bytes, self.depth_raw)
        except ValueError as e:
            self.get_logger().error(str(e))
            return

        out = Float32MultiArray()
        out.data = [float(x) for x in action]
        self.pub.publish(out)

        self._frame_count += 1
        if self._frame_count % 10 == 0 or self._frame_count == 1:
            act_str = ", ".join(f"{v:+.3f}" for v in action)
            self.get_logger().info(
                f"frame {self._frame_count}: action=[{act_str}]  ({latency_ms:.1f} ms)")
