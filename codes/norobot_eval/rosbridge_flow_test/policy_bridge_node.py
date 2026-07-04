#!/usr/bin/env python3
"""ROS2 policy bridge node (Python 3.12 + rclpy, source /opt/ros/jazzy/setup.bash).

Subscribes to the obs topics (rgb/depth Image + JointState), forwards each
time-synced frame to the model_server over a localhost socket, and publishes the
returned action (a joint delta, for a delta-trained policy) on /policy/action.

On the real arm this is the node you keep: it would subscribe to the arm's camera
and joint-state topics and publish action commands. Only the data source changes
(replay node now, real arm later).
"""
import time
import socket
import struct
import pickle

import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float64MultiArray, Int32
import message_filters


def image_to_np(msg):
    """Decode a sensor_msgs/Image to numpy without cv_bridge (avoids cv2/Qt deps)."""
    raw = bytes(msg.data)
    if msg.encoding in ("rgb8", "bgr8"):
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            arr = arr[..., ::-1]
        return np.ascontiguousarray(arr)
    if msg.encoding == "32FC1":
        return np.frombuffer(raw, dtype=np.float32).reshape(msg.height, msg.width).copy()
    raise ValueError(f"unsupported image encoding {msg.encoding!r}")


# ---- socket framing (mirror of model_server) ----
def _recvall(conn, n):
    buf = b""
    while len(buf) < n:
        c = conn.recv(n - len(buf))
        if not c:
            return None
        buf += c
    return buf


def recv_msg(conn):
    hdr = _recvall(conn, 4)
    if hdr is None:
        return None
    (ln,) = struct.unpack(">I", hdr)
    return pickle.loads(_recvall(conn, ln))


def send_msg(conn, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack(">I", len(data)) + data)


class PolicyBridge(Node):
    def __init__(self):
        super().__init__("policy_bridge")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 5599)
        host = self.get_parameter("host").value
        port = int(self.get_parameter("port").value)

        self.sock = self._connect(host, port)
        send_msg(self.sock, {"cmd": "ping"})
        rep = recv_msg(self.sock)
        self.get_logger().info(f"connected to model_server: {rep}")

        self.pub = self.create_publisher(Float64MultiArray, "/policy/action", 10)
        self.create_subscription(Int32, "/eval/reset", self.on_reset, 10)

        rgb_sub = message_filters.Subscriber(self, Image, "/eval/rgb")
        depth_sub = message_filters.Subscriber(self, Image, "/eval/depth")
        js_sub = message_filters.Subscriber(self, JointState, "/eval/joint_states")
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub, js_sub], queue_size=30, slop=0.05)
        self.sync.registerCallback(self.on_obs)
        self.get_logger().info("policy_bridge ready (waiting for obs)")

    def _connect(self, host, port, retries=60, delay=1.0):
        """Retry until model_server is up (it may still be loading the model)."""
        for i in range(retries):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((host, port))
                return s
            except OSError:
                if i == 0:
                    self.get_logger().info(f"waiting for model_server at {host}:{port} ...")
                time.sleep(delay)
        raise RuntimeError(f"could not connect to model_server at {host}:{port}")

    def on_reset(self, msg):
        send_msg(self.sock, {"cmd": "reset"})
        recv_msg(self.sock)
        self.get_logger().info(f"episode reset (expecting {msg.data} frames)")

    def on_obs(self, rgb_msg, depth_msg, js_msg):
        rgb = image_to_np(rgb_msg)
        depth = image_to_np(depth_msg)
        if depth.ndim == 2:
            depth = depth[..., None]
        obs = {
            "joint_pos": np.asarray(js_msg.position[:6], dtype=np.float32),
            "joint_vel": np.asarray(js_msg.velocity[:6], dtype=np.float32),
            "rgb": np.ascontiguousarray(rgb, dtype=np.uint8),
            "depth": np.ascontiguousarray(depth, dtype=np.float32),
        }
        send_msg(self.sock, {"cmd": "step", "obs": obs})
        rep = recv_msg(self.sock)
        if not rep or not rep.get("ok"):
            self.get_logger().error(f"model_server error: {rep}")
            return
        out = Float64MultiArray()
        out.data = [float(x) for x in rep["action"]]
        self.pub.publish(out)


def main():
    try:
        from rclpy.executors import ExternalShutdownException
    except Exception:
        ExternalShutdownException = ()
    rclpy.init()
    node = PolicyBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
