#!/usr/bin/env python3
"""Entry point for the live model interface (robomimic_env / Python 3.12).

Loads a trained Diffusion Policy checkpoint and directly subscribes/publishes
ROS2 topics in a single process -- see ../README.md for why this no longer
needs a socket bridge to a separate torch process.

Settings come from config.json (ckpt path, topic names, depth_max, rate) --
edit that file to reconfigure, no code changes needed. Pass --config for a
different file, or individual flags to override one field for a one-off run.

Usage:
    source /opt/ros/jazzy/setup.bash
    conda run -n robomimic_env python codes/live_inference/run.py
"""
import argparse

import rclpy

import config as config_mod
from ros_node import LiveInterface


def build_parser():
    p = argparse.ArgumentParser(
        description="Live model interface: load model, subscribe sensors, publish delta_action")
    p.add_argument("--config", default=config_mod.DEFAULT_CONFIG,
                    help="JSON config (ckpt/topics/depth_max/rate)")
    p.add_argument("--ckpt", default=None, help="override config's ckpt path")
    p.add_argument("--device", default=None, help="cuda / cpu (overrides config; auto if unset)")
    p.add_argument("--depth-max", type=float, default=None, help="override config's depth_max (mm)")
    p.add_argument("--joint-topic", default=None)
    p.add_argument("--rgb-topic", default=None)
    p.add_argument("--depth-topic", default=None)
    p.add_argument("--action-topic", default=None)
    p.add_argument("--rate", type=float, default=None, help="override config's inference rate (Hz)")
    return p


def main():
    args = build_parser().parse_args()
    cfg = config_mod.load_config(args.config)
    for key in ("ckpt", "device", "joint_topic", "rgb_topic", "depth_topic", "action_topic"):
        val = getattr(args, key)
        if val is not None:
            cfg[key] = val
    if args.depth_max is not None:
        cfg["depth_max"] = args.depth_max
    if args.rate is not None:
        cfg["rate"] = args.rate

    try:
        from rclpy.executors import ExternalShutdownException
    except Exception:
        ExternalShutdownException = ()

    rclpy.init()
    node = LiveInterface(cfg)
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
