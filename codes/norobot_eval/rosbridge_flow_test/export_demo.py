#!/usr/bin/env python3
"""Export one demo (obs + ground-truth actions) from a robomimic hdf5 into a .npz.

Run in robomimic_env (conda, Python 3.12). The ROS2 replay node (system
Python 3.12) then loads the .npz with numpy only, so the ROS side needs no
h5py.
"""
import os
import sys
import argparse

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from norobot_eval_gui import load_demo, list_demos, DEFAULT_DATASET  # noqa: E402

OBS_KEYS = ["joint_pos", "joint_vel", "rgb", "depth"]


def main():
    ap = argparse.ArgumentParser(description="Export one demo to .npz for the ROS replay node")
    ap.add_argument("--dataset", default=DEFAULT_DATASET)
    ap.add_argument("--demo", default=None,
                    help="demo name, e.g. demo_15 (default: first demo of --filter-key)")
    ap.add_argument("--filter-key", default="valid")
    ap.add_argument("--out", default="/tmp/ros_eval_demo.npz")
    args = ap.parse_args()

    name = args.demo or list_demos(args.dataset, args.filter_key)[0]
    obs, actions = load_demo(args.dataset, name, OBS_KEYS)
    np.savez(
        args.out,
        demo_name=name,
        actions=actions,
        joint_pos=obs["joint_pos"],
        joint_vel=obs["joint_vel"],
        rgb=obs["rgb"],
        depth=obs["depth"],
    )
    print(f"[export] {name}: T={actions.shape[0]}  rgb={obs['rgb'].shape} "
          f"depth={obs['depth'].shape} -> {args.out}")


if __name__ == "__main__":
    main()
