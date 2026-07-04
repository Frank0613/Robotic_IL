#!/usr/bin/env python3
"""Compare ROS-run predictions against the dataset ground truth (no ROS needed).

Run in robomimic_env (conda, Python 3.12). Reuses the exact scoring/plotting
from norobot_eval_gui.py, so a ROS-bridge run is judged the same way as the
offline no-ROS eval.
"""
import os
import sys
import argparse

import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import norobot_eval_gui as N  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="Score a ROS-bridge run vs dataset GT")
    ap.add_argument("--demo-npz", default="/tmp/ros_eval_demo.npz",
                    help="exported demo (has GT actions + obs)")
    ap.add_argument("--preds-npz", default="/tmp/ros_eval_preds.npz",
                    help="predictions collected by the ROS replay node")
    ap.add_argument("--action-mode", choices=["absolute", "delta"], default="delta")
    ap.add_argument("--init-obs-key", default="joint_pos")
    ap.add_argument("--threshold-joint", type=float, default=0.01)
    ap.add_argument("--threshold-gripper", type=float, default=0.05)
    ap.add_argument("--out-png", default="/tmp/ros_eval_plot.png")
    args = ap.parse_args()

    demo = np.load(args.demo_npz, allow_pickle=True)
    pred = np.load(args.preds_npz, allow_pickle=True)
    name = str(demo["demo_name"])

    actions = demo["actions"].astype(np.float32)
    preds = pred["preds"].astype(np.float32)
    T = min(len(actions), len(preds))   # replay may have stopped early
    actions, preds = actions[:T], preds[:T]
    obs = {"joint_pos": demo["joint_pos"][:T], "joint_vel": demo["joint_vel"][:T]}

    # same transform + scoring as the offline eval
    preds2, actions2 = N.to_plot_space(preds, actions, obs, args.action_mode, args.init_obs_key)
    res = N.score(preds2, actions2, args.threshold_joint, args.threshold_gripper)

    print(f"\n[ros-eval] demo={name}  mode={args.action_mode}  T={T}")
    print("dim        MSE        thr     result")
    for d in range(len(res["mse"])):
        tag = "PASS" if res["passed"][d] else "FAIL"
        print(f"{N.ACTION_LABELS[d]:<9} {res['mse'][d]:9.4f}  {res['thresholds'][d]:7.3f}   {tag}")
    print(f"OVERALL: {'PASS' if res['overall'] else 'FAIL'}")

    img = N.render_plots(preds2, actions2, res, name, "ROS-run",
                         ac_dim=actions.shape[1], mode_tag=f"ROS/{args.action_mode}")
    cv2.imwrite(args.out_png, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[ros-eval] plot saved -> {args.out_png}")


if __name__ == "__main__":
    main()
