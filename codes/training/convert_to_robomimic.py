#!/usr/bin/env python3
"""
convert_to_robomimic.py
------------------------
Convert an HDF5 collected from a real robot arm into robomimic
(Diffusion Policy) trainable format.

Input structure (per demo):
    arm_joint_pos  (T,6)          <- command side, target joint angles from IK
    gripper_cmd    (T,) or (T,1)  <- command side, gripper open/close (0/1)
    real_joint_pos (T,6)          <- actual joint angles reported by the real arm
    real_joint_vel (T,6)          <- actual joint velocities reported by the real arm
    real_rgb       (T,H,W,3) uint8
    real_depth     (T,H,W)   uint16
    (ee_pos / ee_quat / gripper_pos / arm_joint_vel / real_stamp are ignored)

Output structure (robomimic standard):
    data/ (@total, @env_args)
      demo_0/ (@num_samples)
        obs/
          joint_pos  (T,6)   float32
          joint_vel  (T,6)   float32
          rgb        (T,h,w,3) uint8
          depth      (T,h,w,1) float32  (normalized to [0,1])
        actions      (T,7)   float32
      ...
    mask/
      train  <- list of demo names
      valid  <- list of demo names

Action representation (--action-mode):
    absolute : actions[:, :6] = arm_joint_pos                 (target joint angles)
    delta    : actions[:, :6] = real_joint_pos[t+1] - real_joint_pos[t]
               (actual per-step movement from real feedback, last step zero-padded;
                at deployment: target = current real_joint_pos + delta)
    In both modes the 7th dim (gripper) stays absolute 0/1, no delta applied.

Usage:
    python convert_to_robomimic.py \
        --input  datasets/RealRobot_Lift_rsm_multiArm.hdf5 \
        --output datasets/rsm_robomimic_delta.hdf5 \
        --resize 128 \
        --action-mode delta \
        --val-ratio 0.1
"""

import argparse
import json
import os

import h5py
import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# Keys to read from the source (everything else is ignored)
ACTION_JOINT_KEY = "arm_joint_pos"      # command-side target joint angles
ACTION_GRIPPER_KEY = "gripper_cmd"      # gripper command 0/1
OBS_JOINT_POS_KEY = "real_joint_pos"    # actual joint angles reported by the arm
OBS_JOINT_VEL_KEY = "real_joint_vel"
OBS_RGB_KEY = "real_rgb"
OBS_DEPTH_KEY = "real_depth"


def resize_rgb(rgb, size):
    """rgb: (T,H,W,3) uint8 -> (T,size,size,3) uint8. Use INTER_AREA for less aliasing when downsampling."""
    T = rgb.shape[0]
    out = np.empty((T, size, size, 3), dtype=np.uint8)
    for i in range(T):
        out[i] = cv2.resize(rgb[i], (size, size), interpolation=cv2.INTER_AREA)
    return out


def resize_depth(depth, size):
    """depth: (T,H,W) -> (T,size,size). Use INTER_NEAREST to avoid interpolating fake values across depth discontinuities."""
    T = depth.shape[0]
    out = np.empty((T, size, size), dtype=depth.dtype)
    for i in range(T):
        out[i] = cv2.resize(depth[i], (size, size), interpolation=cv2.INTER_NEAREST)
    return out


def process_depth(depth_raw, depth_max):
    """
    depth_raw: (T,H,W) uint16, raw depth (usually in mm, 0 means invalid reading)
    returns: (T,H,W,1) float32, normalized to [0,1]
    handling: 0 (invalid) -> treat as farthest (=depth_max); clip to [0,depth_max]; divide by depth_max
    """
    d = depth_raw.astype(np.float32)
    # Treat 0 as an invalid reading -> set to the farthest distance so it does
    # not get misread as "closest" and mislead the policy.
    d[d <= 0] = depth_max
    d = np.clip(d, 0.0, depth_max)
    d = d / depth_max
    d = d[..., np.newaxis]  # add channel dim -> (T,H,W,1)
    return d.astype(np.float32)


def build_actions(src_demo, action_mode):
    """
    Build actions (T,7) float32.
      absolute: [arm_joint_pos(6), gripper_cmd(1)]
      delta:    [real_joint_pos[t+1] - real_joint_pos[t] (6), gripper_cmd(1)]
                (actual per-step movement measured from real feedback;
                 last step padded with zeros)
    The gripper dim is always absolute 0/1, never delta.
    """
    jp = np.asarray(src_demo[ACTION_JOINT_KEY][()], dtype=np.float32)   # (T,6) command-side target angles
    g = np.asarray(src_demo[ACTION_GRIPPER_KEY][()], dtype=np.float32)  # (T,) or (T,1)
    if g.ndim == 1:
        g = g[:, np.newaxis]  # -> (T,1)
    assert jp.shape[0] == g.shape[0], (
        f"arm_joint_pos and gripper_cmd length mismatch: {jp.shape} vs {g.shape}"
    )

    if action_mode == "absolute":
        joint_action = jp
    elif action_mode == "delta":
        real = np.asarray(src_demo[OBS_JOINT_POS_KEY][()], dtype=np.float32)  # (T,6) actual angles
        # delta = real_joint_pos[t+1] - real_joint_pos[t]
        # i.e. how much the arm actually moved at each step (measured from real feedback).
        # The last step has no t+1, so pad it with zeros (arm holds still).
        joint_action = np.zeros_like(real)
        joint_action[:-1] = real[1:] - real[:-1]
        # joint_action[-1] stays all-zero
    else:
        raise ValueError(f"Unknown action_mode: {action_mode}")

    actions = np.concatenate([joint_action, g], axis=1)  # (T,7)
    return actions


def auto_depth_max(src, demo_keys, percentile=99.9):
    """Estimate a robust upper bound (percentile) from all demos' non-zero depth, used as depth_max for normalization."""
    samples = []
    for k in demo_keys:
        d = src[f"data/{k}/{OBS_DEPTH_KEY}"]
        # To save memory, sample the first 100 frames of each demo.
        n = min(100, d.shape[0])
        arr = np.asarray(d[:n], dtype=np.float32).ravel()
        arr = arr[arr > 0]
        if arr.size > 0:
            samples.append(arr)
    if not samples:
        return 1.0
    allv = np.concatenate(samples)
    return float(np.percentile(allv, percentile))


def main():
    parser = argparse.ArgumentParser(description="Convert real-robot HDF5 -> robomimic format")
    parser.add_argument("--input", required=True, help="Source HDF5 path")
    parser.add_argument("--output", required=True, help="Output HDF5 path")
    parser.add_argument("--action-mode", choices=["absolute", "delta"], default="absolute",
                        help="Action representation: absolute=target joint angles (default); "
                             "delta=real_joint_pos[t+1]-real_joint_pos[t] (actual per-step movement)")
    parser.add_argument("--resize", type=int, default=None,
                        help="Side length after image resize (e.g. 84 / 128). Keep original resolution if omitted")
    parser.add_argument("--depth-max", type=float, default=None,
                        help="Upper bound for depth normalization (same unit as source, usually mm). Auto-estimated if omitted")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fraction of demos used for validation (default 0.1)")
    parser.add_argument("--gzip", type=int, default=4,
                        help="HDF5 image compression level 0-9 (default 4, 0 means no compression)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/valid split")
    args = parser.parse_args()

    if args.resize is not None and not HAS_CV2:
        raise RuntimeError("--resize was given but opencv is not found. Please run: pip install opencv-python")

    print(f"Action mode: {args.action_mode}")

    src = h5py.File(args.input, "r")
    demo_keys = sorted(src["data"].keys(), key=lambda x: int(x.split("_")[-1])
                       if x.split("_")[-1].isdigit() else x)
    print(f"Source has {len(demo_keys)} demos")

    # Decide the depth normalization upper bound.
    depth_max = args.depth_max if args.depth_max is not None else auto_depth_max(src, demo_keys)
    print(f"Depth normalization upper bound depth_max = {depth_max:.1f}")

    comp = dict(compression="gzip", compression_opts=args.gzip) if args.gzip > 0 else {}

    out = h5py.File(args.output, "w")
    data_grp = out.create_group("data")

    total = 0
    for k in demo_keys:
        src_demo = src[f"data/{k}"]
        T = src_demo[OBS_JOINT_POS_KEY].shape[0]

        # --- actions ---
        actions = build_actions(src_demo, args.action_mode)
        assert actions.shape[0] == T, f"{k}: action length {actions.shape[0]} != T {T}"

        # --- obs ---
        joint_pos = np.asarray(src_demo[OBS_JOINT_POS_KEY][()], dtype=np.float32)
        joint_vel = np.asarray(src_demo[OBS_JOINT_VEL_KEY][()], dtype=np.float32)
        rgb = np.asarray(src_demo[OBS_RGB_KEY][()], dtype=np.uint8)
        depth_raw = np.asarray(src_demo[OBS_DEPTH_KEY][()])

        # Resize (optional)
        if args.resize is not None:
            rgb = resize_rgb(rgb, args.resize)
            depth_raw = resize_depth(depth_raw, args.resize)

        depth = process_depth(depth_raw, depth_max)  # (T,h,w,1) float32 [0,1]

        # --- write ---
        d = data_grp.create_group(k)
        d.attrs["num_samples"] = T
        obs = d.create_group("obs")
        obs.create_dataset("joint_pos", data=joint_pos)
        obs.create_dataset("joint_vel", data=joint_vel)
        obs.create_dataset("rgb", data=rgb,
                           chunks=(1,) + rgb.shape[1:], **comp)
        obs.create_dataset("depth", data=depth,
                           chunks=(1,) + depth.shape[1:], **comp)
        d.create_dataset("actions", data=actions)

        total += T
        print(f"  {k}: T={T:5d}  actions={actions.shape}  rgb={rgb.shape}  depth={depth.shape}")

    data_grp.attrs["total"] = total

    # robomimic's train.py reads data.attrs["env_args"] unconditionally (even when
    # rollout is disabled), so write a minimal dummy. Since rollout.enabled=false in
    # the config, this environment is never actually created; it only lets the
    # metadata read pass.
    data_grp.attrs["env_args"] = json.dumps({
        "env_name": "RealRobotRSM",
        "type": 1,
        "env_kwargs": {}
    })

    # --- train / valid split ---
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(demo_keys))
    n_val = max(1, int(round(len(demo_keys) * args.val_ratio)))
    val_idx = set(perm[:n_val].tolist())
    train_demos = [demo_keys[i] for i in range(len(demo_keys)) if i not in val_idx]
    valid_demos = [demo_keys[i] for i in range(len(demo_keys)) if i in val_idx]

    mask = out.create_group("mask")
    mask.create_dataset("train", data=np.array(train_demos, dtype="S"))
    mask.create_dataset("valid", data=np.array(valid_demos, dtype="S"))

    out.close()
    src.close()

    size = os.path.getsize(args.output)
    print("\n" + "=" * 60)
    print(f"Done -> {args.output}  ({size / 1e9:.2f} GB)")
    print(f"   action mode: {args.action_mode}")
    print(f"   demos: {len(demo_keys)}  (train {len(train_demos)} / valid {len(valid_demos)})")
    print(f"   total transitions: {total}")
    print(f"   action_dim = 7  (6 joints + 1 gripper)")
    if args.action_mode == "delta":
        print("   NOTE (delta mode): at deployment, target angle = current real_joint_pos + first 6 dims of policy output")
    print("=" * 60)
    print("\nNext: run inspect_hdf5.py on the output and make sure there are no errors before training.")


if __name__ == "__main__":
    main()