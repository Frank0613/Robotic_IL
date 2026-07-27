#!/usr/bin/env python3
"""
convert_jetcobot_to_robomimic.py
--------------------------------
Convert the JetCobot real-arm HDF5 (Realjetcobot.hdf5) into robomimic
(Diffusion Policy) trainable format, with **EE-delta** actions.

This is a separate converter from convert_to_robomimic.py because the source
layout is different: demos sit at the TOP level (no `data/` group), the arm is
EE-controlled (not joint-controlled), and `action` is an ABSOLUTE target pose.

Input structure (per demo, at the file's top level):
    action                    (T,10) float32  <- [target ee_pose(9), gripper(1)]  ABSOLUTE
    observations/ee_pose      (T,9)  float32  <- [xyz(3), 6D rotation(6)] actual measured pose
    observations/joint_state  (T,6)  float32  <- joint angles in DEGREES
    observations/gripper_width(T,)   int8     <- binary 0/1 (open/close), not a real width
    observations/wrist_rgb    (T,256,256,3) uint8
    timestamp / timestamps/   (ignored)

Output structure (robomimic standard):
    data/ (@total, @env_args)
      demo_0/ (@num_samples)
        obs/
          rgb         (T,h,w,3) uint8
          joint_state (T,6)  float32   (converted to radians by default)
          ee_pose     (T,9)  float32   (xyz + 6D rotation, as-is)
        actions       (T,10) float32   [ee_delta(9), gripper(1)]
      ...
    (no mask/ group by default -- every demo is used for training)

Action representation:
    The source `action` is the ABSOLUTE commanded target pose, while
    `observations/ee_pose` is the pose the arm actually reached. The delta is
    therefore the command relative to the current measured pose:

        delta[t] = action[t][:9]  (-)  ee_pose[t]

    which is exactly what deployment inverts:  target = current_ee_pose + delta.
    The gripper dim (last) always stays absolute 0/1 -- never a delta.

    --rot-delta linear (default)
        Plain element-wise subtraction on the 6D rotation block. Exactly
        invertible with the additive deployment rule above (the same idiom
        codes/live_inference/ already uses for joint deltas). At deployment,
        re-orthonormalize the resulting 6D block back into a valid rotation.
    --rot-delta so3
        Proper relative rotation: R_rel = R_cur^T @ R_tgt, re-encoded as 6D.
        Mathematically correct on SO(3); deployment must then compose
        R_tgt = R_cur @ R_rel instead of adding.

Usage:
    python codes/training/convert_jetcobot_to_robomimic.py \
        --input  datasets/Realjetcobot.hdf5 \
        --output datasets/jetcobot_robomimic_ee_delta.hdf5
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


# Source keys (everything else in the file is ignored)
SRC_ACTION = "action"
SRC_EE = "observations/ee_pose"
SRC_JOINT = "observations/joint_state"
SRC_GRIPPER = "observations/gripper_width"
SRC_RGB = "observations/wrist_rgb"

DEG2RAD = np.pi / 180.0


# ---------------------------------------------------------------- rotation
def sixd_to_matrix(d6):
    """(N,6) 6D rotation -> (N,3,3) rotation matrices (Zhou et al. Gram-Schmidt).

    Columns are [b1, b2, b3]. The source data is already orthonormal, so this
    is a no-op for it, but we re-orthonormalize anyway so the function is safe
    on any input.
    """
    a1, a2 = d6[:, 0:3], d6[:, 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=1, keepdims=True)
    a2_proj = a2 - (b1 * a2).sum(axis=1, keepdims=True) * b1
    b2 = a2_proj / np.linalg.norm(a2_proj, axis=1, keepdims=True)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)  # (N,3,3), columns = b1,b2,b3


def matrix_to_sixd(R):
    """(N,3,3) -> (N,6): take the first two COLUMNS (inverse of sixd_to_matrix)."""
    return np.concatenate([R[:, :, 0], R[:, :, 1]], axis=1)


def build_actions(act_src, ee, rot_delta):
    """Build (T,10) float32 EE-delta actions.

    act_src : (T,10) absolute [target ee_pose(9), gripper(1)]
    ee      : (T,9)  measured ee_pose
    returns : (T,10) [delta_pos(3), delta_rot6d(6), gripper(1)]
    """
    tgt = act_src[:, :9].astype(np.float32)
    grip = act_src[:, 9:10].astype(np.float32)

    delta_pos = tgt[:, :3] - ee[:, :3]

    if rot_delta == "linear":
        delta_rot = tgt[:, 3:9] - ee[:, 3:9]
    elif rot_delta == "so3":
        R_cur = sixd_to_matrix(ee[:, 3:9].astype(np.float64))
        R_tgt = sixd_to_matrix(tgt[:, 3:9].astype(np.float64))
        R_rel = np.einsum("nji,njk->nik", R_cur, R_tgt)  # R_cur^T @ R_tgt
        delta_rot = matrix_to_sixd(R_rel).astype(np.float32)
    else:
        raise ValueError(f"Unknown rot_delta: {rot_delta}")

    return np.concatenate([delta_pos, delta_rot, grip], axis=1).astype(np.float32)


def resize_rgb(rgb, size):
    """(T,H,W,3) uint8 -> (T,size,size,3) uint8, INTER_AREA (less aliasing when downsampling)."""
    T = rgb.shape[0]
    out = np.empty((T, size, size, 3), dtype=np.uint8)
    for i in range(T):
        out[i] = cv2.resize(rgb[i], (size, size), interpolation=cv2.INTER_AREA)
    return out


def main():
    p = argparse.ArgumentParser(description="Convert JetCobot real-arm HDF5 -> robomimic (EE delta)")
    p.add_argument("--input", default="datasets/Realjetcobot.hdf5")
    p.add_argument("--output", default="datasets/jetcobot_robomimic_ee_delta.hdf5")
    p.add_argument("--rot-delta", choices=["linear", "so3"], default="linear",
                   help="how the 6D rotation delta is computed (see module docstring)")
    p.add_argument("--keep-degrees", action="store_true",
                   help="keep joint_state in degrees; default converts to radians so its "
                        "scale matches ee_pose (obs are NOT normalized by the training config)")
    p.add_argument("--resize", type=int, default=None,
                   help="resize RGB to this side length; omit to keep the source 256")
    p.add_argument("--include-gripper-obs", action="store_true",
                   help="also store gripper_width as an observation (off by default)")
    p.add_argument("--val-ratio", type=float, default=0.0,
                   help="fraction of demos held out for validation; 0 (default) = train on everything "
                        "and write NO mask group")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gzip", type=int, default=4, help="HDF5 image compression 0-9 (0 = off)")
    args = p.parse_args()

    if args.resize is not None and not HAS_CV2:
        raise RuntimeError("--resize needs opencv: pip install opencv-python")

    src = h5py.File(args.input, "r")
    demo_keys = sorted([k for k in src.keys() if k.startswith("demo")])
    print(f"Source: {args.input}  ({len(demo_keys)} demos)")
    print(f"Action: EE delta (rot-delta={args.rot_delta}) + absolute gripper")
    print(f"joint_state unit: {'degrees (kept)' if args.keep_degrees else 'radians (converted)'}")

    comp = dict(compression="gzip", compression_opts=args.gzip) if args.gzip > 0 else {}

    out = h5py.File(args.output, "w")
    data_grp = out.create_group("data")

    total = 0
    for i, k in enumerate(demo_keys):
        g = src[k]
        act_src = np.asarray(g[SRC_ACTION][()], dtype=np.float32)
        ee = np.asarray(g[SRC_EE][()], dtype=np.float32)
        joint = np.asarray(g[SRC_JOINT][()], dtype=np.float32)
        rgb = np.asarray(g[SRC_RGB][()], dtype=np.uint8)
        T = act_src.shape[0]
        assert ee.shape[0] == T and joint.shape[0] == T and rgb.shape[0] == T, \
            f"{k}: length mismatch act={T} ee={ee.shape[0]} joint={joint.shape[0]} rgb={rgb.shape[0]}"

        actions = build_actions(act_src, ee, args.rot_delta)
        if not args.keep_degrees:
            joint = (joint * DEG2RAD).astype(np.float32)
        if args.resize is not None:
            rgb = resize_rgb(rgb, args.resize)

        # robomimic expects demo_0, demo_1, ... (its filter/sort logic splits on "_")
        name = f"demo_{i}"
        d = data_grp.create_group(name)
        d.attrs["num_samples"] = T
        obs = d.create_group("obs")
        obs.create_dataset("rgb", data=rgb, chunks=(1,) + rgb.shape[1:], **comp)
        obs.create_dataset("joint_state", data=joint)
        obs.create_dataset("ee_pose", data=ee)
        if args.include_gripper_obs:
            obs.create_dataset("gripper", data=np.asarray(g[SRC_GRIPPER][()],
                                                          dtype=np.float32)[:, None])
        d.create_dataset("actions", data=actions)

        total += T
        if i < 3 or i == len(demo_keys) - 1:
            print(f"  {k} -> {name}: T={T:5d} actions={actions.shape} rgb={rgb.shape}")
        elif i == 3:
            print("  ...")

    data_grp.attrs["total"] = total
    # train.py reads env_args unconditionally even with rollout disabled; a minimal
    # dummy is enough since no environment is ever constructed.
    data_grp.attrs["env_args"] = json.dumps(
        {"env_name": "JetCobotReal", "type": 1, "env_kwargs": {}})

    # --- optional train/valid split (default: none -> train on everything) ---
    if args.val_ratio > 0:
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(demo_keys))
        n_val = max(1, int(round(len(demo_keys) * args.val_ratio)))
        val_idx = set(perm[:n_val].tolist())
        names = [f"demo_{i}" for i in range(len(demo_keys))]
        mask = out.create_group("mask")
        mask.create_dataset("train", data=np.array(
            [n for i, n in enumerate(names) if i not in val_idx], dtype="S"))
        mask.create_dataset("valid", data=np.array(
            [n for i, n in enumerate(names) if i in val_idx], dtype="S"))
        print(f"\nwrote mask/: train={len(names)-len(val_idx)} valid={len(val_idx)}")
    else:
        print("\nno mask/ group written -- ALL demos are used for training")

    out.close()
    src.close()

    print("=" * 64)
    print(f"Done -> {args.output}  ({os.path.getsize(args.output) / 1e9:.2f} GB)")
    print(f"  demos={len(demo_keys)}  total transitions={total}")
    print(f"  obs: rgb{'' if args.resize is None else f'({args.resize})'}, joint_state(6), ee_pose(9)"
          + (", gripper(1)" if args.include_gripper_obs else ""))
    print("  actions: 10 = [ee_delta_pos(3), ee_delta_rot6d(6), gripper(1)]")
    if args.rot_delta == "linear":
        print("  deployment: target_pose = current_ee_pose + action[:9]  (then re-orthonormalize the 6D rot)")
    else:
        print("  deployment: target_pos = cur_pos + action[:3];  R_tgt = R_cur @ decode6d(action[3:9])")
    print("=" * 64)


if __name__ == "__main__":
    main()
