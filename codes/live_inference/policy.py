"""Model loading + causal-window inference, decoupled from ROS."""
import os
import sys
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> codes/
from model_utils import load_model  # noqa: E402

import sensors


class PolicyRunner:
    """Loads a checkpoint and runs one causal-window inference step per call.

    The policy needs a window of the last `To` obs frames on every call (see
    codes/norobot_eval/norobot_eval_gui.py's module docstring for why); this
    class owns that window so callers just pass in the latest raw frame.
    """

    def __init__(self, ckpt, device=None, depth_max=1000.0):
        self.policy, self.info = load_model(ckpt, device=device)
        self.depth_max = depth_max

        obs_key_shapes = self.policy.policy.obs_key_shapes  # {"rgb": [3,H,W], "depth": [1,H,W], ...}
        self.rgb_hw = (int(obs_key_shapes["rgb"][1]), int(obs_key_shapes["rgb"][2]))
        self.depth_hw = (int(obs_key_shapes["depth"][1]), int(obs_key_shapes["depth"][2]))

        self.window = deque(maxlen=self.info["To"])
        self.policy.start_episode()

    def step(self, joint_pos, joint_vel, rgb_bytes, depth_raw):
        """joint_pos/joint_vel: (6,) float; rgb_bytes: CompressedImage payload;
        depth_raw: (H,W) uint16 mm (already decoded from sensor_msgs/Image).
        Returns (action[ac_dim] float32, latency_ms)."""
        rgb = sensors.decode_rgb(rgb_bytes, self.rgb_hw)
        depth = sensors.decode_depth(depth_raw, self.depth_hw, self.depth_max)

        obs_frame = {
            "joint_pos": np.asarray(joint_pos, dtype=np.float32)[:6],
            "joint_vel": np.asarray(joint_vel, dtype=np.float32)[:6],
            "rgb": rgb,
            "depth": depth,
        }
        self.window.append(obs_frame)
        frames = list(self.window)
        while len(frames) < self.info["To"]:  # front-pad with earliest frame
            frames.insert(0, frames[0])
        win = {k: np.stack([f[k] for f in frames], axis=0) for k in self.info["obs_keys"]}

        t0 = time.time()
        action = np.asarray(self.policy(ob=win), dtype=np.float32)  # [ac_dim], un-normalized
        latency_ms = (time.time() - t0) * 1000.0
        return action, latency_ms
