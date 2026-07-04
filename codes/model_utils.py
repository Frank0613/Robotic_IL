"""Shared checkpoint-loading helper for norobot_eval and live_inference.

Kept at the codes/ top level (not inside either subpackage) because both
offline evaluation and live deployment need it equally -- neither should
import it through the other.
"""
import torch

import robomimic.utils.file_utils as FileUtils


def load_model(ckpt_path, device=None):
    """Load a checkpoint and return (policy, info dict).

    policy is a RolloutPolicy whose __call__ automatically applies obs/action
    normalization (and un-normalizes the output action).
    info contains To / Ta / Tp / obs_keys / ac_dim.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=ckpt_path, device=device, verbose=False
    )
    config, _ = FileUtils.config_from_checkpoint(ckpt_dict=ckpt_dict)

    info = dict(
        To=int(config.algo.horizon.observation_horizon),
        Ta=int(config.algo.horizon.action_horizon),
        Tp=int(config.algo.horizon.prediction_horizon),
        obs_keys=list(ckpt_dict["shape_metadata"]["all_obs_keys"]),
        ac_dim=int(ckpt_dict["shape_metadata"]["ac_dim"]),
        ckpt_path=ckpt_path,
        device=str(device),
    )
    return policy, info
