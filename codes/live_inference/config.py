"""Load live_inference's JSON config (ckpt path, topics, depth_max, rate)."""
import os
import json

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")


def load_config(path=DEFAULT_CONFIG):
    with open(path) as f:
        cfg = json.load(f)
    if not os.path.isabs(cfg["ckpt"]):
        cfg["ckpt"] = os.path.join(REPO_ROOT, cfg["ckpt"])
    return cfg
