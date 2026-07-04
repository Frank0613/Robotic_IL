"""Sensor decode/normalize -- turns raw ROS message payloads into the numpy
arrays the policy expects (matches convert_to_robomimic.py's training-time
preprocessing exactly)."""
import numpy as np
import cv2


def depth_image_to_np(msg):
    """Decode a raw sensor_msgs/Image depth frame to numpy (no cv_bridge)."""
    raw = bytes(msg.data)
    if msg.encoding in ("16UC1", "mono16"):
        return np.frombuffer(raw, dtype=np.uint16).reshape(msg.height, msg.width).copy()
    if msg.encoding == "32FC1":
        return np.frombuffer(raw, dtype=np.float32).reshape(msg.height, msg.width).copy()
    raise ValueError(f"unsupported depth encoding {msg.encoding!r}")


def normalize_depth(depth_raw_mm, depth_max):
    """Same normalization as training (codes/training/convert_to_robomimic.py:
    process_depth): 0 (invalid) -> treated as farthest; clip to [0,depth_max];
    scale to [0,1]. Returns (H,W,1) float32."""
    d = depth_raw_mm.astype(np.float32)
    d[d <= 0] = depth_max
    d = np.clip(d, 0.0, depth_max)
    d = d / depth_max
    return d[..., np.newaxis]


def decode_rgb(rgb_bytes, target_hw):
    """rgb_bytes: sensor_msgs/CompressedImage payload (jpeg/png). Returns
    (h,w,3) uint8 RGB, resized to target_hw."""
    buf = np.frombuffer(rgb_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("failed to decode RGB CompressedImage payload")
    h, w = target_hw
    bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(bgr[..., ::-1])  # BGR -> RGB


def decode_depth(depth_raw, target_hw, depth_max):
    """depth_raw: (H,W) uint16 mm, from depth_image_to_np(). Returns (h,w,1)
    float32 in [0,1], resized to target_hw."""
    h, w = target_hw
    raw = cv2.resize(depth_raw, (w, h), interpolation=cv2.INTER_NEAREST)
    return normalize_depth(raw, depth_max)
