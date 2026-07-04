# Live deployment (real robot integration)

This is the interface for a real robot integration: someone else's node
publishes sensor topics and consumes the action topic; this repo only owns
loading the model and running inference. It's a single ordinary rclpy node
that loads the model directly and calls it inline — no socket, no second
process (see [`../README.md`](../README.md) for why that's possible and how
to set up `robomimic_env`).

## Files

- `run.py`      — entry point: argparse + rclpy init/spin/shutdown (**run this**)
- `ros_node.py` — the `LiveInterface(Node)` class: subscriptions, publisher, the inference tick
- `policy.py`   — `PolicyRunner`: loads the checkpoint, owns the causal obs window, runs one inference step
- `sensors.py`  — decode/resize/normalize raw ROS payloads (RGB, depth) into what the policy expects
- `config.py`   — loads `config.json`
- `config.json` — **edit this** to change the checkpoint path, topic names, depth_max, or rate

Dependency direction: `run.py` → `ros_node.py` → `policy.py` → `sensors.py`.
`policy.py` also reaches up to `../model_utils.py` (shared with
`codes/norobot_eval/`) for the actual checkpoint-loading call.

## Usage

Settings come from **`config.json`** — edit it to reconfigure, no code changes
needed. Pass `--config <path>` to use a different file, or individual flags
(`--ckpt`, `--joint-topic`, `--rgb-topic`, `--depth-topic`, `--action-topic`,
`--depth-max`, `--rate`, `--device`) to override one field for a one-off run.

```bash
source /opt/ros/jazzy/setup.bash
conda run -n robomimic_env python codes/live_inference/run.py
```

It logs a line every 10 frames (`frame N: action=[...] (x.x ms)`). `Ctrl+C`
to stop. To change the model or any topic, edit `config.json` and restart.

## Topics / messages

Default topic names/types (in `config.json`) match what the real delta-arm
data collector already publishes on the robot's ROS graph (see
[`../../delta_arm_data_summary.md`](../../delta_arm_data_summary.md)), so
`run.py` can subscribe directly with no extra remap layer:

| topic (`config.json` key)                                | type                          | content |
|-----------------------------------------------------------|-------------------------------|---------|
| `/left_arm/joint_states` (`joint_topic`)                   | `sensor_msgs/JointState`      | `position`/`velocity` = real joint angles/speeds (rad, rad/s). Velocity is zero-filled if the driver doesn't populate it (same fallback the data collector uses). |
| `/camera_L/color/image_raw/compressed` (`rgb_topic`)       | `sensor_msgs/CompressedImage` | RGB frame, jpeg/png encoded |
| `/camera_L/depth/aligned_depth_to_color` (`depth_topic`)   | `sensor_msgs/Image`           | encoding `16UC1`, raw depth in **mm**, not normalized |
| `/delta_action` (`action_topic`)                           | `std_msgs/Float32MultiArray`  | 7 floats: `[joint_delta_0..5, gripper]`, raw model output (un-normalized). New topic — the collector's own real-robot output is only `/gripper_cmd` (Int32); whoever integrates the real arm computes `target = current_joint_pos + delta` and is responsible for turning it into a joint command. |

Depth normalization (`clip(0, depth_max) / depth_max`, 0 = invalid → treated
as farthest) uses the exact same logic as
`codes/training/convert_to_robomimic.py:process_depth` (reimplemented in
`sensors.py:normalize_depth`), controlled by `depth_max` in the config (mm) —
this must match the working distance range used when the dataset was built,
there is no way to recover it automatically from the checkpoint.

No `message_filters` time-sync is used, matching the data collector's own
`ros_subscriber.py` (its `ApproximateTimeSynchronizer` was removed): each
subscriber just keeps the latest message, and `run.py` runs inference on a
fixed timer (`rate`, default 10 Hz) instead.

The camera topics publish **BEST_EFFORT** QoS (sensor-data profile); a
RELIABLE subscriber silently receives nothing from them. `ros_node.py`
subscribes RGB/Depth with a BEST_EFFORT profile and keeps joint_state
RELIABLE, mirroring `ros_subscriber.py` in `RealRobot_DataCollector` exactly.
