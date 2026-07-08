# ROS bridge flow test (no robot)

Validate the **ROS deployment pipeline** for the Diffusion Policy without a real
arm: dataset obs are published on ROS2 topics, the model consumes them over ROS
and emits an action (delta), and the action is compared offline against the
dataset ground truth.

This is a **historical validation harness** for the old two-process socket
bridge design. The live deployment interface (`codes/live_inference/`) no
longer uses that design (see its README for why), but this harness is still
useful for offline-replaying a dataset demo through a real ROS2 pipeline and
scoring the result against ground truth, without needing a robot.

```
 dataset ─export→ npz ─[ROS replay node, system py3.12]→ /eval/rgb,/eval/depth,/eval/joint_states
                                              │
                     [policy_bridge, system py3.12] ──socket──> [model_server, robomimic_env + torch]
                                              │                          (runs the policy)
                                /policy/action ◀───────────────────────── delta
                                              │
                [replay node collects] → preds.npz → [score_ros_run, robomimic_env] vs dataset GT
```

## Why two processes?

This predates `codes/live_inference/`'s Python 3.12 migration: `rclpy`
(ROS2 Jazzy) needed system Python 3.12 while `robomimic_env` (torch/robomimic)
was Python 3.10 back then, so the model ran as a small **socket server** and a
thin **rclpy bridge node** carried data between ROS topics and that server.
`robomimic_env` is Python 3.12 now too (see [`../../README.md`](../../README.md)),
so that specific version conflict is gone — but `robomimic_env` (conda) and
the system's `/usr/bin/python3.12` (used for the ROS side) are still two
separate installations with different site-packages, and this is just a
validation harness, not something that needs collapsing into one process the
way `codes/live_inference/` did. So it keeps the two-process split.

> Note: the server keeps a **causal** obs window (last `To` frames), like the
> real arm. The offline `norobot_eval_gui.py` uses a forward window, so ROS-run
> MSE can differ slightly — that is expected; this test checks the *pipeline*.

## Topics / messages (standard ROS msgs)

| topic                  | type                         | content                          |
|------------------------|------------------------------|----------------------------------|
| `/eval/rgb`            | `sensor_msgs/Image` (rgb8)   | RGB obs frame                    |
| `/eval/depth`          | `sensor_msgs/Image` (32FC1)  | depth obs frame                  |
| `/eval/joint_states`   | `sensor_msgs/JointState`     | `position`=joint_pos, `velocity`=joint_vel |
| `/eval/reset`          | `std_msgs/Int32`             | start a new episode (frame count)|
| `/policy/action`       | `std_msgs/Float64MultiArray` | predicted action (delta), 7 dims |

`policy_bridge` time-syncs the 3 obs topics with `message_filters`.

## One-shot run

```bash
# quick smoke (first 24 frames):
MAX_FRAMES=24 codes/norobot_eval/rosbridge_flow_test/run_demo.sh train/rsm_fan_lift_dp_delta_256/<run>/last.pth demo_15 delta

# full demo:
codes/norobot_eval/rosbridge_flow_test/run_demo.sh train/rsm_fan_lift_dp_delta_256/<run>/last.pth demo_15 delta
```

Output: per-dim MSE table + `OVERALL PASS/FAIL`, and a plot at
`/tmp/ros_eval_plot.png`.

## Manual run (3 terminals)

```bash
# T1 — export demo + model server (robomimic_env)
conda run -n robomimic_env python codes/norobot_eval/rosbridge_flow_test/export_demo.py \
    --dataset datasets/rsm_robomimic_delta_256.hdf5 --demo demo_15 --out /tmp/ros_eval_demo.npz
conda run -n robomimic_env python codes/norobot_eval/rosbridge_flow_test/model_server.py \
    --ckpt train/rsm_fan_lift_dp_delta_256/<run>/last.pth

# T2 — policy bridge (ROS2, system py3.12)
source /opt/ros/jazzy/setup.bash
/usr/bin/python3.12 codes/norobot_eval/rosbridge_flow_test/policy_bridge_node.py

# T3 — replay + collect (ROS2, system py3.12)
source /opt/ros/jazzy/setup.bash
/usr/bin/python3.12 codes/norobot_eval/rosbridge_flow_test/replay_eval_node.py \
    --npz /tmp/ros_eval_demo.npz --out /tmp/ros_eval_preds.npz   # add --max-frames 24 for a smoke

# then score (robomimic_env)
conda run -n robomimic_env python codes/norobot_eval/rosbridge_flow_test/score_ros_run.py --action-mode delta
```

## Files

- `model_server.py`      — robomimic_env: socket inference server (causal To-window, action queue)
- `policy_bridge_node.py`— system py3.12: ROS obs → socket → `/policy/action` (**the reusable arm node**)
- `replay_eval_node.py`  — system py3.12: publish dataset obs lock-step, collect actions → preds.npz
- `export_demo.py`       — robomimic_env: dataset demo → npz (so the ROS side needs no h5py)
- `score_ros_run.py`     — robomimic_env: compare preds vs GT, reusing norobot_eval_gui scoring/plots
- `run_demo.sh`          — orchestrates all of the above
