# codes/

Custom tooling for the real delta-arm project, on top of the `robomimic`
framework.

## Fresh-machine setup (clone → runnable)

Everything you need after cloning this repo onto a new machine. Steps 1–3 are
one-time; step 4 is a sanity check.

### 0. System prerequisites (install these first, outside this repo)

| requirement | why | notes |
|---|---|---|
| **NVIDIA GPU + driver** (CUDA 12.8-capable) | torch runs the policy on GPU | verified on RTX 5090 / driver 580.159 / CUDA 12.8; a different driver means a different torch `--index-url` in step 2 |
| **Miniconda / Anaconda** | the `robomimic_env` conda env | any recent version |
| **ROS2 Jazzy** (`/opt/ros/jazzy`) | only `live_inference/` needs it (`rclpy`) | skip if you only train / run `norobot_eval` |

### 1. Clone

```bash
git clone <your-repo-url> robomimic
cd robomimic
```

### 2. Create the `robomimic_env` conda env (one time)

```bash
conda create -n robomimic_env python=3.12 -y

# pyyaml/numpy/opencv aren't robomimic deps but rclpy and cv2 decode need them;
# installing them first keeps the next two steps' dependency resolution clean
conda run -n robomimic_env pip install pyyaml numpy opencv-python

# match the driver's CUDA version (cu128 here -- change if your driver differs)
conda run -n robomimic_env pip install torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128

# installs robomimic + diffusers/transformers/huggingface-hub/egl_probe/h5py/etc.
# (versions pinned in setup.py -- see the table further down)
conda run -n robomimic_env pip install -e .
```

### 3. Put the model checkpoint in place

Trained checkpoints are **not** in git (they're ~1.4 GB each). Get
`model_epoch_300.pth` (e.g. from `model.zip`) and place it where
`live_inference/config.json` expects it:

```
train/rsm_fan_lift_dp_delta_256/20260613201942/models/model_epoch_300.pth
```

Or put it anywhere and pass `--ckpt <path>` (and/or edit `config.json`).

> **Inference does NOT need the dataset.** All normalization stats are baked
> into the `.pth` at training time, so `live_inference/` runs from the model
> alone — the `datasets/*.hdf5` files are only needed to *train* or to run
> `norobot_eval/` (offline replay against ground truth).

### 4. Verify it end to end

```bash
# torch + CUDA
conda run -n robomimic_env python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# rclpy (only if you'll use live_inference; needs ROS2 sourced first)
bash -c "source /opt/ros/jazzy/setup.bash; conda run -n robomimic_env python -c 'import rclpy; print(\"rclpy OK\")'"

# training smoke test (2 epochs x 3 steps, output to /tmp, ~30s)
conda run -n robomimic_env python robomimic/scripts/train.py \
    --config diffusion_policy/diffusion_policy_rsm_delta.json --debug
```

To smoke-test the live deployment interface without a robot, see
[`live_inference/README.md`](live_inference/README.md) (§ Testing without a
robot) — it publishes fake sensor topics and opens the status UI.

## Environment: `robomimic_env`

One conda env, Python 3.12, covers everything below: training,
`norobot_eval/`, and `live_inference/` (the only one of the three that also
needs `rclpy`, via `source /opt/ros/jazzy/setup.bash` — that just points
`PYTHONPATH` at ROS2's install and is independent of which conda env is
active). Verified empirically: rclpy Node creation, cv2 decode, a real
checkpoint inference call, and `robomimic/scripts/train.py --debug` (dataset
loading, model build, forward/backward pass, checkpoint saving) all work
under this one env.

Confirmed working versions (GPU: RTX 5090, driver 580.159, CUDA 12.8):

| package | version |
|---|---|
| Python | 3.12.13 |
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| robomimic | 0.5.0 (editable install of this repo) |
| diffusers | 0.11.1 |
| transformers | 4.41.2 |
| huggingface-hub | 0.23.4 |
| egl_probe | 1.0.2 (builds from source; not actually used by inference or this project's training, since `rollout.enabled=false` and there's no robosuite/mujoco env — but `pip install -e .` still needs it to build) |
| h5py | 3.16.0 |
| numpy | 2.5.0 |
| opencv-python | 5.0.0.93 |
| PyYAML | 6.0.3 (rclpy dependency; not in setup.py, installed separately) |

The exact commands to build and verify this env are in **Fresh-machine setup**
above (steps 2 and 4).

`codes/norobot_eval/rosbridge_flow_test/` additionally needs system Python
3.12 + ROS2 Jazzy sourced for its own `rclpy`-based scripts
(`policy_bridge_node.py`, `replay_eval_node.py`) — that harness predates
`robomimic_env`'s migration to Python 3.12 and still uses a two-process
split; see its own README for why.

## Components

| folder | what it's for |
|---|---|
| [`training/`](training/) | Turn a real-robot HDF5 collection into a robomimic-trainable dataset (`convert_to_robomimic.py`), and sanity-check an HDF5's structure (`inspect_hdf5.py`). |
| [`norobot_eval/`](norobot_eval/) | Validate a trained checkpoint **without a robot**: replay a dataset demo through the policy and compare predicted vs. ground-truth actions. `norobot_eval_gui.py` does this directly (offline); `rosbridge_flow_test/` does the same thing but routed through a real ROS2 pipeline, to validate the pipeline itself. |
| [`live_inference/`](live_inference/) | The actual real-robot deployment interface: loads a checkpoint, subscribes to the robot's joint/RGB/depth topics, publishes a predicted `/delta_action`. |

`model_utils.py` (top level) holds `load_model()`, the checkpoint-loading
helper shared by `norobot_eval/` and `live_inference/` — neither imports it
through the other.

For the real-robot topic/message contract (what the data collector already
publishes), see [`../delta_arm_data_summary.md`](../delta_arm_data_summary.md).
