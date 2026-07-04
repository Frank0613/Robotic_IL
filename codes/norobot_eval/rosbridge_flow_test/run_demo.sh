#!/usr/bin/env bash
# End-to-end ROS-bridge flow test (no robot).
#
#   dataset --(export)--> npz --(ROS replay node, system py3.12)--> /eval/* topics
#        --> policy_bridge (system py3.12) --(socket)--> model_server (robomimic_env, conda py3.12, torch)
#        --> /policy/action --> replay node collects --> preds.npz
#        --> score_ros_run (robomimic_env) compares vs dataset GT
#
# Usage:
#   codes/norobot_eval/rosbridge_flow_test/run_demo.sh <ckpt.pth> [demo_name] [action_mode]
#
# Env overrides: DATASET=..., MAX_FRAMES=24 (quick smoke), ROS_SETUP=/opt/ros/jazzy/setup.bash
# (no `set -u`: the ROS setup.bash references unbound vars)
set -eo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
CKPT="${1:?usage: run_demo.sh <ckpt.pth> [demo_name] [action_mode]}"
CKPT="$(readlink -f "$CKPT" || true)"   # conda run may use a different cwd; make absolute
[ -n "$CKPT" ] && [ -f "$CKPT" ] || { echo "error: ckpt not found: $1" >&2; exit 1; }
DEMO="${2:-}"
MODE="${3:-delta}"
DATASET="${DATASET:-$HERE/../../../datasets/rsm_robomimic_delta_256.hdf5}"
ROS_SETUP="${ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
PY312="${PY312:-/usr/bin/python3.12}"
PORT="${PORT:-5599}"
# Resolve the robomimic_env interpreter directly. Backgrounding `conda run ... &`
# spawns a child python that survives killing the wrapper PID, so the model_server
# leaks past the run, keeps holding $PORT, and silently hijacks the NEXT run (the
# bridge connects to the wrong, stale model). Backgrounding the real interpreter
# means $! is the python PID and the cleanup trap can actually kill it.
CONDA_PY="${CONDA_PY:-$(conda run -n robomimic_env python -c 'import sys; print(sys.executable)')}"
MAX_FRAMES="${MAX_FRAMES:-0}"
DEMO_NPZ=/tmp/ros_eval_demo.npz
PREDS_NPZ=/tmp/ros_eval_preds.npz
EVAL_DIR="${EVAL_DIR:-$HERE/../../../evaluation}"
OUT_PNG="${OUT_PNG:-$EVAL_DIR/ros_eval_${DEMO:-firstvalid}_${MODE}.png}"

PIDS=()
free_port() {
    # Kill anything still listening on $PORT (e.g. a stale model_server from a
    # crashed/Ctrl-C'd run) so the bridge can't connect to the wrong model.
    local pids
    # `|| true`: grep returns non-zero when the port is free, which would abort
    # the whole script under `set -eo pipefail`. A free port is the normal case.
    pids="$(ss -ltnp 2>/dev/null | grep ":${PORT} " | grep -oP 'pid=\K[0-9]+' | sort -u || true)"
    if [ -n "$pids" ]; then
        echo "  port ${PORT} busy (pid: $(echo $pids)) -> killing stale server" >&2
        # shellcheck disable=SC2086
        kill -9 $pids 2>/dev/null || true
        sleep 1
    fi
}
cleanup() {
    for p in "${PIDS[@]:-}"; do kill -INT "$p" 2>/dev/null || true; done
    free_port   # belt-and-suspenders: never leave a server holding the port
}
trap cleanup EXIT

echo "== 1. export demo (robomimic_env) =="
conda run -n robomimic_env python "$HERE/export_demo.py" \
    --dataset "$DATASET" ${DEMO:+--demo "$DEMO"} --out "$DEMO_NPZ"

echo "== 2. start model_server (robomimic_env) =="
free_port   # make sure no stale server is squatting on $PORT before we start ours
"$CONDA_PY" "$HERE/model_server.py" --ckpt "$CKPT" --port "$PORT" &
PIDS+=($!)

echo "== 3. start policy_bridge (ROS2, system py3.12) =="
# shellcheck disable=SC1090
source "$ROS_SETUP"
"$PY312" "$HERE/policy_bridge_node.py" &
PIDS+=($!)

echo "== 4. replay + collect (ROS2, system py3.12, blocks until done) =="
"$PY312" "$HERE/replay_eval_node.py" --npz "$DEMO_NPZ" --out "$PREDS_NPZ" --max-frames "$MAX_FRAMES"

echo "== 5. score vs dataset GT (robomimic_env) =="
mkdir -p "$EVAL_DIR"
# clear the ROS PYTHONPATH so it cannot leak into robomimic_env's interpreter
# (both are Python 3.12 now, but separate installs with different site-packages)
PYTHONPATH="" conda run -n robomimic_env python "$HERE/score_ros_run.py" \
    --demo-npz "$DEMO_NPZ" --preds-npz "$PREDS_NPZ" --action-mode "$MODE" \
    --out-png "$OUT_PNG"

echo "== done. plot at $OUT_PNG =="
