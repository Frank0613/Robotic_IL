"""
NoRobot Eval GUI -- validate a trained Diffusion Policy by checking whether it
reproduces a demo trajectory, without any robot / simulator.

Idea: feed a demo's observations (teacher-forcing, always ground-truth) from the
dataset into model.pth, collect the model's per-step output actions via
"deployment-style replay", then compare against the demo's ground-truth actions
dimension-by-dimension (MSE + threshold + overlay plot).

Action modes (--action-mode, or the GUI "Action" chips / hotkey 'd'):
    absolute : compare predicted vs GT actions directly (joint angles).
    delta    : the dataset's joint dims are per-step deltas. Comparing them
               step-by-step is misleading (GT deltas are tiny, so a "predict
               zero" model trivially scores low MSE). In delta mode we instead
               OPEN-LOOP INTEGRATE (cumsum) both predicted and GT joint deltas
               into absolute trajectories -- starting from obs[init_key][0] --
               and compare those. This reveals drift and makes MSE meaningful.
               The gripper dim is always compared as-is.

Correctness note: this config uses frame_stack=1, so rollout does NOT apply the
FrameStackWrapper, but training takes obs[:, :To=2] (global_cond dim = To*obs_feat).
Therefore at inference we must manually stack obs into a forward window obs[t:t+To];
otherwise the dimensions mismatch and the network errors out.

Usage (inside the robomimic_env conda env):
    # CLI only (verify inference works; prints per-dim MSE for every demo)
    conda run -n robomimic_env python codes/norobot_eval/norobot_eval_gui.py --no-gui \
        --action-mode delta \
        --ckpt train/rsm_fan_lift_dp_delta/.../last.pth

    # GUI
    conda run -n robomimic_env python codes/norobot_eval/norobot_eval_gui.py
"""

import os
import sys
import argparse
import threading

import numpy as np
import h5py

import matplotlib
matplotlib.use("Agg")  # headless backend: render to memory, then blit into cv2
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # -> codes/
from model_utils import load_model  # noqa: E402


# ----- default paths -----
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATASET = os.path.join(REPO_ROOT, "datasets", "rsm_robomimic_abs.hdf5")
DEFAULT_CKPT_DIR = os.path.join(REPO_ROOT, "rsm_trained_models")

# action dimension labels (joints 0-5, gripper). In delta mode the joint dims are
# shown as integrated absolute trajectories, so the "rad" unit still applies.
ACTION_LABELS = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]


# =====================================================================
# 1. Model loading -- see model_utils.load_model (shared with live_inference)
# =====================================================================


# =====================================================================
# 2. Demo reading
# =====================================================================
def list_demos(dataset_path, filter_key="valid"):
    """List demo names. Returns all demos when filter_key is None/'all'."""
    with h5py.File(dataset_path, "r") as f:
        if filter_key in (None, "all"):
            demos = list(f["data"].keys())
        else:
            if "mask" not in f or filter_key not in f["mask"]:
                raise KeyError(
                    f"mask/{filter_key} not found in {dataset_path}; try --filter-key all"
                )
            demos = [d.decode() if isinstance(d, bytes) else str(d)
                     for d in f["mask"][filter_key][()]]
        # sort by demo index
        demos = sorted(demos, key=lambda s: int(s.split("_")[-1]) if "_" in s else 0)
    return demos


def load_demo(dataset_path, demo_name, obs_keys):
    """Read a single demo's obs (raw stored format) and actions."""
    out_obs = {}
    with h5py.File(dataset_path, "r") as f:
        grp = f["data"][demo_name]
        for k in obs_keys:
            out_obs[k] = grp["obs"][k][()]
        actions = grp["actions"][()].astype(np.float32)
    return out_obs, actions


# =====================================================================
# 3. NoRobot Eval core (deployment-style replay + correct To stacking)
# =====================================================================
def run_norobot_eval(policy, obs, actions, To, progress_cb=None):
    """Feed ground-truth obs (forward To window) into the policy step by step,
    collecting the per-step predicted action.

    Returns preds, shape [T, ac_dim], already un-normalized (same scale as the
    demo actions). If progress_cb is given, it is called as progress_cb(t, T).
    """
    T = actions.shape[0]
    obs_keys = list(obs.keys())

    policy.start_episode()  # reset obs/action queue
    preds = []
    for t in range(T):
        # forward To window; pad the tail by repeating the last frame (like pad_seq_length)
        win = {
            k: np.stack([obs[k][min(t + i, T - 1)] for i in range(To)], axis=0)
            for k in obs_keys
        }
        a = policy(ob=win)  # __call__ batches to [1, To, ...]; correct dims; auto un-normalize
        preds.append(np.asarray(a, dtype=np.float32))
        if progress_cb is not None and (t % 5 == 0 or t == T - 1):
            progress_cb(t + 1, T)
    return np.stack(preds, axis=0)


def integrate_delta_to_abs(deltas, init_abs):
    """Open-loop integrate per-step joint deltas into an absolute trajectory.

    out[0] = init_abs;  out[t] = init_abs + sum(deltas[0..t-1]).
    Because it chains predictions without re-anchoring to ground truth, any
    per-step bias accumulates -- so a drifting delta policy visibly diverges,
    which is exactly what we want to see.
    """
    T = deltas.shape[0]
    out = np.empty((T, deltas.shape[1]), dtype=np.float32)
    cur = np.asarray(init_abs, dtype=np.float32).copy()
    for t in range(T):
        out[t] = cur
        cur = cur + deltas[t]
    return out


def to_plot_space(preds, actions, obs, action_mode, init_obs_key="joint_pos"):
    """Transform actions into the space used for scoring / plotting.

    absolute mode: returned unchanged.
    delta mode   : joint dims (0..5) of BOTH preds and GT are open-loop integrated
                   into absolute trajectories, starting from obs[init_obs_key][0].
                   The gripper dim (last) is left unchanged.
    """
    if action_mode != "delta":
        return preds, actions
    if init_obs_key not in obs:
        raise KeyError(
            f"init-obs-key '{init_obs_key}' not found in obs keys {list(obs.keys())}; "
            f"pass --init-obs-key to point at the 6-dim joint position obs"
        )
    init = np.asarray(obs[init_obs_key])[0].astype(np.float32)[:6]
    preds_abs = preds.copy()
    actions_abs = actions.copy()
    preds_abs[:, :6] = integrate_delta_to_abs(preds[:, :6], init)
    actions_abs[:, :6] = integrate_delta_to_abs(actions[:, :6], init)
    return preds_abs, actions_abs


def score(preds, actions, threshold_joint, threshold_gripper):
    """Per-dim MSE + threshold PASS/FAIL. Returns a dict."""
    ac_dim = actions.shape[1]
    mse = np.mean((preds - actions) ** 2, axis=0)  # [ac_dim]
    rmse = np.sqrt(mse)
    thresholds = np.array(
        [threshold_joint] * (ac_dim - 1) + [threshold_gripper], dtype=np.float32
    )[:ac_dim]
    passed = mse <= thresholds
    return dict(
        mse=mse, rmse=rmse, thresholds=thresholds,
        passed=passed, overall=bool(np.all(passed)),
    )


# =====================================================================
# 4. Visualization (matplotlib -> numpy image)
# =====================================================================
def render_plots(preds, actions, result, demo_name, ckpt_name, ac_dim=7,
                 fig_w=8.0, fig_h=9.0, dpi=100, span_k=8.0, mode_tag=""):
    """Plot predicted vs ground-truth overlay; return an RGB numpy image [H,W,3] uint8.

    The Y axis of each subplot is the action value for that dimension (joints 0-5 in
    radians, gripper as a 0/1 command). To stop matplotlib from auto-zooming a nearly
    constant signal -- which makes a sub-threshold error look huge -- we enforce a
    minimum Y span tied to the pass/fail threshold: half-span >= span_k * sqrt(thr).
    So an error right at the threshold spans ~1/span_k of the view; PASS looks small.

    In delta mode the joint curves passed in here are already integrated absolute
    trajectories, so they span the real joint range and are not flattened.
    """
    ncols = 2
    nrows = int(np.ceil(ac_dim / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), dpi=dpi)
    axes = np.array(axes).reshape(-1)

    t = np.arange(actions.shape[0])
    for d in range(ac_dim):
        ax = axes[d]
        ax.plot(t, actions[:, d], "k--", linewidth=1.2, label="GT")
        ax.plot(t, preds[:, d], "r-", linewidth=1.0, label="Pred")
        tag = "PASS" if result["passed"][d] else "FAIL"
        color = "green" if result["passed"][d] else "red"
        ax.set_title(
            f"{ACTION_LABELS[d]}  MSE={result['mse'][d]:.4f} [{tag}]",
            fontsize=9, color=color,
        )

        # consistent Y scaling: max(actual data half-span, threshold-based floor)
        combined = np.concatenate([actions[:, d], preds[:, d]])
        lo, hi = float(combined.min()), float(combined.max())
        center = 0.5 * (lo + hi)
        data_half = 0.5 * (hi - lo)
        thr_half = span_k * float(np.sqrt(result["thresholds"][d]))
        half = max(data_half, thr_half) * 1.1 + 1e-6
        ax.set_ylim(center - half, center + half)

        ax.set_ylabel("rad" if d < ac_dim - 1 else "0/1", fontsize=7)
        if d >= ac_dim - ncols:
            ax.set_xlabel("timestep", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.25)
        if d == 0:
            ax.legend(fontsize=7, loc="best")
    # turn off any unused subplots
    for d in range(ac_dim, len(axes)):
        axes[d].axis("off")

    overall = "PASS" if result["overall"] else "FAIL"
    ocolor = "green" if result["overall"] else "red"
    mtag = f"  |  mode: {mode_tag}" if mode_tag else ""
    fig.suptitle(
        f"NoRobot Eval - {demo_name}  |  {ckpt_name}{mtag}  |  OVERALL: {overall}",
        fontsize=11, color=ocolor,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())  # [H,W,4]
    img = buf[:, :, :3].copy()  # RGB
    plt.close(fig)
    return img


# =====================================================================
# 5. CLI batch mode
# =====================================================================
def run_cli(args):
    if not args.ckpt:
        raise SystemExit("--no-gui mode requires --ckpt to point at a model.pth")
    print(f"[load] {args.ckpt}")
    policy, info = load_model(args.ckpt, device=args.device)
    print(f"[info] To={info['To']} Ta={info['Ta']} Tp={info['Tp']} "
          f"obs_keys={info['obs_keys']} ac_dim={info['ac_dim']}")
    print(f"[mode] action-mode={args.action_mode}"
          + (f" (integrate joints from obs['{args.init_obs_key}'][0])"
             if args.action_mode == "delta" else ""))

    demos = list_demos(args.dataset, args.filter_key)
    print(f"[data] {args.dataset}  filter={args.filter_key}  #demos={len(demos)}")

    header = "demo".ljust(12) + "".join(l[:8].rjust(10) for l in ACTION_LABELS) + "   overall"
    print(header)
    print("-" * len(header))

    all_overall = []
    for name in demos:
        obs, actions = load_demo(args.dataset, name, info["obs_keys"])
        preds = run_norobot_eval(policy, obs, actions, info["To"])
        # delta mode: integrate joint deltas into absolute trajectories first
        preds, actions = to_plot_space(preds, actions, obs, args.action_mode, args.init_obs_key)
        res = score(preds, actions, args.threshold_joint, args.threshold_gripper)
        all_overall.append(res["overall"])
        row = name.ljust(12) + "".join(f"{m:10.4f}" for m in res["mse"])
        row += "     PASS" if res["overall"] else "     FAIL"
        print(row)

    n_pass = sum(all_overall)
    print("-" * len(header))
    print(f"[summary] PASS {n_pass}/{len(demos)} demos "
          f"(threshold joint={args.threshold_joint}, gripper={args.threshold_gripper})")


# =====================================================================
# 6. cv2 + tkinter GUI
# =====================================================================
def run_gui(args):
    import cv2
    import tkinter as tk
    from tkinter import filedialog

    # hidden tk root, used only for the native file dialog
    tk_root = tk.Tk()
    tk_root.withdraw()

    # ---- layout constants ----
    WIN = "NoRobot Eval"
    PANEL_W = 410           # left control / status column
    PLOT_W, PLOT_H = 860, 1060   # right plot area
    H = PLOT_H
    W = PANEL_W + PLOT_W
    MARGIN = 20
    BG = (38, 38, 40)       # canvas background (BGR)

    # phase -> (banner text, BGR color). RESULT/ERROR resolved dynamically.
    PHASE_COLOR = dict(
        NO_MODEL=(120, 120, 120),
        MODEL_SELECTED=(40, 170, 235),
        LOADING=(210, 150, 60),
        READY=(80, 190, 80),
        EVALUATING=(210, 150, 60),
        RESULT=(80, 190, 80),
        ERROR=(70, 70, 220),
    )

    # ---- state ----
    state = dict(
        ckpt=args.ckpt,
        dataset=args.dataset,
        filter_key=args.filter_key,
        action_mode=args.action_mode,
        policy=None,
        info=None,
        demos=[],
        demo_idx=0,
        plot_img=None,
        result=None,
        phase="NO_MODEL" if not args.ckpt else "MODEL_SELECTED",
        status="Select a model .pth to begin.",
        busy=False,            # True while a background worker (load/eval) runs
        progress=None,         # (done, total) for the eval progress bar
        pending_render=None,   # raw eval arrays awaiting main-thread rendering
        tick=0,                # frame counter for the indeterminate spinner
        list_offset=0,         # scroll offset of the demo list
        drag_sb=False,         # True while dragging the demo-list scrollbar
    )

    def set_state(phase=None, status=None):
        if phase is not None:
            state["phase"] = phase
        if status is not None:
            state["status"] = status

    # preload demo list
    try:
        state["demos"] = list_demos(state["dataset"], state["filter_key"])
    except Exception as e:
        set_state("ERROR", f"Dataset load failed: {e}")

    # ---- buttons (explicit geometry; supports half-width pairs) ----
    BTN_H = 44
    PITCH = 56              # vertical distance between buttons (BTN_H + gap)
    full_w = PANEL_W - 2 * MARGIN
    half_w = (full_w - 10) // 2
    by = 70                 # first button top
    buttons = [
        dict(x=MARGIN, y=by + 0 * PITCH, w=full_w, h=BTN_H, label="Select Model",   key="select_model"),
        dict(x=MARGIN, y=by + 1 * PITCH, w=full_w, h=BTN_H, label="Select Dataset", key="select_dataset"),
        dict(x=MARGIN, y=by + 2 * PITCH, w=full_w, h=BTN_H, label="Load Model",     key="load"),
        dict(x=MARGIN, y=by + 3 * PITCH, w=full_w, h=BTN_H, label="Run Eval",       key="run"),
        dict(x=MARGIN,               y=by + 4 * PITCH, w=half_w, h=BTN_H, label="< Prev", key="prev"),
        dict(x=MARGIN + half_w + 10, y=by + 4 * PITCH, w=half_w, h=BTN_H, label="Next >", key="next"),
        dict(x=MARGIN, y=by + 5 * PITCH, w=full_w, h=BTN_H, label="Quit", key="quit"),
    ]

    def base(p):
        """Show just the file name, not the full path."""
        return os.path.basename(p) if p else "(none)"

    # ---- background worker plumbing ----
    # Heavy work (model load, eval) runs on a daemon thread so the cv2 window
    # keeps repainting and stays responsive. Only torch/h5py work happens on the
    # worker; all cv2 / matplotlib drawing stays on the main thread.
    def _run_in_thread(fn):
        if state["busy"]:
            return
        state["busy"] = True
        th = threading.Thread(target=fn, daemon=True)
        th.start()

    # ---- actions ----
    def do_select_model():
        if state["busy"]:
            return
        p = filedialog.askopenfilename(
            title="Select model .pth",
            initialdir=DEFAULT_CKPT_DIR if os.path.isdir(DEFAULT_CKPT_DIR) else REPO_ROOT,
            filetypes=[("PyTorch checkpoint", "*.pth"), ("All files", "*.*")],
        )
        if p:
            state["ckpt"] = p
            state["policy"] = None
            set_state("MODEL_SELECTED", f"Model selected: {base(p)}. Press Load Model.")

    def do_select_dataset():
        if state["busy"]:
            return
        p = filedialog.askopenfilename(
            title="Select dataset .hdf5",
            initialdir=os.path.join(REPO_ROOT, "datasets"),
            filetypes=[("HDF5", "*.hdf5"), ("All files", "*.*")],
        )
        if p:
            state["dataset"] = p
            state["plot_img"] = None
            state["result"] = None
            _reload_demos(state["filter_key"])
            set_state(status=f"Dataset: {base(p)}  (#demos={len(state['demos'])})")

    def _load_worker():
        try:
            policy, info = load_model(state["ckpt"], device=args.device)
            state["policy"] = policy
            state["info"] = info
            set_state("READY",
                      f"Model ready (To={info['To']} Ta={info['Ta']}). Press Run Eval.")
        except Exception as e:
            state["policy"] = None
            set_state("ERROR", f"Model load failed: {e}")
        finally:
            state["busy"] = False

    def do_load():
        if state["busy"]:
            return
        if not state["ckpt"]:
            set_state("NO_MODEL", "No model selected. Press Select Model first.")
            return
        set_state("LOADING", f"Loading model: {base(state['ckpt'])} ... (first load may take ~20s)")
        _run_in_thread(_load_worker)

    # Demo selection is drawn directly on the canvas (filter chips + a scrollable
    # list) instead of using cv2 trackbars or a tkinter dialog: this OpenCV build
    # uses the Qt5 backend, where a custom tkinter window crashes the process and
    # trackbars are easy to miss. Self-drawn controls are always visible.
    FILTERS = ["valid", "train", "all"]
    AMODE_MODES = ["absolute", "delta"]

    # control geometry (also used by the mouse hit-testing).
    # An "Action" chip row was inserted above the filter row, so the filter chips
    # and the demo list were nudged down to make room.
    AMODE_Y = 568
    CHIPS_Y, CHIP_H = 620, 30
    LIST_TOP, ROW_H = 672, 24

    def _list_bottom():
        return H - 42

    def _visible_rows():
        return max(1, (_list_bottom() - LIST_TOP) // ROW_H)

    def _max_offset():
        return max(0, len(state["demos"]) - _visible_rows())

    def _ensure_visible(idx):
        vis = _visible_rows()
        if idx < state["list_offset"]:
            state["list_offset"] = idx
        elif idx >= state["list_offset"] + vis:
            state["list_offset"] = idx - vis + 1

    def _chip_rects():
        cw = (full_w - 2 * 8) // 3
        return [(MARGIN + i * (cw + 8), CHIPS_Y, cw, CHIP_H) for i in range(3)]

    def _amode_rects():
        cw = (full_w - 8) // 2
        return [(MARGIN + i * (cw + 8), AMODE_Y, cw, CHIP_H) for i in range(2)]

    def _set_action_mode(m):
        """Switch absolute/delta; clears any stale result (recompute differs)."""
        if m != state["action_mode"]:
            state["action_mode"] = m
            state["plot_img"] = None
            state["result"] = None
            set_state(status=f"action mode = {m}")

    def _toggle_action_mode():
        _set_action_mode("delta" if state["action_mode"] == "absolute" else "absolute")

    def _set_demo(idx):
        """Jump to a specific demo index and clear any stale result."""
        if not state["demos"]:
            return
        state["demo_idx"] = idx % len(state["demos"])
        _ensure_visible(state["demo_idx"])
        state["plot_img"] = None
        state["result"] = None
        ph = "READY" if state["policy"] is not None else state["phase"]
        set_state(ph if ph in PHASE_COLOR else None,
                  f"Demo: {state['demos'][state['demo_idx']]}")

    def _reload_demos(filter_key):
        """Reload the demo list for a filter and reset the list view."""
        try:
            demos = list_demos(state["dataset"], filter_key)
        except Exception as e:
            demos = []
            set_state("ERROR", f"Dataset load failed: {e}")
        state["filter_key"] = filter_key
        state["demos"] = demos
        state["demo_idx"] = 0
        state["list_offset"] = 0
        state["plot_img"] = None
        state["result"] = None
        if demos:
            set_state(status=f"filter={filter_key}  (#demos={len(demos)})")

    def _step_demo(delta):
        if state["busy"] or not state["demos"]:
            return
        _set_demo(state["demo_idx"] + delta)

    def _scroll_list(step):
        if state["demos"]:
            state["list_offset"] = max(0, min(state["list_offset"] + step, _max_offset()))

    def _scrollbar_to(my):
        """Map a y position on the scrollbar track to a list offset."""
        mo = _max_offset()
        if mo <= 0:
            return
        track_h = max(1, _list_bottom() - LIST_TOP)
        frac = (my - LIST_TOP) / track_h
        state["list_offset"] = max(0, min(mo, int(round(frac * mo))))

    SB_X0 = PANEL_W - MARGIN - 12   # scrollbar left edge (wide enough to click)

    def _in_scrollbar(mx, my):
        return (state["demos"] and len(state["demos"]) > _visible_rows()
                and SB_X0 <= mx <= PANEL_W - MARGIN
                and LIST_TOP <= my <= _list_bottom())

    def _eval_worker(name):
        try:
            info = state["info"]
            obs, actions = load_demo(state["dataset"], name, info["obs_keys"])

            def cb(done, total):
                state["progress"] = (done, total)

            preds = run_norobot_eval(state["policy"], obs, actions, info["To"], progress_cb=cb)
            # delta mode: integrate joint deltas into absolute trajectories first
            preds, actions = to_plot_space(preds, actions, obs, state["action_mode"], args.init_obs_key)
            res = score(preds, actions, args.threshold_joint, args.threshold_gripper)
            # hand raw arrays to the main thread for matplotlib rendering
            state["pending_render"] = dict(name=name, preds=preds, actions=actions,
                                           res=res, mode=state["action_mode"])
        except Exception as e:
            set_state("ERROR", f"Eval failed: {e}")
        finally:
            state["progress"] = None
            state["busy"] = False

    def do_run():
        if state["busy"]:
            return
        if state["policy"] is None:
            set_state(status="Load a model before running eval.")
            return
        if not state["demos"]:
            set_state("ERROR", "No demos available.")
            return
        name = state["demos"][state["demo_idx"]]
        state["result"] = None
        state["plot_img"] = None
        state["progress"] = (0, 1)
        set_state("EVALUATING", f"Evaluating {name} ({state['action_mode']}) ... (DDPM inference, ~1-2 min)")
        _run_in_thread(lambda: _eval_worker(name))

    def do_prev():
        _step_demo(-1)

    def do_next():
        _step_demo(+1)

    actions_map = dict(
        select_model=do_select_model, select_dataset=do_select_dataset,
        load=do_load, run=do_run,
        prev=do_prev, next=do_next,
    )

    def on_mouse(event, mx, my, flags, param):
        # mouse wheel scrolls the demo list (may be inert under WINDOW_GUI_NORMAL)
        if event == cv2.EVENT_MOUSEWHEEL:
            try:
                delta = cv2.getMouseWheelDelta(flags)
            except Exception:
                delta = flags
            _scroll_list(-1 if delta > 0 else 1)
            return
        # scrollbar drag: track left-button motion / release
        if event == cv2.EVENT_MOUSEMOVE:
            if state["drag_sb"] and (flags & cv2.EVENT_FLAG_LBUTTON):
                _scrollbar_to(my)
            else:
                state["drag_sb"] = False
            return
        if event == cv2.EVENT_LBUTTONUP:
            state["drag_sb"] = False
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        # 1) buttons
        for b in buttons:
            if b["x"] <= mx <= b["x"] + b["w"] and b["y"] <= my <= b["y"] + b["h"]:
                if b["key"] == "quit":
                    state["phase"] = "__quit__"
                elif not state["busy"]:
                    actions_map[b["key"]]()
                return

        # 2) scrollbar (click to jump / start dragging; view-only, allowed any time)
        if _in_scrollbar(mx, my):
            state["drag_sb"] = True
            _scrollbar_to(my)
            return

        if state["busy"]:
            return

        # 2.5) action-mode chips
        for m, (cx, cy, cw, ch) in zip(AMODE_MODES, _amode_rects()):
            if cx <= mx <= cx + cw and cy <= my <= cy + ch:
                _set_action_mode(m)
                return

        # 3) filter chips
        for fk, (cx, cy, cw, ch) in zip(FILTERS, _chip_rects()):
            if cx <= mx <= cx + cw and cy <= my <= cy + ch:
                if fk != state["filter_key"]:
                    _reload_demos(fk)
                return

        # 4) demo list rows
        if state["demos"] and LIST_TOP <= my < _list_bottom() and MARGIN <= mx < SB_X0:
            row = (my - LIST_TOP) // ROW_H
            idx = state["list_offset"] + row
            if 0 <= idx < len(state["demos"]):
                _set_demo(idx)
            return

    # ---- drawing helpers ----
    def _put(canvas, text, org, scale=0.5, color=(230, 230, 230), thick=1):
        cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def _banner_text():
        ph = state["phase"]
        if ph == "RESULT" and state["result"] is not None:
            ok = state["result"]["overall"]
            return ("RESULT: PASS" if ok else "RESULT: FAIL",
                    (80, 190, 80) if ok else (70, 70, 220))
        labels = dict(
            NO_MODEL="NO MODEL", MODEL_SELECTED="MODEL SELECTED",
            LOADING="LOADING MODEL...", READY="MODEL READY",
            EVALUATING="EVALUATING...", ERROR="ERROR",
        )
        return labels.get(ph, ph), PHASE_COLOR.get(ph, (120, 120, 120))

    def _draw():
        canvas = np.full((H, W, 3), BG, np.uint8)

        # title
        _put(canvas, "NoRobot Eval  -  Diffusion Policy", (MARGIN, 34), 0.62, (240, 240, 240), 1)
        cv2.line(canvas, (MARGIN, 44), (PANEL_W - MARGIN, 44), (80, 80, 80), 1)

        # buttons (dimmed while a task runs, except Quit)
        for b in buttons:
            c = (95, 95, 95)
            if b["key"] == "load" and state["policy"] is not None:
                c = (60, 120, 60)
            if b["key"] == "run" and state["policy"] is not None:
                c = (70, 110, 140)
            if b["key"] == "quit":
                c = (60, 60, 120)
            disabled = state["busy"] and b["key"] != "quit"
            if disabled:
                c = tuple(int(v * 0.45) for v in c)
            txt_col = (130, 130, 130) if disabled else (255, 255, 255)
            cv2.rectangle(canvas, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), c, -1)
            cv2.rectangle(canvas, (b["x"], b["y"]), (b["x"] + b["w"], b["y"] + b["h"]), (190, 190, 190), 1)
            _put(canvas, b["label"], (b["x"] + 14, b["y"] + 27), 0.58, txt_col, 1)

        # ---- STATE banner ----
        by0, by1 = 414, 472
        btext, bcol = _banner_text()
        dark = tuple(int(c * 0.35) for c in bcol)
        cv2.rectangle(canvas, (MARGIN, by0), (PANEL_W - MARGIN, by1), dark, -1)
        cv2.rectangle(canvas, (MARGIN, by0), (PANEL_W - MARGIN, by1), bcol, 2)
        _put(canvas, "STATE", (MARGIN + 12, by0 + 20), 0.45, (200, 200, 200), 1)
        _put(canvas, btext, (MARGIN + 12, by0 + 40), 0.7, bcol, 2)

        # ---- progress bar (eval = determinate, load = indeterminate spinner) ----
        pbx0, pbx1, pby = MARGIN, PANEL_W - MARGIN, by1 + 8
        if state["progress"] is not None:
            done, total = state["progress"]
            frac = done / max(total, 1)
            cv2.rectangle(canvas, (pbx0, pby), (pbx1, pby + 16), (60, 60, 60), -1)
            cv2.rectangle(canvas, (pbx0, pby), (pbx0 + int((pbx1 - pbx0) * frac), pby + 16), bcol, -1)
            _put(canvas, f"step {done}/{total}  ({frac * 100:.0f}%)", (pbx0 + 8, pby + 13), 0.4, (240, 240, 240))
        elif state["busy"]:
            # indeterminate: a moving segment driven by the frame counter
            cv2.rectangle(canvas, (pbx0, pby), (pbx1, pby + 16), (60, 60, 60), -1)
            seg = (pbx1 - pbx0) // 4
            pos = (state["tick"] * 8) % (pbx1 - pbx0)
            x0 = pbx0 + pos
            cv2.rectangle(canvas, (x0, pby), (min(x0 + seg, pbx1), pby + 16), bcol, -1)

        # ---- info block ----
        _put(canvas, f"Model:   {base(state['ckpt'])}", (MARGIN, 510))
        _put(canvas, f"Dataset: {base(state['dataset'])}", (MARGIN, 536))

        # ---- action-mode chips ----
        _put(canvas, "Action:", (MARGIN, AMODE_Y - 8), 0.45, (180, 180, 180))
        for m, (cx, cy, cw, ch) in zip(AMODE_MODES, _amode_rects()):
            active = (m == state["action_mode"])
            c = (140, 90, 60) if active else (70, 70, 70)
            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), c, -1)
            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch),
                          (220, 170, 120) if active else (140, 140, 140), 1)
            tc = (255, 255, 255) if active else (200, 200, 200)
            label = "abs" if m == "absolute" else "delta (integrate)"
            _put(canvas, label, (cx + 10, cy + 20), 0.5, tc, 1)

        # ---- filter chips ----
        _put(canvas, "Filter:", (MARGIN, CHIPS_Y - 8), 0.45, (180, 180, 180))
        for fk, (cx, cy, cw, ch) in zip(FILTERS, _chip_rects()):
            active = (fk == state["filter_key"])
            c = (70, 130, 90) if active else (70, 70, 70)
            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch), c, -1)
            cv2.rectangle(canvas, (cx, cy), (cx + cw, cy + ch),
                          (120, 220, 160) if active else (140, 140, 140), 1)
            tc = (255, 255, 255) if active else (200, 200, 200)
            _put(canvas, fk, (cx + 10, cy + 20), 0.5, tc, 1)

        # ---- demo list (scrollable; click a row to select) ----
        n = len(state["demos"])
        _put(canvas, f"Demo  ({state['demo_idx']+1}/{n}, click to select):" if n else "Demo: (none)",
             (MARGIN, LIST_TOP - 10), 0.45, (180, 180, 180))
        lb = _list_bottom()
        cv2.rectangle(canvas, (MARGIN, LIST_TOP), (PANEL_W - MARGIN, lb), (30, 30, 32), -1)
        vis = _visible_rows()
        off = state["list_offset"]
        for r in range(vis):
            idx = off + r
            if idx >= n:
                break
            ry = LIST_TOP + r * ROW_H
            if idx == state["demo_idx"]:
                cv2.rectangle(canvas, (MARGIN, ry), (PANEL_W - MARGIN, ry + ROW_H), (70, 110, 140), -1)
            tc = (255, 255, 255) if idx == state["demo_idx"] else (200, 200, 200)
            _put(canvas, f"{idx:>3}  {state['demos'][idx]}", (MARGIN + 8, ry + 17), 0.45, tc, 1)
        # scrollbar (click/drag the track to scroll)
        if n > vis:
            track_h = lb - LIST_TOP
            cv2.rectangle(canvas, (SB_X0, LIST_TOP), (PANEL_W - MARGIN, lb), (55, 55, 58), -1)
            thumb_h = max(24, int(track_h * vis / n))
            thumb_y = LIST_TOP + int((track_h - thumb_h) * off / max(1, _max_offset()))
            tc = (170, 170, 175) if state["drag_sb"] else (120, 120, 125)
            cv2.rectangle(canvas, (SB_X0 + 1, thumb_y),
                          (PANEL_W - MARGIN - 1, thumb_y + thumb_h), tc, -1)

        # ---- status bar (bottom) ----
        cv2.rectangle(canvas, (0, H - 30), (PANEL_W, H), (28, 28, 30), -1)
        _put(canvas, state["status"], (MARGIN, H - 10), 0.46, (160, 220, 255), 1)

        # vertical divider
        cv2.line(canvas, (PANEL_W, 0), (PANEL_W, H), (70, 70, 70), 1)

        # ---- right plot area ----
        if state["plot_img"] is not None:
            pimg_bgr = cv2.cvtColor(state["plot_img"], cv2.COLOR_RGB2BGR)
            scale = min(PLOT_W / pimg_bgr.shape[1], PLOT_H / pimg_bgr.shape[0])
            nw, nh = int(pimg_bgr.shape[1] * scale), int(pimg_bgr.shape[0] * scale)
            pimg_bgr = cv2.resize(pimg_bgr, (nw, nh), interpolation=cv2.INTER_AREA)
            ox = PANEL_W + (PLOT_W - nw) // 2
            oy = (PLOT_H - nh) // 2
            canvas[oy:oy + nh, ox:ox + nw] = pimg_bgr
        else:
            _put(canvas, "Press Run Eval to show predicted vs ground-truth overlay.",
                 (PANEL_W + 40, H // 2), 0.6, (160, 160, 160))

        cv2.imshow(WIN, canvas)

    # ---- main loop ----
    # WINDOW_GUI_NORMAL disables the Qt highgui zoom/pan toolbar (otherwise the
    # mouse wheel zooms the whole image instead of scrolling our demo list).
    try:
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
    except Exception:
        cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, on_mouse)

    key_actions = {
        ord("l"): do_load, ord("r"): do_run,
        ord("n"): do_next, ord("p"): do_prev,
        ord("d"): _toggle_action_mode,
    }
    while True:
        # render finished eval results here (matplotlib must run on main thread)
        pend = state["pending_render"]
        if pend is not None:
            state["pending_render"] = None
            res = pend["res"]
            state["result"] = res
            state["plot_img"] = render_plots(
                pend["preds"], pend["actions"], res, pend["name"],
                base(state["ckpt"]), ac_dim=state["info"]["ac_dim"],
                mode_tag=pend.get("mode", ""),
            )
            ov = "PASS" if res["overall"] else "FAIL"
            set_state("RESULT", f"{pend['name']}: OVERALL {ov}  (T={pend['actions'].shape[0]} steps)")

        state["tick"] += 1
        _draw()
        if state["phase"] == "__quit__":
            break
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):  # q / ESC
            break
        if not state["busy"] and k in key_actions:  # ignore hotkeys while busy
            key_actions[k]()

    cv2.destroyAllWindows()
    tk_root.destroy()


# =====================================================================
def build_parser():
    p = argparse.ArgumentParser(description="NoRobot Eval for Diffusion Policy")
    p.add_argument("--ckpt", default=None, help="path to model.pth (optional in GUI; pick it later)")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help="robomimic hdf5 dataset")
    p.add_argument("--filter-key", default="valid", help="valid / train / all")
    p.add_argument("--action-mode", choices=["absolute", "delta"], default="absolute",
                   help="absolute: compare actions directly. "
                        "delta: integrate joint deltas (cumsum) into an absolute "
                        "trajectory before compare/plot (reveals drift)")
    p.add_argument("--init-obs-key", default="joint_pos",
                   help="obs key whose first frame is the integration start for delta mode (default joint_pos)")
    p.add_argument("--threshold-joint", type=float, default=0.01,
                   help="joint-dim MSE threshold (rad^2; default 0.01 => RMSE ~0.1 rad)")
    p.add_argument("--threshold-gripper", type=float, default=0.05,
                   help="gripper-dim MSE threshold")
    p.add_argument("--device", default=None, help="cuda / cpu (auto by default)")
    p.add_argument("--no-gui", action="store_true", help="CLI batch over all demos")
    return p


def main():
    args = build_parser().parse_args()
    if args.no_gui:
        run_cli(args)
    else:
        run_gui(args)


if __name__ == "__main__":
    main()