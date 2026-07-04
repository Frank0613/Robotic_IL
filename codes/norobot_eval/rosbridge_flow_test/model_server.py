#!/usr/bin/env python3
"""Model inference server (robomimic_env, conda Python 3.12).

Hosts the trained Diffusion Policy behind a tiny localhost socket so a ROS2
(rclpy, system Python 3.12) node can reach it. Historically this split existed
because ROS2 Jazzy's rclpy needed Python 3.12 while robomimic_env was Python
3.10 -- that's no longer true (robomimic_env is Python 3.12 too now, see
../../README.md), but this harness still keeps the two processes separate:
`robomimic_env` and the system's `/usr/bin/python3.12` are different
installations with different site-packages, so mixing them via a shared
PYTHONPATH is still avoided (see run_demo.sh).

Protocol: every message is a 4-byte big-endian length prefix followed by a
pickled dict (localhost only, trusted). Requests:
    {"cmd": "ping"}
    {"cmd": "reset"}
    {"cmd": "step", "obs": {"joint_pos":[6], "joint_vel":[6],
                            "rgb":[H,W,3] uint8, "depth":[H,W,1] float32}}
Replies:
    {"ok": True, "action": np.ndarray[ac_dim]}  or  {"ok": False, "error": str}

The server keeps a CAUSAL observation window (deque of the last To frames, front-
padded with the first frame). This matches how the policy will run on the real
arm, where only past frames are available -- unlike the offline forward-window
eval in norobot_eval_gui.py. So ROS-run predictions may differ slightly from the
offline eval; that is expected, and the point here is to validate the pipeline.
"""
import os
import sys
import socket
import struct
import pickle
import argparse
from collections import deque

import numpy as np

# reuse the shared checkpoint loader (codes/model_utils.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model_utils import load_model  # noqa: E402


def _recvall(conn, n):
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_msg(conn):
    hdr = _recvall(conn, 4)
    if hdr is None:
        return None
    (length,) = struct.unpack(">I", hdr)
    data = _recvall(conn, length)
    if data is None:
        return None
    return pickle.loads(data)


def send_msg(conn, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    conn.sendall(struct.pack(">I", len(data)) + data)


class PolicyService:
    def __init__(self, ckpt, device=None):
        self.policy, self.info = load_model(ckpt, device=device)
        self.To = self.info["To"]
        self.obs_keys = self.info["obs_keys"]
        self.window = deque(maxlen=self.To)

    def reset(self):
        self.policy.start_episode()
        self.window.clear()

    def step(self, obs_frame):
        """obs_frame: dict key -> single-frame np array (raw stored format)."""
        self.window.append(obs_frame)
        frames = list(self.window)
        while len(frames) < self.To:          # front-pad with earliest frame
            frames.insert(0, frames[0])
        win = {k: np.stack([f[k] for f in frames], axis=0) for k in self.obs_keys}
        action = self.policy(ob=win)           # [ac_dim], already un-normalized
        return np.asarray(action, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser(description="Diffusion Policy socket inference server")
    ap.add_argument("--ckpt", required=True, help="path to model .pth")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--device", default=None, help="cuda / cpu (auto by default)")
    args = ap.parse_args()

    print(f"[model_server] loading {args.ckpt} ...", flush=True)
    svc = PolicyService(args.ckpt, device=args.device)
    print(f"[model_server] ready: To={svc.To} obs_keys={svc.obs_keys} "
          f"ac_dim={svc.info['ac_dim']}", flush=True)

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[model_server] listening on {args.host}:{args.port}", flush=True)

    try:
        while True:
            conn, addr = srv.accept()
            print(f"[model_server] client connected {addr}", flush=True)
            try:
                while True:
                    req = recv_msg(conn)
                    if req is None:
                        break
                    cmd = req.get("cmd")
                    try:
                        if cmd == "ping":
                            send_msg(conn, {"ok": True,
                                            "info": {k: svc.info[k]
                                                     for k in ("To", "Ta", "Tp", "ac_dim")}})
                        elif cmd == "reset":
                            svc.reset()
                            send_msg(conn, {"ok": True})
                        elif cmd == "step":
                            a = svc.step(req["obs"])
                            send_msg(conn, {"ok": True, "action": a})
                        else:
                            send_msg(conn, {"ok": False, "error": f"unknown cmd {cmd!r}"})
                    except Exception as e:  # keep the server alive on a bad request
                        send_msg(conn, {"ok": False, "error": repr(e)})
            finally:
                conn.close()
                print("[model_server] client disconnected", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        srv.close()


if __name__ == "__main__":
    main()
