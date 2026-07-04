#!/usr/bin/env python3
"""
inspect_hdf5.py
----------------
檢查實體手臂收集的 HDF5 結構，印出樹狀架構、每個 dataset 的 shape / dtype、
attributes，並對照 robomimic (Diffusion Policy) 期望的格式，標出潛在缺漏。

用法:
    python inspect_hdf5.py /path/to/your_dataset.hdf5
    python inspect_hdf5.py /path/to/your_dataset.hdf5 --max-demos 3
"""

import argparse
import sys

import h5py
import numpy as np


# robomimic 期望的頂層結構（用來對照檢查）
ROBOMIMIC_EXPECTED = {
    "data": "所有 demo 的容器 group",
    "data/<demo>/obs": "observation group (low_dim / rgb / depth 各 key)",
    "data/<demo>/actions": "動作序列, shape (T, action_dim)",
}

# robomimic 常見/建議的 attributes
ROBOMIMIC_EXPECTED_ATTRS = {
    "data": ["total", "env_args"],          # total=總transition數, env_args=環境meta(可選)
    "demo": ["num_samples"],                 # 每個 demo 的 timestep 數
}


def human_bytes(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


def describe_dataset(ds):
    """回傳 dataset 的簡短描述字串"""
    info = f"shape={ds.shape}, dtype={ds.dtype}"
    # 對小型數值資料額外印出數值範圍，方便檢查 NaN / Inf / 數值範圍
    try:
        if ds.size > 0 and np.issubdtype(ds.dtype, np.number):
            sample = ds[()] if ds.size < 1e6 else ds[tuple(slice(0, min(s, 50)) for s in ds.shape)]
            arr = np.asarray(sample)
            has_nan = np.isnan(arr).any() if np.issubdtype(arr.dtype, np.floating) else False
            has_inf = np.isinf(arr).any() if np.issubdtype(arr.dtype, np.floating) else False
            vmin, vmax = float(np.nanmin(arr)), float(np.nanmax(arr))
            flags = []
            if has_nan:
                flags.append("⚠️NaN")
            if has_inf:
                flags.append("⚠️Inf")
            flag_str = (" " + " ".join(flags)) if flags else ""
            info += f", range=[{vmin:.4g}, {vmax:.4g}]{flag_str}"
    except Exception:
        pass
    return info


def print_attrs(obj, indent):
    if len(obj.attrs) == 0:
        return
    for k, v in obj.attrs.items():
        # env_args / 長字串只印出前段，避免洗版
        v_str = str(v)
        if len(v_str) > 120:
            v_str = v_str[:120] + " ...(truncated)"
        print(f"{indent}@attr  {k} = {v_str}")


def walk(name, obj, indent="", max_children=None):
    """遞迴印出 group / dataset"""
    if isinstance(obj, h5py.Group):
        print(f"{indent}📁 {name}/  (group, {len(obj.keys())} keys)")
        print_attrs(obj, indent + "    ")
        keys = list(obj.keys())
        shown = keys if max_children is None else keys[:max_children]
        for k in shown:
            walk(k, obj[k], indent + "    ", max_children=max_children)
        if max_children is not None and len(keys) > max_children:
            print(f"{indent}    ... ({len(keys) - max_children} more keys 省略)")
    else:  # Dataset
        print(f"{indent}📄 {name}  ({describe_dataset(obj)})")
        print_attrs(obj, indent + "    ")


def check_robomimic_compat(f):
    """對照 robomimic 期望格式，印出檢查結果"""
    print("\n" + "=" * 70)
    print("  Robomimic 相容性檢查")
    print("=" * 70)

    problems = []
    warnings = []

    # 1. 頂層必須有 'data'
    if "data" not in f:
        problems.append("缺少頂層 'data' group — robomimic 所有 demo 必須放在 data/ 底下")
        print("❌ 找不到 'data' group，無法繼續結構檢查")
        return
    data = f["data"]
    print(f"✅ 找到 'data' group，內含 {len(data.keys())} 個 demo")

    # 2. data 的 attrs
    if "total" not in data.attrs:
        warnings.append("data 缺少 @attr 'total'（總 transition 數）— 訓練時通常需要")
    else:
        print(f"✅ data/@total = {data.attrs['total']}")
    if "env_args" not in data.attrs:
        warnings.append("data 缺少 @attr 'env_args'（環境 meta）— 實體手臂可不需要，"
                        "但若用 rollout 評估則需要")

    # 3. 逐一檢查 demo
    demo_keys = list(data.keys())
    if len(demo_keys) == 0:
        problems.append("data/ 底下沒有任何 demo")
        return

    # 命名慣例檢查（robomimic 預設用 demo_0, demo_1, ...）
    non_standard = [k for k in demo_keys if not k.startswith("demo_")]
    if non_standard:
        warnings.append(f"demo 命名非標準（robomimic 慣例為 demo_0, demo_1, ...）"
                        f"，例如: {non_standard[:3]}")

    # 用第一個 demo 當代表做詳細檢查
    rep = demo_keys[0]
    demo = data[rep]
    print(f"\n--- 以 '{rep}' 為代表檢查 demo 內部結構 ---")

    # 3a. actions
    if "actions" not in demo:
        problems.append(f"demo 缺少 'actions' — 這是訓練必要的 key")
    else:
        act = demo["actions"]
        print(f"✅ actions: shape={act.shape}, dtype={act.dtype}")
        if act.ndim != 2:
            warnings.append(f"actions 維度為 {act.ndim}，robomimic 期望 (T, action_dim) 2D")

    # 3b. obs group
    if "obs" not in demo:
        problems.append(f"demo 缺少 'obs' group — observation 必須放在 obs/ 底下")
    else:
        obs = demo["obs"]
        print(f"✅ obs group，內含 keys: {list(obs.keys())}")
        # 檢查每個 obs key 的時間長度是否一致
        lengths = {}
        for k in obs.keys():
            lengths[k] = obs[k].shape[0]
        if "actions" in demo:
            lengths["actions"] = demo["actions"].shape[0]
        unique_lengths = set(lengths.values())
        if len(unique_lengths) > 1:
            warnings.append(f"各 key 的時間長度 T 不一致: {lengths} — "
                            f"obs 與 actions 的 T 必須相同")
        else:
            print(f"✅ 所有 key 時間長度一致 T={list(unique_lengths)[0]}")

        # 嘗試辨識 modality
        print("\n--- obs key modality 推測 ---")
        for k in obs.keys():
            shp = obs[k].shape
            dt = obs[k].dtype
            if len(shp) == 4 or (len(shp) == 4):  # (T, H, W, C)
                guess = "rgb / depth (影像)"
            elif len(shp) == 2:
                guess = "low_dim (向量狀態)"
            else:
                guess = "需確認"
            print(f"    {k:20s} shape={str(shp):20s} dtype={str(dt):12s} → {guess}")

    # 3c. 常見可選 key
    for opt_key in ["rewards", "dones"]:
        if opt_key not in demo:
            warnings.append(f"demo 缺少 '{opt_key}'（Diffusion Policy 通常不需要，BC 也可省略）")

    # 3d. num_samples attr
    if "num_samples" not in demo.attrs:
        warnings.append(f"demo 缺少 @attr 'num_samples'（每個 demo 的長度）— 建議補上")

    # 總結
    print("\n" + "=" * 70)
    print("  檢查總結")
    print("=" * 70)
    if not problems:
        print("✅ 沒有發現嚴重缺漏（structural problems）")
    else:
        print(f"❌ 發現 {len(problems)} 個嚴重問題（需修正才能訓練）:")
        for p in problems:
            print(f"    ❌ {p}")
    if warnings:
        print(f"\n⚠️  {len(warnings)} 個提醒（不一定要修，但建議注意）:")
        for w in warnings:
            print(f"    ⚠️  {w}")
    print()


def main():
    parser = argparse.ArgumentParser(description="檢查 HDF5 結構並對照 robomimic 格式")
    parser.add_argument("path", help="HDF5 檔案路徑")
    parser.add_argument("--max-demos", type=int, default=2,
                        help="樹狀結構最多展開幾個 demo（預設 2，避免洗版）")
    parser.add_argument("--full-tree", action="store_true",
                        help="展開全部 demo（會很長）")
    args = parser.parse_args()

    try:
        f = h5py.File(args.path, "r")
    except Exception as e:
        print(f"無法開啟檔案: {e}")
        sys.exit(1)

    import os
    size = os.path.getsize(args.path)
    print("=" * 70)
    print(f"  檔案: {args.path}")
    print(f"  大小: {human_bytes(size)}")
    print("=" * 70)

    # 頂層 attrs
    if len(f.attrs) > 0:
        print("\n[頂層 root attributes]")
        print_attrs(f, "  ")

    # 樹狀結構
    print("\n" + "=" * 70)
    print("  HDF5 樹狀結構")
    print("=" * 70)
    max_children = None if args.full_tree else args.max_demos
    for name in f.keys():
        walk(name, f[name], max_children=max_children)

    # 相容性檢查
    check_robomimic_compat(f)

    f.close()


if __name__ == "__main__":
    main()