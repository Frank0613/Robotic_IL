# Delta 手臂資料收集總結

對應設定檔：[`presets/delta.json`](../exts/frank.realrobot.collector/frank/realrobot/collector/presets/delta.json)

## 1. 機器人 / 場景設定

| 項目 | 值 |
|------|----|
| USD 場景 | `rsm_multiArm.usd`（RSM_Multi 雙臂場景） |
| 控制的 articulation（Robot prim） | `/World/rsm_multiArm/WorkSpace/RS_M90E7A_Left`（左手臂） |
| URDF / Descriptor | `rsm_multiArm.urdf` / `rsm_multiArm_description.yaml` |
| EE frame | `gripper_center` |
| IK marker 起始姿態 | pos `(-0.45732, 0.90739, 0.67568)`　euler xyz(deg) `(-0.585, 0.772, 91.48)` |
| 收集 / 控制頻率 | `30 Hz` |
| Dataset 輸出 | `datasets/rsm_multiArm.hdf5` |
| Sim-only 模式 | 否（`sim_only: false`，會嘗試連接真實手臂） |

## 2. ROS2 Topics

### 訂閱（real → sim，接收真實手臂資料）

| Topic | 訊息型別 | QoS | 說明 |
|-------|---------|-----|------|
| `/left_arm/joint_states` | `sensor_msgs/JointState` | RELIABLE | 真實左手臂關節角度 / 角速度 |
| `/camera_L/color/image_raw/compressed` | `sensor_msgs/CompressedImage`（因 `rgb_compressed: true`） | BEST_EFFORT | RGB 彩色影像（壓縮） |
| `/camera_L/depth/aligned_depth_to_color` | `sensor_msgs/Image` | BEST_EFFORT | 深度影像（對齊到彩色相機座標） |
| `ee_pose_topic`（空字串） | `geometry_msgs/PoseStamped` | — | **未啟用**：此 preset 未設定 EE pose topic，因此不訂閱、也不會收集 `real_ee_pos` / `real_ee_quat` |

### 發布（sim → real，送出控制指令）

| Topic | 訊息型別 | 說明 |
|-------|---------|------|
| `/gripper_cmd` | `std_msgs/Int32` | 每個 step 發布，`1 = 打開`、`0 = 關閉`，讓真實夾爪跟隨模擬端狀態 |

> **同步方式**：目前程式（`ros_subscriber.py`）**沒有做時間對齊**——每個 topic 各自保存最新一筆訊息，`get_latest()` 只是把 joint / rgb / depth 三者目前的最新值合併回傳（`ApproximateTimeSynchronizer` 已被移除）。設定檔中的 `slop: 0.05` 目前只是保留在 UI／設定中，實際上未被使用。

## 3. 每一幀（frame）收集的欄位

### Sim 端（一定會有）

| 欄位 | Shape / dtype | 說明 |
|------|---------------|------|
| `arm_joint_pos` | `(n,) float32` | 模擬左手臂關節角度 |
| `arm_joint_vel` | `(n,) float32` | 模擬左手臂關節角速度 |
| `gripper_pos` | `float32`（scalar） | 夾爪關節角度（若該機器人有夾爪才會有） |
| `gripper_cmd` | `int8`（scalar） | 夾爪指令狀態，`0=open / 1=closed` |
| `ee_pos` | `(3,) float32` | 模擬 EE 位置（Lula FK，base frame） |
| `ee_quat` | `(4,) float32` xyzw | 模擬 EE 方向 |

### Real 端（只有在 ROS 有連上，且 joint/rgb/depth 三個必要 topic 都至少收到一次資料時才會寫入）

| 欄位 | Shape / dtype | 說明 |
|------|---------------|------|
| `real_joint_pos` | `(n_real,) float32` | 真實手臂關節角度，來自 `/left_arm/joint_states` |
| `real_joint_vel` | `(n_real,) float32` | 真實手臂關節角速度（若訊息沒帶 velocity，則為全 0） |
| `real_rgb` | `(256, 256, 3) uint8` | RGB 影像，解壓縮後 resize 至 256×256（`cv2.INTER_AREA`） |
| `real_depth` | `(256, 256) uint16 / float32` | 深度影像，resize 至 256×256（`cv2.INTER_NEAREST`，避免深度值被內插混合） |
| `real_stamp` | `float64`（scalar） | RGB 訊息的 `header.stamp`（秒） |

> 因為此 preset 的 `ee_pose_topic` 為空，**不會**收集 `real_ee_pos` / `real_ee_quat`。

### Action（存檔當下才計算，每個 frame 一列）

| 欄位 | Shape / dtype | 說明 |
|------|---------------|------|
| `action` | `(n,) float32` | 關節角度的 delta：`action[t] = q[t+1] - q[t]`。優先用 `real_joint_pos` 計算，若沒有真實關節資料才退回用 `arm_joint_pos`；最後一幀因無下一幀可比較，全為 0 |

## 4. HDF5 檔案結構

```
rsm_multiArm.hdf5
└── data/                          (attrs: total, created)
    ├── demo_0/                    (attrs: num_samples, robot_name,
    │                               timestamp, arm_joint_names,
    │                               real_joint_names, action_source)
    │   ├── arm_joint_pos   (T, n)
    │   ├── arm_joint_vel   (T, n)
    │   ├── gripper_pos     (T,)
    │   ├── gripper_cmd     (T,)
    │   ├── ee_pos          (T, 3)
    │   ├── ee_quat         (T, 4)
    │   ├── real_joint_pos  (T, n_real)
    │   ├── real_joint_vel  (T, n_real)
    │   ├── real_rgb        (T, 256, 256, 3)   [gzip compressed]
    │   ├── real_depth      (T, 256, 256)      [gzip compressed]
    │   ├── real_stamp      (T,)
    │   └── action          (T, n)
    ├── demo_1/
    └── ...
```

- 每次 `Save demo (B)` 會新增一個 `demo_<N>` group（`N` 從 0 遞增）。
- 影像資料（≥3 維陣列）以 gzip（level 4）壓縮並分塊儲存，其餘欄位不壓縮。

## 5. 備註

- `rgb_compressed: true` → RGB 走 `CompressedImage`，需靠 `cv2.imdecode` 解碼。
- 讀取 / 檢視工具：`tools/hdf5_reader.py`（結構檢查）、`tools/image_viewer.py`（互動式播放 RGB/depth + 關節 + EE + 夾爪）。
