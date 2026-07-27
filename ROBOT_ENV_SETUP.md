# 機器人端環境復現(venv)

在**機器人那台電腦**上重建跑模型所需的 Python 環境。這份只涵蓋**環境安裝**——把 `robomimic` + torch 這套裝起來、能 import、能載入 checkpoint。實際部署節點怎麼跑,見 [`codes/live_inference/README.md`](codes/live_inference/README.md)。

> 這台開發機是用 **conda**(env 名 `robomimic_env`);機器人端改用 **venv**,虛擬環境名稱用 **`robomimic_venv`**。內容物完全相同,只是建立方式不同。

---

## 0. 前置需求

| 需求 | 說明 |
|---|---|
| **Python 3.12** | 必須。開發機是 3.12.13。**若機器人端也要在同一個 venv 裡用 `rclpy`(ROS2),這個 Python 必須跟 ROS2 建置時用的系統 Python 3.12 是同一版**(見最後一節)。 |
| **NVIDIA 驅動 + CUDA**(選用) | 有 GPU 才需要。開發機是 CUDA 12.8 → 裝 `cu128` 的 torch。**機器人端要依它自己的驅動/CUDA 版本挑對應 torch**(見步驟 3)。沒有 GPU 也能純 CPU 跑推論。 |
| **git** | 用來 clone 這個 repo(`robomimic` 要以本 repo 的 0.5.0 安裝,才能對上你訓練出來的 checkpoint)。 |

---

## 1. 取得專案

```bash
git clone <你的-repo-url> robomimic
cd robomimic
```

> 只需要**程式碼**和你的**模型 `.pth`**。**不需要 dataset**——正規化統計已烤進 checkpoint,推論不吃資料集。

## 2. 建立並啟用 venv

```bash
# 用 Python 3.12 建立;名稱 robomimic_venv
python3.12 -m venv robomimic_venv

# 啟用(每次開新 terminal 都要)
source robomimic_venv/bin/activate

# 升級 pip(避免舊 pip 解析依賴出錯)
python -m pip install --upgrade pip
```

啟用後提示字會出現 `(robomimic_venv)`。之後所有 `pip` / `python` 都指向這個環境。

## 3. 安裝 PyTorch(先裝,並對齊機器人的 CUDA)

```bash
# 開發機用的版本 = CUDA 12.8。若機器人 CUDA 也是 12.x,直接照抄:
pip install torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128
```

**若機器人環境不同**,把 `--index-url` 換掉:
- CUDA 12.1 → `https://download.pytorch.org/whl/cu121`
- CUDA 11.8 → `https://download.pytorch.org/whl/cu118`
- **沒有 GPU(純 CPU)** → `pip install torch torchvision`(不加 index-url,裝 CPU 版即可;DP 推論在 CPU 上也能跑,只是慢一點)
- 用 `nvidia-smi` 右上角可查驅動支援的 CUDA 版本。

## 4. 安裝其餘依賴 + robomimic 本體

```bash
# pyyaml/numpy/opencv 不是 robomimic 的依賴,但 rclpy 與 cv2 影像解碼需要;
# 先裝好可讓下一步依賴解析乾淨
pip install pyyaml numpy opencv-python

# 以「可編輯模式」安裝本 repo 的 robomimic 0.5.0,
# 會一併帶入 diffusers/transformers/huggingface-hub/egl_probe/h5py 等(版本見下表)
pip install -e .
```

## 5. 驗證環境

```bash
# torch 有沒有裝好、能不能用 GPU
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

# robomimic 能不能 import
python -c "import robomimic; print('robomimic', robomimic.__version__)"

# 能不能載入你的 checkpoint(把路徑換成你複製過來的 .pth)
python -c "
import sys; sys.path.insert(0, 'codes')
from model_utils import load_model
_, info = load_model('train/jetcobot_dp_ee_delta/<你的timestamp>/models/model_epoch_600.pth', device='cpu')
print('checkpoint OK:', info['obs_keys'], 'ac_dim=', info['ac_dim'])
"
```

看到 `checkpoint OK: [...] ac_dim= 10` 就代表環境可以跑模型了。

---

## 版本對照表(復現目標)

開發機實測可運作的版本。`pip install -e .` 預設就會裝到這些(torch 由步驟 3 指定):

| 套件 | 版本 |
|---|---|
| Python | 3.12.13 |
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 |
| robomimic | 0.5.0(本 repo 可編輯安裝) |
| diffusers | 0.11.1 |
| transformers | 4.41.2 |
| huggingface-hub | 0.23.4 |
| h5py | 3.16.0 |
| numpy | 2.5.0 |
| opencv-python | 5.0.0.93 |
| egl_probe | 1.0.2(會從原始碼編譯;推論其實用不到,但 `pip install -e .` 需要它才能 build) |
| PyYAML | 6.0.3 |

若機器人端的 CUDA/torch 被迫換版,以上 torch/torchvision 版本可跟著變;其餘(diffusers/transformers/huggingface-hub)**請維持一致**,這幾個版本對 checkpoint 的相容性較敏感。

---

## 重要:venv 與 ROS2 `rclpy` 的關係

真機部署(`codes/live_inference/`)需要 `rclpy`,但 **`rclpy` 不能用 pip 裝進 venv**——它來自系統的 ROS2 安裝。作法跟開發機一樣:**先 source ROS2,再用這個 venv 的 python 跑**。ROS2 的 `PYTHONPATH` 會讓 venv 也找得到系統的 `rclpy`:

```bash
source /opt/ros/jazzy/setup.bash      # 機器人端 ROS2 版本(可能不是 jazzy,換成實際的)
source robomimic_venv/bin/activate
python -c "import rclpy, robomimic; print('rclpy + robomimic 同時可用')"
```

**關鍵前提**:建立 venv 用的 `python3.12` 必須跟 ROS2 建置時的系統 Python 是**同一版**(ABI 要相容),否則 `import rclpy` 會失敗。
- 如果上面那行 import rclpy 失敗,改用讓 venv 看得到系統套件的方式重建:
  ```bash
  python3.12 -m venv robomimic_venv --system-site-packages
  ```
  然後重跑步驟 2~4。

> 若你機器人端跑模型與跑 ROS 是**分兩台/兩個環境**,那 venv 這邊就不用管 rclpy,只要 robomimic + torch 能載入模型即可。
