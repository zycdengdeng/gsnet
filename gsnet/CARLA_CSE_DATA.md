# CARLA CSE Benchmark — 评测规格（给合作者跑其它方法做对比）

> 用途：让合作者把**任意方法（尤其其它 feed-forward NVS 方法）**放到我们的 **CSE（Cross-Sensor /
> 跨相机合成）** benchmark 上，得到与我们**口径完全一致**的 PSNR/SSIM/LPIPS。
> 本文档给：① benchmark 定义 ② 数据/位姿在哪、怎么读 ③ 你的方法要产出什么 ④ **评测代码与指标（直接可跑）**。

---

## 0. 一句话
**5 条序列** `110 / 210 / 310 / 410 / 510`，每条：**用 60 张「源」相机图重建 → 在另外 60 张「目标」相机图上合成并评测**（目标相机的位置在重建时从未出现）。报告 PSNR/SSIM/LPIPS（逐序列 + 5 条均值）。

---

## 1. 协议（每条序列）
- **源 / source（重建输入）= 60 张**：6 个奇数相机 × 10 帧（CARLA 12 相机环视 rig 的奇数相机 cam01/03/05/07/09/11）。带 COLMAP 位姿 + 内参。
- **目标 / target（测试）= 60 张**：6 个偶数相机 × 10 帧（cam02/04/06/08/10/12，yaw 各偏 30°）。**重建时不可见**，仅用于评测。
- 任务：方法用 60 源视图建好表示后，在 **60 个目标位姿**渲染，与 60 张目标 GT 比。

> 这是 CARLA 合成数据，12 相机为半径 0.75m 的十二边形环视 rig（见 `gsnet/carla_rig.json`）。

---

## 2. 数据在哪、怎么读
每条序列一个标准 **COLMAP 文本模型**：`runs/cse_scenes/<id>/`
```
runs/cse_scenes/<id>/
  sparse/0/cameras.txt   # 1 个 PINHOLE 内参（12 相机共享）：fx fy cx cy W H
  sparse/0/images.txt    # 120 张图的位姿(qvec,tvec=world->camera)；含源(60)+目标(60)
  sparse/0/test.txt      # 60 个【目标】图名（评测集；其余 60 = 源/训练）
  sparse/0/points3D.*    # 源(奇数)稀疏 SfM 点（如果你的方法想用点云初始化，可用；否则忽略）
  images/                # 全部 120 张图（软链接）。目标 GT = images/<name>，name∈test.txt
```
- **源 vs 目标的区分**：`test.txt` 里的 = 目标（测试）；`images.txt` 里其余 = 源（重建输入）。
- **目标 GT 图**：`runs/cse_scenes/<id>/images/<name>`，其中 `<name>` 遍历 `test.txt`（形如 `e02_03.png`）。
- **图像分辨率/内参**：从 `cameras.txt` 读（所有相机同一内参）。

### 位姿约定（COLMAP，务必按此解）
`images.txt` 每行：`IMAGE_ID qw qx qy qz tx ty tz CAMERA_ID NAME`
- `q=(qw,qx,qy,qz)`、`t=(tx,ty,tz)` 是 **world→camera**。
- 旋转 `R = qvec2rotmat(q)`（world→cam）；相机中心 `C = -R^T t`；**cam→world** 外参 `[R^T | C]`。
- 内参矩阵 `K = [[fx,0,cx],[0,fy,cy],[0,0,1]]`。
- 投影：`x_pix ~ K · (R · X_world + t)`。
- 渲染你的方法时，对每个目标图名，用其 `(q,t)` + `K` 作为目标视点。

> 想要更省事的 JSON（intrinsics/extrinsics + train/test 划分）我可以加个导出脚本，说一声。

---

## 3. 你的方法要产出什么
对每条序列，把你的方法在 **60 个目标位姿**渲染出的图，**用与目标 GT 完全相同的文件名**（即 `test.txt` 里的名字）存到一个 `renders/` 目录：
```
<your_method>/<id>/renders/e02_00.png, e02_01.png, ..., e12_09.png   # 60 张
```
分辨率需与 GT 一致（评测脚本会检查 shape）。

---

## 4. 评测代码与指标（直接用 `gsnet/eval_cse.py`）
**指标定义（与我们 3DGS 主表完全一致）：**
- **PSNR** `= 20·log10(1/√MSE)`，图像归一到 [0,1]、RGB 前 3 通道、逐图算再平均。
- **SSIM** = 3DGS 实现：11×11 高斯窗(σ=1.5)、`C1=0.01²`、`C2=0.03²`。
- **LPIPS** = **VGG** backbone（脚本优先用本仓库 `lpipsPyTorch`，与我们数字逐位一致；无仓库时回退 pip 包 `lpips`(net='vgg')，数值基本一致）。

**跑评测：**
```bash
# 单条序列（GT 自动从该 scene 的 test.txt 拉取）
python gsnet/eval_cse.py --renders <your_method>/110/renders --scene runs/cse_scenes/110

# 一次性 5 条 + 自动出均值（'renders目录:scene目录' 成对）
python gsnet/eval_cse.py --multi \
  <your_method>/110/renders:runs/cse_scenes/110 \
  <your_method>/210/renders:runs/cse_scenes/210 \
  <your_method>/310/renders:runs/cse_scenes/310 \
  <your_method>/410/renders:runs/cse_scenes/410 \
  <your_method>/510/renders:runs/cse_scenes/510 \
  --out <your_method>/cse_scores.json
```
依赖：`torch torchvision pillow`（+ 本仓库的 `lpipsPyTorch`，或 `pip install lpips`）。
输出：逐序列 + **5 条均值**的 PSNR/SSIM/LPIPS 表。

> ⚠️ **所有方法（含我们）都用这同一个 `eval_cse.py` 打分**，保证 apples-to-apples。
> 渲染图名必须与 `test.txt` 一致，否则脚本报"无匹配"。

---

## 5. 报告格式
| Seq | PSNR | SSIM | LPIPS |
|---|---|---|---|
| 110 | … | … | … |
| 210 | … | … | … |
| 310 | … | … | … |
| 410 | … | … | … |
| 510 | … | … | … |
| **Avg** | … | … | … |

我们的数字（参考，同一评测口径）：**GS-Net + 3DGS（densify=2000）Avg PSNR ≈ 19.89 vs 3DGS-baseline 18.00（Δ+1.89）。**

---

## 6. 我们的方法怎么跑的（参考，理解 benchmark 来历）
- **GS-Net**（本工作，encoder=**concat**）：稀疏 SfM 点 → 一次前向 → 稠密 3DGS 高斯，作 3DGS 初始化；再用 60 源视图优化 3DGS，渲染 60 目标视图评测。
- **baseline**：同样流程，但 3DGS 用源稀疏 SfM 点直接初始化（不接 GS-Net）。
- 两者都用 `train.py --eval`（在非 test 图上优化、在 test.txt 图上评测）→ `render.py` → 我们用 `eval_cse.py` 同口径打分。

### benchmark 是怎么造出来的（数据 provenance，可复现）
| 步骤 | 命令 / 文件 | 作用 |
|---|---|---|
| 源数据 | `/mnt/zihanw/carla/input_output/<id>_base/` | 60 奇数相机图的 COLMAP SfM（源位姿）|
| 目标图 | `/mnt/zihanw/carla/paired_120/<id>_dense/camXX/` | 偶数相机的稠密渲染图（目标 GT）|
| 目标位姿 | `gsnet/cse_poses.py`（rig + Umeyama，**不重跑 SfM**）| 在源 COLMAP 坐标系下推导偶数相机位姿，自校验亚度 |
| 组装场景 | `gsnet/make_cse_scene.py` | 拼成 `runs/cse_scenes/<id>/`（60源+60目标+test.txt+源点云）|

---

## 7. 关键文件清单
| 文件 | 作用 |
|---|---|
| `runs/cse_scenes/<id>/` | **benchmark 数据**（COLMAP 模型 + 120 图 + test.txt）|
| `gsnet/eval_cse.py` | **统一评测脚本**（PSNR/SSIM/LPIPS-vgg）|
| `gsnet/carla_rig.json` | 12 相机 rig（位置/朝向）|
| `gsnet/cse_poses.py` | 推导目标相机位姿 |
| `gsnet/make_cse_scene.py` | 组装 CSE 场景 |
| `gsnet/run_cse.py` | 我们方法 + baseline 的端到端跑分驱动（参考）|
