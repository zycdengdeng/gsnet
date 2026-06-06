# CARLA 数据组织 & CSE 训练/评测流程（给合作者）

> 目的：让合作者完整理解 GS-Net 在 CARLA 上的 **CSE（Cross-Sensor Evaluation，跨相机合成）** 是怎么组织数据的——
> ① GS-Net 网络的训练资料怎么来；② 喂进 3DGS 做监督/测试的视角与初始化是什么；③ 位姿从哪来；④ 对应磁盘路径。

---

## 0. 一句话
GS-Net = **稀疏 SfM 点 → 一次前向 → 稠密 3DGS 高斯**，作即插即用初始化。
CARLA 用一个 **12 相机的十二边形环视 rig**（绕 ego 一圈）。CSE = **用 6 个奇数相机重建、合成并测 6 个偶数相机**（偶数相机的位置在重建时完全没出现过）= 跨传感器/跨视角合成。

---

## 1. 相机 rig（`gsnet/carla_rig.json`）
半径 0.75m 的十二边形，12 个相机均匀环视，每个 yaw 差 30°：

| 相机 | 角色 | yaw |
|---|---|---|
| cam01,03,05,07,09,11（**奇数**）| **source / 训练视角** | 0,60,120,180,240,300° |
| cam02,04,06,08,10,12（**偶数**）| **target / 测试视角** | 30,90,150,210,270,330° |

- 12 相机共享同一组内参（intrinsics）。
- 坐标系：CARLA 为 x-前、y-右、z-上（左手系）；CARLA↔COLMAP 的轴变换在 `cse_poses.py` 里自动求解。

---

## 2. 序列 / 场景编号约定
- 编号 `<id>`（3 位）：**场景 = id // 100**，**序列 = id % 100**。
- 5 个场景 `S01..S05`；每个场景多条序列。
- **测试序列 = 每场景的第 10 条：`110 / 210 / 310 / 410 / 510`**（CSE 与 SSE 共用这 5 条做测试）。
- **训练序列 = 其余（如 101–109、201–209 …）**。
- ⚠️ **务必确认 GS-Net 训练用的 `CORR/train` 里不含 110/210/310/410/510**（测试序列必须从训练中剔除，否则泄漏）。

---

## 3. 磁盘上的原始数据（路径表）
根目录：`/mnt/zihanw/carla/`

| 路径 | 内容 | 在流程里的角色 |
|---|---|---|
| `input_output/<id>_base/sparse/0/` | 60 张**奇数相机**图（6 奇相机 × 10 帧）的 **COLMAP SfM 模型**（`cameras.bin`/`images.bin`/`points3D.*`）| 提供奇数相机的**真实 COLMAP 位姿** + source 重建 |
| `input_output/<id>_base/images/` | 那 60 张奇数相机图 | 3DGS 的**训练（监督）视角** |
| `input_output/output_<id>_dense/point_cloud/iteration_30000/point_cloud.ply` | 在该序列上跑满 30k 的 **3DGS 稠密高斯** | **G_dense = GS-Net 的监督目标（伪GT）** |
| `sparse_point/S0X/<id>_sparse.ply` | 该序列的**稀疏 SfM 点** | **GS-Net 的输入**（也是 baseline 3DGS 的初始化点云）|
| `paired_120/<id>_dense/cam00../cam11/` | 全 12 相机的稠密渲染图 | 取**偶数相机的测试图**（GT）|

> 奇数图命名约定（`<id>_base`）：`<n>.png`，n=1..60，其中 `cam = 2*((n-1)//10)+1`、`frame = (n-1)%10`。

---

## 4. GS-Net 训练资料（correspondences）= 监督怎么来的
对每条**训练序列**，把"稀疏 SfM 点（输入）"与"G_dense（目标）"配对，生成一个 `.npz`：

```
python -m gsnet.build_correspondences --batch \
    --io_dir /mnt/zihanw/carla/input_output \
    --sparse_root /mnt/zihanw/carla/sparse_point \
    --out_dir CORR/train --M 3 --K 5
```
- 对每个稀疏点 p：取它的 **M=3 个最近稀疏邻居**（给几何编码器）+ 在 G_dense 里**最近的 K=5 个稠密高斯**（作为该点的监督目标集：位置/颜色/尺度/旋转/不透明度）。
- **逐序列归一化**（`compute_normalization`：center=点中位数、scale=到 center 距离的 95 分位）→ 所有序列搬到同一标准尺度，网络才学得动 offset/尺寸。归一化参数 `norm_center/norm_scale` 存进 npz，推理时反变换回真实尺度。
- 产物：`CORR/train/<id>.npz`（每条训练序列一个）。

**训练 GS-Net（最终配方 geom:tanh:0.1:10:1 @ M=3, T=5）：**
```
python -m gsnet.train_gsnet --corr_dir CORR/train \
    --encoder_type geom --color_activation tanh \
    --w_rot 0.1 --w_pos 10 --M 3 --T 5 --in_memory 1 \
    --out_dir runs/ours
```
→ ckpt：`runs/ours/gsnet_latest.pt`（CSE 论文里用的 ckpt 见 `run_cse --ckpt`）。

---

## 5. 喂进 3DGS：监督视角 vs 测试视角（CSE 的核心）
每条测试序列构建一个 COLMAP 场景（`runs/cse_scenes/<id>/`），含 **60 奇（训练）+ 60 偶（测试）**：

| | 图来源 | 位姿 | 在 3DGS 里 |
|---|---|---|---|
| **奇数 60 张** | `input_output/<id>_base/images/` | COLMAP 原始位姿 | **训练/监督视角**（重建用这些）|
| **偶数 60 张** | `paired_120/<id>_dense/camXX/` | 由 rig 推导（见 §6）| **测试视角**（写进 `test.txt`，重建时不参与）|

- **初始化点云**：
  - baseline 3DGS → **奇数稀疏 SfM 点**（`<id>_base` 的 points3D 复制进场景）。
  - GS-Net+3DGS → **GS-Net 从奇数稀疏点预测的高斯**（`infer` 产出 `gsnet_init.ply`，经 `--gsnet_init` 注入）。
- 评测 = 在 60 个**偶数相机位置**（重建中从未出现的位置）上 render + 算 PSNR/SSIM/LPIPS。这就是"跨传感器"：用一组相机重建、去合成另一组相机。

---

## 6. 位姿从哪来（关键，避免误解）
- **奇数（训练）位姿**：直接来自 `<id>_base` 的 COLMAP SfM（`images.bin`）。
- **偶数（测试）位姿**：**不重新跑 SfM**（重跑会改坐标系、破坏与 G_dense/GS-Net 的一致性）。改用 `cse_poses.py`：
  - 已知 CARLA rig（12 相机相对 ego 的固定安装）+ 每帧用 6 个奇数相机的 COLMAP 位姿，**Umeyama 拟合每帧 ego→COLMAP 相似变换**，再作用到偶数相机的已知 rig 安装上 → 得到偶数相机在**奇数 COLMAP 坐标系**下的位姿。
  - CARLA↔COLMAP 轴变换自动搜索（48×48 signed-permutation），用奇数相机自校验挑残差最小的。
  - **自校验残差应 ~亚度、亚百分比**（日志会打印 `rot_residual_deg` / `center_residual_rel`）——这是位姿正确性的体检。

---

## 7. 端到端复现命令（CSE）
```
# (a) 推导偶数相机测试位姿
python -m gsnet.cse_poses --io_dir /mnt/zihanw/carla/input_output \
    --rig gsnet/carla_rig.json --ids 110 210 310 410 510 --out_dir runs/cse_poses

# (b) 构建 CSE COLMAP 场景（60奇训练 + 60偶测试 + test.txt + 初始化点云）
python -m gsnet.make_cse_scene --io_dir /mnt/zihanw/carla/input_output \
    --paired_dir /mnt/zihanw/carla/paired_120 \
    --poses_dir runs/cse_poses --out_dir runs/cse_scenes --ids 110 210 310 410 510

# (c) 跑 CSE：baseline(稀疏init) vs GS-Net(--gsnet_init)，train.py --eval -> render -> metrics
python -m gsnet.run_cse --scenes_dir runs/cse_scenes \
    --sparse_root /mnt/zihanw/carla/sparse_point \
    --ckpt runs/ours/gsnet_latest.pt --out_dir runs/cse --gpus 2 3 4 5 6 7
#   甜点设置（论文主表）：加 --train_extra "--densify_until_iter 2000"
```
→ 结果：`runs/cse/cse_results.{json,md}`（逐序列 + 均值 PSNR/SSIM/LPIPS）。
**当前主表数字：CSE densify=2000，gsnet 19.89 / baseline 18.00 → Δ=+1.89±0.1（多seed）。**

---

## 8. 关键文件清单
| 文件 | 作用 |
|---|---|
| `gsnet/carla_rig.json` | 12 相机 rig（位置/朝向）|
| `gsnet/build_correspondences.py` | 稀疏点↔G_dense 配对 → GS-Net 训练 npz |
| `gsnet/train_gsnet.py` | 训练 GS-Net |
| `gsnet/infer.py` | 稀疏点 → GS-Net 高斯 init ply（含反归一化）|
| `gsnet/cse_poses.py` | 由 rig 推导偶数相机测试位姿（不重跑 SfM）|
| `gsnet/make_cse_scene.py` | 组装 CSE COLMAP 场景（奇训练+偶测试）|
| `gsnet/run_cse.py` | CSE 评测驱动（baseline vs GS-Net）|
| `train.py`（仓库根）| 3DGS 优化；`--gsnet_init` 注入 GS-Net 初始化 |

---

### 数据流总览
```
稀疏SfM点(sparse_point/S0X/<id>_sparse.ply) ──┐
                                              ├─ build_correspondences ─ CORR/train/<id>.npz ─ train_gsnet ─ ckpt
G_dense(output_<id>_dense/.../point_cloud.ply)┘                                                              │
                                                                                                            ▼
CSE 测试序列:                                                                                          infer → gsnet_init.ply
  奇数图+COLMAP位姿  ── 训练/监督视角 ─┐                                                                       │
  偶数图+rig推导位姿 ── 测试视角(test.txt)├─ make_cse_scene → runs/cse_scenes/<id> ─ run_cse(train.py --eval)─┤
  奇数稀疏点 ── baseline 初始化 ─────────┘                                          (baseline 用稀疏点 / ours 用 gsnet_init)
```
