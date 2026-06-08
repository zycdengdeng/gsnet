# CARLA SSE Benchmark — 数据 / 训练 / 评测划分 / 评测代码（给合作者）

> 用途：让合作者在我们的 **SSE（Same-Sensor Evaluation，同传感器·帧留出）** benchmark 上跑任意方法、得到与我们**口径一致**的指标。
> SSE 与 CSE 的区别：**SSE 留出帧**（同一批相机、时间内插），CSE 留出整个相机（跨视角）。本文件只讲 SSE；CSE 见 `CARLA_CSE_DATA.md`。
> 我们的主表硬底就是这个：**CARLA SSE GS-Net+3DGS vs 3DGS = +1.69±0.38**（多seed，排崩坏场景310；densify-on/off 都成立=不挑密化）。

---

## 0. 一句话
每条测试序列 60 张图（6 相机 × 10 帧）：**留出 12 张测试**（每相机第 4、9 帧），**用其余 48 张重建**，在 12 张留出帧上评测。报告 PSNR/SSIM/LPIPS。

---

## 1. 协议（每条测试序列）
- **5 条测试序列**：`110 / 210 / 310 / 410 / 510`（=每个场景 S01..S05 的第 10 条序列）。
- **60 张图 = 6 个源相机 × 10 帧**（CARLA 12 相机环视 rig 的 6 个奇数相机 cam1/3/5/7/9/11）。
- **测试 = 每相机第 4、9 帧 → 12 张**；**训练 = 其余 48 张**。
- 任务：方法用 48 张训练图重建 → 在 12 个留出帧的相机位姿渲染 → 与 12 张 GT 比。

> ⚠️ `310` 是优化病理崩坏场景，论文里**剔除**（报均值时排除 310）。

---

## 2. 数据在哪、怎么读
根：`/mnt/zihanw/carla/`

| 路径 | 内容 | 角色 |
|---|---|---|
| `input_output/<id>_base/images/` | 60 张图，**扁平命名 `1.png`…`60.png`** | 训练(48)+测试(12)图 |
| `input_output/<id>_base/sparse/0/` | COLMAP SfM 模型（`cameras.bin`/`images.bin`/`points3D.bin`）| 60 图的**位姿+内参**；`points3D`=baseline 3DGS 的稀疏初始化 |
| `input_output/<id>_base/sparse/0/test.txt` | 12 个测试图名（`make_sse_split` 写）| 训练/测试划分 |
| `sparse_point/S0<scene>/<id>_sparse.ply` | 稀疏 SfM 点 | **GS-Net 的输入** |

### 图名 ↔ 相机/帧映射（关键）
60 张图按 `1.png`…`60.png` 顺序，每 10 张一个相机块：

| 图号 | 相机 | | 图号 | 相机 |
|---|---|---|---|---|
| 1–10 | cam1 | | 31–40 | cam7 |
| 11–20 | cam3 | | 41–50 | cam9 |
| 21–30 | cam5 | | 51–60 | cam11 |

**测试帧** = 每块内第 4、9 张（1-indexed）→ 图号 `{4,9, 14,19, 24,29, 34,39, 44,49, 54,59}.png`（共 12 张）。其余 48 张为训练。

### 位姿约定（COLMAP）
`images.bin` 每张图：`qvec(qw,qx,qy,qz), tvec` = **world→camera**。`R=qvec2rotmat(q)`（world→cam），相机中心 `C=-R^T t`，cam→world 外参 `[R^T|C]`。内参 `K=[[fx,0,cx],[0,fy,cy],[0,0,1]]`（从 `cameras.bin`，PINHOLE）。

> 想要 JSON 格式的内外参 + train/test 划分（很多 feed-forward 方法不直接读 COLMAP），说一声我加个导出脚本。

---

## 3. GS-Net 训练资料（监督怎么来的）
**与 CSE 共用同一个 GS-Net**（一个模型，SSE/CSE 都用它）。训练对应关系：对每条**训练序列**把稀疏 SfM 点（输入）与 G_dense（3DGS 在该序列上跑满 30k 的稠密高斯=伪GT）配对：
```
python -m gsnet.build_correspondences --batch \
    --io_dir /mnt/zihanw/carla/input_output \
    --sparse_root /mnt/zihanw/carla/sparse_point \
    --out_dir CORR/train --M 3 --K 5
```
- 每稀疏点取 M=3 最近邻（给几何编码器）+ K=5 最近稠密高斯（监督目标：位置/颜色/尺度/旋转/不透明度）。逐序列归一化（center=中位数、scale=95分位）。
- **测试序列 110/210/310/410/510 必须从训练中剔除**（确认 `CORR/train/` 不含这些 id）。

**训练 GS-Net（论文配方 encoder=concat, T=5, M=3）：**
```
python -m gsnet.train_gsnet --corr_dir CORR/train \
    --encoder_type concat --color_activation tanh \
    --w_rot 0.1 --w_pos 10 --T 5 --M 3 --in_memory 1 --out_dir runs/gsnet
```
→ ckpt `runs/gsnet/gsnet_latest.pt`。

---

## 4. 喂进 3DGS：训练视角 vs 测试视角
| | 图 | 位姿 | 在 3DGS 里 |
|---|---|---|---|
| 48 张训练图 | `<id>_base/images/`（不在 test.txt 的）| COLMAP | **训练/监督视角** |
| 12 张测试图 | `<id>_base/images/`（test.txt 里的）| COLMAP | **测试视角**（重建不参与）|

- **初始化点云**：baseline → `<id>_base` 的稀疏 SfM 点（`points3D`）；GS-Net → GS-Net 从 `<id>_sparse.ply` 预测的高斯（`infer` → `--gsnet_init`）。
- 评测 = 在 12 个留出帧上 render + 算 PSNR/SSIM/LPIPS。

---

## 5. 评测代码与指标（与 CSE 同一把尺子）
**指标定义**（与我们 3DGS 主表完全一致）：PSNR=`20·log10(1/√MSE)`（[0,1] RGB）、SSIM=3DGS 的 11×11 高斯窗(C1=.01²,C2=.03²)、**LPIPS=VGG**。

**你的方法产出**：在 12 个测试位姿渲染，**用与 test.txt 相同的文件名**存进 `renders/`，再用统一脚本打分：
```bash
# GT = <id>_base/images/ 里 test.txt 列出的 12 张
python gsnet/eval_cse.py --renders <your_method>/110/renders --gt <gt>/110/gt
# 或一次性 5 条 + 自动均值
python gsnet/eval_cse.py --multi \
  <m>/110/renders:<gt>/110/gt <m>/210/renders:<gt>/210/gt ... --out scores.json
```
> `eval_cse.py` 是 SSE/CSE 通用的统一评测（按文件名配对，PSNR/SSIM/LPIPS-vgg）。**所有方法都用它打分**保证可比。

---

## 6. 端到端复现命令（我们的 SSE）
```bash
# 写 SSE 划分(test.txt) + baseline(稀疏init) vs GS-Net(--gsnet_init) + render + metrics
python -m gsnet.run_sse \
    --io_dir /mnt/zihanw/carla/input_output \
    --sparse_root /mnt/zihanw/carla/sparse_point \
    --ckpt runs/gsnet/gsnet_latest.pt \
    --out_dir runs/sse --gpus 4 5 6 7
#   (--holdout 4 9 / --block 10 / --num_images 60 为默认；--train_extra 透传给 train.py 两边)
```
→ `runs/sse/sse_results.{json,md}`（逐序列 + 均值 PSNR/SSIM/LPIPS + 时间）。
**主表（剔310）：GS-Net+3DGS vs 3DGS = +1.69±0.38（多seed）。**

---

## 7. 关键文件
| 文件 | 作用 |
|---|---|
| `gsnet/make_sse_split.py` | 写 SSE 的 test.txt（帧留出 [4,9]）|
| `gsnet/run_sse.py` | SSE 评测驱动（baseline vs GS-Net）|
| `gsnet/build_correspondences.py` | 稀疏点↔G_dense → GS-Net 训练数据 |
| `gsnet/train_gsnet.py` / `gsnet/infer.py` | 训练 / 推理(稀疏点→init高斯)|
| `gsnet/eval_cse.py` | **统一评测**(PSNR/SSIM/LPIPS-vgg)|
| `train.py`（仓库根）| 3DGS；`--gsnet_init` 注入 GS-Net 初始化 |

### 数据流总览
```
稀疏SfM点(sparse_point/S0X/<id>_sparse.ply) ─┐
                                            ├ build_correspondences ─ CORR/train ─ train_gsnet ─ ckpt
G_dense(output_<id>_dense/.../point_cloud.ply)┘                                                 │
                                                                                              ▼
SSE 测试序列 <id>_base (60图,扁平1-60.png):                                            infer→gsnet_init.ply
  48训(非test.txt) ── 监督视角 ─┐                                                              │
  12测(test.txt,每相机4/9帧) ── 测试视角 ├ run_sse(train.py --eval)→render→metrics(eval_cse.py)┤
  稀疏SfM点 ── baseline 初始化 ──┘                                  (baseline 稀疏点 / ours gsnet_init)
```
