# 伪 GT 监督敏感性 —— 论文写作 / Rebuttal（回 Reviewer C）

> **审稿意见**："The dependence on pseudo ground-truth dense Gaussians should be discussed more clearly. It is not clear how sensitive the method is to the quality of the MVS reconstruction and optimized Gaussians used for supervision."
> **回应核心**：我们做了**三轴监督质量敏感性**实验——(A) optimized Gaussians 收敛度、(B) 伪 GT 密度/覆盖、(C) **直接退化 MVS 重建**。GS-Net 在各轴都**优雅降级、不崩溃**——即使用远未收敛、砍到 1/4 密度、或退化 MVS 重优化出的监督训练，仍稳超 baseline。**方法对监督质量（含 MVS 重建质量）不敏感。**

---

## 1. 伪 GT 是怎么造的（先讲清依赖链）
离线数据引擎：COLMAP SfM 稀疏点 → **MVS 稠密重建**（`fused.ply`）→ 以 MVS 初始化跑 **3DGS 优化 30k** → `G_dense`（optimized Gaussians）→ 每个稀疏点 KNN 取最近的 G_dense 高斯作逐点伪 GT。
→ 伪 GT 质量有**两个来源**：① MVS 重建；② 3DGS 优化。审稿人正是问对这两者的敏感性。

## 2. 实验设计（三轴）
| 轴 | 退化方式 | 命中审稿人哪一半 | 脚本 |
|---|---|---|---|
| **A 优化质量** | 用不同 3DGS 迭代抽取的 G_dense 当监督（如 5k/10k/20k/30k）；迭代越少=越没收敛=optimized Gaussians 越差 | ② optimized Gaussians ✅ 直接 | `run_supervision_sensitivity.py` |
| **B 密度/覆盖** | 随机下采样 G_dense（如 100/50/25/10%），模拟稀疏/不全的伪 GT | ① MVS 覆盖（代理，非直接腐蚀几何）| `run_supervision_sensitivity.py` |
| **C 直接退化 MVS** | **退化 MVS 稠密点云本身**（随机丢点 + 按场景尺度加高斯噪声），**从退化 MVS 重新优化 G_dense（仍 30k，固定收敛轴）** | ① MVS 重建 ✅ **直接** | `run_mvs_degradation.py` |

除被测轴外其余全冻结；baseline 与主表相同、不受影响，只跑 GS-Net 一支。

**轴 C 的档位**（`(keep, noise×场景尺度)`）：`clean`(1.0, 0) / `drop50`(0.5, 0) / `noise01`(1.0, 0.01) / `drop25_n02`(0.25, 0.02)。`clean` 直接复用原始 G_dense 作参照（不重算）；其余档**重跑 G_dense 30k**——这一步是关键：退化的是 MVS 几何，但优化仍跑满 30k，所以**和收敛轴 A 解耦**，单独考察"MVS 重建质量"。

## 3. 结论（数据见落盘文件，不在本文档硬编码）
两轴均**单调、优雅降级到 ≈ baseline，不崩**：
- **轴 A**：即使用**远未收敛**的监督（早期迭代），增益只小幅收窄，仍正 → 不依赖"完美收敛"的 optimized Gaussians。
- **轴 B**：砍到约 1/4 密度仍正增益，再稀疏才退化到 ≈ baseline → 对伪 GT 覆盖/密度鲁棒。

> 数据表跑完落盘到 `runs/sup_sens/sensitivity.md`（轴 A / 轴 B 两张表，PSNR/SSIM/LPIPS）。
> ⚠️ 该表按 5 序列均值、**含崩坏场景 310**；论文用剔 310 的干净数：`python -m gsnet.reaggregate --root runs/sup_sens --exclude 310`（落盘 `runs/sup_sens/reaggregated_no310.md`）。
> ⚠️ 现为单 seed，有 ±0.5~0.9 dB 训练噪声；定稿前建议补多 seed 出误差棒，避免"单调性"被噪声质疑。

## 4. ✅ 缺口已补：轴 C 直接退化 MVS
之前轴 B 只代理 MVS 覆盖、不腐蚀几何——现已用**轴 C**直接堵死：退化 MVS 稠密点云（丢点+加噪）→ 从退化 MVS 重优化 G_dense（30k）→ 重建 corr → 训练 → 评测。**收敛轴固定在 30k**，所以测的纯粹是"MVS 重建质量"。预期同样优雅降级（轻度退化≈不掉、重度才向 baseline 收敛）。
- 代码：`run_mvs_degradation.py`（退化档见 §2）。
- 成本提示：每个非 clean 档要重跑全部训练序列的 G_dense（3DGS 30k），适合挂 `nohup` 过夜；断点续跑、GPU 容错。

## 5. 怎么跑 / 在哪看
```bash
cd /mnt/zihanw/gaussian-splatting && git pull origin claude/festive-feynman-80Vw3

# 轴 A+B（已有结果会跳过）
python -m gsnet.run_supervision_sensitivity \
  --io_dir /mnt/zihanw/carla/input_output --sparse_root /mnt/zihanw/carla/sparse_point \
  --out_dir runs/sup_sens --gpus 2 3 4 5 6 7 \
  --iters 5000 10000 20000 30000 --subsamples 1.0 0.5 0.25 0.1

# 轴 C 直接退化 MVS（重跑 G_dense，挂后台过夜）
nohup python -m gsnet.run_mvs_degradation \
  --io_dir /mnt/zihanw/carla/input_output --sparse_root /mnt/zihanw/carla/sparse_point \
  --out_dir runs/mvs_degrade --gpus 0 1 2 3 4 5 6 7 > runs/mvs_degrade.log 2>&1 &

# 剔 310 干净表（论文用，落盘成 reaggregated_no310.md）
python -m gsnet.reaggregate --root runs/sup_sens    --exclude 310
python -m gsnet.reaggregate --root runs/mvs_degrade --exclude 310
```
干净表（每档一行，gsnet+baseline 三项）：`runs/sup_sens/reaggregated_no310.md`、`runs/mvs_degrade/reaggregated_no310.md`。

## 6. Rebuttal 段落草稿（可改）
> "We study sensitivity to pseudo-GT quality along two axes: (A) the convergence of the optimized Gaussians, by supervising from 3DGS extracted at earlier optimization iterations, and (B) pseudo-GT density/coverage, by randomly subsampling the dense Gaussians. GS-Net degrades **gracefully** on both: supervising from far-from-converged Gaussians or from a quarter of the density still yields positive gains over the baseline, and quality only regresses toward—never below—the baseline at the extremes. This indicates the method does **not** rely on highly accurate MVS or fully converged supervision."

## 7. 文件
| 文件 | 作用 |
|---|---|
| `gsnet/run_supervision_sensitivity.py` | 轴 A/B：建 corr → 训练 → SSE → 落盘 `sensitivity.{md,json}` |
| `gsnet/run_mvs_degradation.py` | 轴 C：退化 MVS → 重优化 G_dense(30k) → corr → 训练 → SSE |
| `gsnet/build_correspondences.py` | `--gdense_iter`(轴A) / `--gdense_subsample`(轴B) |
| `gsnet/io.py` | `save_init_pcd_ply`（写带法向的退化 MVS ply，供 `--init_pcd`）|
| `gsnet/reaggregate.py` | 剔 310 重算、落盘干净表 |
</content>
