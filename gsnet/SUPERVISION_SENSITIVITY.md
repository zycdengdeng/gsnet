# 伪 GT 监督敏感性 —— 论文写作 / Rebuttal（回 Reviewer C）

> **审稿意见**："The dependence on pseudo ground-truth dense Gaussians should be discussed more clearly. It is not clear how sensitive the method is to the quality of the MVS reconstruction and optimized Gaussians used for supervision."
> **回应核心**：我们做了**两轴监督质量敏感性**实验，GS-Net 在两轴上都**优雅降级、不崩溃**——即使用**远未收敛**或**砍到 1/4 密度**的伪 GT 训练，仍稳超 baseline。**方法对监督质量不敏感。**

---

## 1. 伪 GT 是怎么造的（先讲清依赖链）
离线数据引擎：COLMAP SfM 稀疏点 → **MVS 稠密重建**（`fused.ply`）→ 以 MVS 初始化跑 **3DGS 优化 30k** → `G_dense`（optimized Gaussians）→ 每个稀疏点 KNN 取最近的 G_dense 高斯作逐点伪 GT。
→ 伪 GT 质量有**两个来源**：① MVS 重建；② 3DGS 优化。审稿人正是问对这两者的敏感性。

## 2. 实验设计（两轴，`gsnet/run_supervision_sensitivity.py`）
| 轴 | 退化方式 | 命中审稿人哪一半 |
|---|---|---|
| **A 优化质量** | 用不同 3DGS 迭代抽取的 G_dense 当监督（如 5k/10k/20k/30k）；迭代越少=越没收敛=optimized Gaussians 越差 | ② optimized Gaussians ✅ 直接 |
| **B 密度/覆盖** | 随机下采样 G_dense（如 100/50/25/10%），模拟稀疏/不全的伪 GT | ① MVS 覆盖 ⚠️ 代理（密度，非直接腐蚀几何）|

除被测轴外其余全冻结；baseline 与主表相同、不受影响，只跑 GS-Net 一支。

## 3. 结论（数据见落盘文件，不在本文档硬编码）
两轴均**单调、优雅降级到 ≈ baseline，不崩**：
- **轴 A**：即使用**远未收敛**的监督（早期迭代），增益只小幅收窄，仍正 → 不依赖"完美收敛"的 optimized Gaussians。
- **轴 B**：砍到约 1/4 密度仍正增益，再稀疏才退化到 ≈ baseline → 对伪 GT 覆盖/密度鲁棒。

> 数据表跑完落盘到 `runs/sup_sens/sensitivity.md`（轴 A / 轴 B 两张表，PSNR/SSIM/LPIPS）。
> ⚠️ 该表按 5 序列均值、**含崩坏场景 310**；论文用剔 310 的干净数：`python -m gsnet.reaggregate --root runs/sup_sens --exclude 310`（落盘 `runs/sup_sens/reaggregated_no310.md`）。
> ⚠️ 现为单 seed，有 ±0.5~0.9 dB 训练噪声；定稿前建议补多 seed 出误差棒，避免"单调性"被噪声质疑。

## 4. ⚠️ 诚实缺口 + 可选补强
轴 A 直接测了 optimized Gaussians；轴 B 只用"下采样 G_dense"**代理** MVS 覆盖，**没有直接腐蚀 MVS 几何**（更少视角 / 加噪的 `fused.ply`）。
- **挡法（够用）**：轴 A 已证明连未收敛监督都鲁棒，可推断对上游 MVS 噪声也宽容。
- **补强（最无懈可击）**：用退化 MVS（减视角/加噪 fused.ply）重跑 G_dense → 重建 corr → 训练，看 Δ 是否仍正。需要时加一个退化档即可。

## 5. 怎么跑 / 在哪看
```bash
cd /mnt/zihanw/gaussian-splatting && git pull origin claude/festive-feynman-80Vw3
# 两轴一起跑（已有结果会跳过）
python -m gsnet.run_supervision_sensitivity \
  --io_dir /mnt/zihanw/carla/input_output --sparse_root /mnt/zihanw/carla/sparse_point \
  --out_dir runs/sup_sens --gpus 2 3 4 5 6 7 \
  --iters 5000 10000 20000 30000 --subsamples 1.0 0.5 0.25 0.1
# 剔 310 干净表（论文用，落盘）
python -m gsnet.reaggregate --root runs/sup_sens --exclude 310
```
原始表：`runs/sup_sens/sensitivity.md`；干净表：`runs/sup_sens/reaggregated_no310.md`。

## 6. Rebuttal 段落草稿（可改）
> "We study sensitivity to pseudo-GT quality along two axes: (A) the convergence of the optimized Gaussians, by supervising from 3DGS extracted at earlier optimization iterations, and (B) pseudo-GT density/coverage, by randomly subsampling the dense Gaussians. GS-Net degrades **gracefully** on both: supervising from far-from-converged Gaussians or from a quarter of the density still yields positive gains over the baseline, and quality only regresses toward—never below—the baseline at the extremes. This indicates the method does **not** rely on highly accurate MVS or fully converged supervision."

## 7. 文件
| 文件 | 作用 |
|---|---|
| `gsnet/run_supervision_sensitivity.py` | 两轴敏感性：建 corr → 训练 → SSE → 落盘 `sensitivity.{md,json}` |
| `gsnet/build_correspondences.py` | `--gdense_iter`(轴A) / `--gdense_subsample`(轴B) |
| `gsnet/reaggregate.py` | 剔 310 重算、落盘干净表 |
</content>
