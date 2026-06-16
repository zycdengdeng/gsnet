# Encoder 消融 —— 论文写作决策

> **目的**：回 Reviewer A（"encoder 太简单，应该用更强的"）。
> **结论**：在最终配方下，**5 个 encoder 打平**（前 4 名差 0.32 dB，在 ±0.5 dB 训练噪声内），朴素的 concat-MLP 与重型 geom/attention 同档。**增益来自损失 formulation，不是 encoder 复杂度——这恰好把审稿人的质疑变成卖点：简单 encoder 是有意为之，且足够。**

---

## 1. 实验设计
只换 encoder，其余全部冻结（扩张头、解耦损失、T=5、M=3、归一化、训练 schedule、SSE 协议、3DGS 30k）。所有 encoder 吃相同输入（中心点 `[xyz;rgb]` + 3 邻居），输出同维 `F_n`。→ 任何 PSNR 差异**只能**归因于 encoder 架构。
代码：`gsnet/encoders.py`、`gsnet/run_encoder_ablation.py`。

## 2. 五个变体（论文主表 a–e）
| | Encoder | 思想 | 归纳偏置 | ~参数 |
|---|---|---|---|---|
| (a) | MLP-only | 逐点 MLP，不看邻居 | 无（下界） | ~124K |
| (b) | **Concat-MLP (Ours)** | 中心+邻居拼接→MLP | 无 | ~222K |
| (c) | EdgeConv | 边特征+max-pool | 置换不变/相对特征 | ~157K |
| (d) | Self-Attention | 中心 query、邻居 KV、相对坐标 PE | 注意力 | ~207K |
| (e) | Explicit-Geometry | 邻居拼相对坐标+距离 | 显式几何 | ~223K |

> 参数为 encoder 子模块近似值；精确总参数见脚本输出的 `#Params(K)` 列。仓库另有 `attention_v2`/`geoedge` 两个加强版（更强注意力/几何图卷积），实测也无稳定提升，正文不必放。

## 3. 结果（最终配方 tanh:0.1:10:1，剔崩坏场景 310）
跑完用 §6 命令出表（数字从 `reaggregated_no310.md` 填，不在本文档硬编码以免过时）：

| Encoder | PSNR | SSIM | LPIPS | Δ PSNR |
|---|---|---|---|---|
| **Concat-MLP (Ours)** | | | | |
| Explicit-Geometry | | | | |
| MLP-only | | | | |
| Self-Attention | | | | |
| EdgeConv | | | | |

**预期/历史结论**：前 4 名差 <0.3 dB = 统计打平（在 ±0.5 dB 训练噪声内）；只有 EdgeConv 明显掉队（见 §4）。
**对照**：换成默认配方（sigmoid, 等权）时 encoder 反而有别，几何 encoder 领先约 1 dB → **好 formulation 把 encoder 的差异吸收掉了**，这本身就是机制证据（默认配方表同样用 §6 命令出）。

## 4. 为什么打平 / 为什么 EdgeConv 掉队（回 Reviewer A 的论据）
- **输入只有 24 维**（中心+3邻居），concat-MLP 已无损吃下全部信息，几何归纳偏置无处发挥。
- **归一化**让旋转/置换不变性变冗余。
- **任务要绝对定位**，而 EdgeConv 用相对特征 `f_j−f_i`+max-pool **丢弃绝对位置** → 归纳偏置与任务错配，所以最差。这是"不是越复杂越好、而是要匹配任务"的干净反例。
- **瓶颈在 formulation**（tanh 颜色 + 位置/旋转权重 10:0.1），不在 encoder——§3 的双配方对照即证。

## 5. 论文写法（选一个）
- **A（推荐，正文）**：一张 §3 表 + 3 句话——"五种由简到繁的 encoder 在最终 formulation 下统计打平，朴素 concat-MLP 匹配重型图/注意力 encoder → 增益源于损失设计而非 encoder 容量，故用轻量 encoder。"配 EdgeConv 错配作反例。
- **B（附录，增强）**：并列默认/最终两套配方的两张表，论证"encoder 差异被好 formulation 吸收"。
- **C（最省版面）**：只一句"a lightweight encoder suffices; stronger encoders bring no consistent gain"。

**建议：正文 A + 附录 B。**

## 6. 怎么查 / 复现结果
结果在服务器 `runs/`（不在仓库）。

**查已有结果（论文用的剔 310 干净数，秒出、不重跑，三项指标齐，且会存盘）：**
```bash
cd /mnt/zihanw/gaussian-splatting && git pull origin claude/festive-feynman-80Vw3
python -m gsnet.reaggregate --root runs/encoder_ablation_final --exclude 310   # 最终配方
python -m gsnet.reaggregate --root runs/encoder_ablation        --exclude 310   # 默认配方
```
打印的同时存到 `runs/encoder_ablation_final/reaggregated_no310.md`（可用 `--out` 改路径）——§3 的表直接从这个文件抄。
含 310 的原始聚合表：`runs/encoder_ablation{,_final}/encoder_ablation.md`。

**从零复现（一条命令切配方）：**
```bash
# 最终配方（论文主表）
python -m gsnet.run_encoder_ablation --corr_dir CORR/train \
  --io_dir /mnt/zihanw/carla/input_output --sparse_root /mnt/zihanw/carla/sparse_point \
  --out_dir runs/encoder_ablation_final --gpus 4 5 6 7 \
  --color_activation tanh --w_rot 0.1 --w_pos 10 --M 3
```
默认配方去掉 `--color_activation/--w_rot/--w_pos` 即可。`--M 3` 必须与 corr 构建一致。

## 7. Caveat（写之前心里有数）
- 打平在噪声内 → 用 "statistically tied"，**别**单独吹 concat 第一；要更稳就加多 seed 出误差棒（脚本未内置，需要我加）。
- EdgeConv 掉队是真的，别说成打平——用作反例更有力。
- 310 必须剔并在文中注明。
- 只在 CARLA SSE 做的，不外推到 nuScenes/跨场景。
</content>
