# Encoder 消融实验 —— 论文写作决策文档

> **用途**：回应 **Reviewer A**（"GS-Net 的 encoder 太简单 / 应该用更强的 encoder"）。本文给出实验设计、全部结果、解读、**三种论文写法供你选**、诚实 caveat，以及**结果在哪查 / 怎么复现**。
> **一句话结论**：在**最终配方**（tanh + 损失权重 0.1:10:1）下，**5 个 encoder 几乎打平**（前 4 名差 <0.32 dB，在噪声内）；轻量的 `concat-MLP` 与重型的 `geom / attention` 同档 → **增益来自损失/输出formulation，不是 encoder 复杂度**。这正面回答了 Reviewer A。

---

## 1. 这个实验到底在比什么（控制变量）

**只换 encoder，其余全部冻结**：扩张头（T 个高斯/点）、解耦损失、T=5、M=3、逐序列归一化、训练 schedule（200 epoch, in-memory）、SSE 评测协议、3DGS 30k 优化，全部一致。
所有 encoder 吃**完全相同的输入**：中心点 `[xyz; rgb]`(6 维) + 它的 M=3 个最近邻的 `[xyz; rgb]` + 邻居坐标；输出统一维度 `context_dim=256` 的 `F_n`。
→ 任何 PSNR 差异**只能**归因于 encoder 架构本身。这是干净的 A/B。

代码：`gsnet/encoders.py`（7 个 encoder 定义）、`gsnet/run_encoder_ablation.py`（训练→SSE→聚合一条龙）。

---

## 2. Encoder 变体（论文里就写这 5 个 a–e）

| 标签 | 名称 | 核心思想 | 归纳偏置 | 邻域聚合 | encoder 模块参数(约) |
|---|---|---|---|---|---|
| (a) | **MLP-only** | 逐点 MLP，**不看邻居** | 无（下界 baseline） | ❌ | ~124K |
| (b) | **Concat-MLP（Ours）** | 中心+M邻居特征拼接→MLP | 无（最朴素聚合） | ✅ 拼接 | ~222K |
| (c) | **EdgeConv / DGCNN** | 边特征 `h([f_i; f_j−f_i])` + max-pool | 置换不变 + 相对特征 | ✅ 图 | ~157K |
| (d) | **Self-Attention** | 中心 query、邻居 key/value，相对坐标做位置编码 | 注意力 + 位置编码 | ✅ 注意力 | ~207K |
| (e) | **Explicit-Geometry** | 邻居特征**额外拼相对坐标+距离**再聚合 | 显式几何 | ✅ 拼接+几何 | ~223K |

> 备注：仓库里还有两个"加强版"（`attention_v2`=向量注意力 Point-Transformer 式、`geoedge`=几何 EdgeConv），是探索性变体，**论文主表不必放**，可在 rebuttal 文字里提一句"更强的向量注意力/几何图卷积也未带来稳定提升"。
> 参数量是 **encoder 子模块**的近似值（不含共享 decoder + T 个扩张头，那部分各 encoder 相同）。要精确总参数：跑 `run_encoder_ablation.py`，输出表的 `#Params(K)` 列是**整网**参数。

**关键观察**：`concat-MLP` 并不是参数最少的（`mlp_only`/`edgeconv` 更少），它的卖点是**最朴素、零几何归纳偏置**——所以"朴素 encoder 打平精巧 encoder"说的是**架构精巧度**，不是单纯参数多少。论文措辞建议用"architecturally simple / without geometric inductive bias"，**不要**说"fewest parameters"。

---

## 3. 结果（两种"配方"下结论不同——这点很重要）

实验跑过**两套**，结论相反，必须分清：

### 3a. 默认配方（sigmoid 颜色、损失等权 1:1:1）—— encoder **有**区别
此设定下几何 encoder 占优：
- **含场景 310**（5 序列，原始 `encoder_ablation`）：geom **25.12** 最好、attention **23.79** 最差。
- **剔除 310**（4 序列，`reaggregate --exclude 310`）：geom **26.51** 最好，领先约 0.8–1.5 dB。
→ 当 formulation 没调好时，更强的几何 encoder 确实帮得上。

### 3b. 最终配方（tanh 颜色、损失权重 0.1:10:1）—— encoder **打平**（★ 论文用这个）
这是 GS-Net 真正采用的配方。**剔除 310、4 个测试序列**（`encoder_ablation_final`，baseline=24.56）：

| 标签 | Encoder | PSNR | Δ vs baseline | 排名 |
|---|---|---|---|---|
| (b) | **Concat-MLP (Ours)** | **26.65** | **+2.09** | 1 |
| (e) | Explicit-Geometry | 26.51 | +1.95 | 2 |
| (a) | MLP-only | 26.40 | +1.84 | 3 |
| (d) | Self-Attention | 26.33 | +1.77 | 4 |
| (c) | EdgeConv | 25.37 | +0.81 | 5 |

- **前 4 名差 0.32 dB**（concat 26.65 ↔ attention 26.33）——**落在 GS-Net 训练随机性 ±0.5~0.9 dB 之内 = 统计打平**。
- 只有 **EdgeConv 明显偏低**（见 §4 解释）。
- 含 310 的同一组（5 序列）数字：concat 25.44 / mlp 25.24 / attn 25.09 / geom 25.07 / edgeconv 24.11——排序一致，结论一致。

> ⚠️ **指标纪律**：上表只列了 PSNR。论文/rebuttal 里**必须同时报 SSIM/LPIPS**（三项都在 `sse_results.json` 里，§6 给了拉取命令）。先把三项拉全再定稿，别只放 PSNR。

---

## 4. 解读（为什么会打平 / 为什么 EdgeConv 偏低）—— 这是回 Reviewer A 的核心论据

1. **输入信息量极小**：每个样本 = 中心点 + 3 个邻居 = 24 维。`concat + MLP` 已经能无损吃下全部信息，几何 encoder 的归纳偏置**无处发挥**——没有复杂的局部结构需要精巧归纳偏置去提炼。
2. **逐序列归一化让"几何不变性"变冗余**：输入已被归一化到统一尺度/中心，置换/旋转不变性这类设计带来的好处被抵消。
3. **任务需要"绝对定位"**：GS-Net 要预测每个点周围高斯的**绝对位置/属性**。EdgeConv 用**相对特征 `f_j−f_i` + max-pool**，恰恰**丢弃绝对位置信息**——它的不变性与任务目标错配，所以**反而最差**（−1.3 dB vs concat）。这是个很好的"反例论据"：不是"越复杂越好"，而是"归纳偏置要和任务匹配"。
4. **真正的瓶颈在别处**：增益对 encoder 不敏感，却对**损失/输出 formulation 敏感**（tanh 颜色可变暗、位置/旋转权重 10:0.1）——§3a vs §3b 的对比就是证据：同样 5 个 encoder，换了 formulation，从"有差别"变成"打平"，且整体抬高 ~1.5 dB。**⇒ 论文应把篇幅给 formulation，encoder 用轻量的就够。**

---

## 5. 论文写法——三个选项（你来选）

> 都是诚实的，差别在**强调点**和**给审稿人的姿态**。

### 选项 A（推荐）：正面"复杂度不必要"叙事
> "We ablate five encoder designs of increasing sophistication (per-point MLP, concat-MLP, EdgeConv, self-attention, explicit-geometry), holding everything else fixed. Under our final formulation, the four strongest variants are **statistically tied** (within 0.32 dB, below the ±0.5 dB training variance); the architecturally simple concat-MLP matches heavier graph/attention encoders. This shows the gain stems from our **loss formulation and Gaussian-expansion design**, not encoder capacity—justifying a lightweight encoder."
- **优点**：直接回 Reviewer A 的质疑（"简单 encoder 是有意为之、且足够"），叙事最干净。
- **配图**：§3b 那张 5 行表（PSNR/SSIM/LPIPS/#Params）。

### 选项 B：双配方对照，强调"是 formulation 不是 encoder"
> 同时放 §3a（默认配方，encoder 有别）和 §3b（最终配方，打平）两张表，论证"**encoder 的作用被好的 formulation 吸收**"。
- **优点**：论据更强、更有"机制洞察"；正好把 encoder 与 formulation 的因果讲透。
- **缺点**：占版面、要解释两套配方，可能引出"为什么默认配方下有差别"的追问。建议放**附录/补充材料**。

### 选项 C：保守"简单即足够"
> 只放最终配方表 + 一句"a lightweight encoder suffices; stronger encoders bring no consistent gain"，不展开机制。
- **优点**：最省版面、最不容易被反问。
- **缺点**：说服力弱于 A。

**我的建议**：**正文用选项 A**（一张表 + 2~3 句解读，含 EdgeConv 错配的反例），**附录放选项 B 的双配方对照**增强说服力。

---

## 6. 结果在哪查 / 怎么复现

> 结果文件在**服务器**的 `runs/` 下（不在仓库）。代码在仓库，已更新可一条命令复现两套配方。

### 6.1 直接查已有结果（最快）
| 想看 | 路径 |
|---|---|
| 最终配方聚合表（PSNR/SSIM/LPIPS/#Params/Infer/Train） | `runs/encoder_ablation_final/encoder_ablation.md` |
| 默认配方聚合表 | `runs/encoder_ablation/encoder_ablation.md` |
| 某 encoder 的逐序列三项指标（拉 SSIM/LPIPS 用这个） | `runs/encoder_ablation_final/<enc>/sse/sse_results.json` |
| 机器可读聚合 | `runs/encoder_ablation_final/encoder_ablation.json` |

`<enc>` ∈ `mlp_only / concat / edgeconv / attention / geom`。

### 6.2 ⭐ 关键：剔除场景 310 再平均（论文用的就是这个）
`encoder_ablation.md` 里的均值**包含崩坏场景 310**（每个 config 都崩到 ~19 dB，把均值拖低且加噪）。论文里的"剔 310"干净数字用 `reaggregate` 重算（**不重跑、秒出**）：
```bash
cd /mnt/zihanw/gaussian-splatting && git pull origin claude/festive-feynman-80Vw3
python -m gsnet.reaggregate --root runs/encoder_ablation_final --exclude 310
python -m gsnet.reaggregate --root runs/encoder_ablation        --exclude 310
```
→ 打印每个 encoder 剔 310 后的 gsnet PSNR / baseline / Δ / 保留的 ids。**这就是 §3 表的来源。**

### 6.3 拉 SSIM / LPIPS（定稿前必做）
`reaggregate.py` **现已直接打印三项**（PSNR/SSIM/LPIPS，gsnet 与 baseline 各一列）——§6.2 的命令一跑就同时给出剔 310 后的三项指标，无需再手动翻 json。
（若想看**含 310**的三项均值，直接看聚合表 `encoder_ablation.md`。）

### 6.4 从零复现（若结果丢了 / 想加 seed）
脚本已支持用**一个 `--color_activation` / `--w_*` 开关**切换两套配方：
```bash
# 最终配方（tanh:0.1:10:1）—— 论文主表
python -m gsnet.run_encoder_ablation --corr_dir CORR/train \
  --io_dir /mnt/zihanw/carla/input_output --sparse_root /mnt/zihanw/carla/sparse_point \
  --out_dir runs/encoder_ablation_final --gpus 4 5 6 7 \
  --color_activation tanh --w_rot 0.1 --w_pos 10 --w_scale 1 --M 3

# 默认配方（sigmoid:1:1:1）—— 对照
python -m gsnet.run_encoder_ablation --corr_dir CORR/train \
  --io_dir /mnt/zihanw/carla/input_output --sparse_root /mnt/zihanw/carla/sparse_point \
  --out_dir runs/encoder_ablation --gpus 4 5 6 7 --M 3
```
> ⚠️ `--M 3` 必须与 `--corr_dir` 构建时的 M 一致（CORR/train 是 M=3）。
> 跑完再用 §6.2 的 `reaggregate --exclude 310` 出干净数。
> 训练随机 → 单次 ±0.5~0.9 dB。若审稿人较真，可加 `--seeds`（当前脚本未内置多 seed，需要的话我加，给每个 encoder 出 mean±std，彻底坐实"打平"）。

---

## 7. 诚实 caveat（写进论文前自己心里有数）
1. **打平靠的是"在噪声内"**：前 4 名差 0.32 dB < 训练方差 ±0.5~0.9 dB。最稳妥的写法是"statistically tied"，**别**说"concat 最好/最优 encoder"——它名义第一，但不显著。**最好补多 seed 出误差棒**。
2. **EdgeConv 偏低是真的**（−1.3 dB），不是噪声——别把它也说成"打平"，要么不强调，要么用作"归纳偏置错配"的反例（更有说服力，见 §4.3）。
3. **310 必须剔且要交代**：论文/附录注明"excluding one degenerate sequence (310) that collapses under 3DGS optimization for all methods"。
4. **默认配方下 encoder 有别**——若只报最终配方表，不要在文字里说"encoder 从来不重要"，准确说法是"under our formulation"。
5. **指标要三项齐**：目前手头是 PSNR；SSIM/LPIPS 拉全前不要定稿（见 §6.3）。
6. **只在 CARLA SSE 上做的**：这套 encoder 消融是 CARLA SSE。够回 Reviewer A，但不要外推到 nuScenes/跨场景。

---

## 8. 文件索引
| 文件 | 作用 |
|---|---|
| `gsnet/encoders.py` | 7 个 encoder 定义（a–e + attention_v2 + geoedge） |
| `gsnet/run_encoder_ablation.py` | 训练→SSE→聚合一条龙；`--color_activation/--w_rot/--w_pos/--w_scale` 切配方 |
| `gsnet/reaggregate.py` | 不重跑、剔 310 重算均值 |
| `gsnet/run_design_sweep.py` | encoder×color×权重 的廉价排序（7k 迭代、子集），找配方用的 |
| `runs/encoder_ablation_final/` | 最终配方结果（服务器） |
| `runs/encoder_ablation/` | 默认配方结果（服务器） |
</content>
