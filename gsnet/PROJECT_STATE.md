# GS-Net 项目状态与交接文档 (PROJECT STATE)

> **用途**：记忆压缩后的"续命"文档。读完本文件即可完整理解任务、数据、代码、流程、进度、当前关键路径。
> **最后更新**：2026-05-30
> **开发分支**：`claude/festive-feynman-80Vw3`（所有改动 push 到这；用户在服务器 `git pull`）

---

## 0. 协作方式（重要）
- 我（Claude）**无法访问用户服务器**。代码 push 到 GitHub 分支，用户 `git pull` 后运行，再把输出贴回来。
- 数据在服务器、不在仓库；`runs/`、`CORR/` 等产物未被 git 跟踪。
- 用户机器：8× A100-80GB（卡 0–7）。**卡 4–7 常被别人的 RL 任务(`actor_rollout`)占**（用户说显存够、共卡不会崩，但算力时间片共享会变慢）；以 `nvidia-smi` 为准。
- **务必记录时间/结果**（用户反复强调）：每个脚本都写 `*_times.json` / `*_results.{json,md}`。
- 多行命令粘贴易因续行 `\` 后混入空行而断；给用户命令尽量**写成单行**。

## 0.5 当前关键路径（最重要）
**GS-Net 增益远低于论文**（见 §7.5）：encoder 消融发现 Ours(concat)=24.46 ≈ 基线 24.67，最好的 (e)Explicit-Geometry 也仅 25.12（+0.45），论文是 +2.08。**用户已同意改进网络设计/改论文**。当前在跑**设计 sweep**（`run_design_sweep.py`：encoder×颜色激活×损失权重，7k/3序列快速排序）找更好配置；若不够，下一步上**集合级匹配损失**。

## 0.6 运行中的实验 / tmux（用户维护，尽力同步）
| tmux | 内容 | 状态 |
|---|---|---|
| zyc1 | finalist M=3 (30k/5序列) → `runs/final_m3/design.md` | 🔄 |
| zyc3 | finalist M=16 (30k/5序列) → `runs/final_m16/design.md` | 🔄 |
> 已完成回收：design/design2/design_attn(@7k)、Waymo SSE(+0.07,弱模型)、监督敏感性(优雅降级)。
> 待回收：final_m3 / final_m16 两表(30k/5序列 vs 基线24.67) → 定最终 Ours。
> 之后统一收尾：CARLA主表SSE / CSE全量 / Waymo重训 + CARLA→Waymo零样本迁移（均用最终 Ours）。

## 1. 任务背景
- 论文《GS-Net: Heterogeneous Vehicle Data Reuse via Generalizable Plug-and-Play 3DGS Module》**代码丢失，按论文重建**。仓库初始是官方 3DGS(Inria)。
- 两大贡献：① CARLA-NVS 数据集（12 相机跨传感器 NVS benchmark）；② **GS-Net**：从稀疏 SfM 点云一次前向预测稠密高斯，作 3DGS 即插即用初始化。
- 处于 **rebuttal**，需补回应审稿人：(A) encoder 设计太简单→消融；(B) 只有 CARLA→真实数据(Waymo)；(C) 对伪 GT 质量的依赖/敏感性。另含 CSE(跨传感器)评测。

## 2. 方法核心（GS-Net）
**离线数据引擎(造伪GT)**：COLMAP SfM→稀疏点 `P_sfm`；MVS→稠密点；以稠密点初始化跑 3DGS 30k→收敛稠密高斯 `G_dense`；每个稀疏点 KD-tree 取 K=5 最近 `G_dense` 高斯做伪 GT。
**在线 GS-Net(单次前向)**：输入稀疏点 6 维(xyz+rgb)→(a)几何感知编码(逐点共享MLP+邻域聚合M=3)→`F_n`→(b)多头扩张:每点→T=5 高斯，增量预测 `μ̂=μ_in+Tanh(Δμ)`、`Ĉ=C_in+σ/tanh(ΔC)`、`R̂=quat2rot`、`Ŝ=σ(s)`、`α̂=Tanh`→(c)解耦几何/外观损失。训练 200ep/batch512/Adam1e-3/T5/M3。推理→导出标准3DGS ply→作初始化。

**关键工程决策（论文外、为让它 work）**：
1. **逐序列归一化**：COLMAP 尺度大(CARLA~数百)，否则 σ scale∈(0,1) 无法表示。中心=median、尺度=稀疏点到中心距 p95（仅用稀疏点，推理可复现）；目标同变换；推理后反归一化导出。
2. **G_dense 过滤**：去远处漂浮点(>1.5×p99)+近透明(opacity<0.005)。
3. **GT scale 裁剪**到 (0,0.999)（防超 sigmoid 范围）。
4. **损失权重可调** `--w_pos/--w_rot/--w_scale/--w_rgb/--w_opacity`（默认 1.0）。
5. **3DGS 集成**：`--gsnet_init <ply>`(`create_from_ply`)；`--init_pcd <ply>`(用任意点云如 MVS fused.ply 初始化，造 G_dense 用)。
6. **测试集划分**：`dataset_readers` 优先读 `sparse/0/test.txt`，否则 LLFF hold。
7. **训练提速** `--in_memory`(默认开)：整集驻留 GPU、permutation 批处理、不走 DataLoader，~100x 快（200ep 几分钟）。

## 3. 代码结构（`gsnet/` 包 + 少量主仓库改动）
| 文件 | 作用 |
|---|---|
| `model.py` | GS-Net 网络；encoder 可插拔(`cfg.encoder_type`)；`cfg.color_activation`(sigmoid/tanh) |
| `encoders.py` | **6** 个 encoder：`mlp_only`/`concat`(原Ours)/`edgeconv`/`attention`/`geom`(显式几何,消融最好)/`geoedge`(几何EdgeConv,新) |
| `losses.py` | 解耦几何/外观损失 + 可调权重(`weights`) |
| `common.py` | 点云/高斯读取(.ply/.bin/.txt)、G_dense 过滤、归一化 |
| `build_correspondences.py` | KD-tree 伪 GT 生成(`--batch`自动发现`output_*_dense`)；`--gdense_iter`(监督迭代)、`--gdense_subsample`(密度)；含 `build_for_sequence` |
| `dataset.py` / `train_gsnet.py` | 跨场景拼接 npz / 训练(`--encoder_type --color_activation --w_* --in_memory`，计时) |
| `infer.py` / `io.py` | 单次前向→反归一化→导出 ply / 写标准3DGS ply |
| `make_sse_split.py` | CARLA SSE：每相机10帧抽第4、9帧→test.txt |
| `make_cam_split.py` | 通用：读 images.bin 按相机(名字目录前缀)抽内部均匀N帧→test.txt（Waymo用）|
| `run_sse.py` | CARLA SSE 驱动：基线 vs GS-Net+3DGS，多卡，merge/resume，`--test_ids --iterations` |
| `run_encoder_ablation.py` | encoder 消融：训全变体→全量SSE→出表 |
| **`run_design_sweep.py`** | **设计 sweep：encoder×color×权重，配置级并行(一卡一配置)，7k/子集快速排序** |
| `run_weight_sweep.py` | 仅损失权重 sweep |
| `run_supervision_sensitivity.py` | 伪GT质量敏感性：监督迭代{5k..30k}×下采样{100..10%}→训+评→表 |
| `cse_poses.py` | **CSE：从CARLA rig+奇数COLMAP位姿推算偶数位姿**(per-frame Umeyama+暴力搜坐标约定+自校验) |
| `make_cse_scene.py` | CSE：拼 60奇训练+60偶测试 的文本COLMAP模型+图像软链+test.txt(偶数)；相机强制PINHOLE；点云=奇数稀疏 |
| `run_cse.py` | CSE 驱动：基线 vs GS-Net+3DGS（重建60奇，评测60偶）|
| `carla_rig.json` | 实际 CARLA rig（半径0.75十二边形，yaw0..330，pitch/roll0）|
| `waymo.py`/`waymo_gdense.py`/`waymo_diag.py`/`waymo_corr.py`/`waymo_sse.py` | Waymo 路径助手/造G_dense/sfm-vs-mvs诊断/建对应/SSE |
| `inspect_ply.py` | 诊断：ply字段/尺度/对齐 |

**主仓库改动**：`arguments/__init__.py`(+`gsnet_init`,`init_pcd`)、`scene/__init__.py`(分支)、`scene/gaussian_model.py`(+`create_from_ply`)、`scene/dataset_readers.py`(优先test.txt)。
**⚠️ checkpoint 兼容**：encoder 重构后子模块名 `encoder.*`，旧 `runs/gsnet`(point_encoder/...) 加载不了，已弃用。

## 4. 数据布局（服务器绝对路径）
### CARLA-NVS（`/mnt/zihanw/carla/`）
- 稀疏 SfM(GS-Net输入)：`sparse_point/S0<scene>/<id>_sparse.ply`，`id=100*scene+seq`，scene1–5,seq1–10。
- 稠密MVS+3DGS工作区：`input_output/<id>_dense/`（注意其`sparse/0/points3D.ply`是**稠密MVS**！）
- **G_dense**：`input_output/output_<id>_dense/point_cloud/iteration_{5000..30000}/point_cloud.ply`（多阶段都存了：42个有5k–25k，43个有30k）。
- 测试序列：`input_output/<id>_base/`(110/210/310/410/510)，图像扁平`1-60.png`；相机映射 1-10→cam1,11-20→cam3,…,51-60→cam11（6奇数相机×10帧）。
- GS-Net 训练序列：101-109…501-509（实际**43**个；102、109缺G_dense被跳过）。
- SSE 划分：每相机抽第4、9帧→`{4,9,…,59}.png`（12测试/48训练）。
- **CSE 偶数图像**：`paired_120/<id>_dense/cam{01..12}/<帧>.png`（12相机全有；偶数=cam02..12）。帧按**目录排序第f张**取（兼容命名）。
- **CARLA rig**：`gsnet/carla_rig.json`。位姿推算自校验残差 rot0.26–0.86°/center0.22–0.67%（SfM噪声量级）。

### Waymo（`/mnt/zihanw/EmerNeRF/data/waymo/colmap_input/`）
- 10个`segment-*`，前视3相机×20帧=60张。每场景`colmap/dense/`：`fused.ply`(MVS)、`images/cam{0,1,2}/0NN.jpg`、`sparse/0/*.bin`。稀疏SfM=`colmap/dense/sparse/0/points3D.bin`(图像名`cam0/000.jpg`式)。
- **G_dense**→`runs/waymo_gdense/<seg>/point_cloud/iteration_30000/point_cloud.ply`（全10已生成）。
- 测试场景(2)：`...10275144660749673822_5755_561...`、`...15868625208244306149_4340_000...`；其余8训练。诊断场景：`...3425716115468765803_977_756...`。
- SSE 划分：每相机20帧抽内部均匀4帧`[4,8,11,15]`。

## 5. 完整流程命令（均可单行）
```bash
# CARLA: 建对应→训练→SSE
python -m gsnet.build_correspondences --batch --io_dir /mnt/zihanw/carla/input_output --sparse_root /mnt/zihanw/carla/sparse_point --out_dir CORR/train --workers 8
python -m gsnet.train_gsnet --corr_dir CORR/train --out_dir runs/gsnet --encoder_type geom --in_memory 1
python -m gsnet.run_sse --io_dir /mnt/zihanw/carla/input_output --sparse_root /mnt/zihanw/carla/sparse_point --ckpt <ckpt> --out_dir runs/sse --gpus 1 2 3
# 设计 sweep（当前关键路径）
python -m gsnet.run_design_sweep --corr_dir CORR/train --io_dir /mnt/zihanw/carla/input_output --sparse_root /mnt/zihanw/carla/sparse_point --out_dir runs/design --gpus 0 1 2 3 4 6 --iterations 7000 --eval_ids 110 310 510 --configs geom:sigmoid:1:1:1 geom:sigmoid:0.1:10:1 geom:tanh:0.1:10:1 geoedge:sigmoid:0.1:10:1 geoedge:tanh:0.1:10:1 geoedge:tanh:0.1:10:10
# 监督敏感性
python -m gsnet.run_supervision_sensitivity --io_dir /mnt/zihanw/carla/input_output --sparse_root /mnt/zihanw/carla/sparse_point --out_dir runs/sup_sens --gpus <空> --iters 5000 10000 20000 --subsamples 0.5 0.25 0.1
# CSE: 推位姿→建场景→评测
python -m gsnet.cse_poses --io_dir /mnt/zihanw/carla/input_output --rig gsnet/carla_rig.json --ids 110 210 310 410 510 --out_dir runs/cse_poses
python -m gsnet.make_cse_scene --io_dir /mnt/zihanw/carla/input_output --paired_dir /mnt/zihanw/carla/paired_120 --poses_dir runs/cse_poses --out_dir runs/cse_scenes --ids 110 210 310 410 510
python -m gsnet.run_cse --scenes_dir runs/cse_scenes --sparse_root /mnt/zihanw/carla/sparse_point --ckpt <ckpt> --out_dir runs/cse --gpus <空>
# Waymo: G_dense→建对应(排除2测试)→训练→SSE
python -m gsnet.waymo_gdense --root /mnt/zihanw/EmerNeRF/data/waymo/colmap_input --out_dir runs/waymo_gdense --gpus <空>
python -m gsnet.waymo_corr --root /mnt/zihanw/EmerNeRF/data/waymo/colmap_input --gdense_dir runs/waymo_gdense --out_dir CORR/waymo_train --workers 8 --test_scenes <seg10275> <seg15868>
python -m gsnet.train_gsnet --corr_dir CORR/waymo_train --out_dir runs/waymo_gsnet --in_memory 1
python -m gsnet.waymo_sse --root /mnt/zihanw/EmerNeRF/data/waymo/colmap_input --test_scenes <seg10275> <seg15868> --ckpt runs/waymo_gsnet/gsnet_latest.pt --out_dir runs/waymo_sse --gpus <空>
```

## 6. 进度
### ✅ 已完成
- 全部核心代码 + CARLA/Waymo/CSE/敏感性/设计sweep 全套驱动。
- CARLA 建对应：43序列(scale56–181,无clip,对齐好)。**基线SSE**：110=26.08/210=24.53/310=26.60/410=22.59/510=23.53，**Avg24.67**(SSIM0.919,LPIPS0.168)。
- **encoder 消融**（见§7.5）。
- **Waymo**：诊断 sfm28.18/mvs29.33(+1.15)；G_dense全10；建对应8场景(scale24–51,无clip)；GS-Net训练done(1916s,233852点)。**Waymo SSE 待跑**。
- **CSE**：位姿推算(残差亚度/亚%)；5场景构建好；**110基线自检18.71(SSIM0.722)→偶数位姿验证通过**(SSE→CSE降7.4dB合理)。**CSE全量待跑**。
### 🔄 进行中
- 设计 sweep（关键路径）；监督敏感性。
### ⬜ 待办
- 设计 sweep 选优→全5序列30k确认→定新 Ours→**重跑** CARLA主表SSE/CSE全量/Waymo SSE。
- 若设计 sweep 不够：上**集合级匹配损失**。
- 用真实数字补全 4 段 rebuttal。
- Waymo CSE(外推可视化)——外推角度待与用户讨论。

## 7. 关键发现
- **训练 loss 形态**：`loss_mu`极小(稠密高斯贴稀疏点)；`loss_scale`≈0(高斯小)；`loss_rgb/opacity`~0.13在降；**`loss_rot`卡~1.5不降**(近各向同性高斯旋转病态、目标即噪声)。
- `inspect_ply`：CARLA坐标~数百单位；稀疏↔G_dense对齐好(NN p50≈0.19)；G_dense有远漂浮点(已过滤)；SH degree3。
- Waymo metric尺度；归一化/过滤自适应，无clip警告。

## 7.5 ⚠️ GS-Net 增益偏低 + 网络设计 R&D（核心）
- **encoder 消融**(SSE,5测试,30k)：(a)MLP-only23.91 / (b)Concat=原Ours24.46 / (c)EdgeConv23.95 / (d)Attention23.79 / **(e)Explicit-Geometry25.12(最好)**；参数207.9/306.2/240.6/290.7/306.4K；infer~6s；train~2h。基线=24.67。
- **问题**：原Ours(concat)24.46≈基线(甚至略低)，最好(e)仅+0.45，**远低于论文+2.08**。重建未达论文效果。
- **诊断**：①旋转损失=不可学噪声地板、淹没梯度；②位置项归一化后目标极小、梯度弱；③颜色σ只能变亮；④通用强encoder(图/注意力)无效→表达力非瓶颈、几何先验才是。
- **已实现改进**：`geoedge`(几何EdgeConv)；`color_activation=tanh`(可变暗)；损失权重；`--in_memory`提速。
- **R&D流程**：`run_design_sweep`(encoder×color×权重,7k/3序列快排,配置级并行)→选优→全5序列30k确认；`run_weight_sweep`(仅权重)。
- **下一招(若不够)**：集合级匹配损失（预测T↔GT K 最近邻/匈牙利匹配，替代任意 t-to-t 配对）。可能位置参数化改局部尺度、颜色去正偏置等。

## 7.6 设计/敏感性/Waymo 结果（2026-05-30）
- **设计 sweep@7k/3序列**（runs/design）：**geoedge:tanh:0.1:10:1=24.47(最优)** > geoedge:tanh:0.1:10:10 24.23 > geoedge:sigmoid:0.1:10:1 23.73 > geom:tanh:0.1:10:1 23.43 > geom:sigmoid:0.1:10:1 23.33 > geom:sigmoid:1:1:1 23.25。**三改动(geoedge/tanh/调权)各有用且叠加，+1.22**。待 design2/design_attn 合并选全局最优 → 全5序列30k确认(vs基线24.67) → 定最终 Ours。
- **Waymo SSE**（用弱 concat 模型）：baseline 30.13 / gsnet 30.20(+0.07)，LPIPS 0.244→0.235。≈基线(符合预期)，**待用最优配置重训 Waymo GS-Net 重测**。注:gsnet优化50min>基线29min(5×初始高斯)。
- **监督敏感性**(30k/5序列,concat)：轴A 监督迭代 5k=23.80/10k=24.13/20k=23.96/(30k≈24.46)；轴B 密度 50%=24.03/25%=24.00/10%=23.64。**两轴优雅降级→对伪GT质量鲁棒**(reviewer C)。
- **设计 sweep 合并(3批,@7k/3序列)**：top=geoedge:tanh:0.1:10:10(M3)24.76 / geoedge:tanh:0.1:10:1(M3)24.47 / attention:tanh(M16)24.43 / geom:tanh:0.1:10:1(M16)24.41。**结论**：tanh≫sigmoid、调权(0.1:10:*)≫默认、geoedge 最强 encoder、**M=16 明显帮 geom(+0.98)**、注意力即使 M16 也不更好。**⚠️7k/3序列噪声~0.5dB(同配置24.23 vs24.76)**→ 需 30k/5序列确认。**最有希望 geoedge×M16 未测**。
- **finalist 30k/5序列确认中**（runs/final_m3: geoedge:tanh:0.1:10:{1,10} @M3；runs/final_m16: geoedge:tanh:0.1:10:{1,10}+geom:tanh:0.1:10:1 @M16）vs 基线24.67 → 定最终 Ours。
- **CARLA→Waymo 零样本迁移**（reviewer 最关注的真实数据迁移）：`waymo_sse --ckpt <CARLA ckpt> --skip_baseline`；机制上归一化使其可迁移；先用 geom 预览、最终用 Ours 重跑。三方对比：Waymo基线30.13/Waymo自训30.20/CARLA→Waymo零样本=?

## 8. 给审稿人的回应（草稿，待真实数字填充）
- **(A) Encoder 太简单**：补 encoder 设计消融(a–e)。结论：邻域必要(a最差)；**通用强encoder(图c/注意力d)不帮忙**→表达力非瓶颈；**审稿人建议的"显式几何"确有效((e)最好)，已采纳并进一步改进(geoedge)**。报告 PSNR/SSIM/LPIPS+参数+推理时间。
- **(B) 真实数据**：Waymo 10场景，SSE 量化(基线vs GS-Net+3DGS)；后续 CSE 外推可视化。
- **(C) 对伪GT质量的依赖**：监督质量敏感性两轴——①优化后高斯质量(监督迭代5k–30k)；②伪GT密度/MVS覆盖(下采样100–10%)。预期平滑/优雅降级→方法不脆弱。
- **(CSE) 跨传感器**：用 CARLA rig 推算偶数相机位姿(不重跑120 SfM，保奇数重建/G_dense一致)，自校验残差亚度级；基线vs GS-Net 在 60 偶数视角评测。
