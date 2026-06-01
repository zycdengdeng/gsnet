# GS-Net 项目状态与交接文档 (PROJECT STATE)

> **用途**：记忆压缩后的"续命"文档。读完即可完整理解任务/数据/代码/流程/结果/当前路径，性能不退化。
> **最后更新**：2026-05-30（最终收尾 + 多种子方差研究阶段）
> **分支**：`claude/festive-feynman-80Vw3`（push 到此；用户服务器 `git pull`）

---

## 0. 协作方式
- 我（Claude）**无法访问服务器**。代码 push 到 GitHub，用户 pull 后运行、贴回输出。数据在服务器、不在仓库；`runs/`、`CORR/` 未跟踪。
- 机器：8× A100-80GB。卡 4–7 有时被别人 RL 占（显存够、共卡不崩，但算力时间片共享变慢）；以 `nvidia-smi` 为准。**用户现在常说 8 卡全空、可大并行**。
- **务必记录时间/结果**：脚本都写 `*_times.json` / `*_results.{json,md}`。
- 多行命令易因续行 `\` 后空行而断 → 给用户命令**写成单行**。

## 0.5 当前状态 & 下一步（最重要）
- **最终 Ours（CARLA）已定：`geom:tanh:0.1:10:1 @ M=3`**，ckpt=`runs/final_m3/model/geom_tanh_wr0.1_wp10_ws1/gsnet_latest.pt`。CARLA 主表：基线 24.67 → Ours ~25-26（方差±0.5-0.9，单次不可信）。
- **⚠️ 方差大**：基线 3DGS 确定性(safe_state固定种子)→基线可靠；GS-Net 训练随机→下游 PSNR ±0.5-0.9dB。结论看趋势、>0.8dB 才算真涨。
- **核心未决问题**：GS-Net 在 **Waymo / CSE** 上没起效（CARLA SSE +1.3 有效，但 CARLA CSE≈基线、Waymo 3相机≈基线、Waymo稀疏更差）。诊断=GS-Net 是"锚定式局部密化"，只在"有覆盖空洞+视角够"的 regime 有效；Waymo 前视无空洞。**当前正用 5 相机 Waymo（前视→侧视跨传感器）验证 regime 假设**。
- **用户外出，已把所有能跑的挂上自动队列**（见 §0.6）。回来收 5 份结果。
- **训练速度更正**：`--in_memory` 不是"几分钟"，CARLA(87万点)实测 **~1hr/训练**(kernel-launch 受限)，比旧 DataLoader 2h 快一倍而已。

## 0.6 自动队列 / 运行中（2026-06，用户外出无人盯）
按**产出目录**认（比 tmux 名可靠）。全部会自动训完→自动评测出结果：
| 实验 | 卡 | 产出 | 备注 |
|---|---|---|---|
| CARLA T扫描{3,5,8,12,16} | 0-4 | `runs/Tsweep_carla/Tsweep.md` | 5个T并行训练中→评测 |
| Waymo5 Step3 主模型训练 | 0 | `runs/waymo5_gsnet/`(train_times.json=完成标志) | 完成后触发4a/4b |
| Waymo 跨传感器 T扫描{3,5,8,12,16} | 4-7 | `runs/Tsweep_waymo_cse/Tsweep.md` | --target_cams cam3 cam4 |
| **4a 5相机SSE**(等Step3) | 0 1 | `runs/waymo5_sse/sse_results.md` | 自动waiter:`until [ -f runs/waymo5_gsnet/train_times.json ]` |
| **4b 跨传感器**(等Step3) | 2 3 | `runs/waymo5_cse/sse_results.md` | 同上;**最关键结果** |
> Waymo5 数据: `/mnt/zihanw/EmerNeRF/data/waymo/colmap_input_5cam`(5相机×20帧,SfM+去畸变PINHOLE+MVS,10 segment)。G_dense=runs/waymo5_gdense(全10 OK)。corr=CORR/waymo5(8训练,排除10275/15868两测试,scale29-57无clip)。相机 cam0=FRONT,1=FL,2=FR,3=SL,4=SR;前视=cam0/1/2,侧视=cam3/4。
> **回来要看的 5 份**: Tsweep_carla, Tsweep_waymo_cse, waymo5_gsnet/train_times, waymo5_sse, waymo5_cse。**最关键=waymo5_cse(跨传感器)**:若 GS-Net 明显超基线→证明网络/设计没问题、只是之前 Waymo 前视 regime 选错。
> 卡有重叠共用(0,4)→慢但不崩(显存80G够)。
> **第二批(闸门队列,等上面两个Tsweep.md后自动跑)**：补充1 Waymo5相机SSE的T扫描→`runs/Tsweep_waymo_sse/Tsweep.md`；补充2 CARLA多种子方差(geom×5种子,SSE+CSE)→`runs/multiseed/multiseed.md`(给主表 mean±std)。
> 历史完成：encoder消融(调损后各encoder打平,edgeconv偏低)、伪GT敏感性(优雅降级=reviewer C搞定)、Waymo诊断(sfm28.18/mvs29.33)、CARLA→Waymo零样本(负迁移,已弃用)、路径2 input_subsample(死路20.05<基线21.31)。

## 1. 任务背景
- 论文《GS-Net: Heterogeneous Vehicle Data Reuse via Generalizable Plug-and-Play 3DGS Module》**代码丢失、按论文重建**（仓库初始=官方 3DGS Inria）。
- 两贡献：① CARLA-NVS（12相机跨传感器NVS benchmark）；② **GS-Net**：稀疏SfM点→一次前向预测稠密高斯，作3DGS即插即用初始化。
- **rebuttal 阶段**。审稿人四大实验诉求：(A)encoder太简单→消融；(B)只有CARLA→真实数据(Waymo)；(C)对伪GT质量的依赖/敏感性；(CSE)跨传感器。**feed-forward/generalizable对比(QuickSplat等)、NeRF baseline 用户在别处跑，本窗口不做**。

## 2. 方法核心
**离线数据引擎(造伪GT)**：COLMAP SfM→稀疏`P_sfm`；MVS→稠密；以稠密初始化跑3DGS 30k→`G_dense`；每稀疏点 KD-tree取 K=5 最近`G_dense`高斯为伪GT。
**在线GS-Net(单次前向)**：稀疏点6维(xyz+rgb)→几何感知编码(逐点MLP+M邻域聚合)→`F_n`→多头扩张:每点→T=5高斯，增量预测 `μ̂=μ_in+Tanh(Δμ)`、`Ĉ=C_in+σ/tanh(ΔC)`、`R̂=quat2rot`、`Ŝ=σ(s)`、`α̂=Tanh`→解耦几何/外观损失。推理→反归一化→导出标准3DGS ply→作初始化。

**关键工程决策（论文外、为让它 work）**：
1. **逐序列归一化**（中心=median、尺度=稀疏点p95距；仅用稀疏点、推理可复现；目标同变换、推理后反归一化）——CARLA坐标~数百，否则σ scale/Tanh偏移失效。
2. **G_dense过滤**（远漂浮>1.5×p99 + opacity<0.005）。3. **GT scale裁剪**到(0,0.999)。
4. **损失权重** `--w_pos/--w_rot/--w_scale/--w_rgb/--w_opacity`（默认1）。**最终用 w_rot=0.1,w_pos=10**（旋转是不可学噪声地板、淹没梯度；位置归一化后目标小、需抬权）。
5. **颜色激活** `--color_activation sigmoid|tanh`（最终用 **tanh**，可变暗；σ只能变亮）。
6. **3DGS集成**：`--gsnet_init`(create_from_ply)；`--init_pcd`(任意点云如MVS fused.ply初始化，造G_dense)。
7. **测试划分**：`dataset_readers`优先`sparse/0/test.txt`。
8. **训练提速** `--in_memory`(默认，整集驻GPU、~100x快)；`--seed`(多种子)。

## 3. 代码结构（`gsnet/` + 主仓库改动）
| 文件 | 作用 |
|---|---|
| `model.py` | 网络；`cfg.encoder_type`、`cfg.color_activation` |
| `encoders.py` | 7 encoder：mlp_only/concat/edgeconv/attention/**attention_v2**(向量注意力)/geom/**geoedge**(几何EdgeConv) |
| `losses.py` | 解耦损失+可调权重 | `common.py` | 读点云/高斯、过滤、归一化 |
| `build_correspondences.py` | 伪GT生成(`--batch`;`--gdense_iter`监督迭代;`--gdense_subsample`密度;`--M`) |
| `dataset.py`/`train_gsnet.py` | 拼npz / 训练(`--encoder_type --color_activation --w_* --M --in_memory --seed`) |
| `infer.py`/`io.py` | 前向→反归一化→导ply / 写3DGS ply |
| `make_sse_split.py`/`make_cam_split.py` | CARLA SSE(帧4,9) / 通用按相机抽帧(Waymo) |
| `run_sse.py` | CARLA SSE驱动(基线vs GS-Net,多卡,resume,`--test_ids --iterations`) |
| `run_encoder_ablation.py` | encoder消融 |
| `run_design_sweep.py` | 设计sweep(encoder×color×权重,配置级并行,`--M --iterations --eval_ids`) |
| `run_weight_sweep.py`/`run_supervision_sensitivity.py` | 权重sweep / 伪GT敏感性 |
| **`run_multiseed.py`** | **多种子方差(geom×N种子→SSE/CSE mean±std)** |
| `cse_poses.py` | CSE偶数位姿推算(rig+奇数COLMAP+Umeyama+约定搜索+自校验) |
| `make_cse_scene.py`/`run_cse.py` | 拼CSE COLMAP模型(60奇训+60偶测,PINHOLE) / CSE评测 |
| `carla_rig.json` | CARLA rig(半径0.75十二边形,yaw0..330) |
| `waymo.py`/`waymo_gdense.py`/`waymo_diag.py`/`waymo_corr.py`/`waymo_sse.py` | Waymo 路径/G_dense/诊断/对应/SSE |
| `inspect_ply.py` | 诊断 ply 字段/尺度/对齐 |

**主仓库改动**：`arguments`(+gsnet_init,init_pcd)、`scene/__init__.py`(分支)、`scene/gaussian_model.py`(+create_from_ply)、`scene/dataset_readers.py`(优先test.txt)。
**⚠️ ckpt兼容**：encoder重构后子模块名`encoder.*`，旧`runs/gsnet`(point_encoder)加载不了，弃用。

## 4. 数据布局（服务器绝对路径）
### CARLA-NVS（`/mnt/zihanw/carla/`）
- 稀疏SfM(GS-Net输入)：`sparse_point/S0<scene>/<id>_sparse.ply`，`id=100*scene+seq`，scene1–5,seq1–10。
- MVS+3DGS工作区：`input_output/<id>_dense/`（其`sparse/0/points3D.ply`是**稠密MVS**！）
- **G_dense**：`input_output/output_<id>_dense/point_cloud/iteration_{5000..30000}/point_cloud.ply`（42个有5k–25k，43个有30k）。
- 测试序列：`input_output/<id>_base/`(110/210/310/410/510)，图`1-60.png`；1-10→cam1…51-60→cam11（6奇数相机×10帧）。
- 训练序列：43个(101-509，缺102/109)。SSE划分：每相机抽第4、9帧。
- **CSE偶数图**：`paired_120/<id>_dense/cam{01..12}/<帧>.png`（偶=02..12，按目录排序第f张取）。rig=`gsnet/carla_rig.json`。
### Waymo（`/mnt/zihanw/EmerNeRF/data/waymo/colmap_input/`）
- 10 segment，前视3相机×20帧=60张。`colmap/dense/`：`fused.ply`(MVS)、`images/cam{0,1,2}/`、`sparse/0/*.bin`。稀疏SfM=`sparse/0/points3D.bin`(名`cam0/000.jpg`)。
- G_dense→`runs/waymo_gdense/<seg>/.../iteration_30000/point_cloud.ply`（全10已生成）。
- 测试场景(2)：`...10275144660749673822_5755_561...`、`...15868625208244306149_4340_000...`；其余8训练。诊断场景`...3425716115468765803_977_756...`。SSE划分：每相机抽内部均匀4帧[4,8,11,15]。

## 5. 关键命令（单行）
```bash
# 多种子(当前): python -m gsnet.run_multiseed --config geom:tanh:0.1:10:1 --M 3 --corr_dir CORR/train --io_dir /mnt/zihanw/carla/input_output --sparse_root /mnt/zihanw/carla/sparse_point --scenes_dir runs/cse_scenes --eval sse cse --seeds 0 1 2 3 4 --gpus 0 1 2 3 4 5 6 7 --out_dir runs/multiseed
# 单训最终Ours: python -m gsnet.train_gsnet --corr_dir CORR/train --encoder_type geom --color_activation tanh --w_rot 0.1 --w_pos 10 --M 3 --in_memory 1 --out_dir runs/ours
# CSE: cse_poses → make_cse_scene → run_cse --ckpt <ours> --gpus ...
# Waymo: waymo_gdense → waymo_corr(--test_scenes 2个) → train_gsnet(geom配方) → waymo_sse
```

## 6. 结果汇总（基线确定性可靠；Ours 单次有±0.5–0.9dB噪声，多种子进行中）
| 实验 | 基线 | Ours(geom:tanh:0.1:10:1@M3) | 备注 |
|---|---|---|---|
| **CARLA SSE** | 24.67 | 25.99 / 25.07（两次）| **稳定超基线 +0.4~1.3**；待多种子定值。论文 +2.08 |
| **CARLA CSE** | 19.66 | **19.52(−0.14)** | ⚠️**未复现论文+1.86**，≈基线略低；逐场景2升3降 |
| **Waymo SSE(in-domain)** | 30.16 | 30.22(+0.06) | **真实数据答复(采用)**；基线已高、空间小；LPIPS 0.243→0.235 |
| ~~CARLA→Waymo零样本~~ | 30.16 | 26.63(−3.5) | **弃用**(负迁移)。理由:已做真实in-domain实验,无需用零样本讨论gap;runs/waymo_transfer 仅留档 |
- **encoder 消融**：默认配方(sigmoid,1:1:1,M3)→geom最好(25.12)、attention最差(23.79)；**最终配方(tanh,0.1:10:1,M3)→各encoder打平**(concat25.44/mlp25.24/attn25.09/geom25.07/edgeconv24.11)。结论：**增益来自损失设计(tanh+调权)，非encoder复杂度→支持轻量encoder**；attention/attention_v2/大M均不更优(且7k排序噪声大，30k才可信)。
- **伪GT敏感性**(reviewer C, concat,30k/5)：监督迭代5k=23.80/10k=24.13/20k=23.96/(30k≈24.46)；密度100%≈24.46/50%=24.03/25%=24.00/10%=23.64。**两轴优雅降级→鲁棒**。
- **Waymo诊断**：sfm28.18/mvs29.33(+1.15)。**CSE位姿自校验**残差rot0.26–0.86°/center0.22–0.67°；110自检18.71。

## 7. 给 rebuttal 的诚实素材映射（数字待多种子定稿）
- **(A)encoder简单**：消融a–e+geoedge。结论：通用强encoder(图/注意力)不帮忙；调好损失后轻量encoder足够；增益来自损失设计与几何感知聚合。
- **(B)真实数据**：**采用 Waymo in-domain(训+测)** 作为真实数据答复（≈基线、LPIPS改善；待"稀疏SfM"实验体现更大增益，见§7.8）。**零样本迁移弃用**（审稿人本意：不做真实实验才用迁移讨论gap，我们已做）。
- **(C)伪GT依赖**：监督质量两轴敏感性，优雅降级→不脆弱。**这块最强**。
- **(CSE)**：rig 推位姿(不重跑SfM)，自校验亚度；但**GS-Net在CSE≈基线(未复现+1.86)**→ rebuttal 需谨慎：可强调困难外推 + LPIPS、或作为 limitation 讨论。
- **本窗口不做**：QuickSplat/generalizable对比、NeRF baseline（用户别处）、下游AD任务(R2.7)。

## 7.8 待议想法：Waymo 基线偏高
- 假设：Waymo 前视3相机+20帧 覆盖过稠密→SfM/MVS稠密→基线~30近天花板→GS-Net稠密化无用武之地（+高PSNR区增益压缩）。
- **正确杠杆=让SfM稀疏=减少视角(重跑稀疏COLMAP)**；减初始点无效(3DGS密化会长回来)。GS-Net已训好、稀疏SSE无需重训，只需更少视角的稀疏SfM+图。
- 方案选项（用户暂未拍板）：①重跑更少帧(6–8/相机,保3相机)COLMAP ②视角数sweep{3,6,12,20} ③加"仅用K训练帧"代理(弱) ④换更少/更宽相机重生成。**等用户指令**。

## 7.9 让 GS-Net 在真实数据起效果（regime 分析，2026-06）
- **诊断**：学习式init/densify先验只在"稀疏SfM初始化不足"区间起效(少视角/宽基线/覆盖空洞/弱纹理)。**Waymo 前视3相机=密集重叠+高基线+无空洞=GS-Net反区间**。CARLA环视有效正因有空洞。nuScenes 6环视(重叠~10%)是天然主场。文献:LoopSparseGS/EAP-GS/DUSt3R稠密init/SDI-GS;DrivingForward/MapGS(环视+传感器配置gap)。
- **关键发现**：Waymo"稀疏视角"≠点稀疏——10帧SfM(18–22k点)≈60帧(19–38k点)。痛点是**少视角过参数化**(GS-Net加5×高斯反而有害),非点密度。**稀疏Waymo: baseline 21.31→GS-Net 20.64(−0.67)**。
- **路径(用户选了 1+2)**：
  - **路径1(主推,用户能抽5相机)**：用 Waymo **5相机**(front/FL/FR/SL/SR)制造宽基线+覆盖空洞。(1a)5相机SSE;(1b)**前视重建→侧视合成 的真实跨传感器**(最贴论文主旨,CARLA CSE真实版)。流程相机无关、复用现有(make_cam_split按相机名分组);1b需相机级留出划分(test=侧视相机)+用户给相机命名。
  - **路径2(便宜但低预期)**：`--input_subsample`(已加)训"稀疏输入→稠密目标";但Waymo非点稀疏,大概率无效。**已验证:input_subsample版20.05 < 全密度版20.64 < 基线21.31 → 路径2确认死路**(少视角痛点=过参数化,非点稀疏)。**只剩路径1。**
- **诚实风险**：GS-Net只吃SfM点(无图像特征),极稀疏下不如图像条件前馈法(pixelSplat/MVSplat/DUSt3R)。备选:加轻量图像/LiDAR特征(Waymo/CARLA都有LiDAR)。

## 7.10 路径1：5相机 Waymo（数据已就绪 2026-06）
- 数据：`/mnt/zihanw/EmerNeRF/data/waymo/colmap_input_5cam/<seg>/colmap/dense/{fused.ply,images/cam0..4,sparse/0}`，5相机×20帧，10 segment，SfM+去畸变PINHOLE+MVS 齐全。
- 相机约定(待用户确认)：cam0=FRONT,1=FL,2=FR,3=SL,4=SR；前视组=cam0/1/2，侧视组=cam3/4。
- 流程：waymo_gdense(runs/waymo5_gdense)→waymo_corr(CORR/waymo5,排除2测试)→train_gsnet(geom:tanh:0.1:10:1→runs/waymo5_gsnet)→ **1a 5相机SSE**(runs/waymo5_sse,帧留出) + **1b 跨传感器**(runs/waymo5_cse,`--target_cams cam3 cam4`,前视训练→侧视测试)。
- 代码新增：`make_cam_split.write_camera_split`(整相机留出) + `waymo_sse --target_cams`。
- 测试场景:10275/15868。预期:1b侧视有覆盖空洞→GS-Net应起效(验证"网络没问题、之前regime错")。

## 8. 待办
1. **多种子结果** → 定 SSE/CSE 的 mean±std；据此最终决定 CSE/真实数据怎么写。
2. 若想拉高 CSE/整体：可试**集合级匹配损失**（预测T↔GT K最近邻/匈牙利匹配，替代任意t-to-t配对）——尚未实现。
3. 定性图（R2.5/R3图4）：渲染测试视角+局部放大、修GT/3DGS列。
4. 用最终数字填 rebuttal（用户负责写文字，我负责实验/数字）。
