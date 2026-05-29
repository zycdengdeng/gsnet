# GS-Net 项目状态与交接文档 (PROJECT STATE)

> **用途**：记忆压缩后的"续命"文档。读完本文件即可完整理解任务、数据、代码、流程、进度。
> **最后更新**：2026-05-29
> **开发分支**：`claude/festive-feynman-80Vw3`（所有改动推到这里；用户在服务器 `git pull`）

---

## 0. 协作方式（重要）
- 我（Claude）**无法访问用户服务器**。代码我 push 到 GitHub 分支，用户在服务器 `git pull` 后运行，再把输出粘贴回来。
- 数据都在服务器，不在仓库里（仓库只放代码）。`runs/`、`CORR/` 等产物未被 git 跟踪。
- 用户机器：8× A100-80GB（卡 0–7，常有其他人占用部分卡，按 `nvidia-smi` 安排）。
- 时间/实验结果一定要记录（用户强调多次）：每个脚本都写 `*_times.json` / `*_results.{json,md}`。

## 0.5 运行中的实验 / tmux 位置（用户维护，尽力同步）
| tmux | 内容 | 状态 |
|---|---|---|
| zyc1 | 最初的 concat 训练（旧代码，已被 encoder 消融取代）| 已弃用/可 kill |
| zyc3 | 先后跑过 CARLA 基线 SSE、Waymo 诊断 | 已完成 |
| zyc4 | CSE 110 基线自检（`run_cse --ids 110 --skip_gsnet`）| 进行中 |
| (未知名) | encoder 消融 `run_encoder_ablation`（GPU 3-6）| 进行中 |
> 备注：GPU 占用以 `nvidia-smi` 为准；CSE/Waymo 等重活按当时空闲卡填 `--gpus`。

## 1. 任务背景
- 论文《GS-Net: Heterogeneous Vehicle Data Reuse via Generalizable Plug-and-Play 3DGS Module》**代码丢失，需按论文重建**。仓库初始是官方 3DGS（Inria）。
- 论文两大贡献：① CARLA-NVS 数据集（12 相机，跨传感器视图合成 benchmark）；② **GS-Net**：从稀疏 SfM 点云一次前向预测稠密高斯，作为 3DGS 的即插即用初始化。
- 当前处于 **rebuttal 阶段**，需补：(A) encoder 设计消融；(B) 真实数据集（Waymo）实验。

## 2. 方法核心（GS-Net）
**离线数据引擎（造伪 GT）**：COLMAP SfM → 稀疏点 `P_sfm`；MVS → 稠密点；以稠密点初始化跑 3DGS 30k → 收敛稠密高斯 `G_dense`；对每个稀疏点用 KD-tree 取 K=5 个最近 `G_dense` 高斯做伪 GT。

**在线 GS-Net（单次前向）**：
- 输入：稀疏点 6 维（xyz+rgb）。
- (a) 几何感知编码：逐点共享 MLP → 邻域聚合（M=3）→ `F_n`。
- (b) 多头扩张：每点 → T=5 个高斯，**增量预测**：`μ̂=μ_in+Tanh(Δμ)`、`Ĉ=C_in+σ(ΔC)`、`R̂=quat2rot`、`Ŝ=σ(s)`、`α̂=Tanh(·)`。
- (c) 损失：几何 `‖Δμ̂-Δμ^gt‖²+‖R̂-R^gt‖²+‖Ŝ-S^gt‖²` + 外观 `‖ΔĈ-ΔC^gt‖²+(max(α̂,0)-α^gt)²`，按 N·T 平均。
- 训练：200 epoch，batch 512，Adam 1e-3，T=5，M=3。
- 推理：稀疏点 → GS-Net → 导出标准 3DGS `.ply` → 作为初始化跑 3DGS 优化。

**我做的关键工程决策（论文外、为让它真能 work）**：
1. **逐序列归一化**：COLMAP 坐标尺度很大（CARLA ~数百单位），否则 sigmoid scale∈(0,1) 无法表示、Tanh 偏移无意义。归一化中心=median、尺度=稀疏点到中心距离的 p95（**仅用稀疏点算，推理可复现**）；G_dense 目标同样变换；推理后反归一化再导出（让 3DGS 在原 COLMAP 系下优化、与相机匹配）。
2. **G_dense 杂点过滤**：去掉离稀疏质心 > `radius_margin(1.5)×p99` 的远处漂浮点 + `opacity<0.005` 的近透明点（可选统计离群 `--sor_k`）。
3. **GT scale 裁剪**：归一化后个别超大高斯 scale>1 会超出 sigmoid 范围，clip 到 (0,0.999) 并在 >1% 时警告。
4. **损失权重可调**：`--w_pos/--w_rot/...` 默认全 1.0（论文一致），备用于再平衡。
5. **3DGS 集成**：`--gsnet_init <ply>`（`GaussianModel.create_from_ply`，复用 `load_ply` 再补 exposure/max_radii2D 等训练态）；`--init_pcd <ply>`（用任意点云如 MVS fused.ply 做 `create_from_pcd` 初始化，用于造 G_dense）。
6. **测试集划分**：`dataset_readers` 优先读 `sparse/0/test.txt`（否则回退 LLFF hold）。

## 3. 代码结构（全部在 `gsnet/` 包内 + 少量对主仓库的改动）
| 文件 | 作用 |
|---|---|
| `gsnet/model.py` | GS-Net 网络；encoder 可插拔（`cfg.encoder_type`）；增量预测+各激活 |
| `gsnet/encoders.py` | 5 个 encoder 变体：`mlp_only`/`concat`(Ours)/`edgeconv`/`attention`/`geom` |
| `gsnet/losses.py` | 解耦几何/外观损失 + 可调权重 |
| `gsnet/common.py` | 点云/高斯读取（.ply/.bin/.txt）、G_dense 过滤、归一化 |
| `gsnet/build_correspondences.py` | CARLA：KD-tree 伪 GT 生成（`--batch` 自动发现 `output_*_dense`）；含 `build_for_sequence` |
| `gsnet/dataset.py` | 跨场景拼接 npz |
| `gsnet/train_gsnet.py` | 训练（`--encoder_type`、`--w_*`、计时） |
| `gsnet/infer.py` | 单次前向 → 反归一化 → 导出 3DGS ply |
| `gsnet/io.py` | 写标准 3DGS ply（f_dc/opacity=inv_sigmoid/scale=log/rot=quat，丢弃负 opacity） |
| `gsnet/make_sse_split.py` | CARLA SSE：每相机 10 帧抽第 4、9 帧 → test.txt |
| `gsnet/run_sse.py` | CARLA SSE 驱动：基线 vs GS-Net+3DGS，多卡，merge/resume |
| `gsnet/run_encoder_ablation.py` | encoder 消融：训全部变体→全量 SSE→出表 |
| `gsnet/make_cam_split.py` | 通用：读 images.bin 按相机（按名字目录前缀分组）抽内部均匀 N 帧 → test.txt |
| `gsnet/waymo.py` | Waymo 路径助手（dense_dir/fused_ply/sparse_points/gdense_path/discover_scenes） |
| `gsnet/waymo_gdense.py` | Waymo：MVS 初始化跑 3DGS 造全 10 场景 G_dense，多卡，可续跑 |
| `gsnet/waymo_diag.py` | Waymo：单场景 sfm_3dgs vs mvs_3dgs（带划分）对照 |
| `gsnet/waymo_corr.py` | Waymo：8 训练场景建对应（排除 2 测试场景） |
| `gsnet/waymo_sse.py` | Waymo：2 测试场景 SSE（每相机抽 4 帧），基线 vs GS-Net+3DGS |
| `gsnet/inspect_ply.py` | 诊断：ply 字段/坐标尺度/稀疏↔G_dense 对齐 |

**对主仓库的改动**：`arguments/__init__.py`(+`gsnet_init`,`init_pcd`)、`scene/__init__.py`(分支用 gsnet_init/init_pcd)、`scene/gaussian_model.py`(+`create_from_ply`)、`scene/dataset_readers.py`(优先 test.txt)。

## 4. 数据布局（服务器，绝对路径）
### CARLA-NVS（`/mnt/zihanw/carla/`）
- 稀疏 SfM（GS-Net 输入）：`sparse_point/S0<scene>/<id>_sparse.ply`，`id=100*scene+seq`，scene 1–5，seq 1–10。
- 稠密 MVS + 3DGS 工作区：`input_output/<id>_dense/`（注意：其 `sparse/0/points3D.ply` 是**稠密 MVS**，不是稀疏！）
- **G_dense**（学习目标）：`input_output/output_<id>_dense/point_cloud/iteration_30000/point_cloud.ply`
- 测试序列：`input_output/<id>_base/`（110/210/310/410/510），图像扁平 `1-60.png`；相机映射：1-10→cam1, 11-20→cam3, …, 51-60→cam11（6 个奇数相机×10 帧）。
- GS-Net 训练序列：101-109…501-509（实际 **43** 个可用；102、109 缺 G_dense 被跳过）。
- SSE 划分：每相机 10 帧抽第 4、9 帧 → 测试集 `{4,9,14,…,59}.png`（12 测试/48 训练）。

### Waymo（`/mnt/zihanw/EmerNeRF/data/waymo/colmap_input/`）
- 10 个 `segment-*`，每个前视 3 相机×20 帧=60 张。
- 每场景 `colmap/dense/`：`fused.ply`(MVS)、`images/cam{0,1,2}/0NN.jpg`、`sparse/0/{cameras,images,points3D}.bin`。
- 稀疏 SfM = `colmap/dense/sparse/0/points3D.bin`（COLMAP 图像名形如 `cam0/000.jpg`）。
- **G_dense** → `runs/waymo_gdense/<seg>/point_cloud/iteration_30000/point_cloud.ply`。
- **测试场景（2 个）**：`segment-10275144660749673822_5755_561_5775_561_with_camera_labels`、`segment-15868625208244306149_4340_000_4360_000_with_camera_labels`。其余 8 个训练。
- 诊断场景：`segment-3425716115468765803_977_756_997_756_with_camera_labels`。
- SSE 划分：每相机 20 帧抽内部均匀 4 帧（`[4,8,11,15]`）。

## 5. 完整流程命令
### CARLA
```bash
# 1 建对应(CPU)  2 训练  3 SSE
python -m gsnet.build_correspondences --batch --io_dir /mnt/zihanw/carla/input_output \
  --sparse_root /mnt/zihanw/carla/sparse_point --out_dir CORR/train --workers 8
CUDA_VISIBLE_DEVICES=X python -m gsnet.train_gsnet --corr_dir CORR/train --out_dir runs/gsnet
python -m gsnet.run_sse --io_dir ... --sparse_root ... --ckpt <concat ckpt> --out_dir runs/sse --gpus ...
```
### encoder 消融（rebuttal A）
```bash
python -m gsnet.run_encoder_ablation --corr_dir CORR/train --io_dir ... --sparse_root ... \
  --out_dir runs/encoder_ablation --gpus 2 3 4 5 6 7
# 产出 runs/encoder_ablation/encoder_ablation.md；其 concat=Ours 也用于主表
```
### Waymo（rebuttal B）
```bash
python -m gsnet.waymo_gdense --root /mnt/zihanw/EmerNeRF/data/waymo/colmap_input --out_dir runs/waymo_gdense --gpus 0 1 2 7
python -m gsnet.waymo_corr  --root ... --gdense_dir runs/waymo_gdense --out_dir CORR/waymo_train --workers 8 \
  --test_scenes <seg_10275...> <seg_15868...>
CUDA_VISIBLE_DEVICES=0 python -m gsnet.train_gsnet --corr_dir CORR/waymo_train --out_dir runs/waymo_gsnet
python -m gsnet.waymo_sse --root ... --test_scenes <seg_10275...> <seg_15868...> \
  --ckpt runs/waymo_gsnet/gsnet_latest.pt --out_dir runs/waymo_sse --gpus 0 1 2 7
```

## 6. 进度
### ✅ 已完成
- GS-Net 全部核心代码 + CARLA/Waymo 全套驱动 + encoder 消融。
- CARLA 建对应：43 序列 → `CORR/train`（scale 56–181，无 scale-clip 警告，对齐良好）。
- CARLA 基线 SSE：用户已跑完（`runs/sse` 基线行）。
- Waymo 诊断：sfm_3dgs **28.18** / mvs_3dgs **29.33**（+1.15dB），证明 60 张/2s 够用、MVS 增益明显。
- **CARLA 基线 SSE 已跑完**（`runs/sse/sse_results.md`）：110=26.08, 210=24.53, 310=26.60, 410=22.59, 510=23.53，**Avg 24.67**（SSIM 0.919, LPIPS 0.168）。待合并 Ours。
- **Waymo G_dense 已生成**（全 10 场景，`runs/waymo_gdense/`，zyc3 跑完）。
- **CARLA CSE 位姿已推算**（`runs/cse_poses/`，自校验 rot<0.86°/center<0.67%）；CSE 场景已构建（`runs/cse_scenes/`）；110 CSE 基线自检进行中（zyc4）。

### 🔄 进行中
- **encoder 消融**（`run_encoder_ablation`）在跑（GPU 3-6 是它）。训 5 变体→全量 SSE→出表。

### ⬜ 待办
- encoder 消融出表后：用其 `concat`(Ours) ckpt 跑 `run_sse --skip_baseline` 合并出 CARLA 主表（基线 vs GS-Net）。
- Waymo：① 全 10 场景 G_dense → ② 8 训练建对应 → ③ 训 Waymo GS-Net → ④ 2 测试 SSE。
- Waymo **CSE**（外推/差值视角可视化）——外推角度待与用户讨论后再做（需新驱动）。
- CARLA **CSE**：🔄 位姿推算已完成（`cse_poses.py`，自校验残差 rot 0.26–0.86°、center 0.22–0.67%，仅 SfM 噪声量级）；`make_cse_scene.py`(拼 60 奇训练+60 偶测试 COLMAP 模型) 与 `run_cse.py`(基线 vs GS-Net+3DGS) 已实现，待跑。CSE 数据：偶数图像在 `/mnt/zihanw/carla/paired_120/<id>_dense/cam{02..12}/`；rig=`gsnet/carla_rig.json`(半径0.75十二边形)。流程：`cse_poses`→`make_cse_scene`→`run_cse`(--ids 110 210 310 410 510)。
- 用真实数字补全 rebuttal 文本（encoder 消融段、Waymo 段）。

## 7. 关键发现 / 注意事项
- **CARLA GS-Net 训练 loss 形态**（正常、符合论文）：`loss_mu`极小（稠密高斯几乎贴稀疏点，偏移≈0）；`loss_scale`≈0（高斯本就小，归一化后易拟合）；`loss_rgb/opacity`~0.13 在降；**`loss_rot` 卡在~1.5 不降**——近各向同性高斯旋转病态、目标即噪声，网络收敛到均值；与论文 Table IV「R 贡献最小（+0.28dB）」一致。如位置"摊开"不足可试 `--w_rot 0.1`。
- **⚠️ checkpoint 兼容性**：encoder 重构把子模块从 `point_encoder/context_encoder` 改名为 `encoder.*`。**旧 checkpoint 新代码加载不了**。早于该重构训的 `runs/gsnet` 已弃用；以 encoder 消融重训的 `concat` 为准。
- `inspect_ply`：CARLA 坐标 ~数百单位；稀疏↔G_dense 对齐好（NN p50≈0.19）；G_dense 有远处漂浮点（已过滤）；SH degree 3。
- Waymo 是 metric 尺度，量级与 CARLA 不同；归一化/过滤自适应，应无碍——但**首次 Waymo 建对应要看 scale 与是否有 scale-clip 警告**。

## 8. 给审稿人的回应（草稿，待数字补全）
- **Encoder 简单**：强调 GS-Net 是即插即用初始化模块、推理速度关键（50× 加速定位）；补 encoder 消融（a–e 五种），报告 PSNR/SSIM/LPIPS + 参数量 + 推理时间。预期结论：邻域聚合必要（a 最差）；更强 encoder（c/d/e）增益边际但参数/延迟上升 → 支撑轻量设计；残差由 per-scene 优化吸收。
- **真实数据**：Waymo 10 场景，先 SSE 量化（基线 vs GS-Net+3DGS），后续 CSE 做外推可视化。
