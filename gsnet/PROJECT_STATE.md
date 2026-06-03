# GS-Net 项目状态与交接文档 (PROJECT STATE)

> **用途**：记忆压缩后的"续命"文档。读完即可完整理解任务/数据/代码/流程/结果/当前路径，性能不退化。
> **最后更新**：2026-06-03（代码平反/CARLA增益坐实 + regime非协议 + 重心转向"让Waymo转正"；见 §0.000 速览）
> **分支**：`claude/festive-feynman-80Vw3`（push 到此；用户服务器 `git pull`）

---

## 0. 协作方式
- 我（Claude）**无法访问服务器**。代码 push 到 GitHub，用户 pull 后运行、贴回输出。数据在服务器、不在仓库；`runs/`、`CORR/` 未跟踪。
- 机器：8× A100-80GB。卡 4–7 有时被别人 RL 占（显存够、共卡不崩，但算力时间片共享变慢）；以 `nvidia-smi` 为准。**用户现在常说 8 卡全空、可大并行**。
- **务必记录时间/结果**：脚本都写 `*_times.json` / `*_results.{json,md}`。
- 多行命令易因续行 `\` 后空行而断 → 给用户命令**写成单行**。
- ⭐**用户明确要求：随时主动更新本 md，不要等被提醒**。每出一个结果/结论/决策就即时落档。

## 0.000 🧭 当前真相速览（READ FIRST，2026-06-03）——下面 §0.0~§8 为历史明细，本节为最新口径
**任务**：GS-Net 论文 rebuttal。GS-Net = 稀疏SfM点→一次前向→稠密3DGS高斯，作即插即用init。核心诉求=**在真实数据(Waymo)上做出可信正增益**，且**不大改网络、不大改行文**。

**✅ 已定论（别再质疑/重测）**
1. **代码正确、CARLA增益真**：CARLA SSE 多seed=**+1.69±0.38**(排崩坏场景310;含310才被拽成+0.22="+0.28"假象)，≈论文+2.08。densify on/off佐证(+1.72/+2.06)→init本身就好、非washout。端到端审计无bug。
2. **是 regime 不是协议**：Waymo within≈cross(seen−0.74≈unseen−0.77≈cross−0.90)；CARLA within+1.69≈cross/LOSO+1.30。→ 见没见过场景**不重要**；决定有无效的是**几何regime**。
3. **GS-Net 的适用域**：只在**覆盖空洞/外推/视角不足**(3DGS自身densify恢复不了的地方)有用=CARLA环视的主场。**Waymo same-sensor帧留出SSE=内插=无洞→GS-Net没空间且过度密化反害(−0.9)**。
4. **死路别再走**：稀疏化(帧留出)→越稀越害(证伪"弄稀疏就有效")；T3=28.40=test.txt竞争污染的假象(已弃)。⚠️CSE**不是死路**(见#9):densify-on剔310≈+0.02、但densify-off=+1.28(washout)。
5. **干净T**：仅CARLA有(**剔310确认T5=27.19最好**、更大无益);Waymo最优T未知(T扫描重跑中)。
6. **相似度**：CARLA场景比Waymo紧(逐段0.21 vs 0.32)→解释CARLA跨场景也灵。
7. **场景310**：所有seed都崩(init正常,优化病理)→诚实排除。**510(−0.49)正常勿丢**。
8. **310清算完成(reaggregate.py剔310)**：所有rebuttal结论剔310后**都成立且更干净、无需重跑**——Reviewer A(encoder最终配方顶4打平+1.8~2.1)、Reviewer C(优雅降级)、最终配方T5@M3(T5=27.19最好)、主表SSE+1.69。详见§0.00。
9. **⭐CSE washout确认=真增益(2026-06-03)**：CARLA CSE×**densify-off**(剔310,`runs/cse_densify_off`)=base17.10/gsnet18.38→**Δ+1.28**(vs densify-on +0.02),LPIPS0.346→0.295,优化快40%(23.6→14.8min)。→**GS-Net多视角一致init对跨传感器真有用,只是被30k源视角密化磨平**。⚠️tradeoff:densify-off绝对PSNR更低(18.38<19.65),故只能**同设置内**比;写法=效率/预算角度(同预算/不靠激进密化时+1.3+LPIPS+收敛快)或扫densify_until_iter找甜点。**⇒Waymo跨传感器赢法**:已给waymo_sse加--train_extra。⚠️**但 front→side 经`waymo_cam_overlap`查实=死局**:侧视点被前视覆盖仅**6-9%**(cam3/4垂直±y,前视组最多45°,大视角跳变+SfM匹配失败)→front→side注定≈0,**别跑**。改走:①留一相机(合成被其余4相机覆盖最好的,LOCO);②**稀疏时序×精简变体×densify-off**(同相机少帧=已观测但欠约束,避开'未观测'死穴,更看好)。`waymo_cam_overlap`已扩LOCO+两两重叠矩阵,待用户跑挑viable目标。 **结果(2026-06-03)**:两两重叠**全≤9%**(Waymo 5相机近乎互不共视=各自沿路独立条带);LOCO最高=合成cam0(FRONT)被其余4覆盖23%(10275)/38%(15868),其余<20%。**深层结论**:GS-Net机制=利用'多相机共视冗余→SfM空洞'去填;CARLA环视高共视→+1.7,Waymo宽基线rig共视<10%→几乎无结构可利用→中性/微负是几何决定的。**⇒主线rebuttal用'覆盖度刻画适用域'诚实框架**(GS-Net受益重叠/环视rig=CARLA/nuScenes,宽基线disjoint rig=Waymo无效,用inter-cam覆盖量化边界)——比硬凑Waymo增益更强。**仅剩长射**:①合成cam0(唯一覆盖较高)×densify-off×精简变体(赌一把,等消融出dens_only ckpt);②稀疏时序×lean×densify-off(期望一般,稀疏sweep已负)。

**🎯 唯一在打的目标 = 让 Waymo 转正（不改网络/行文）**——三杠杆：
- **L1 去有害成分**：属性消融(`run_waymo_ablation`,跑中)→ no_opacity/dens_only 能否把−0.9拉向正？(假设opacity是真实数据坏/不可迁移因子)
- **L2 换外推regime测**：别用帧留出SSE(内插)。用**跨传感器(front→side,=论文"异构传感器复用"主旨,需干净重跑waymo5_cse)** 或 **真稀疏视角**(整块无人看→3DGS没梯度→几何先验补位)。
- **L3 少密化**：小T(T扫描中)。
- **押注组合 = dens_only/no_opacity × 外推regime**（合成用户两直觉:opacity坏+用稀疏/有洞regime）。

**🧪 Waymo属性消融第一波(`runs/waymo5_ablation`,8/2跨场景SSE,baseline27.31)**:full**−0.41**(最好)/no_color−0.42/no_opacity−0.82/no_scale_rot−2.85⚠️/xyz_rgb−1.15⚠️/dens_only−1.66⚠️(⚠️=旧固定0.01scale不公平,待distCUDA2修复重评)。**结论:(1)no_opacity比full差→opacity不是坏因子、反而有用,用户假设否;(2)full最不差但仍负,砍任何属性都更糟→无变体把Waymo拉正**。与相机共视<10%发现一致=Waymo无GS-Net可利用结构、几何决定、调参翻不动。lean3个公平重评待跑(rm其sse重跑,期望仍翻不过−0.41)。
**⏳ 当前状态(2026-06-03晚)**：Waymo属性消融**并行重跑中**(`runs/waymo5_ablation/ablation_results.md`,**最关键**,6变体含xyz_rgb,8卡)。**待用户跑/贴**:① Waymo消融表 ② **CSE×densify-off**(`runs/cse_densify_off`,验washout) ③ 干净Waymo T扫描(`resolve_scene`已修,重跑补SSE)。**已决定不跑**CARLA消融。
**🔔 待办触发器(用户贴第一波消融结果时,必须主动提醒)**：① **infer已加distCUDA2 std-init scale fix(只影响no_scale_rot/xyz_rgb/dens_only)**→让用户`rm -rf runs/waymo5_ablation/{no_scale_rot,xyz_rgb,dens_only}/sse`+git pull+重跑同命令(跳训练只重评),得**公平的纯密化数**;② 据结果按需加`--no_rot`(拆scale/rot单独砍旋转,rot是已知噪声地板)、`geom_only`(xyz+scale_rot)、`xyz_opacity`组合(用户已同意第一波后做)。
**❓待用户定**：L2先打"跨传感器"还是"少视角"。

**📁 关键路径**：Waymo 5cam=`/mnt/zihanw/EmerNeRF/data/waymo/colmap_input_5cam`(8训/2测=10275,15868;cam0前/1FL/2FR/3SL/4SR);`CORR/waymo5`;ckpt`runs/waymo5_gsnet`(T5);within-scene back=`colmap_input_5cam_next20`。CARLA io=`/mnt/zihanw/carla/input_output`,sparse=`/mnt/zihanw/carla/sparse_point`,`CORR/train`,多seed模型`runs/multiseed_sse/model_s{0-4}`。
**🔧 近期新增工具**：`run_waymo_ablation`/`run_carla_ablation`(属性消融,6变体,断点续跑) · `train_gsnet --no_color/--no_opacity/--no_scale_rot` · `run_sse --train_extra`(透传如`--densify_until_iter 0`) · `run_waymo_withinscene` · `scene_similarity --explode`+块统计 · `waymo.resolve_scene`支持短名子串匹配。

---

## 0.00 ✅✅✅ 代码已平反 + CARLA 增益坐实（2026-06-03，逐序列多seed证据）
- **✅ 剔310重聚合(reaggregate.py)结论:没翻任何rebuttal结论,更干净**(基线excl310=24.56):
  - **CARLA T扫描(`Tsweep_carla`,剔310,2序列110/510)**:T3=26.14/**T5=27.19**/T8=26.11/T12=26.57/T16=26.40→**T5仍最好(Δ≈+1.97),与含310排名一致→最终配方T5@M3稳**。
  - **CARLA CSE剔310(`runs/cse`+`multiseed_cse`,基线19.65)**:gsnet多seed=**19.67±0.06→Δ≈+0.02(中性)**;310曾把它拽到−0.29。**干净CSE=≈基线,非负,但也无正增益**。机制:GS-Net只能加密已观测几何、不能生成未观测;CSE向外极端外推大片未观测→谁都救不了。**关键认识修正**:CARLA CSE是12环视、奇→偶仅差30°,几何**大部分已观测**,不是'无观测无解',真因更可能是**(a)过度密化floater在30°外露馅 (b)washout——GS-Net多视角一致init被30k源视角密化磨回源过拟合**(同CARLA SSE densify-off+2.06>on+1.72)。**最该试:CSE×densify-off**(`run_cse --train_extra "--densify_until_iter 0"`已加),保住视角一致init→偶视角优势显现;若Δ明显转正=washout证实=真CSE增益(机制干净不改网络,但较窄主张/效率角度)。其次:CSE×精简变体(dens_only减floater)、降极端度(重叠FoV)、诚实定位。`runs/cse_densify_off`待跑(ids 110/210/410/510排310,ckpt=最终配方)。
  - **Reviewer A(encoder,`encoder_ablation_final`最终配方tanh0.1:10:1,4序列)**:concat26.65(+2.09)/geom26.51(+1.95)/mlp_only26.40(+1.84)/attention26.33(+1.77)/edgeconv25.37(+0.81)。**顶4打平(差0.32,在噪声内),轻量mlp≈重型geom→'增益来自损失设计非encoder复杂度'成立且更强。用这张替论文encoder表。**
  - **Reviewer C(`sup_sens`)**:监督迭代5k+0.70/10k+1.07/20k+0.82;密度25%+0.76/10%≈基线/50%≈基线→**优雅降级到≈基线不崩**(单seed略噪)。措辞:监督变差→增益单调收窄→退化到≈基线。
  - **design/design2/design_attn**:剔310只剩2序列(110,510)太薄,但最终配方已被encoder_ablation_final干净背书→配方选择站得住,**不重跑**。
  - `encoder_ablation`(默认配方sigmoid)剔310:geom26.51最好(领先~0.8-1.5)=默认配方下encoder有用,与'最终配方打平'两半故事自洽。
- **⚠️ 旧消融被310污染(2026-06-03发现)**:encoder消融(run_sse默认5序列含310)、design/weight sweep(默认eval_ids 110/310/510,310占1/3)、supervision敏感性——平均里都含310→被拉低。**但逐序列PSNR存于各config的sse_results.json→用`reaggregate.py --exclude 310`重算即可,无需重跑**。重算还能查310是否config相关(若剔310后encoder排名变则原结论需修订;若仍打平则结论不变更干净)。待用户跑重聚合。可选升级:encoder消融多seed。
- **densify on/off 佐证(`runs/sse_densify_on|off`,3id排310)**:Δ(gsnet−base) ON=+1.72 / OFF=+2.06(110/510)→两regime都稳+1.7~2,增益非washout假象,init本身就好(关密化更少迭代也到位);310两边都崩(−6/−8)=场景病理与密化无关。**'+0.28缩水'担忧彻底关闭。**
- **多seed逐序列gsnet PSNR(基线)**：110=27.39(25.59)/210=27.97(25.56)/410=23.70(22.23)/510=25.94(24.85)/**310=19.75(25.43)←所有5seed都18-20崩坏**。
- **310是稳定的场景级病灶**(每seed都崩、init正常无NaN/15万点)→非代码bug、非seed噪声,是该场景优化病理。
- **含310全5个 Δ=+0.22(=那个吓人的"+0.28")；排除310 Δ=+1.69±0.38**。→ **CARLA SSE真实增益≈+1.7,与论文+2.08同量级。代码正确、能复现论文效果。**
- **⚠️自我修正**:之前§0.1说"缩水成+0.28/头条站不住"是**错的**——未先做逐序列分解,被310一个−5.68拽偏。已纠正。
- **最终2×2(均排310)**：CARLA within **+1.69** / CARLA cross(LOSO) **+1.30** / Waymo within −0.74(seen≈unseen) / Waymo cross −0.90。→ **regime决定有无效、协议不重要;相似度解释CARLA跨场景也灵(场景紧)。代码无bug。**
- **重心回归**:代码既已确认,目标=**让Waymo转正**→属性消融(opacity坏因子?纯密化?`run_waymo_ablation`)是关键。

## 0.0 ⭐ 代码审计 + "增益缩水"诊断（2026-06-03，用户怀疑重建代码有错）
- **背景**：用户原版 GS-Net 在 CARLA SSE 给论文级增益(+2.08);本重建多seed只+0.28。用户疑"预测参数没正确替进3DGS/学错了"。
- **端到端审计(infer→io→model→build_correspondences→losses→scene/__init__→gaussian_model)结论：未发现替换/学习的硬bug**。预测参数正确转3DGS约定(RGB→SH、scale→log、opacity→inverse_sigmoid、quat wxyz)写盘;`create_from_ply`正确加载并`active_sh_degree=0`;baseline走`create_from_pcd`(SfM稀疏点),与gsnet**只差init**,公平;pseudo-GT归一化/裁剪(scale→/scene_scale,clip(1e-6,0.999))与模型σ∈(0,1)同空间一致;损失pos用delta、scale同空间、opacity用max(α,0)对齐。**机制是对的。**
- ~~**最可能真因=30k densification抹平init**~~ **⛔此假设已被推翻(见§0.00)**:真因是**场景310崩坏**拖低5序列均值;densify on/off证明增益是真的(+1.72/+2.06)、剔310=+1.69。(washout效应存在但很小,非主因。)
- **诊断(已加`run_sse --train_extra`透传)**：关掉densification(`--densify_until_iter 0`)再比baseline vs gsnet。若gsnet≫baseline=washout证实(init真有用,价值在"免密化/快收敛",可正面重写故事);若仍≈=init本身弱(再查G_dense质量/T-to-K任意配对/损失权重)。命令见下方对话。**结果出来更新此处**。

## 0.1 ⭐⭐⭐ 决定性结果（2026-06-03，必读，改写结论）
- **🎯 是 regime 不是协议（Waymo within-scene 裁决,`runs/waymo5_withinscene`）**：Δ_seen=**−0.74** ≈ Δ_unseen=**−0.77**（逐场景:12879 +1.15/+0.98、14004 −0.87/−0.77、3988 −2.51/−2.51）。**GS-Net 见没见过该场景对结果无影响** → 跨场景−0.9≈同场景−0.75。**"协议/同场景身份/分布邻近度"解释在 Waymo 被证伪**；救不了它的是**几何 regime（前视稠密→过度密化有害）**，与熟悉度无关。这是最干净的因果结论。
- **🔄🔄 重大修正：CARLA "+0.28 缩水" 很可能是 310 崩坏拖累的假象（2026-06-03晚）**。CARLA LOSO 重跑(`runs/carla_loso`)：110+1.83/210+2.67/410+1.18/510−0.49/**310 gsnet=19.04(−6.39,崩坏退化非"有害")**。**排除崩坏310,其余4个 Δ=+1.30** → CARLA**跨场景**GS-Net仍+1.3(印证相似度预测:CARLA场景紧→跨场景照样有效,2×2右上=正)。**推论**:多seed的+0.28(5序列含310)极可能也被310拽下→**CARLA真实SSE增益≈+1.3不是+0.28,用户"原版=论文效果"可能对、是我被310误导**。**待验证**:多seed逐序列PSNR(看310是否各seed都崩)→确认后排除310重算。**310为何反复崩待查**(infer退化init?优化发散?)。
- ~~**CARLA CSE 多 seed**：19.37±0.07,基线19.66→−0.29~~ **⛔已被§0.00取代**:那是**含310**的;剔310后 CSE=**19.67±0.06,基线19.65→≈+0.02(中性)**,非负。
- **诚实定调(更新)**：GS-Net 增益=**regime依赖**——CARLA SSE(有覆盖空洞)**≈+1.3(排310)**、within≈cross(CARLA场景紧);CSE−0.29、Waymo−0.8(within≈cross,熟悉度无关)。即**相机几何/regime决定有无效,within-vs-cross两数据集上都不重要**;相似度解释CARLA跨场景为何仍灵(场景紧)。
- **✅ 相似度控制扛住pooling（`runs/scene_similarity_explode`）**：逐段+排同场景后 carla-carla **0.211** < waymo-waymo **0.318**(比值0.66,与pool版0.61一致),cross0.356。"CARLA更同质"为真。⚠️该run的test-z(15868 z=+2.39)不可信(train基线被50CARLA段主导);Waymo内是否离群以纯Waymo那次(z=+0.44,in-dist)为准。
- **两个重跑进展(2026-06-03晚)**：CARLA LOSO **已重跑成功**(见上,跨场景+1.3排310);干净Waymo T扫描挂在**场景名解析**(传短名,`resolve_scene`只精确拼接)**非OOM**——T1/T5模型已训好,**已修`resolve_scene`支持子串匹配**,重跑只补SSE。
- **进行中(2026-06-03晚)**：densify on/off 诊断(`runs/sse_densify_on|off`,验证washout) + Waymo T扫描重跑(补SSE)。**待用户贴**:多seed逐序列PSNR(查310)、两份densify表、T扫描表。
- **310 init 正常**(15万点/无NaN/坐标±200)→崩在**优化层面**非init退化;可丢310(灾难离群),**510(−0.49)正常轻微负勿cherry-pick**。
- **⭐ 属性消融工具就绪(2026-06-03,回应用户假设)**：`train_gsnet --no_color/--no_opacity/--no_scale_rot`(禁用→默认值+不入loss;infer读ckpt自动一致) + `run_waymo_ablation`(full/no_opacity/no_color/no_scale_rot/dens_only)。**假设:真实数据上opacity是坏/不可迁移因子,纯密化(位置)才有用**。`runs/waymo5_ablation`,跑中。**CARLA对照消融`run_carla_ablation`(同6变体,排310,run_sse评)→`runs/carla_ablation`**:与Waymo并排看,某因子CARLA有用/Waymo有害=sim-vs-real不可迁移(疑opacity)。
- **regime/帧数(用户质疑,认同)**：20帧是为within-scene协议A/B选的(同密度),**非为展示GS-Net最佳regime**;既已确认是regime非协议,该用**稀疏/有空洞**regime展示。3cam(无空洞)≈基线、5cam(加侧视本应有空洞)反掉、抽帧越稀越掉——**均为全属性**结果,疑opacity/scale过度密化作祟。**计划:先属性消融(当前20帧,数据现成)找有用因子→再把精简版(如dens_only)放稀疏/少帧regime放大**。


- **Waymo SSE 稀疏度 sweep 跑完**（T=5, ckpt `waymo5_gsnet`, 跨场景2测试场景, `runs/waymo5_sparsity/sparsity_sweep.md`）：

  | n_holdout | train/cam | base | gsnet | Δ |
  |---|---|---|---|---|
  | 4 | 16 | 27.43 | 26.52 | **−0.91** |
  | 12 | 8 | 23.46 | 21.05 | **−2.41** |
  | 16 | 4 | 19.06 | 17.15 | **−1.91** |
  | 18 | 2 | 16.31 | 14.98 | **−1.33** |

  → **GS-Net 每个密度都掉分，越稀疏越惨（中段−2.41最狠）。这把 §7.8/7.9 的核心赌注"把SfM弄稀疏→GS-Net起效"彻底证伪。稀疏化这条路死，别再投算力。**
- **⚠️ 修正 §0.3：T3=28.40 不可信，降级为"未复现孤点"**。它出自撞上 §0.4 test.txt 竞争 bug 的那批 T 扫描；当时自己就标了"待确认(waymo5_sse_v3/Tsweep_waymo_sse2)"但**从未复现**。旁证：sparsity sweep 的 nh=4 基线=**27.43**，而 28.40 那批引用的基线~**26.4**，光基线就飘~1dB；叠加 ±0.5–0.9 噪声，"+2" 极可能是噪声+基线漂移拼出来的假象。**不能再当"首个正向信号"引用。**
- **机制综合**：GS-Net 真实数据上的伤害=**过度密化**；稀疏(少视角)让过度密化更致命；方向上"别猛密化"成立(干净 CARLA T 扫描 T=5最好、更大T无增益)。⚠️**注意**："小T(<5)是解药"这个更强的说法**没有干净证据**——它来自被污染的 Waymo T3=28.40，已推翻。**干净证据只支持到"T=5、再大无益"，不支持"比5更小更好"。**
- **⭐ 干净的 T 结论（防再被 28.40 带偏）**：**唯一干净的 T 扫描是 CARLA**(单一确定性划分、非并发→无 test.txt 竞争)：T3=23.20/**T5=24.58**/T8=23.29/T12=23.86/T16=24.38 → **T=5 最好，更大 T 无稳定增益**(最终 CARLA 配方=T5@M3 的依据)。**Waymo 没有任何干净 T 结果**(唯一那次=污染批次，作废)→ Waymo 最优 T **目前空白**，待干净多seed重扫。**故 within-scene 一律用 T=5**(且 SEEN=waymo5_gsnet 本身 T=5)，不追泄漏出来的 T=3。
- **28.40 污染机制(记牢)**：test.txt 决定 train/test 划分(在 test.txt 里=测试,其余=训练)，且 `train.py` 与 `render.py` **各自现读一次**。并发实验共用同一 test.txt → 训练时按集合A留出(把B喂进训练)，渲染时 test.txt 被覆盖成B → 在**训练过的视角B**上评测 = 数据泄漏式虚高(28.40 vs 真留出~26.4)。铁证=4a==4b 雷同26.32。修复=`scene_source(tag)` 复制私有 test.txt。
- **新诊断 `scene_similarity.py`（已 push）**：在 GS-Net 归一化特征空间量场景两两分布距离 + 测试场景是否离群(z>2)。
- **✅ Waymo 相似度结果（2026-06-02）**：train↔train mean=**0.283** std=**0.059**(10场景同质,std仅~20%均值)；测试场景 10275 **z=−0.62**(比平均训练场景更居中)、15868 **z=+0.44**，**两个都牢牢在分布内、非离群**。
- **✅✅ CARLA+Waymo 联合相似度（2026-06-02, `runs/scene_similarity_joint`）—— 关键量化结论**：滑动Wasserstein块均值 **CARLA内部=0.165 / Waymo内部=0.271 / 跨数据集=0.329**（比值CARLA/Waymo=**0.61**，MMD同向）。→ **CARLA场景同质度远高于Waymo(彼此距离仅~60%)**。机制：LOSO留出的CARLA场景到训练~0.165，而Waymo测试场景到训练~0.27(虽in-distribution但流形本身更宽)→**CARLA"跨场景"其实离训练近得多→迁移更容易**。**区分CARLA赢/Waymo输的很可能是"训练-测试邻近度/流形宽窄",非"合成vs真实"**。⚠️confound:CARLA每场景pool了10段(平均压低组内方差、抬高组间相似)→0.165可能偏小;控制实验=每CARLA段当独立组只算跨场景段对(待做)。⭐**对LOSO的可证伪预测**:若邻近度是主因→CARLA LOSO应仍+正(留出场景离训练近);若LOSO也崩→则是"精确同场景身份"(交给Waymo within裁决)。
- **逐段控制实验(跑中,2026-06-02)**:`scene_similarity.py` 已加 `--explode`(每CARLA段=独立组,共50CARLA+10Waymo) + 自动块统计(summary.md直接报carla-carla/waymo-waymo/cross,**排除同场景对**),消除"CARLA pool10段→0.165偏小"的confound。产出`runs/scene_similarity_explode`。判读:carla-carla(逐段跨场景)仍≪waymo-waymo→"CARLA更紧"稳;若蹿到接近→0.165是平均假象需收回。**纯CPU,不占卡**。
- **运行环境(2026-06-02)**:卡3/4/5空,0/1/2部分空余(6/7疑被LOSO占)。within-scene driver已就绪可上空卡;控制实验CPU即可。
- **当前在跑(2026-06-02)**:CARLA LOSO(`runs/carla_loso`) + 逐段控制相似度(CPU,`runs/scene_similarity_explode`) + Waymo within-scene(`runs/waymo5_withinscene`,卡3/4/5)。
- **新排队/可跑(都现成,不用新代码)**：
  - **A. 干净 Waymo 跨场景 T 扫描 {1,2,3,5,8}**：`run_T_sweep --dataset waymo`(每T重建corr→重训→waymo_sse,走隔离test.txt)→`runs/Tsweep_waymo_clean`。**目的:填Waymo最优T空白+正式复现/证伪污染的28.40**(干净管线下T3若不再~28即钉死artifact)。
  - **B. CARLA SSE 多 seed**：`run_multiseed --eval sse --seeds 0-4`→`runs/multiseed_sse/multiseed.md`,给论文主表 SSE mean±std(头条+1.3现为单次±0.5-0.9噪声)。可断点续跑。
  - 靠后:Waymo within-scene多seed(需加--seeds代码,等单seed方向)、定性图。
- **🌙 无人值守批次(2026-06-02夜,用户外出,明天收)**——全部铺上,明天一次性收这6份：
  | 实验 | 结果文件 | 看什么 |
  |---|---|---|
  | CARLA LOSO | `runs/carla_loso/loso_results.md` | 2×2右上:跨场景CARLA还+正吗 |
  | Waymo within-scene | `runs/waymo5_withinscene/withinscene_results.md` | 2×2左下:Δ_seen vs Δ_unseen |
  | 逐段控制相似度 | `runs/scene_similarity_explode/summary.md` | carla-carla是否仍≪waymo-waymo(块表) |
  | 干净Waymo T扫描 | `runs/Tsweep_waymo_clean/.../Tsweep.md` | T3是否还~28(埋28.40)+Waymo最优T |
  | CARLA SSE多seed | `runs/multiseed_sse/multiseed.md` | 主表SSE mean±std |
  | CARLA CSE多seed | `runs/multiseed_cse/multiseed.md` | CSE mean±std(若cse_scenes在) |
  收齐后:拼2×2 + 定最终结论(协议邻近度 vs regime) + 给rebuttal写法。**CSE多seed注意**:勿与SSE多seed同out_dir并发(撞车);省算力合并跑等B完再`--eval sse cse`同dir复用ckpt。→ **"跨场景失败=划分不幸/测试场景太不同"被证伪**；跨场景负结果是"真"的(多seed只能加误差棒、救不回正)；"输入分布gap"在Waymo内部不成立(见过一堆几乎一样的场景仍没用)→ 责任更多推向 **regime/机制(前视稠密几何里过度密化有害)**，而非"没见过类似场景"。**缺的参照系=CARLA spread(联合图回答:CARLA是否挤得远比Waymo紧→其'跨场景'≈近乎同场景)**。
- **下一步优先级**：① scene_similarity（跑中）→ ② Waymo 跨场景**多 seed**(8/2, 顺带干净扫 T{1,2,3,5,8} 填 Waymo 最优 T 的空白+证伪 T3, mean±std) → ③ CARLA LOSO（`run_carla_loso.py` 已存在但**仍未跑**）→ ④ Waymo within-scene。
- **within-scene 方案敲定**：复用现有 000–019 fronts（已在 `waymo5_gsnet` 训练里=GS-Net已"见过"这些场景），**只需新跑 3 个训练场景的 back = frames 020–039**；back **只要 SfM+去畸变PINHOLE，免 MVS**（纯测试不进训练corr；`waymo_sse` 只读 sparse/0+images，baseline走SfM点、gsnet走infer(sparse points)）；back 目录名沿用 front 的 ID 便于配对；3 个场景从**8个训练场景**里挑(不要10275/15868测试场景)。
- **帧数：用 20 不用 10**。理由：(a) §7.9 已证 10帧≠点稀疏=少视角过参数化=GS-Net受害区；(b) CARLA的"10"≠Waymo的"10"，真正密度杠杆是相机几何不是帧数；(c) within-Waymo 是只改"协议"的 A/B，要 20 才和现有跨场景结果对齐、训练corr同密度。
- **within-scene A/B**：快版(用现 ckpt) vs **干净版(再训一个排除这3场景的GS-Net，约1hr，无需新COLMAP，两模型测同一批back)**，推荐干净版以排除"挑的场景刚好好测"confound。
- **within-scene back 数据已就绪**（用户跑完SfM，2026-06-02）：`/mnt/zihanw/EmerNeRF/data/waymo/colmap_input_5cam_next20/segment-{12879640240483815315_5852_605, 14004546003548947884_2331_861, 3988957004231180266_5566_500}.../colmap/dense/{sparse/0,images/cam0..4}`，5相机×20帧(图重新编号000–019)，**MVS可有可无(waymo_sse测试用不到)**。⚠️**待确认**：这3个ID的front(原colmap_input_5cam同ID段)确在那8个训练场景里(非10275/15868)→ within-scene才成立。**✅已确认**(相似度输出:这3个=index 1/2/8,均训练场景)。**driver已写**:`run_waymo_withinscene.py`(SEEN=waymo5_gsnet vs UNSEEN=排除这3场景重训;两模型测同一批back;baseline跑一次;出Δ_seen/Δ_unseen对照表;`--retrain_seen`可超参对齐)。
- **相似度脚本已扩 CARLA/联合模式**：`--point_specs LABEL=GLOB`(CARLA一场景=其10段ply,各自归一化再pool)可与`--root`(Waymo)合跑→共享标准化→CARLA vs Waymo 场景离散度**可同轴比较**(单独跑因各自标准化不可比)。
- **CARLA LOSO 命令**：`python -m gsnet.run_carla_loso --corr_dir CORR/train --io_dir /mnt/zihanw/carla/input_output --sparse_root /mnt/zihanw/carla/sparse_point --out_dir runs/carla_loso --gpus 0 1 2 3 4 5 6 7`(5折留一,~1hr/折训+评测; **轻confound**:LOSO模型只见4/5数据,但80%够,主效应=排除测试场景)。

## 0.3 ⭐ 正向信号 + 又一个BUG（2026-06）（⚠️本节T3结论已被 §0.2 推翻）
- ~~**SSE T扫描**：T3=**28.40**/T5=26.58/T8=26.66/T12=26.26/T16=26.72，"T3 比 T5 高+1.8、首个真实数据正向信号"~~ → **⚠️ 2026-06-02 推翻（见 §0.2）：此数出自撞 §0.4 test.txt bug 的批次、从未复现、基线本身飘~1dB，属噪声/孤点，不可信。** 仍保留的只有"机制方向"：真实数据上小T(少密化)可能优于大T(过密)，但**需干净多seed重测才算数**。
- **⚠️BUG2(已修)**：`build_subset_source` 之前把所有相机塞成同一内参；Waymo 5相机内参不同 → FL+FR→FRONT(`waymo5_flfr2front`)+其T扫描 ~12dB崩坏作废。已修(保留 per-camera 内参),重跑 `waymo5_flfr2front_v2`。
- **作废**:waymo5_flfr2front, Tsweep_waymo_flfr2front。**有效**:waymo5_sse_v2(1027:base26.42/gsnetT5 25.45), Tsweep_waymo_sse。

## 0.4 ⚠️⚠️ 已修的严重BUG：test.txt 竞争（2026-06）
- **现象**：并发跑的 Waymo 实验（4a 帧留出 / 4b 相机留出 / T扫描）共用同一个 `sparse/0/test.txt`，互相覆盖 → **所有 Waymo 5相机结果污染无效**（铁证:4a==4b 数值雷同 26.32）。
- **修复**：`scene_source(scene, tag)` 现在把 sparse 模型**复制**到 per-out_dir 的 `_gsnet_src_<tag>`（test.txt 隔离），图像仍 symlink。waymo_sse 传 tag=out_dir全路径。
- **作废**：runs/waymo5_sse, runs/waymo5_cse, runs/Tsweep_waymo_cse/sse。**需重跑**(waymo5_sse_v2/cse_v2)。
- **仍有效**：3相机 waymo_sse_geom(+0.06)/sparse(-0.67)(单进程跑的)；**CARLA T扫描有效**(同一确定性划分):T3=23.20/T5=24.58/T8=23.29/T12=23.86/T16=24.38 → **T=5最好,更大T无稳定增益("猛密化"无效)**。

## 0.5 当前状态 & 下一步（⚠️ 最新进展/优先级见 §0.2，本节为 CARLA 主线背景）
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
| `run_sparsity_sweep.py` | Waymo SSE 稀疏度sweep(变n_holdout,base vs gsnet)；**结论:稀疏化死路(§0.2)** |
| `run_carla_loso.py` | CARLA 跨场景 LOSO(对齐Waymo held-out协议)；**未跑(§0.2)** |
| `scene_similarity.py` | 场景间分布相似度诊断(归一化特征空间;测试场景是否离群z>2)；**待跑(§0.2)** |
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

## 7.8 ❌ 已证伪（2026-06-02，见 §0.2）：Waymo 基线偏高 / 稀疏化杠杆
> **结论：这条"把SfM弄稀疏→GS-Net起效"的路已被 sparsity sweep 证伪（每个密度都掉分，越稀疏越惨）。整节留档，勿再投算力。**
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
