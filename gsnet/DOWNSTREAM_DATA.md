# 下游感知评测 — 数据说明（给 Depth_Seg_eval 那边）

> 目的：回应审稿"只在 3DGS 内评估、没证 data reuse 价值"。用 Depth_Seg_eval 库对**我们的渲染**和**真值**各跑预训练感知模型（深度/分割/SAM），比 `metric(gen, gt)`。
> **gen = 我们的 3DGS 渲染**（两种 init：baseline=SfM 点 / gsnet=GS-Net）；**gt = 真值图**。GS-Net 的一致性更高 = 更好的重建翻译成更好的下游感知。

---

## 0. 怎么比（关键）
对**每种 init 各跑一次库**，比两次的指标：
- `baseline`：gen=SfM-init 渲染，gt=真值 → 一组指标。
- `gsnet`：gen=GS-Net-init 渲染，gt=真值 → 一组指标。
- **GS-Net 应：depth abs_rel/rmse 更低、seg mIoU/consistency 更高、SAM edge_f1 更高。**

> 注意：两次的 **gt 完全相同**（同一批测试视角的真值），变的只是 gen（不同 init 的渲染）。

---

## 1. 我们渲染的原始位置（renders/ 和 gt/，按文件名一一对应）
| 数据集 | 路径模板 |
|---|---|
| **CARLA CSE**（大增益，合成）| `runs/cse_d2000_s0/<id>/<method>/test/ours_30000/{renders,gt}/` ；`<id>`=110/210/410/510（**跳过崩坏的 310**），`<method>`=`baseline`/`gsnet` |
| **nuScenes SSE**（真实域，无域 gap）| `runs/nusc_filt/eval/<clip>/<init>_d15000/test/ours_30000/{renders,gt}/` ；`<clip>`=`*_clip_09`×5，`<init>`=`sfm`/`gsnet` |
- `renders/00000.png…` = 我们的渲染（gen）；`gt/00000.png…` = 真值（gt）；同名对应。

---

## 2. 一键整理成库要的格式 `_eval_frames/<camera>/{gen,gt}/`
（`<camera>` 用序列 id / clip 名做分组，只影响 per-group 报告，不影响 per-image 指标。）

```bash
cd /mnt/zihanw/gaussian-splatting && git pull origin claude/festive-feynman-80Vw3

# CARLA CSE -> /mnt/zihanw/downstream_cse/{baseline,gsnet}/_eval_frames/<id>/{gen,gt}/
python -m gsnet.prep_downstream --mode cse \
  --run runs/cse_d2000_s0 --groups 110 210 410 510 \
  --methods baseline gsnet --out /mnt/zihanw/downstream_cse

# nuScenes SSE -> /mnt/zihanw/downstream_nusc/{sfm,gsnet}/_eval_frames/<clip>/{gen,gt}/
python -m gsnet.prep_downstream --mode nusc \
  --run runs/nusc_filt/eval \
  --groups 348_clip_09 332_clip_09 331_clip_09 299_clip_09 325_clip_09 \
  --methods sfm gsnet --densify d15000 --out /mnt/zihanw/downstream_nusc
```
（默认软链接，不占空间；要拷贝加 `--copy`。脚本会打印每个 `_eval_frames` root 路径。）

---

## 3. 库要 point 的位置（config.yaml 里填 `_eval_frames` root）
| 跑哪次 | 数据集 | `_eval_frames` root |
|---|---|---|
| CARLA baseline | CSE | `/mnt/zihanw/downstream_cse/baseline/_eval_frames` |
| CARLA gsnet | CSE | `/mnt/zihanw/downstream_cse/gsnet/_eval_frames` |
| nuScenes baseline | SSE | `/mnt/zihanw/downstream_nusc/sfm/_eval_frames` |
| nuScenes gsnet | SSE | `/mnt/zihanw/downstream_nusc/gsnet/_eval_frames` |

每个 root 下是 `<camera>/gen/*.png` + `<camera>/gt/*.png`，正是库的图像帧模式（`evaluate.py` + `config.yaml`）所需。

---

## 4. 跑哪几个 task（你定的三个）
- **深度一致性**（`depth_eval.py`，Depth-Anything-V2）：`abs_rel / rmse / delta_1 / pearson`——**最该重点看**（GS-Net 改的就是几何，深度=几何，因果最直接，且跨域稳）。
- **语义分割一致性**（`seg_eval.py`，Mask2Former/Cityscapes）：`consistency / mIoU / fwIoU`。
- **SAM 结构一致性**（`sam_eval.py`）：`edge_f1 / edge_correlation`。

> ⚠️ CARLA 是**合成**，Cityscapes 分割可能有域 gap；**深度模型最稳**，建议先看深度，分割/SAM 作补充。nuScenes 是真实图、无域 gap，三个都可信。

---

## 5. 预期结果表（论文里这么呈现）
| 数据集 | init | depth abs_rel↓ | seg mIoU↑ | SAM edge_f1↑ |
|---|---|---|---|---|
| CARLA CSE | baseline | … | … | … |
| CARLA CSE | **gsnet** | **更低** | **更高** | **更高** |
| nuScenes SSE | baseline | … | … | … |
| nuScenes SSE | **gsnet** | **更低** | **更高** | **更高** |

**结论一句话**：GS-Net 的更好重建（data reuse）在三个下游感知任务上一致优于 baseline → 证明实际价值，正面回应审稿人。

---

## 6. 注意/确认
1. **GT 来源**：CARLA CSE 的 gt = 偶数相机真值图（CARLA 渲染的目标视角）；nuScenes SSE 的 gt = 真实留出帧。都是 render.py 存在 `gt/` 里的，已是真值。
2. **跳 310**：CARLA 崩坏场景，不要放进去。
3. **分辨率**：CARLA 与 nuScenes 渲染尺寸不同（库内部 resize，一般无碍）。
4. 若某 method/group 的 renders/gt 缺失，`prep_downstream` 会打印 `[skip]`，检查那次 eval 是否跑全。
