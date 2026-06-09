# nuScenes 外插视角定性 Benchmark（给合作者测别的方法）

> 用途：在**完全相同的新视角（训练没见过的外插位姿）**下，渲染各方法的重建结果做**定性对比**（无 GT、无量化指标，纯视觉证据：哪个方法在外插视角下空洞/floater 更少）。
> 合作者：用你的方法重建同一个 clip，在**本文件给定的那组新位姿**渲染，与我们的 GS-Net / baseline 并排比。

---

## 0. 一句话
取一个 clip 的 **FRONT 相机**为基准，把它**平移 + 旋转**到 20 个训练集没有的位姿（外插），用各方法的重建在这些位姿渲染、对比完整度。

---

## 1. 数据 / 基准相机
- **数据**：`/mnt/zihanw/gsnet_nusc/<clip>/colmap/dense/sparse/0/`（COLMAP PINHOLE 模型，`cameras.bin`/`images.bin`）。
- **基准相机**：`cam0/005.jpg`（FRONT 相机第 5 帧）。其内参（`cameras.bin`，1600×900 PINHOLE：fx,fy,cx,cy）+ 位姿（`images.bin` 的 `qvec,tvec` = world→cam）。
- 推荐 clip：`348_clip_09 / 332_clip_09 / 331_clip_09 / 299_clip_09 / 325_clip_09`（和我们 SSE 测试集一致）。

---

## 2. 偏移表（20 个新位姿）
相机坐标系 = **OpenCV**（x 右、y 下、z 前）。平移单位米（在**相机坐标系**下），旋转：`yaw` 绕 y(下)轴、`pitch` 绕 x(右)轴。

| 文件名 | 平移 (dx,dy,dz) m | yaw° | pitch° | 含义 |
|---|---|---|---|---|
| `00_orig` | (0,0,0) | 0 | 0 | 基准（训练见过）|
| `01_left1` | (-1,0,0) | 0 | 0 | 左移 1m |
| `02_left2` | (-2,0,0) | 0 | 0 | 左移 2m |
| `03_right1` | (1,0,0) | 0 | 0 | 右移 1m |
| `04_right2` | (2,0,0) | 0 | 0 | 右移 2m |
| `05_up1` | (0,-1,0) | 0 | 0 | 上移 1m |
| `06_up2` | (0,-2,0) | 0 | 0 | 上移 2m |
| `07_fwd2` | (0,0,2) | 0 | 0 | 前移 2m |
| `08_fwd4` | (0,0,4) | 0 | 0 | 前移 4m |
| `09_back2` | (0,0,-2) | 0 | 0 | 后移 2m |
| `10_yawL10` | (0,0,0) | **-10** | 0 | **左转 10°** |
| `11_yawL25` | (0,0,0) | **-25** | 0 | **左转 25°** |
| `12_yawR10` | (0,0,0) | **+10** | 0 | **右转 10°** |
| `13_yawR25` | (0,0,0) | **+25** | 0 | **右转 25°** |
| `14_pitchUp10` | (0,0,0) | 0 | -10 | 上仰 10° |
| `15_pitchDn10` | (0,0,0) | 0 | +10 | 下俯 10° |
| `16_left2_yawR15` | (-2,0,0) | +15 | 0 | 左移2m + 右转15° |
| `17_up1_pitchDn10` | (0,-1,0) | 0 | +10 | 上移1m + 下俯10° |
| `18_fwd3_left1` | (-1,0,3) | 0 | 0 | 前移3m + 左移1m |
| `19_right2_yawL15` | (2,0,0) | -15 | 0 | 右移2m + 左转15° |

> ⚠️ yaw 的正负=绕 y(下)轴的右手系旋转；视觉上的"左/右"以本表的 `yaw°` 数值为准（标签仅助记，若视觉左右相反不影响对比）。
> **重点外插视角**（差异最大、最该比）：`02_left2 / 04_right2 / 08_fwd4 / 11_yawL25 / 13_yawR25`。

---

## 3. 从基准位姿算新位姿（合作者照此在自己 pipeline 里建相机）
设基准（来自 COLMAP）：`R = qvec2rotmat(qvec)`（world→cam 3×3），`t = tvec`（world→cam 平移）。给定偏移 `dpos`(相机系)、`dyaw`、`dpitch`：

```
C   = -Rᵀ · t                          # 相机中心(世界系)
C'  =  C + Rᵀ · dpos                    # 平移(相机系→世界)
R'  =  Rx(dpitch) · Ry(dyaw) · R        # 旋转相机朝向
t'  = -R' · C'                          # 新的 world→cam 平移
# 渲染位姿 = (R', t')，内参用基准相机的 fx,fy,cx,cy
```
其中
```
Rx(a)=[[1,0,0],[0,cos a,-sin a],[0,sin a,cos a]]      # 绕 x(右)
Ry(a)=[[cos a,0,sin a],[0,1,0],[-sin a,0,cos a]]      # 绕 y(下)
```
投影约定：`x_cam = R'·x_world + t'`，像素 `~ K·x_cam`，`K=[[fx,0,cx],[0,fy,cy],[0,0,1]]`。

---

## 4. 直接拿到这 20 个确切位姿（推荐，免自己算）
我们提供导出：每个新视角的 **`world_to_cam` 4×4 + fx/fy/cx/cy + 宽高** 存成 JSON：
```bash
python -m gsnet.render_novel \
  --colmap /mnt/zihanw/gsnet_nusc/348_clip_09/colmap/dense/sparse/0 --base cam0/005.jpg \
  --ply_baseline <any.ply> --ply_gsnet <any.ply> \
  --out_dir /tmp/_ignore --dump_poses /mnt/zihanw/nusc_novel/348_poses.json
```
→ `348_poses.json`：
```json
{"base_image":"cam0/005.jpg","convention":"OpenCV ... world_to_cam [R|t], x_cam=R·x_world+t",
 "views":[{"tag":"13_yawR25","world_to_cam":[[...4x4...]],"fx":...,"fy":...,"cx":...,"cy":...,
           "width":1600,"height":900,"dpos_cam_m":[0,0,0],"dyaw_deg":25,"dpitch_deg":0}, ...]}
```
**合作者只需：加载 `world_to_cam` + 内参，在自己的方法的重建上渲染这 20 个视角即可**（不依赖我们的代码/坐标推导）。

---

## 5. 怎么比 / 出图
- 每个方法在这 20 个位姿各渲染一张 → 与我们的并排。
- 我们的并排图（左=baseline SfM-init / 右=GS-Net-init）由 `render_novel` 生成：
```bash
python -m gsnet.render_novel \
  --colmap /mnt/zihanw/gsnet_nusc/348_clip_09/colmap/dense/sparse/0 --base cam0/005.jpg \
  --ply_baseline runs/nusc_filt/eval/348_clip_09/sfm_d15000/point_cloud/iteration_30000/point_cloud.ply \
  --ply_gsnet   runs/nusc_filt/eval/348_clip_09/gsnet_d15000/point_cloud/iteration_30000/point_cloud.ply \
  --out_dir /mnt/zihanw/nusc_novel/348_clip_09_front_d15000
```
→ `00_orig.png … 19_*.png`，每张**左=baseline | 右=GS-Net**，同一新视角。
- 看点：外插越大（left2/fwd4/yawR25…），哪个方法**空洞/拉丝/floater 更少、结构更完整**。

---

## 6. 关键文件
| 文件 | 作用 |
|---|---|
| `gsnet/render_novel.py` | 生成新位姿 + 渲染并排图 + `--dump_poses` 导位姿 JSON |
| `<clip>/colmap/dense/sparse/0/{cameras,images}.bin` | 基准相机内参 + 位姿 |
| `runs/nusc_filt/eval/<clip>/{sfm,gsnet}_d15000/.../point_cloud.ply` | 我们的 baseline / GS-Net 重建（供并排）|

> 换基准相机：`--base cam5/005.jpg`(BACK) 等；换稀疏对比：路径 `d15000`→`d0`（baseline 更稀，差异更刺眼）。
