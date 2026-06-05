# nuScenes data spec for GS-Net (hand to data team)

**Goal**: produce nuScenes surround-view scenes in the **same COLMAP layout** the
CARLA/Waymo GS-Net pipeline already consumes, so we can run it unchanged. The
6-camera ring (high cross-camera overlap, like CARLA) is the regime where GS-Net
is expected to help — especially **cross-camera synthesis (CSE)**.

## Why nuScenes (not more Waymo)
Waymo's 5 cameras are forward strips with <10% inter-camera overlap → no
exploitable coverage holes → GS-Net neutral (verified: closed regime).
nuScenes has a **6-camera 360° ring**; adjacent cameras overlap → a held-out
camera's region is covered by its neighbors → GS-Net can fill it. This is the
real-data analog of CARLA CSE (+1.89).

## Cameras (6), fixed dir mapping (ring order)
| dir | nuScenes camera |
|-----|-----------------|
| cam0 | CAM_FRONT |
| cam1 | CAM_FRONT_LEFT |
| cam2 | CAM_FRONT_RIGHT |
| cam3 | CAM_BACK_LEFT |
| cam4 | CAM_BACK_RIGHT |
| cam5 | CAM_BACK |

## Frames
- Use the **2 Hz keyframes** of each nuScenes scene (~40 per 20 s scene).
- **Do NOT over-sparsify.** (Lesson from Waymo: too few frames → default 3DGS
  densification over-fits → baseline craters to the teens, which is not a
  credible baseline.) Use all keyframes, or every-other if 3DGS memory is tight
  (~20/cam). The regime here comes from the **ring geometry**, not from sparsity
  — so keep the baseline credible (target ~25–27 PSNR, matching published
  nuScenes 3DGS).
- Use **static-ish scenes** if possible (fewer moving objects) — our 3DGS is
  static; heavy dynamics hurt the baseline for everyone (orthogonal to GS-Net).

## Scenes
- **20–30 different nuScenes scenes** (different logs). We will split e.g. 25
  train / 5 test.

## COLMAP per scene
1. **SfM** over all 6×N images. Use nuScenes calibrated intrinsics/extrinsics as
   priors if convenient (more robust on the ring), else SfM from scratch.
   Per-camera intrinsics (the 6 cameras differ — do **not** collapse to one).
2. `image_undistorter` → **PINHOLE**, producing the dense layout.
3. `patch_match_stereo` + `stereo_fusion` → `fused.ply` (MVS = our pseudo-GT
   supervision target and the densification-ceiling reference).

## Output layout (mirror `colmap_input_5cam`)
```
<root>/scene-<token>/
  colmap/dense/
    sparse/0/{cameras.bin, images.bin, points3D.bin}   # SfM model (PINHOLE)
    images/cam0/000.jpg cam0/001.jpg ... cam5/NNN.jpg   # per-camera, frame-indexed
    fused.ply                                            # MVS pseudo-GT
```
- Scene dir name: anything starting `scene-` (our loader now auto-discovers any
  dir containing `colmap/`).
- Image names **must** be `camK/<frameidx>.jpg` (the directory prefix = camera,
  the number = frame), so per-camera train/test splitting works.
- `points3D.bin` (SfM sparse points) is the **GS-Net input**; `fused.ply` (MVS)
  is the **target**. Both must be in the **same (undistorted) COLMAP frame**.

## What we run once data lands (no new data work for you)
1. `waymo_cam_overlap` → confirm adjacent-camera overlap ≫ Waymo's <10%
   (sanity: the ring really provides redundancy).
2. `waymo_gdense` → `waymo_corr` (test scenes excluded) → `train_gsnet`.
3. **CSE (the main bet)**: `waymo_sse --source_cams ... --target_cams camK`
   (train on 5 cameras, synthesize the held-out one) — CARLA-CSE real analog.
4. SSE (frame holdout) + init-spectrum + densify-sweep for completeness, always
   compared at the **credible (tuned) baseline**.
