# GS-Net

Reimplementation of **GS-Net** (from the paper *"GS-Net: Heterogeneous Vehicle
Data Reuse via Generalizable Plug-and-Play 3DGS Module"*): a lightweight,
plug-and-play module that predicts **dense Gaussian primitives directly from
sparse SfM point clouds** in a single forward pass, used as initialization for
standard 3DGS optimization.

This package lives alongside the original 3D Gaussian Splatting repo and is
fully decoupled from it: GS-Net produces a standard 3DGS `.ply`, which the
training pipeline consumes via the new `--gsnet_init` flag.

## Components

| File | Role |
|------|------|
| `model.py` | GS-Net network: geometry-aware encoder (Eq. 3) + multi-head Gaussian expansion (Eq. 4). |
| `losses.py` | Decoupled geometric + appearance training objective (Eq. 5–7). |
| `build_correspondences.py` | Offline data engine step **we own**: KD-tree sparse→dense K-NN, plus M-NN sparse graph → pseudo-GT `.npz`. |
| `dataset.py` | Concatenates all correspondence `.npz` across scenes for cross-scene training. |
| `train_gsnet.py` | Training loop: 200 epochs, batch 512, Adam lr 1e-3, T=5, M=3. |
| `infer.py` | Single forward pass: sparse SfM → dense Gaussian init `.ply`. |
| `io.py` | Writes predictions as a standard 3DGS `.ply` (consumed by `GaussianModel.load_ply`). |

## End-to-end workflow

The offline data engine up to `G_dense` (COLMAP SfM, MVS, per-scene 3DGS
optimization) is produced on the server side. We consume `G_dense` + sparse SfM.

```bash
# 1. Build pseudo-GT correspondences for every TRAINING sequence
python -m gsnet.build_correspondences \
    --sfm    SCENE/seq/sparse/0/points3D.bin \
    --gdense SCENE/seq/gaussians/point_cloud.ply \
    --out    CORR/train/<scene>_<seq>.npz --K 5 --M 3

# 2. Train GS-Net across all 15 training scenes
python -m gsnet.train_gsnet --corr_dir CORR/train --out_dir runs/gsnet

# 3. Inference: densify a (test) scene's sparse SfM into an init .ply
python -m gsnet.infer \
    --ckpt runs/gsnet/gsnet_latest.pt \
    --sfm  SCENE/seq/sparse/0/points3D.bin \
    --out  SCENE/seq/gsnet_init.ply

# 4. Standard 3DGS optimization, initialized from GS-Net (plug-and-play)
python train.py -s SCENE/seq --gsnet_init SCENE/seq/gsnet_init.ply --eval
```

## Conventions / design notes (please confirm)

- **Color ↔ SH.** SfM colors and `G_dense` diffuse colors are handled in RGB
  space `[0,1]`. `G_dense` `f_dc` is converted via `SH2RGB`; on export the
  predicted color is converted back via `RGB2SH`. Only the 0-order SH (diffuse)
  is predicted; higher-order SH are left to per-scene optimization (paper).
- **Scale.** `S_hat = sigmoid(s_hat) ∈ (0,1)` per the paper. The GT scale is the
  *actual* scale `exp(_scaling)` from `G_dense`. If scenes are metric and many
  dense Gaussians have scale `> 1`, the sigmoid cap would clip them — in that
  case we may need to rescale or relax the activation. Flagged for review.
- **Position offset.** Bounded by `Tanh ∈ (-1,1)` (`pos_offset_scale=1.0`). This
  assumes the K nearest dense Gaussians lie within ~1 unit of each SfM point. If
  the COLMAP coordinate scale makes this too tight, raise `--pos_offset_scale`.
- **Loss offsets.** The geometric/appearance losses compare the *applied*
  (post-activation) offsets `Tanh(Δμ̂)` and `σ(ΔĈ)` against the GT offsets, so
  that `μ̂ ≈ μ^gt` and `Ĉ ≈ C^gt` directly.
- **Opacity.** `α̂ = Tanh(·) ∈ (-1,1)`; non-positive opacity primitives are
  discarded on export.
- **Ablations (Table IV).** `GSNetConfig` toggles `predict_color`,
  `predict_opacity`, `predict_scale_rot` for the attribute ablation study.
