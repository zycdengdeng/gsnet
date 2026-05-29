# GS-Net

Reimplementation of **GS-Net** (*"GS-Net: Heterogeneous Vehicle Data Reuse via
Generalizable Plug-and-Play 3DGS Module"*): a lightweight, plug-and-play module
that predicts **dense Gaussian primitives directly from sparse SfM point clouds**
in a single forward pass, used as initialization for standard 3DGS optimization.

Fully decoupled from the base 3DGS repo: GS-Net produces a standard 3DGS `.ply`,
consumed via the new `--gsnet_init` flag.

## Components

| File | Role |
|------|------|
| `model.py` | GS-Net network: geometry-aware encoder (Eq. 3) + multi-head Gaussian expansion (Eq. 4). |
| `losses.py` | Decoupled geometric + appearance objective (Eq. 5–7). |
| `common.py` | Shared readers, G_dense outlier filtering, per-sequence normalization. |
| `build_correspondences.py` | KD-tree sparse→dense K-NN (pseudo-GT) + M-NN sparse graph → `.npz`. |
| `dataset.py` / `train_gsnet.py` | Cross-scene dataset and training loop. |
| `infer.py` | Single forward pass: sparse SfM → dense Gaussian init `.ply`. |
| `io.py` | Export predictions as a standard 3DGS `.ply`. |
| `make_sse_split.py` | Write the SSE train/test split as `sparse/0/test.txt`. |
| `run_sse.py` | SSE driver: baseline 3DGS vs GS-Net+3DGS, with metrics + timing. |
| `inspect_ply.py` | Diagnostic for ply fields / coordinate scale / alignment. |

## Dataset layout (CARLA-NVS, 5 scenes × 10 sequences)

```
sparse_point/S0<scene>/<id>_sparse.ply                 # sparse SfM  (P_sfm, GS-Net input)
input_output/<id>_dense/sparse/0/points3D.ply          # dense MVS   (3DGS input for targets)
input_output/output_<id>_dense/point_cloud/iteration_30000/point_cloud.ply   # G_dense target
input_output/<id>_base/{images,sparse/0}               # test sequence COLMAP workspace
```
`id = 100*scene + seq`. Train seqs: `seq 1–9`. Test seqs: `seq 10` → `110,210,310,410,510`.
Images 1–60 per sequence map to source cameras `1,3,5,7,9,11` (10 frames each).

## Workflow

```bash
# 0. (optional) inspect a ply's fields / scale / sparse-vs-dense alignment
python -m gsnet.inspect_ply --ply sparse_point/S01/101_sparse.ply \
    --gdense input_output/output_101_dense/point_cloud/iteration_30000/point_cloud.ply

# 1. Build pseudo-GT correspondences for all training sequences
python -m gsnet.build_correspondences --batch \
    --io_dir /mnt/zihanw/carla/input_output \
    --sparse_root /mnt/zihanw/carla/sparse_point \
    --out_dir CORR/train          # -> CORR/train/*.npz + build_times.json

# 2. Train GS-Net (200 epochs, batch 512, Adam 1e-3; logs train_times.json)
python -m gsnet.train_gsnet --corr_dir CORR/train --out_dir runs/gsnet

# 3+4. SSE evaluation: baseline 3DGS vs GS-Net+3DGS on the 5 test sequences
python -m gsnet.run_sse \
    --io_dir /mnt/zihanw/carla/input_output \
    --sparse_root /mnt/zihanw/carla/sparse_point \
    --ckpt runs/gsnet/gsnet_latest.pt \
    --out_dir runs/sse            # -> runs/sse/sse_results.{json,md}
```

`run_sse.py` records GS-Net inference time + 3DGS optimization time + PSNR/SSIM/
LPIPS per sequence and the averages, mirroring the paper's SSE table and
efficiency comparison.

## Design notes

- **Sparse source.** GS-Net always consumes the *sparse* SfM (`*_sparse.ply`),
  never the dense MVS (`<id>_dense/.../points3D.ply`).
- **Coordinate normalization.** Each sequence is normalized by a similarity
  transform (median center + p95 radius) computed from its sparse points only —
  reproducible at inference. Targets are normalized identically; predictions are
  denormalized before export so 3DGS optimizes in the original COLMAP frame. This
  unifies cross-sequence scale and keeps the Tanh offset / sigmoid scale bounds
  meaningful.
- **G_dense filtering.** Far floating junk (distance > `radius_margin`×p99 of the
  sparse extent) and near-transparent Gaussians (`opacity < opacity_min`) are
  removed before building correspondences. Optional statistical outlier removal
  (`--sor_k`).
- **Color ↔ SH.** Colors handled in RGB `[0,1]`; only 0-order SH (diffuse) is
  predicted; `G_dense` `f_dc`→RGB via `SH2RGB`, exported back via `RGB2SH`.
- **Loss offsets.** Geometric/appearance losses compare the *applied*
  (post-activation) offsets `Tanh(Δμ̂)` / `σ(ΔĈ)` against GT offsets.
- **Opacity.** `α̂ = Tanh(·) ∈ (-1,1)`; non-positive opacity primitives dropped on export.
- **Ablations (Table IV).** `GSNetConfig` toggles `predict_color`,
  `predict_opacity`, `predict_scale_rot`.
- **Timing.** Every stage logs wall-clock time (`build_times.json`,
  `train_times.json`, `sse_results.json`).
