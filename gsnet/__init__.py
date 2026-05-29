#
# GS-Net: Heterogeneous Vehicle Data Reuse via Generalizable Plug-and-Play 3DGS Module
#
# Lightweight, plug-and-play module that predicts dense Gaussian primitives
# directly from sparse Structure-from-Motion (SfM) point clouds in a single
# forward pass, providing a transferable geometric prior for 3DGS
# initialization across scenes.
#
# Submodules are imported directly (e.g. `from gsnet.model import GSNet`) so that
# torch-free utilities (make_sse_split, inspect_ply, build_correspondences) do
# not pull in heavy dependencies.
#
