#
# GS-Net: Heterogeneous Vehicle Data Reuse via Generalizable Plug-and-Play 3DGS Module
#
# Lightweight, plug-and-play module that predicts dense Gaussian primitives
# directly from sparse Structure-from-Motion (SfM) point clouds in a single
# forward pass, providing a transferable geometric prior for 3DGS
# initialization across scenes.
#

from .model import GSNet, GSNetConfig, quat_to_rot
from .losses import gsnet_loss

__all__ = ["GSNet", "GSNetConfig", "quat_to_rot", "gsnet_loss"]
