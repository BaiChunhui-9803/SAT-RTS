"""
距离计算模块

包含各种距离度量方法的实现:
- 劶态距离计算 (用于游戏状态比较)
- 点云距离计算 (Wasserstein, Chamfer, Hausdorff, EMD)
"""

from .base import CustomDistance, DistributionDistance
from .custom import (
    CustomDistance as CustomDistanceForCustom,
    DistributionDistance as DistributionDistanceForCustom,
)
from .wasserstein import WassersteinDistance, WassersteinDistributionDistance
from .chamfer import ChamferDistance, ChamferDistributionDistance
from .hausdorff import ModifiedHausdorffDistance, ModifiedHausdorffDistributionDistance
from .emd import PointCloudEMDDistance, PointCloudEMDDistance

__all__ = [
    "CustomDistance",
    "DistributionDistance",
    "CustomDistanceForCustom",
    "DistributionDistanceForCustom",
    "WassersteinDistance",
    "WassersteinDistributionDistance",
    "ChamferDistance",
    "ChamferDistributionDistance",
    "ModifiedHausdorffDistance",
    "ModifiedHausdorffDistributionDistance",
    "PointCloudEMDDistance",
    "PointCloudEMDDistance",
]
