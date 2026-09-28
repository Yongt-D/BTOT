"""
DinoBCD Losses Package
"""

from .combined_loss import (
    FocalLoss,
    DiceLoss,
    EdgeLoss,
    DinoBCDLoss
)

__all__ = [
    'FocalLoss',
    'DiceLoss',
    'EdgeLoss',
    'DinoBCDLoss'
]
