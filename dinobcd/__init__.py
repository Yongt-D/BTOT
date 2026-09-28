"""
DinoBCD: DINOv3-based Building Change Detection

核心创新：双重差异建模 (Dual-Stream Difference Modeling)
- Change Stream: 捕获时序变化
- Invariant Stream: 捕获语义相似度
- Adaptive Fusion: 动态加权融合
"""

__version__ = '2.0.0'
__author__ = 'Your Name'

from .models import DinoBCD, build_dinobcd
from .losses import DinoBCDLoss

__all__ = ['DinoBCD', 'build_dinobcd', 'DinoBCDLoss']
