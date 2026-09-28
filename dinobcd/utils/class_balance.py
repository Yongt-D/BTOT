"""
DinoBCD - Dataset-Adaptive Class Balancing (DACB)
数据集自适应类别平衡模块

核心创新：
根据数据集的类别分布自动计算最优的Focal Loss α值，
无需人工调参即可适应不同变化检测数据集的类别不平衡程度。

使用方法:
    from dinobcd.utils.class_balance import compute_adaptive_alpha
    
    alpha = compute_adaptive_alpha(train_loader, num_samples=1000)
    criterion = DinoBCDLoss(focal_alpha=alpha)
"""

import torch
import numpy as np
from tqdm import tqdm
from typing import Optional, Tuple


def compute_class_distribution(
    dataloader,
    num_samples: Optional[int] = None,
    show_progress: bool = True
) -> Tuple[float, float, int, int]:
    """
    统计数据集的类别分布
    
    Args:
        dataloader: 数据加载器
        num_samples: 采样数量，None表示使用全部数据
        show_progress: 是否显示进度条
        
    Returns:
        change_ratio: 变化像素占比
        no_change_ratio: 不变像素占比
        total_change_pixels: 总变化像素数
        total_pixels: 总像素数
    """
    total_change_pixels = 0
    total_pixels = 0
    samples_processed = 0
    
    iterator = tqdm(dataloader, desc="Analyzing class distribution") if show_progress else dataloader
    
    for batch in iterator:
        label = batch['label']  # [B, H, W]
        
        # 统计变化像素
        change_pixels = (label == 1).sum().item()
        batch_pixels = label.numel()
        
        total_change_pixels += change_pixels
        total_pixels += batch_pixels
        samples_processed += label.size(0)
        
        # 检查是否达到采样数量
        if num_samples is not None and samples_processed >= num_samples:
            break
    
    change_ratio = total_change_pixels / (total_pixels + 1e-6)
    no_change_ratio = 1 - change_ratio
    
    return change_ratio, no_change_ratio, total_change_pixels, total_pixels


def compute_adaptive_alpha(
    dataloader,
    num_samples: Optional[int] = 1000,
    min_alpha: float = 0.55,
    max_alpha: float = 0.85,
    method: str = 'sqrt_inverse'
) -> float:
    """
    根据数据集类别分布自动计算Focal Loss的最优α值

    Dataset-Adaptive Class Balancing (DACB)

    核心思想：
    - 自动分析数据集的类别不平衡程度
    - 根据不平衡程度计算最优的Focal Loss α值
    - 使用保守的范围限制避免极端权重

    Args:
        dataloader: 训练数据加载器
        num_samples: 用于统计的样本数量(默认1000，None表示全部)
        min_alpha: α的最小值(默认0.55，保守下限避免欠加权)
        max_alpha: α的最大值(默认0.85，保守上限避免过度加权导致高Recall低Precision)
        method: 计算方法
            - 'sqrt_inverse': α = sqrt(1 - change_ratio) (推荐，平方根平滑更稳健)
            - 'inverse_freq': α = 1 - change_ratio (简单反频率)
            - 'effective_samples': 基于有效样本数的方法

    Returns:
        alpha: 计算得到的最优α值（为变化类的权重）
    """
    print("\n" + "="*60)
    print("[DACB] Dataset-Adaptive Class Balancing")
    print("="*60)
    
    # 统计类别分布
    change_ratio, no_change_ratio, change_pixels, total_pixels = compute_class_distribution(
        dataloader, num_samples, show_progress=True
    )
    
    print(f"\n[Class Distribution]")
    print(f"  Change pixels:    {change_pixels:,} ({change_ratio*100:.2f}%)")
    print(f"  No-change pixels: {total_pixels - change_pixels:,} ({no_change_ratio*100:.2f}%)")
    print(f"  Imbalance ratio:  1:{no_change_ratio/change_ratio:.1f}")
    
    # 计算α值
    if method == 'inverse_freq':
        # 简单反频率: 变化越少，权重越高
        alpha = 1 - change_ratio
        
    elif method == 'sqrt_inverse':
        # 平方根平滑: 减少极端情况的影响
        alpha = np.sqrt(1 - change_ratio)
        
    elif method == 'effective_samples':
        # 基于有效样本数 (参考 Class-Balanced Loss)
        beta = 0.9999
        effective_change = (1 - beta**change_pixels) / (1 - beta)
        effective_no_change = (1 - beta**(total_pixels - change_pixels)) / (1 - beta)
        alpha = effective_no_change / (effective_change + effective_no_change)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 限制范围
    alpha = np.clip(alpha, min_alpha, max_alpha)
    
    print(f"\n[Adaptive Alpha Calculation]")
    print(f"  Method: {method}")
    print(f"  Computed α: {alpha:.4f}")
    print(f"  (α={alpha:.2f} means {alpha*100:.0f}% weight for change class)")
    print("="*60 + "\n")
    
    return float(alpha)


def get_class_weights(dataloader, num_samples: Optional[int] = 1000) -> torch.Tensor:
    """
    计算类别权重用于CrossEntropyLoss
    
    Args:
        dataloader: 数据加载器
        num_samples: 采样数量
        
    Returns:
        weights: [2] tensor，[背景权重, 变化权重]
    """
    change_ratio, no_change_ratio, _, _ = compute_class_distribution(
        dataloader, num_samples, show_progress=True
    )
    
    # 使用反频率作为权重
    weights = torch.tensor([change_ratio, no_change_ratio], dtype=torch.float32)
    
    # 归一化
    weights = weights / weights.sum() * 2
    
    return weights


if __name__ == "__main__":
    # 测试代码
    print("Dataset-Adaptive Class Balancing Test")
    print("="*50)
    
    # 模拟不同数据集的类别分布
    test_cases = [
        ("LEVIR-CD (极度不平衡)", 0.03),
        ("WHU-CD (较平衡)", 0.15),
        ("S2Looking (中等)", 0.08),
    ]
    
    for name, change_ratio in test_cases:
        no_change_ratio = 1 - change_ratio
        
        # 模拟计算
        alpha_inverse = 1 - change_ratio
        alpha_sqrt = np.sqrt(1 - change_ratio)
        
        print(f"\n{name}:")
        print(f"  Change ratio: {change_ratio*100:.1f}%")
        print(f"  α (inverse_freq): {alpha_inverse:.4f}")
        print(f"  α (sqrt_inverse): {alpha_sqrt:.4f}")
