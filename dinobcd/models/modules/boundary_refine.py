"""
DinoBCD - Soft Boundary Refinement Module
软边界细化模块

使用可微分的软形态学操作进行边界优化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftMorphology(nn.Module):
    """
    可微分的软形态学操作
    
    使用logsumexp近似min/max操作，保持可微分性
    """

    def __init__(self, kernel_size=3, tau=0.05):
        super().__init__()

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

        # 可学习的温度参数
        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau))))

        # 可学习的卷积核权重
        num_elements = kernel_size * kernel_size
        self.weight_erode = nn.Parameter(torch.zeros(1, num_elements))
        self.weight_dilate = nn.Parameter(torch.zeros(1, num_elements))

    def soft_dilate(self, x, weight):
        """软膨胀: max(x) ≈ τ·logsumexp(x/τ)"""
        B, C, H, W = x.shape
        tau = torch.exp(self.log_tau).clamp_min(1e-4)

        # Unfold为patches
        x_unfold = F.unfold(
            x,
            kernel_size=self.kernel_size,
            padding=self.padding
        )  # [B, C*K*K, H*W]

        x_unfold = x_unfold.view(B, C, self.kernel_size**2, H*W)

        # Soft max: τ·logsumexp((x+w)/τ)
        logsum = torch.logsumexp((x_unfold + weight.unsqueeze(-1)) / tau, dim=2)
        result = logsum * tau

        return result.view(B, C, H, W)

    def soft_erode(self, x, weight):
        """软腐蚀: min(x) = -max(-x)"""
        return -self.soft_dilate(-x, -weight)

    def forward(self, x):
        """
        Args:
            x: [B, 1, H, W] 概率图

        Returns:
            refined: [B, 1, H, W] 细化后的概率图
        """
        # 开运算: 腐蚀 -> 膨胀 (去除小噪点)
        x = self.soft_erode(x, self.weight_erode)
        x = self.soft_dilate(x, self.weight_dilate)

        # 闭运算: 膨胀 -> 腐蚀 (填充小孔洞)
        x = self.soft_dilate(x, self.weight_dilate)
        x = self.soft_erode(x, self.weight_erode)

        return x


class SoftBoundaryRefinement(nn.Module):
    """
    软边界细化模块
    
    结合软形态学操作优化预测边界
    """

    def __init__(self, kernel_size=3):
        super().__init__()

        # 软形态学
        self.morph = SoftMorphology(kernel_size=kernel_size)

        # 融合权重 (可学习)
        self.alpha = nn.Parameter(torch.tensor(0.3))

    def forward(self, logits):
        """
        Args:
            logits: [B, 2, H, W] 分类logits

        Returns:
            refined_logits: [B, 2, H, W] 细化后的logits
        """
        # 获取前景概率
        prob = F.softmax(logits, dim=1)[:, 1:2]  # [B, 1, H, W]

        # 软形态学细化
        prob_morph = self.morph(prob)

        # 转回logits
        prob_morph = torch.clamp(prob_morph, 1e-6, 1-1e-6)
        fg_logit_refined = torch.log(prob_morph / (1 - prob_morph + 1e-6))

        # 与原始logits加权融合
        alpha = torch.sigmoid(self.alpha)
        refined_logits = logits.clone()
        refined_logits[:, 1:2] = logits[:, 1:2] + alpha * (fg_logit_refined - logits[:, 1:2])

        return refined_logits


class EfficientLMM(nn.Module):
    """
    Efficient Learnable Morphology Module (EfficientLMM)
    
    借鉴ChangeDINO的LMM，但使用深度可分离卷积实现效率优化
    
    特点:
    1. 多尺度边界感知
    2. 深度可分离卷积 (-70%参数)
    3. 更锋利的边界输出
    """
    
    def __init__(self, num_classes=2, hidden_dim=8):
        super().__init__()
        
        # 多尺度边界检测 (使用不同kernel size)
        self.edge_detect_3x3 = nn.Sequential(
            nn.Conv2d(num_classes, hidden_dim, 3, 1, 1, groups=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.edge_detect_5x5 = nn.Sequential(
            nn.Conv2d(num_classes, hidden_dim, 5, 1, 2, groups=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 特征融合 (深度可分离)
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, 3, 1, 1, groups=hidden_dim * 2, bias=False),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 边界细化输出
        self.refine_conv = nn.Conv2d(hidden_dim, num_classes, 1)
        
        # 可学习的融合权重
        self.gamma = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, logits):
        """
        Args:
            logits: [B, 2, H, W] 分类logits
        Returns:
            refined_logits: [B, 2, H, W] 细化后的logits
        """
        # 多尺度边界特征
        edge_3x3 = self.edge_detect_3x3(logits)
        edge_5x5 = self.edge_detect_5x5(logits)
        
        # 融合
        edge_feat = torch.cat([edge_3x3, edge_5x5], dim=1)
        edge_feat = self.fuse(edge_feat)
        
        # 边界细化
        refinement = self.refine_conv(edge_feat)
        
        # 残差连接
        refined_logits = logits + self.gamma * refinement
        
        return refined_logits


if __name__ == "__main__":
    # 测试代码
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 60)
    print("Testing SoftBoundaryRefinement")
    print("=" * 60)

    refiner = SoftBoundaryRefinement(kernel_size=3).to(device)
    logits = torch.randn(2, 2, 64, 64).to(device)

    refined = refiner(logits)
    print(f"Input: {logits.shape}")
    print(f"Output: {refined.shape}")

    total_params = sum(p.numel() for p in refiner.parameters())
    print(f"Parameters: {total_params:,}")
