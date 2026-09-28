"""
FCCR - Feature Consistency Contrastive Regularization
特征一致性对比正则化损失

核心思想:
传统变化检测损失 (Focal/Dice/Tversky) 仅作用于最终预测mask,
对中间特征层没有显式约束. 这导致:
- 未变化区域的T1/T2特征可能差异很大 → 产生False Positive
- 变化区域的T1/T2特征差异可能不够显著 → 产生False Negative
- 模型倾向于记忆训练集特有的像素模式, 而非通用的变化/不变特征表示

FCCR在金字塔特征层面施加对比约束:
- 未变化像素: 宽松拉近T1和T2特征 (sim > pull_threshold即可)
- 变化像素:   推远T1和T2特征 (f1_changed ≠ f2_changed)

数学形式:
    给定金字塔层 l 的T1特征 f1_l, T2特征 f2_l, 和下采样到对应尺度的GT label m_l:

    像素级余弦相似度:
        sim(i,j) = cos(f1_l[:,i,j], f2_l[:,i,j])   ∈ [-1, 1]

    未变化区域约束 (Relaxed Pull):
        L_pull = mean( max(0, θ - sim) * (1 - m_l) )
        → 只要求未变化区域sim > θ (默认0.5), 不强制sim→1.0
        → 容忍T1/T2间的成像条件差异 (光照/季节/传感器)

    变化区域约束 (Push):
        L_push = mean( max(0, sim - τ) * m_l )
        → 迫使变化区域的特征不相似, τ为margin (默认-0.3)

    总损失:
        L_fccr = Σ_l [ λ_pull * L_pull + λ_push * L_push ]

设计细节:
- 使用cosine similarity而非L2距离: 对特征尺度不敏感, 更稳定
- pull_threshold θ = 0.5: 宽松一致性, 容忍成像差异导致的特征偏移
- push_margin τ = -0.3: 不要求变化区域特征完全相反, 只要求不相似
- 多尺度: 在所有4个金字塔层级施加约束, 低层(细节)和高层(语义)都有监督
- 权重warmup: 前5个epoch逐步增加FCCR权重, 避免干扰初期学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCCRLoss(nn.Module):
    """
    Feature Consistency Contrastive Regularization Loss

    在金字塔特征空间施加像素级对比学习约束,
    提升未变化区域的特征一致性和变化区域的特征可分辨性.
    """

    def __init__(
        self,
        pull_weight=1.0,
        push_weight=1.0,
        push_margin=-0.3,
        pull_threshold=0.5,
        warmup_epochs=5,
    ):
        """
        Args:
            pull_weight: 未变化区域pull loss的权重
            push_weight: 变化区域push loss的权重
            push_margin: push loss的margin阈值. sim < margin时不产生loss.
                         默认-0.3, 即只惩罚sim > -0.3的变化像素
            pull_threshold: pull loss的宽松阈值. sim > threshold时不产生loss.
                           默认0.5, 容忍T1/T2成像差异导致的特征偏移.
                           设为1.0等价于原始严格约束 (1-sim).
            warmup_epochs: FCCR权重warmup轮数 (前N轮线性增加)
        """
        super().__init__()
        self.pull_weight = pull_weight
        self.push_weight = push_weight
        self.push_margin = push_margin
        self.pull_threshold = pull_threshold
        self.warmup_epochs = warmup_epochs

    def forward(self, pyramid1, pyramid2, label, current_epoch=0):
        """
        Args:
            pyramid1: list of [B, C, H_l, W_l] T1金字塔特征
            pyramid2: list of [B, C, H_l, W_l] T2金字塔特征
            label:    [B, H, W] GT标签 (0=未变化, 1=变化), long类型
            current_epoch: 当前epoch, 用于warmup

        Returns:
            fccr_loss: scalar, 总FCCR损失
            loss_dict: dict, 各分量损失值 (用于日志)
        """
        # warmup: 前N轮线性增加权重, 避免干扰主损失的初期优化
        if self.warmup_epochs > 0 and current_epoch < self.warmup_epochs:
            warmup_factor = (current_epoch + 1) / self.warmup_epochs
        else:
            warmup_factor = 1.0

        total_pull = 0.0
        total_push = 0.0
        num_levels = len(pyramid1)

        for l in range(num_levels):
            f1 = pyramid1[l]  # [B, C, H_l, W_l]
            f2 = pyramid2[l]  # [B, C, H_l, W_l]

            H_l, W_l = f1.shape[2], f1.shape[3]

            # 下采样label到当前金字塔尺度
            # label: [B, H, W] → [B, 1, H, W] → interpolate → [B, 1, H_l, W_l]
            label_float = label.float().unsqueeze(1)  # [B, 1, H, W]
            label_l = F.interpolate(
                label_float, size=(H_l, W_l), mode='nearest'
            ).squeeze(1)  # [B, H_l, W_l]

            # 像素级余弦相似度
            # f1, f2: [B, C, H_l, W_l]
            sim = F.cosine_similarity(f1, f2, dim=1, eps=1e-6)  # [B, H_l, W_l]

            # 未变化区域mask
            unchanged_mask = (label_l < 0.5).float()  # [B, H_l, W_l]
            # 变化区域mask
            changed_mask = (label_l >= 0.5).float()    # [B, H_l, W_l]

            # Pull loss: 未变化区域应该相似 → 但只要求sim > threshold
            # 宽松约束: sim > threshold时无惩罚, 容忍成像条件差异
            unchanged_count = unchanged_mask.sum().clamp(min=1.0)
            pull_loss = (
                F.relu(self.pull_threshold - sim) * unchanged_mask
            ).sum() / unchanged_count

            # Push loss: 变化区域应该不相似 → 惩罚 sim > margin
            # 使用hinge loss: max(0, sim - margin)
            changed_count = changed_mask.sum().clamp(min=1.0)
            push_loss = (
                F.relu(sim - self.push_margin) * changed_mask
            ).sum() / changed_count

            total_pull += pull_loss
            total_push += push_loss

        # 平均所有层级
        total_pull = total_pull / num_levels
        total_push = total_push / num_levels

        # 加权求和
        fccr_loss = warmup_factor * (
            self.pull_weight * total_pull + self.push_weight * total_push
        )

        loss_dict = {
            'fccr_total': fccr_loss.item(),
            'fccr_pull': total_pull.item(),
            'fccr_push': total_push.item(),
            'fccr_warmup': warmup_factor,
        }

        return fccr_loss, loss_dict
