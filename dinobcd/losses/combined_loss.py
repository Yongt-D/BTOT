"""
DinoBCD - Adaptive Loss Functions
自适应损失函数模块

核心创新 - Adaptive Precision-Recall Balance (APRB):
针对变化检测任务中普遍存在的类别不平衡问题，提出一种通用的自适应损失函数框架。
该框架无需针对特定数据集调参，能够自动适应不同的类别分布。

包含：
1. Focal Loss - 处理类别不平衡
2. Dice Loss - 优化IoU
3. Edge Loss - 增强边界 (可选)
4. AdaptiveDiceLoss - 自适应Dice损失，根据batch统计动态调整
5. TverskyLoss - 可控的FP/FN权衡, 直接提升召回率 (新增)
6. DinoBCDLoss - 支持APRB的综合损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


class FocalLoss(nn.Module):
    """
    Focal Loss - 处理类别不平衡
    
    FL = -α(1-p)^γ * log(p)
    
    对于变化检测：
    - alpha 是前景(变化)类的权重，应该 > 0.5 因为变化像素是少数
    - gamma 控制难样本的聚焦程度，越大越关注难样本
    """

    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean', label_smoothing=0.0):
        """
        Args:
            alpha: 前景类(变化)的权重，默认0.75 (因为变化是少数类)
            gamma: 聚焦参数，默认2.0
            label_smoothing: 标签平滑系数，默认0.0 (不平滑)
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, 2, H, W] logits
            targets: [B, H, W] labels (0=背景, 1=变化)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none',
                                   label_smoothing=self.label_smoothing)
        p = torch.exp(-ce_loss)
        
        focal_weight = (1 - p) ** self.gamma
        
        # alpha权重: 前景(变化=1)用alpha, 背景(不变=0)用1-alpha
        alpha_weight = torch.where(
            targets == 1,
            torch.tensor(self.alpha, device=targets.device, dtype=inputs.dtype),
            torch.tensor(1 - self.alpha, device=targets.device, dtype=inputs.dtype)
        )
        
        focal_loss = alpha_weight * focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss - 直接优化Dice/IoU
    
    Loss = 1 - (2*|X∩Y| / (|X|+|Y|))
    """

    def __init__(self, smooth=1.0):  # 增大smooth避免零除
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)[:, 1]
        targets_float = (targets == 1).float()
        
        intersection = (probs * targets_float).sum()
        union = probs.sum() + targets_float.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class EdgeLoss(nn.Module):
    """
    Edge Loss - 边界区域监督
    
    使用Sobel算子提取边界，计算边界BCE Loss
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(-2, -1)
        
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def extract_edge(self, mask):
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        mask = mask.float()

        edge_x = F.conv2d(mask, self.sobel_x, padding=1)
        edge_y = F.conv2d(mask, self.sobel_y, padding=1)
        # 使用数值稳定的sqrt: sqrt(x^2 + y^2 + eps)
        edge = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        return (edge > 0.1).float()

    def forward(self, inputs, targets):
        target_edge = self.extract_edge(targets)
        pred_prob = F.softmax(inputs, dim=1)[:, 1:2]
        pred_edge = self.extract_edge(pred_prob.squeeze(1))

        # 使用AMP安全的损失函数
        # 将pred_edge转换为logits (通过反sigmoid)以使用binary_cross_entropy_with_logits
        # 但由于pred_edge是二值边缘(0或1)，直接使用MSE更稳定且AMP安全
        return F.mse_loss(pred_edge, target_edge, reduction='mean')


class AdaptiveDiceLoss(nn.Module):
    """
    Adaptive Dice Loss - 自适应Dice损失

    核心创新：根据当前batch的预测统计动态调整Dice计算方式，
    避免在极度不平衡数据上过度优化Recall而牺牲Precision。

    动态调整机制：
    - 监控预测的正样本比例 vs 真实正样本比例
    - 当预测过多正样本时（高Recall低Precision倾向），增加对假阳性的惩罚
    """

    def __init__(self, smooth=1.0, adaptive=True):
        super().__init__()
        self.smooth = smooth
        self.adaptive = adaptive

    def forward(self, inputs, targets):
        probs = F.softmax(inputs, dim=1)[:, 1]
        targets_float = (targets == 1).float()

        # 基础Dice计算
        intersection = (probs * targets_float).sum()
        pred_sum = probs.sum()
        target_sum = targets_float.sum()

        if self.adaptive and target_sum > 0:
            # 计算预测/真实比例
            pred_ratio = pred_sum / (targets.numel() + 1e-6)
            target_ratio = target_sum / (targets.numel() + 1e-6)

            # 自适应因子：当预测过多时增加惩罚
            # 如果pred_ratio >> target_ratio，说明假阳性过多
            ratio = pred_ratio / (target_ratio + 1e-6)

            # 使用soft clamping，避免极端值
            adaptive_factor = torch.clamp(ratio, 0.5, 2.0)

            # 调整分母中pred_sum的权重
            union = adaptive_factor * pred_sum + target_sum
        else:
            union = pred_sum + target_sum

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class TverskyLoss(nn.Module):
    """
    Tversky Loss - 直接控制FP/FN的权衡

    数学公式:
    TI(α,β) = TP / (TP + α·FP + β·FN)
    Loss = 1 - TI

    关键参数:
    - α: 假阳性(FP)惩罚权重 (误检)
    - β: 假阴性(FN)惩罚权重 (漏检)
    - 当α < β时, 更重视减少漏检 → 提升Recall
    - 当α > β时, 更重视减少误检 → 提升Precision

    推荐设置:
    - α=0.3, β=0.7: 召回率优先 (适合当前Recall偏低的问题)
    - α=0.5, β=0.5: 等价于Dice Loss
    - α=0.7, β=0.3: 精确率优先

    优势 vs Dice Loss:
    - Dice: 隐式假设FP和FN等权重
    - Tversky: 显式可调, 能直接控制P-R平衡
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        """
        Args:
            inputs: [B, 2, H, W] logits
            targets: [B, H, W] labels (0=背景, 1=变化)
        """
        probs = F.softmax(inputs, dim=1)[:, 1]
        targets_float = (targets == 1).float()

        tp = (probs * targets_float).sum()
        fp = (probs * (1 - targets_float)).sum()
        fn = ((1 - probs) * targets_float).sum()

        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        return 1.0 - tversky_index


class OnlinePRMonitor:
    """
    Online Precision-Recall Monitor
    在线P-R监控器

    核心功能：
    1. 滑动窗口统计训练过程中的P-R指标
    2. 检测P-R失衡状态（如Recall过高Precision过低）
    3. 为自适应损失调整提供信号
    """

    def __init__(self, window_size=100):
        """
        Args:
            window_size: 滑动窗口大小（batch数量）
        """
        self.window_size = window_size
        self.tp_history = deque(maxlen=window_size)
        self.fp_history = deque(maxlen=window_size)
        self.fn_history = deque(maxlen=window_size)

    def update(self, predictions, targets):
        """
        更新统计（使用预测概率，不改变计算图）

        Args:
            predictions: [B, 2, H, W] logits
            targets: [B, H, W] labels
        """
        with torch.no_grad():
            probs = F.softmax(predictions, dim=1)[:, 1]
            pred_mask = (probs > 0.5).long()

            tp = ((pred_mask == 1) & (targets == 1)).sum().item()
            fp = ((pred_mask == 1) & (targets == 0)).sum().item()
            fn = ((pred_mask == 0) & (targets == 1)).sum().item()

            self.tp_history.append(tp)
            self.fp_history.append(fp)
            self.fn_history.append(fn)

    def get_pr_stats(self):
        """
        获取当前窗口的P-R统计

        Returns:
            precision, recall, pr_balance_factor
        """
        if len(self.tp_history) < 10:  # 至少需要一定样本
            return 0.5, 0.5, 1.0

        total_tp = sum(self.tp_history)
        total_fp = sum(self.fp_history)
        total_fn = sum(self.fn_history)

        precision = total_tp / (total_tp + total_fp + 1e-6)
        recall = total_tp / (total_tp + total_fn + 1e-6)

        # P-R平衡因子：用于调整损失权重
        # 当Recall >> Precision时，因子 > 1，增加对假阳性的惩罚
        # 当Precision >> Recall时，因子 < 1，增加对假阴性的惩罚
        if precision > 0 and recall > 0:
            pr_balance = recall / precision
            # Soft clamping to [0.5, 2.0]
            pr_balance_factor = max(0.5, min(2.0, pr_balance))
        else:
            pr_balance_factor = 1.0

        return precision, recall, pr_balance_factor

    def is_recall_biased(self, threshold=1.5):
        """
        检测是否存在Recall偏向（高Recall低Precision）
        """
        _, _, pr_balance = self.get_pr_stats()
        return pr_balance > threshold


class DinoBCDLoss(nn.Module):
    """
    DinoBCD 综合损失函数 - 支持自适应P-R平衡 (APRB)

    核心创新 - Adaptive Precision-Recall Balance (APRB):
    1. 自动根据数据集的类别分布调整focal_alpha（通过外部DACB模块）
    2. 使用自适应Dice Loss，动态调整对假阳性的惩罚
    3. 在线监控P-R平衡，检测训练异常

    这是一个通用框架，无需针对特定数据集手动调参。
    """

    def __init__(
        self,
        focal_alpha=0.75,
        focal_gamma=2.0,
        use_edge_loss=True,
        edge_loss_weight=0.2,
        dice_weight=1.0,
        focal_weight=1.0,
        # APRB相关参数
        use_adaptive_dice=True,  # 使用自适应Dice Loss
        use_pr_monitor=True,     # 启用P-R监控
        pr_window_size=100,      # P-R监控窗口大小
        # Tversky Loss参数
        use_tversky=False,       # 启用Tversky Loss (替代Dice)
        tversky_alpha=0.3,       # FP惩罚权重
        tversky_beta=0.7,        # FN惩罚权重 (>alpha → 提升Recall)
        tversky_weight=1.0,      # Tversky损失权重
        label_smoothing=0.0,     # 标签平滑 (减少过拟合)
        ohem_ratio=0.0           # ⭐ OHEM: 只对最难的K%像素计算focal loss (0=关闭)
    ):
        super().__init__()

        self.focal_alpha = focal_alpha
        self.label_smoothing = label_smoothing
        self.ohem_ratio = ohem_ratio
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma,
                                     label_smoothing=label_smoothing)
        if label_smoothing > 0:
            print(f"[Loss] Label Smoothing: {label_smoothing}")
        if ohem_ratio > 0:
            print(f"[Loss] ⭐ OHEM: 保留最难的 {ohem_ratio*100:.0f}% 像素计算Focal Loss")

        # 选择Dice/Tversky Loss类型
        self.use_tversky = use_tversky
        self.use_adaptive_dice = use_adaptive_dice

        if use_tversky:
            # Tversky Loss: 直接控制P-R平衡
            self.tversky_loss = TverskyLoss(alpha=tversky_alpha, beta=tversky_beta)
            self.tversky_weight = tversky_weight
            # 同时保留轻量Dice作为辅助
            self.dice_loss = DiceLoss()
        else:
            if use_adaptive_dice:
                self.dice_loss = AdaptiveDiceLoss(adaptive=True)
            else:
                self.dice_loss = DiceLoss()

        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

        self.use_edge_loss = use_edge_loss
        self.edge_loss_weight = edge_loss_weight
        if use_edge_loss:
            self.edge_loss = EdgeLoss()

        # P-R监控器
        self.use_pr_monitor = use_pr_monitor
        if use_pr_monitor:
            self.pr_monitor = OnlinePRMonitor(window_size=pr_window_size)

    def forward(self, predictions, targets, current_epoch=0):
        """
        Args:
            predictions: dict with 'refined' and optional 'p1'-'p4'
            targets: [B, H, W] labels
            current_epoch: current training epoch

        Returns:
            total_loss: scalar
            loss_dict: dict with loss components
        """
        loss_dict = {}

        # 主预测损失
        if isinstance(predictions, dict):
            logits = predictions['refined']
        else:
            logits = predictions

        # 更新P-R监控
        if self.use_pr_monitor:
            self.pr_monitor.update(logits, targets)
            precision, recall, pr_balance = self.pr_monitor.get_pr_stats()
            loss_dict['online_precision'] = precision
            loss_dict['online_recall'] = recall
            loss_dict['pr_balance'] = pr_balance

        # 计算各损失分量
        if self.ohem_ratio > 0:
            # OHEM: 计算逐像素focal loss, 只保留最难的K%
            focal = self._ohem_focal(logits, targets)
        else:
            focal = self.focal_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        loss_dict['focal'] = focal.item()
        loss_dict['dice'] = dice.item()

        if self.use_tversky:
            # Tversky模式: Focal + Tversky + 轻量Dice
            tversky = self.tversky_loss(logits, targets)
            loss_dict['tversky'] = tversky.item()

            total_loss = (
                self.focal_weight * focal +
                self.tversky_weight * tversky +
                self.dice_weight * 0.5 * dice  # Dice作为辅助, 权重减半
            )
        else:
            # 原始模式: Focal + Dice (with adaptive)
            # 自适应权重调整
            if self.use_pr_monitor and self.pr_monitor.is_recall_biased(threshold=1.5):
                effective_dice_weight = self.dice_weight * 0.7
                loss_dict['dice_weight_adjusted'] = True
            else:
                effective_dice_weight = self.dice_weight
                loss_dict['dice_weight_adjusted'] = False

            total_loss = self.focal_weight * focal + effective_dice_weight * dice

        # 边界损失
        if self.use_edge_loss:
            edge = self.edge_loss(logits, targets)
            loss_dict['edge'] = edge.item()
            total_loss = total_loss + self.edge_loss_weight * edge

        # ⭐ Deep Supervision: FPN各层独立辅助头的辅助loss
        # 与旧版的区别: 独立1x1头(非共享output_conv) + 递减权重(浅层高/深层低)
        if isinstance(predictions, dict):
            aux_keys = ['aux_p2', 'aux_p3', 'aux_p4']
            aux_weights = [0.4, 0.2, 0.1]  # 浅层权重高: P2最重要(小目标)
            for key, w in zip(aux_keys, aux_weights):
                if key in predictions:
                    aux_logits = predictions[key]
                    aux_focal = self.focal_loss(aux_logits, targets)
                    aux_dice = self.dice_loss(aux_logits, targets)
                    aux_loss = w * (aux_focal + aux_dice)
                    total_loss = total_loss + aux_loss
                    loss_dict[f'{key}_loss'] = aux_loss.item()

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict

    def _ohem_focal(self, logits, targets):
        """Online Hard Example Mining for Focal Loss

        只对最难的K%像素计算loss, 迫使模型关注难样本.
        对小目标漏检(FN)特别有效: 小建筑的变化像素通常是"难像素",
        OHEM确保它们不会被大量简单背景像素的梯度淹没.
        """
        # 计算逐像素focal loss (不做reduction)
        ce_loss = F.cross_entropy(logits, targets, reduction='none',
                                   label_smoothing=self.label_smoothing)
        p = torch.exp(-ce_loss)
        focal_weight = (1 - p) ** self.focal_loss.gamma

        alpha_weight = torch.where(
            targets == 1,
            torch.tensor(self.focal_loss.alpha, device=targets.device, dtype=logits.dtype),
            torch.tensor(1 - self.focal_loss.alpha, device=targets.device, dtype=logits.dtype)
        )

        pixel_loss = alpha_weight * focal_weight * ce_loss  # [B, H, W]

        # 选择最难的K%像素
        pixel_loss_flat = pixel_loss.view(-1)
        num_pixels = pixel_loss_flat.numel()
        num_keep = max(int(num_pixels * self.ohem_ratio), 1)

        # topk选择loss最大的像素
        loss_sorted, _ = pixel_loss_flat.sort(descending=True)
        ohem_loss = loss_sorted[:num_keep].mean()

        return ohem_loss

    def get_pr_status(self):
        """获取当前P-R状态，用于日志记录"""
        if self.use_pr_monitor:
            return self.pr_monitor.get_pr_stats()
        return None, None, None


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("Testing DinoBCDLoss")
    print("=" * 50)
    
    logits = torch.randn(2, 2, 64, 64).to(device)
    targets = torch.randint(0, 2, (2, 64, 64)).to(device)
    
    criterion = DinoBCDLoss().to(device)
    
    predictions = {'refined': logits}
    loss, loss_dict = criterion(predictions, targets)
    
    print(f"Total Loss: {loss.item():.4f}")
    print(f"Loss Dict: {loss_dict}")
