"""
EMA - Exponential Moving Average for Model Parameters
指数移动平均模型参数

核心原理:
    θ_ema = decay × θ_ema + (1 - decay) × θ_model

在训练过程中, 模型参数 θ_model 因为SGD噪声而波动.
EMA 通过指数加权平均平滑参数轨迹, 得到更稳定的参数估计.

数学分析:
    - 等效窗口大小 ≈ 1/(1-decay)
    - decay=0.999 → 等效平均最近1000步
    - decay=0.9999 → 等效平均最近10000步

实践效果:
    - 分割任务: +0.5~1.0% mIoU (几乎免费的提升)
    - 变化检测: 预期 +0.5~1.0% IoU
    - 无额外训练成本, 仅增加少量内存 (存储一份参数副本)

参考:
    - Polyak 1992: "Acceleration of Stochastic Approximation by Averaging"
    - PyTorch ImageNet examples, timm library, DINO/BYOL自监督学习
"""

import copy
import torch
import torch.nn as nn


class ModelEMA:
    """
    Exponential Moving Average of model parameters.

    Usage:
        ema = ModelEMA(model, decay=0.9998)

        for epoch in range(num_epochs):
            for batch in train_loader:
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()
                ema.update(model)  # 每步更新EMA

            # 验证时使用EMA模型
            ema.apply(model)          # 暂时替换模型参数为EMA参数
            val_metrics = validate(model)
            ema.restore(model)        # 恢复原始模型参数

    Args:
        model: 要跟踪的模型
        decay: EMA衰减率. 推荐0.9998 (100 epochs) 或 0.9999 (200+ epochs)
        warmup_steps: EMA热身步数. 前N步decay从0线性增长到目标值, 避免初始EMA被随机初始化主导.
    """

    def __init__(self, model, decay=0.9998, warmup_steps=2000):
        # 创建模型参数的深拷贝 (EMA参数)
        self.ema_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.ema_params[name] = param.data.clone()

        # 同时跟踪BN的running_mean/running_var (它们不是parameters但影响推理)
        self.ema_buffers = {}
        for name, buf in model.named_buffers():
            if 'running_mean' in name or 'running_var' in name:
                self.ema_buffers[name] = buf.data.clone()

        self.decay = decay
        self.warmup_steps = warmup_steps
        self.num_updates = 0

        # 备份区 (用于apply/restore)
        self.backup_params = {}
        self.backup_buffers = {}

    def _get_decay(self):
        """
        计算当前步的有效decay (含warmup)

        warmup: decay从0线性增长到目标值
        这确保训练初期EMA快速跟上模型变化,
        后期才开始慢慢平滑.
        """
        if self.warmup_steps > 0 and self.num_updates < self.warmup_steps:
            # 线性warmup: 0 → decay
            return min(self.decay, self.num_updates / self.warmup_steps * self.decay)
        return self.decay

    @torch.no_grad()
    def update(self, model):
        """
        更新EMA参数 (每个训练step调用一次)

        θ_ema = d × θ_ema + (1-d) × θ_model
        其中 d = effective_decay
        """
        self.num_updates += 1
        d = self._get_decay()

        # 更新可训练参数
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.ema_params:
                self.ema_params[name].mul_(d).add_(param.data, alpha=1.0 - d)

        # 更新BN统计量
        for name, buf in model.named_buffers():
            if name in self.ema_buffers:
                self.ema_buffers[name].mul_(d).add_(buf.data, alpha=1.0 - d)

    def apply(self, model):
        """
        将EMA参数应用到模型 (验证/测试前调用)
        同时备份当前模型参数以便恢复
        """
        # 备份当前参数
        self.backup_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.ema_params:
                self.backup_params[name] = param.data.clone()
                param.data.copy_(self.ema_params[name])

        # 备份并替换BN统计量
        self.backup_buffers = {}
        for name, buf in model.named_buffers():
            if name in self.ema_buffers:
                self.backup_buffers[name] = buf.data.clone()
                buf.data.copy_(self.ema_buffers[name])

    def restore(self, model):
        """
        恢复模型的原始参数 (验证/测试后调用)
        """
        for name, param in model.named_parameters():
            if name in self.backup_params:
                param.data.copy_(self.backup_params[name])

        for name, buf in model.named_buffers():
            if name in self.backup_buffers:
                buf.data.copy_(self.backup_buffers[name])

        self.backup_params = {}
        self.backup_buffers = {}

    def state_dict(self):
        """保存EMA状态 (用于checkpoint)"""
        return {
            'ema_params': self.ema_params,
            'ema_buffers': self.ema_buffers,
            'decay': self.decay,
            'num_updates': self.num_updates,
            'warmup_steps': self.warmup_steps
        }

    def load_state_dict(self, state_dict):
        """加载EMA状态 (从checkpoint恢复)"""
        self.ema_params = state_dict['ema_params']
        self.ema_buffers = state_dict['ema_buffers']
        self.decay = state_dict['decay']
        self.num_updates = state_dict['num_updates']
        self.warmup_steps = state_dict['warmup_steps']
