"""
Bi-Temporal MixStyle (BTM) - 双时序特征统计混合

基于 MixStyle (Zhou et al., ICLR 2021) 改造为双时序 CD 场景:
训练时以概率 p 随机混合 T1, T2 的通道级 (μ, σ) 统计,
让模型在训练中见过"任意风格组合"的 (T1, T2) 对,
缓解 train-test 风格分布偏移。

测试时模块自动 pass-through, 不做任何操作。

与 v9 BIW 的区别:
- v9 BIW: 确定性白化, 抹掉 style 信号 (但保留 content)
- BTM:    随机扰动, 保留 style 信号但扩大训练域分布

Reference: https://arxiv.org/abs/2104.02008
"""
import random
import torch
import torch.nn as nn


class BiTemporalMixStyle(nn.Module):
    """对 (f1, f2) 一对特征做联合 style 扰动."""

    def __init__(self, p=0.5, alpha=0.1, eps=1e-6, mix_mode='random'):
        """
        Args:
            p: 触发概率 (每个 batch 以 p 概率触发整个 batch 的 mix)
            alpha: Beta(alpha, alpha) 分布参数, 越小越极端 (0.1 推荐)
            eps: 数值稳定
            mix_mode: 'random' 在 batch 内随机配对, 'crossdomain' 强制 T1↔T2 配对
        """
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps
        self.mix_mode = mix_mode
        self._beta = torch.distributions.Beta(alpha, alpha)

    def __repr__(self):
        return f"BiTemporalMixStyle(p={self.p}, alpha={self.alpha}, mode={self.mix_mode})"

    def forward(self, f1, f2):
        """
        Args:
            f1, f2: [B, C, H, W] T1, T2 特征
        Returns:
            f1_aug, f2_aug: style 扰动后的特征 (训练时), 或原样 (eval/概率不触发)
        """
        if not self.training:
            return f1, f2
        if random.random() > self.p:
            return f1, f2

        B = f1.size(0)
        # 计算通道级 (μ, σ) 跨空间维
        mu1 = f1.mean(dim=[2, 3], keepdim=True)
        var1 = f1.var(dim=[2, 3], keepdim=True, unbiased=False)
        sig1 = (var1 + self.eps).sqrt()
        mu2 = f2.mean(dim=[2, 3], keepdim=True)
        var2 = f2.var(dim=[2, 3], keepdim=True, unbiased=False)
        sig2 = (var2 + self.eps).sqrt()

        # 标准化
        f1_norm = (f1 - mu1) / sig1
        f2_norm = (f2 - mu2) / sig2

        # 采样 mix 系数 (per sample, per channel-mean/std)
        lmda = self._beta.sample((B, 1, 1, 1)).to(f1.device)

        if self.mix_mode == 'crossdomain':
            # 直接把 T1, T2 的 style 互相混合
            mu_mix_1 = lmda * mu1 + (1 - lmda) * mu2
            sig_mix_1 = lmda * sig1 + (1 - lmda) * sig2
            mu_mix_2 = lmda * mu2 + (1 - lmda) * mu1
            sig_mix_2 = lmda * sig2 + (1 - lmda) * sig1
        else:  # 'random' - 在 batch 内随机配对
            perm = torch.randperm(B, device=f1.device)
            mu_mix_1 = lmda * mu1 + (1 - lmda) * mu1[perm]
            sig_mix_1 = lmda * sig1 + (1 - lmda) * sig1[perm]
            mu_mix_2 = lmda * mu2 + (1 - lmda) * mu2[perm]
            sig_mix_2 = lmda * sig2 + (1 - lmda) * sig2[perm]

        # 用混合后的 style 重建特征
        f1_aug = f1_norm * sig_mix_1 + mu_mix_1
        f2_aug = f2_norm * sig_mix_2 + mu_mix_2
        return f1_aug, f2_aug
