"""
BTOT: Bi-Temporal Optimal Transport change mechanism.

核心思想 (区别于一切 difference / attention / SSM 方法的全新归纳偏置):
    "变化 = 双时相特征在最优传输下的不可匹配质量"

传统 CD 用 |F1 - F2| 差分, 在外观变化微弱 / nuisance 干扰时会塌缩成零激活
(这正是 complete-miss 的根源)。BTOT 改问 "T2 能否被 T1 解释":
  - 在局部窗口内对 T1/T2 的 token 做熵正则最优传输 (Sinkhorn) 软匹配;
  - 加一个可学习的 dustbin (垃圾桶, 借鉴 SuperGlue), 无法匹配的质量被吸收到 dustbin;
  - dustbin 质量场 = appeared/disappeared 的建筑变化, 即使差分很弱也能被捕获。

纯 PyTorch 实现 (log-domain Sinkhorn = 矩阵运算), 无自定义 CUDA, Windows 友好。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def log_sinkhorn_dustbin(sim: torch.Tensor, bin_score: torch.Tensor, n_iters: int = 5):
    """Entropic OT with a learnable dustbin (SuperGlue-style), log-domain.

    Args:
        sim: [B, N, N] (已温度缩放的) 相似度, 越大越匹配. N = 窗口内 token 数.
        bin_score: 标量可学习参数, dustbin 的相似度阈值
                   (介于"好匹配"与"坏匹配"之间, 决定不可匹配判定).
        n_iters: Sinkhorn 迭代次数.
    Returns:
        P: [B, N+1, N+1] 传输计划 (概率). 最后一行/列为 dustbin.
    """
    b, m, n = sim.shape
    # 增广: 加 dustbin 行列
    bins = bin_score.expand(b, 1, 1)
    sim_aug = torch.cat(
        [
            torch.cat([sim, bins.expand(b, m, 1)], dim=2),
            torch.cat([bins.expand(b, 1, n), bins.expand(b, 1, 1)], dim=2),
        ],
        dim=1,
    )  # [B, M+1, N+1]

    # 边缘分布 (log): 每个真实 token 质量 1, dustbin 吸收对侧全部 token 的质量
    ms, ns = float(m), float(n)
    norm = -torch.log(torch.tensor(ms + ns, device=sim.device, dtype=sim.dtype))
    log_mu = torch.cat([norm.expand(m), (torch.log(torch.tensor(ns, device=sim.device, dtype=sim.dtype)) + norm).view(1)])
    log_nu = torch.cat([norm.expand(n), (torch.log(torch.tensor(ms, device=sim.device, dtype=sim.dtype)) + norm).view(1)])
    log_mu = log_mu.view(1, m + 1).expand(b, m + 1)
    log_nu = log_nu.view(1, n + 1).expand(b, n + 1)

    u = torch.zeros(b, m + 1, device=sim.device, dtype=sim.dtype)
    v = torch.zeros(b, n + 1, device=sim.device, dtype=sim.dtype)
    for _ in range(n_iters):
        u = log_mu - torch.logsumexp(sim_aug + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(sim_aug + u.unsqueeze(2), dim=1)
    log_P = sim_aug + u.unsqueeze(2) + v.unsqueeze(1)
    return torch.exp(log_P)


def log_sinkhorn_balanced(sim: torch.Tensor, n_iters: int = 5):
    """Entropic OT without a dustbin (ablation): uniform marginals, all mass must be transported.

    Returns:
        P: [B, N, N] 传输计划 (概率).
    """
    b, m, n = sim.shape
    log_mu = torch.full((b, m), -math.log(m), device=sim.device, dtype=sim.dtype)
    log_nu = torch.full((b, n), -math.log(n), device=sim.device, dtype=sim.dtype)
    u = torch.zeros_like(log_mu)
    v = torch.zeros_like(log_nu)
    for _ in range(n_iters):
        u = log_mu - torch.logsumexp(sim + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(sim + u.unsqueeze(2), dim=1)
    return torch.exp(sim + u.unsqueeze(2) + v.unsqueeze(1))


class WindowedDustbinOT(nn.Module):
    """对一个金字塔层, 用窗口化 dustbin-OT 产生变化证据场.

    forward(pre, post) -> change_field [B, 2, H, W]
        通道0 = dustbin 不可匹配质量 (appeared+disappeared)
        通道1 = 传输代价残差 (软变化幅度)

    消融: use_dustbin=False 用无 dustbin 的平衡 OT (通道0 恒为 0, 只剩代价残差);
          learn_bin=False 把 dustbin 分数固定为 0 (不参与学习).
    """

    def __init__(self, channels: int, ot_dim: int = 48, window: int = 8, n_iters: int = 5,
                 use_dustbin: bool = True, learn_bin: bool = True):
        super().__init__()
        self.window = window
        self.n_iters = n_iters
        self.ot_dim = ot_dim
        self.use_dustbin = use_dustbin
        # 投影到低维匹配空间, 双时相共享参数 (对称性)
        self.proj = nn.Sequential(
            nn.Conv2d(channels, ot_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(ot_dim),
        )
        # 温度: 把 cosine([-1,1]) 放大, 让 Sinkhorn 能锐化匹配 (SuperGlue 用 sqrt(d) 量级)
        self.temp = nn.Parameter(torch.tensor(10.0))
        # dustbin 分数 (相似度空间; 初始 0 介于好匹配 >0 与坏匹配 <0 之间)
        if learn_bin:
            self.bin_score = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('bin_score', torch.tensor(0.0))

    def _partition(self, x: torch.Tensor):
        # x: [B, C, H, W] -> windows [B*nW, ws*ws, C], 记录 padding
        b, c, h, w = x.shape
        ws = self.window
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        hp, wp = x.shape[-2:]
        x = x.view(b, c, hp // ws, ws, wp // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).contiguous()  # [B, nH, nW, ws, ws, C]
        x = x.view(-1, ws * ws, c)  # [B*nWindows, N, C]
        return x, (b, hp, wp, pad_h, pad_w)

    def _unpartition(self, tokens: torch.Tensor, meta, ch: int):
        # tokens: [B*nW, ws*ws, ch] -> [B, ch, H, W]
        b, hp, wp, pad_h, pad_w = meta
        ws = self.window
        x = tokens.view(b, hp // ws, wp // ws, ws, ws, ch)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()  # [B, ch, nH, ws, nW, ws]
        x = x.view(b, ch, hp, wp)
        if pad_h or pad_w:
            x = x[:, :, : hp - pad_h, : wp - pad_w]
        return x

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        # OT 在 fp32 下做, 避免 bf16/fp16 Sinkhorn 数值不稳
        f1 = self.proj(pre).float()
        f2 = self.proj(post).float()
        f1 = F.normalize(f1, dim=1)
        f2 = F.normalize(f2, dim=1)

        t1, meta = self._partition(f1)  # [BW, N, d]
        t2, _ = self._partition(f2)
        n = t1.shape[1]

        # cosine 相似度 (token 已 L2 归一化), 温度缩放后喂 Sinkhorn
        raw_sim = torch.bmm(t1, t2.transpose(1, 2))      # [BW, N, N], in [-1,1]
        sim = raw_sim * self.temp.float()
        cost = 1.0 - raw_sim                              # 用于传输代价残差报告

        if self.use_dustbin:
            P = log_sinkhorn_dustbin(sim, self.bin_score.float(), self.n_iters)  # [BW, N+1, N+1]
            # disappeared: T1 token i 的质量被分到 dustbin 列
            disappeared = P[:, :n, n]            # [BW, N]
            # appeared: T2 token j 被 dustbin 行吸收
            appeared = P[:, n, :n]               # [BW, N]
            unmatched = disappeared + appeared   # [BW, N]
            # 传输代价残差: token i 实际匹配质量 * 代价
            cost_res = (P[:, :n, :n] * cost).sum(dim=2)  # [BW, N]
        else:
            P = log_sinkhorn_balanced(sim, self.n_iters)  # [BW, N, N], 无处可"不匹配"
            unmatched = torch.zeros_like(raw_sim[:, :, 0])
            cost_res = (P * cost).sum(dim=2)

        field = torch.stack([unmatched, cost_res], dim=-1)  # [BW, N, 2]
        field = self._unpartition(field, meta, ch=2)        # [B, 2, H, W]
        return field.to(pre.dtype)
