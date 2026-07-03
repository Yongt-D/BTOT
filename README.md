# BTOT: Bi-Temporal Optimal Transport for Change Detection

Official implementation of **"From Appearance Differencing to Correspondence Reasoning:
A Bi-Temporal Optimal-Transport Operator for Change Detection on Vision Foundation Models."**

> **TL;DR** — Almost all deep change-detection (CD) models decide *what changed* with the same
> shallow mechanism: an appearance **difference** between spatially aligned features. This conflates
> photometric dissimilarity with semantic change and, in the worst case, produces **complete misses**
> — an entire changed structure that receives *no activation*. BTOT instead infers change by explicit
> **correspondence reasoning**: a region has changed precisely when it cannot be matched to any
> counterpart in the other acquisition. We realize this as a differentiable, dustbin-augmented optimal
> transport operator, and read change as the **unmatchable transport mass** routed to a learnable
> dustbin — a structured, interpretable change representation rather than an opaque score.

## Highlights

- **Correspondence reasoning, not differencing.** Change is the mass that cannot be transported
  between the two acquisitions, recovered by an entropy-regularized, windowed, dustbin-augmented
  log-domain Sinkhorn solver.
- **Interpretable by construction.** The unmatchable-mass field localizes *why* a region is flagged
  — it has no counterpart — and is directional (appeared / disappeared).
- **Lightweight & CUDA-free.** A drop-in bi-temporal interaction operator (~1.4M parameters) on top
  of a **frozen** DINOv3 vision foundation model; no custom CUDA kernels, no task-specific backbone.
- **Recovers silent misses.** Highest recall on four optical building-CD benchmarks; best single-pass
  IoU on LEVIR-CD.

## Results (single-pass, no test-time augmentation)

| Benchmark   | IoU   | F1    | Precision | Recall |
|-------------|-------|-------|-----------|--------|
| WHU-CD      | 86.60 | 92.82 | 92.87     | 92.76  |
| LEVIR-CD    | 85.47 | 92.17 | 91.21     | 93.15  |
| LEVIR-CD+   | 85.64 | 92.26 | 89.78     | 94.89  |
| S2Looking   | 53.07 | 69.34 | 68.29     | 70.42  |

Over three random seeds on WHU-CD, BTOT attains **86.78 ± 0.85%** IoU (seeds 42/123/456:
86.60 / 86.04 / 87.71), confirming the reported run is representative.

## Method

Bi-temporal images are encoded by a shared, frozen DINOv3-Large backbone (last blocks fine-tuned)
with a lightweight CNN, bidirectional temporal interaction (BTI) on the raw tokens, and hierarchical
feature aggregation (HFA), then fused into a four-level pyramid. At each level the **BTOT operator**
replaces the difference module: it projects and L2-normalizes tokens, partitions them into local
windows, builds a temperature-scaled cosine cost augmented with a learnable dustbin, runs a
log-domain Sinkhorn solver, and reads change as the directional unmatchable mass. A γ-gated bounded
residual merges it with a difference branch (γ initialized so training starts from pure differencing).
A multi-scale feature aggregation (MSFA) decoder produces the final change map.

## Status

The paper is currently **under review**. Full training/evaluation code, configurations, and
pretrained checkpoints will be released here upon acceptance.

## Citation

```bibtex
@article{deng2026btot,
  title   = {From Appearance Differencing to Correspondence Reasoning: A Bi-Temporal
             Optimal-Transport Operator for Change Detection on Vision Foundation Models},
  author  = {Deng, Yongtao and Lei, Dajiang and Zhang, Liping and Peng, Yidong and Li, Weisheng},
  journal = {Under review},
  year    = {2026}
}
```

## License

To be determined upon release.
