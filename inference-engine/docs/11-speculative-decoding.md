# Lesson 10: Speculative Decoding — 投机采样加速

## 核心思想

Autoregressive 生成的瓶颈：**每步只生成 1 个 token，GPU 利用率低（memory-bound）。**

Speculative Decoding 的思路：
```
传统 decode (每步 1 token):
  Target Model: [t1] → [t2] → [t3] → [t4] → ...  (4 步)

Speculative Decode (每步可能接受多个):
  Draft Model:  [d1, d2, d3, d4]  ← 小模型快速猜 4 个 token
  Target Model: verify([d1,d2,d3,d4]) → 接受 [d1,d2,d3], 拒绝 d4
  结果: 1 步生成 3 个 token! (3x 加速)
```

## 为什么能加速？

- **Draft model 很小**（如 68M），生成快
- **Target model 验证是并行的** — 一次 forward 同时检查所有 draft token
- **数学保证**: 输出分布与直接用 target model 完全一致（无损）

## SGLang 实现

- `eagle_worker.py` — EAGLE 算法（用 target 的 hidden state 训练 draft head）
- `ngram_worker.py` — N-gram 猜测（不需要额外模型）
- `standalone_worker.py` — 独立 draft model

## 代码

见 `mini_engine/speculative.py`
