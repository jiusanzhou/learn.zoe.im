# Lesson 4: Continuous Batching

## 目标

实现真正的 Continuous Batching — 多个请求在同一个 forward pass 中并行推理。

## 核心概念

### Static Batching vs Continuous Batching

```
Static Batching (传统方式):
  ┌─────────────────────────┐
  │ Req A: [tok tok tok tok] │  ← 全部等最长的请求完成
  │ Req B: [tok tok PAD PAD] │  ← B 已完成但等 A
  │ Req C: [tok PAD PAD PAD] │  ← 更多浪费
  └─────────────────────────┘
  
Continuous Batching:
  Step 1: [A₁, B₁, C₁]  ← 三个请求同时 decode
  Step 2: [A₂, B₂]       ← C 完成了，B 也快了
  Step 3: [A₃, D₁]       ← B 完成，新请求 D 插入！
  Step 4: [A₄, D₂, E₁]   ← 又来新请求
```

**关键区别**：请求可以随时加入和退出 batch，不需要对齐。

### Prefill 和 Decode 混合

SGLang 的策略：
1. **Prefill 优先**：新请求先做 prefill（整个 prompt 的 forward）
2. **Decode batch**：所有 running 请求合并成一个 batch 做 decode
3. **混合 chunked prefill**：大 prompt 分块 prefill，和 decode 交替

## SGLang 对照

| 概念 | SGLang 实现 |
|------|------------|
| Batch 管理 | `ScheduleBatch` — 维护 batch 内所有请求的 tensor |
| 调度策略 | `get_next_batch_to_run()` — prefill 优先 |
| Decode batch | `update_running_batch()` — 更新 running batch |
| Forward | `run_batch()` → `model_worker.forward_batch_generation()` |
| 资源检查 | `check_decode_mem()` — 检查 KV Cache 是否够 |
| Retract | `retract_decode()` — OOM 时踢出请求 |

## 代码

见 `mini_engine/batch_engine.py`
