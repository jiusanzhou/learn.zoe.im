# Lesson 5: PagedAttention

## 目标

1. 理解 PagedAttention 的内存管理思想
2. 实现 Page 级的 KV Cache 分配器
3. 手写 Paged Attention 计算（纯 PyTorch 模拟）
4. 对比 contiguous vs paged 内存管理

## 核心问题

### 为什么需要 PagedAttention？

Lesson 2-4 的 KV Cache 已经是 slot-based 了，但有个隐藏问题：

```
传统方式（预分配连续内存）：
  请求 A: 预分配 max_seq_len 的 KV 空间
  → 实际只用了 30%，70% 浪费！
  
  多请求时：
  ┌─────────────────────────────────┐
  │ Req A: [■■■■■□□□□□□□□□□□□□□□] │  ← 70% 浪费
  │ Req B: [■■■■■■■■□□□□□□□□□□□□] │  ← 60% 浪费
  │ Req C: [■■□□□□□□□□□□□□□□□□□□] │  ← 90% 浪费
  └─────────────────────────────────┘

PagedAttention（按需分配 page）：
  ┌──────────────────────────────┐
  │ Page 0: [A₀ A₁ A₂ A₃]      │
  │ Page 1: [A₄ B₀ B₁ B₂]      │  ← 不同请求可以共享 page
  │ Page 2: [B₃ B₄ B₅ B₆]      │
  │ Page 3: [B₇ C₀ C₁ □]       │  ← 只有最后一个 page 可能有浪费
  │ Page 4-N: [空闲]             │
  └──────────────────────────────┘
  
  内存利用率: ~95%+ vs ~30%
```

### Page Table

每个请求维护一个 **page table**，记录它的 token 分布在哪些 page 的哪些 slot：

```
Req A page table: [page 0: slot 0-3, page 1: slot 0]
Req B page table: [page 1: slot 1-3, page 2: slot 0-3, page 3: slot 0]
```

Attention kernel 在计算时，按 page table 索引访问 KV，不需要物理连续。

## SGLang 对照

| 概念 | SGLang 实现 |
|------|------------|
| Page 分配 | `PagedTokenToKVPoolAllocator` |
| Page 大小 | `server_args.page_size`（默认 1） |
| Extend 分配 | `alloc_extend()` — prefill 时分配 |
| Decode 分配 | `alloc_decode()` — 跨 page 边界时分配新 page |
| Page 释放 | `free()` — 按 page 粒度释放 |

## 代码

见 `mini_engine/paged_attention.py`
