# Lesson 2: KV Cache — 理解内存管理

## 目标

1. 理解 KV Cache 的原理和必要性
2. 手写一个简单的 KV Cache 管理器
3. 手写带 KV Cache 的 Attention 前向推理
4. 对比性能：无 Cache vs 有 Cache vs 我们的手写 Cache

## 背景知识

### 为什么需要 KV Cache？

在 Lesson 1 中我们看到，无 KV Cache 时每步都重算整个序列。问题在于 Self-Attention：

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V
```

对于已经生成的 token，它们的 K 和 V 不会变（因为 causal mask 保证后面的 token 不影响前面的）。所以可以把之前算过的 K、V 缓存下来，decode 时只算新 token 的 Q、K、V。

### KV Cache 内存布局

```
没有 KV Cache:
  每步: input_ids[0:N] → 模型 → logits (重算所有 K,V)

有 KV Cache:
  Prefill: input_ids[0:N] → K[0:N], V[0:N] 存入 cache
  Decode:  input_ids[N]   → Q[N] @ concat(K_cache, K[N])
```

### SGLang 的两级内存管理

```
Level 1: ReqToTokenPool
  请求 → token 位置映射 (哪些 slot 属于这个请求)
  
Level 2: TokenToKVPoolAllocator  
  管理 KV Cache 的 slot 分配和释放
  
底层: MHATokenToKVPool (KVCache)
  实际的 GPU tensor，存储每层的 K/V
  k_buffer[layer][slot, head, dim]
  v_buffer[layer][slot, head, dim]
```

这种设计的精妙之处：**slot 是逻辑的，不需要物理连续**。
这就是 PagedAttention 的核心思想——像操作系统的虚拟内存一样管理 KV Cache。

## SGLang 对照

| 我们做的 | SGLang 对应 | 说明 |
|---------|------------|------|
| KVCachePool | `MHATokenToKVPool` | 实际存储 K/V tensor |
| SlotAllocator | `TokenToKVPoolAllocator` | slot 分配/释放 |
| 请求管理 | `ReqToTokenPool` | 请求→slot 映射 |

## 代码

见 `mini_engine/kv_cache.py`
