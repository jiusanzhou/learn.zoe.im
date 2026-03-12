# Lesson 6: Radix Tree Cache — SGLang 的核心创新

## 目标

1. 理解 Radix Tree（前缀树）用于 KV Cache 共享的原理
2. 实现一个简化版 RadixCache
3. 演示多请求间的前缀复用
4. 实现 LRU 驱逐策略

## 核心问题

### 为什么 PagedAttention 不够？

PagedAttention 解决了内存碎片问题，但没解决**重复计算**：

```
请求 A: "You are a helpful assistant. What is Python?"
请求 B: "You are a helpful assistant. What is Rust?"
请求 C: "You are a helpful assistant. Explain machine learning."

三个请求的 system prompt "You are a helpful assistant." 完全相同，
但 PagedAttention 会为每个请求独立计算和存储这部分的 KV Cache。
```

### Radix Tree 的解决方案

把所有请求的 token 序列组织成一棵前缀树（Radix Tree），
**相同前缀的 KV Cache 只存一份，所有请求共享**：

```
Root
├── "You are a helpful assistant."  [KV Cache: 共享!]
│   ├── " What is Python?"          [Req A 独有]
│   ├── " What is Rust?"            [Req B 独有]  
│   └── " Explain machine learning." [Req C 独有]
└── "The capital of"                 [另一组请求共享]
    ├── " France is"                 
    └── " Germany is"                
```

**效果**：system prompt 的 KV 只算一次，后续请求直接复用。

## SGLang 对照

| 概念 | SGLang 实现 |
|------|------------|
| 树节点 | `TreeNode` — 存 key(token_ids), value(kv_indices), children |
| 前缀匹配 | `match_prefix()` — 找最长公共前缀 |
| 插入 | `insert()` → `_insert_helper()` |
| 节点分裂 | `_split_node()` — 匹配到一半时分裂 |
| 驱逐 | `evict()` — LRU/LFU/FIFO 等策略 |
| 引用计数 | `lock_ref` — 正在使用的节点不能驱逐 |

## 代码

见 `mini_engine/radix_cache.py`
