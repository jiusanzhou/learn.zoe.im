# SGLang 架构分析

> 对标源码版本：latest main (2026-03)  
> 源码路径：`/Users/zoe/projects/learn.zoe.im/sglang/python/sglang/srt/`

## 1. 整体架构

SGLang Runtime (SRT) 是一个多进程架构的 LLM 推理引擎：

```
                    ┌─────────────────────────────────────────────────┐
                    │                  主进程                          │
                    │  ┌─────────────┐     ┌──────────────────────┐  │
   HTTP Request ──► │  │ HTTP Server │ ──► │  TokenizerManager    │  │
                    │  └─────────────┘     │  - tokenize          │  │
                    │                      │  - 分发请求            │  │
                    │                      │  - 收集结果            │  │
                    │                      └──────────┬───────────┘  │
                    └─────────────────────────────────┼──────────────┘
                                                      │ ZMQ (IPC)
                              ┌────────────────────────┼────────────────────────┐
                              │                        ▼                        │
                    ┌─────────┴──────────┐   ┌────────────────────┐   ┌────────┴─────────┐
                    │  Scheduler (GPU 0) │   │ Scheduler (GPU 1)  │   │  Detokenizer     │
                    │  - 请求调度          │   │  (Tensor Parallel) │   │  - detokenize    │
                    │  - Batch 管理       │   │                    │   │  - 流式输出       │
                    │  - KV Cache 管理    │   │                    │   │                  │
                    │  - 前向推理          │   │                    │   │                  │
                    └────────────────────┘   └────────────────────┘   └──────────────────┘
```

### 三大组件

| 组件 | 进程 | 职责 | 源码 |
|------|------|------|------|
| **TokenizerManager** | 主进程 | Tokenize 输入、分发请求、收集结果 | `managers/tokenizer_manager.py` |
| **Scheduler** | 子进程(每 GPU 一个) | 调度 batch、管理 KV Cache、执行前向推理 | `managers/scheduler.py` |
| **DetokenizerManager** | 子进程 | 将 token ids 转回文字、处理流式输出 | `managers/detokenizer_manager.py` |

### 进程间通信

使用 **ZMQ (ZeroMQ)** 做 IPC，每个进程分配独立端口：
- TokenizerManager → Scheduler: 发送 tokenized 请求
- Scheduler → Detokenizer: 发送生成的 token ids
- Detokenizer → TokenizerManager: 返回 detokenized 文本

## 2. 请求生命周期

一个请求从进入到返回的完整流程：

```
1. HTTP 请求到达
   └─► HTTP Server (FastAPI/uvicorn)

2. TokenizerManager 处理
   ├─ tokenize(prompt) → input_ids
   ├─ 构建 GenerateReqInput
   └─ 通过 ZMQ 发送给 Scheduler

3. Scheduler 接收并调度
   ├─ recv_requests()          # 接收请求
   ├─ process_input_requests() # 验证、入等待队列
   ├─ get_next_batch_to_run()  # 核心：决定下一个 batch
   │   ├─ 尝试合并 prefill batch 到 running batch
   │   ├─ 检查是否有新请求可以 prefill
   │   └─ 否则继续 decode running batch
   └─ run_batch()              # 执行前向推理

4. 模型前向推理 (TpModelWorker)
   ├─ Prefill: 处理完整输入序列，生成 KV Cache
   └─ Decode: 基于 KV Cache 逐 token 生成

5. Detokenizer 处理
   ├─ 收到新 token ids
   ├─ 增量 detokenize
   └─ 返回给 TokenizerManager

6. 返回给客户端
   ├─ 非流式：等全部完成后一次返回
   └─ 流式：通过 SSE 逐步返回
```

## 3. Scheduler 核心循环

Scheduler 是整个引擎的大脑。它的主循环极其简洁：

```python
# scheduler.py: event_loop_normal()
while True:
    # 1. 接收请求
    recv_reqs = self.recv_requests()
    self.process_input_requests(recv_reqs)

    # 2. 决定下一个 batch
    batch = self.get_next_batch_to_run()

    # 3. 执行 batch
    if batch:
        result = self.run_batch(batch)
        self.process_batch_result(batch, result)
```

### 调度策略：Prefill 优先

`get_next_batch_to_run()` 的核心逻辑：

1. **合并上一轮 prefill 到 running batch** — prefill 完成的请求进入持续解码
2. **尝试新 prefill** — 如果等待队列有请求且资源够，创建新 prefill batch
3. **否则 decode** — 继续对 running batch 做 decode

这就是 **Continuous Batching** 的精髓：新请求可以随时插入，不需要等当前 batch 全部完成。

### Overlap 模式

SGLang 还有个 `event_loop_overlap()` 变体，把 CPU 处理（调度、采样后处理）和 GPU 计算重叠，进一步隐藏延迟。

## 4. KV Cache 管理

SGLang 的一大创新是 **RadixTree 前缀缓存**：

```
传统方式：每个请求独立 KV Cache
SGLang：用 Radix Tree 共享公共前缀

例：
  请求A: "The capital of France is"
  请求B: "The capital of France is Paris. The capital of Germany is"
  
  B 可以复用 A 的 KV Cache 前缀！
```

相关源码：
- `mem_cache/radix_cache.py` — Radix Tree 实现
- `mem_cache/memory_pool.py` — GPU 显存池管理
- `mem_cache/hiradix_cache.py` — 分层缓存（扩展到 CPU/磁盘）

## 5. 模型执行

```
ModelRunner (model_executor/model_runner.py)
├─ 加载模型权重
├─ 管理 CUDA Graph（减少 kernel launch 开销）
├─ forward() 统一入口
│   ├─ Prefill 模式：处理完整序列
│   └─ Decode 模式：只处理最新 token
└─ 调用具体模型实现 (models/ 目录)

TpModelWorker (managers/tp_worker.py)
├─ 封装 ModelRunner
├─ 处理 Tensor Parallel 通信
└─ 管理 token → KV Cache 的映射
```

## 6. 关键数据结构

### Req（请求）
```python
# managers/schedule_batch.py
class Req:
    rid: str                    # 请求 ID
    input_ids: List[int]        # 输入 token ids
    sampling_params: ...        # 采样参数 (temperature, top_p 等)
    output_ids: List[int]       # 已生成的 token ids
    prefix_indices: List[int]   # KV Cache 中的前缀位置
    ...
```

### ScheduleBatch（调度批次）
```python
class ScheduleBatch:
    reqs: List[Req]             # 本 batch 的请求
    forward_mode: ForwardMode   # PREFILL / DECODE
    ...
```

### ForwardMode（前向模式）
```python
class ForwardMode:
    PREFILL   # 完整序列前向（首次处理）
    EXTEND    # 扩展序列（追加 prefill）
    DECODE    # 单 token 解码
```

## 7. 我们要实现什么

理解了 SGLang 的架构后，我们的 mini engine 将按以下顺序实现核心功能：

| 阶段 | 目标 | 对应 SGLang |
|------|------|------------|
| **v0.1** | 单请求推理（无 batch，无 cache） | model_runner + 简单 forward |
| **v0.2** | 加 KV Cache（避免重算） | memory_pool 简化版 |
| **v0.3** | Continuous Batching | scheduler 核心循环 |
| **v0.4** | Radix Cache | radix_cache |
| **v0.5** | HTTP API | http_server + openai adapter |

每一步都是可运行的完整版本，逐步接近 SGLang 的完整能力。

---

*下一篇：[02-transformer-forward.md](02-transformer-forward.md) — 手写 Transformer 前向推理*
