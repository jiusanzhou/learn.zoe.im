# Mini Inference Engine

从零实现一个推理引擎，边学 SGLang 源码边动手。

## 学习路线

通过实现一个精简版推理引擎来理解核心概念。每个阶段对照 SGLang 源码学习。

### Phase 1: 基础 — 能跑起来
- [ ] **Lesson 1**: Transformer Forward Pass — 手写一个最简单的前向推理
- [ ] **Lesson 2**: KV Cache — 理解为什么需要 KV Cache，怎么管理
- [ ] **Lesson 3**: Tokenizer + Sampling — 从 token 到文字的完整链路

### Phase 2: 性能 — 让它快
- [ ] **Lesson 4**: Continuous Batching — 动态 batch 调度
- [ ] **Lesson 5**: PagedAttention — 内存分页管理（vLLM 的核心思想）
- [ ] **Lesson 6**: Radix Tree Cache — SGLang 的前缀缓存

### Phase 3: 工程 — 让它能用
- [ ] **Lesson 7**: Scheduler — 请求调度策略
- [ ] **Lesson 8**: Tensor Parallelism — 多 GPU 并行
- [ ] **Lesson 9**: OpenAI-compatible API — HTTP 服务层

### Phase 4: 进阶 — 让它更快
- [ ] **Lesson 10**: CUDA Graph — 减少 kernel launch 开销
- [ ] **Lesson 11**: Speculative Decoding — 投机采样加速
- [ ] **Lesson 12**: Quantization — 量化推理

## SGLang 源码对照

| 我们的模块 | SGLang 对应 | 路径 |
|-----------|------------|------|
| Engine 入口 | `Engine` | `srt/entrypoints/engine.py` |
| 调度器 | `Scheduler` | `srt/managers/scheduler.py` |
| Batch 管理 | `ScheduleBatch` | `srt/managers/schedule_batch.py` |
| 调度策略 | `SchedulePolicy` | `srt/managers/schedule_policy.py` |
| KV Cache | `RadixCache` | `srt/mem_cache/radix_cache.py` |
| 内存池 | `memory_pool` | `srt/mem_cache/memory_pool.py` |
| 模型执行 | `ModelRunner` | `srt/model_executor/model_runner.py` |
| CUDA Graph | `CudaGraphRunner` | `srt/model_executor/cuda_graph_runner.py` |
| Tokenizer | `TokenizerManager` | `srt/managers/tokenizer_manager.py` |
| Detokenizer | `DetokenizerManager` | `srt/managers/detokenizer_manager.py` |
| Sampling | `sampling_batch_info` | `srt/sampling/sampling_batch_info.py` |
| 投机解码 | `eagle_worker` | `srt/speculative/eagle_worker.py` |
| HTTP API | `http_server` | `srt/entrypoints/http_server.py` |
| OpenAI 适配 | `adapter` | `srt/openai_api/adapter.py` |

## 技术栈

- Python 3.10+
- PyTorch 2.x
- Transformers (仅用于加载模型权重和 tokenizer)

## 目录结构

```
inference-engine/
├── README.md
├── docs/                    # 学习笔记 + 源码分析
│   └── 01-architecture.md   # SGLang 整体架构分析
├── mini_engine/             # 我们的实现
│   ├── __init__.py
│   ├── engine.py            # 引擎入口
│   ├── model.py             # 模型前向
│   ├── kv_cache.py          # KV Cache 管理
│   ├── scheduler.py         # 调度器
│   ├── sampling.py          # 采样策略
│   └── server.py            # HTTP 服务
└── tests/
    └── __init__.py
```

## 运行

```bash
# TODO: Phase 1 完成后补充
```
