# Lesson 3: 完整请求链路 — Tokenize → Infer → Detokenize

## 目标

把前两课的组件串成完整的推理管线：
1. 请求接收和管理
2. Tokenize → 调度 → Forward → Sample → Detokenize
3. 流式输出（增量 detokenize）
4. 多请求排队和生命周期管理

## SGLang 的请求链路

```
HTTP Request
  → GenerateReqInput (io_struct.py)
  → TokenizerManager.generate_request()
    → tokenize
    → 发送 TokenizedGenerateReqInput 给 Scheduler
  → Scheduler.process_input_requests()
    → 加入等待队列
    → 被选入 batch → forward → sample
    → 发送 BatchTokenIDOut 给 Detokenizer
  → DetokenizerManager
    → 增量 decode → BatchStrOutput
    → 返回给 TokenizerManager
  → 返回给客户端
```

## 关键设计

### 增量 Detokenize

SGLang 的 DetokenizerManager 不是等全部生成完再 decode，而是：
- 维护 DecodeStatus（已解码文本 + 偏移量）
- 每收到新 token，增量 decode
- 处理 surrogate pair 边界问题（UTF-8 多字节字符）
- 支持流式输出

### 请求生命周期

```
NEW → WAITING → RUNNING(prefill) → RUNNING(decode) → FINISHED
                                         ↓
                                   (每步生成一个 token)
                                         ↓
                                   (EOS / max_tokens / stop_str)
```

## 代码

见 `mini_engine/engine.py` — 把所有组件串起来的引擎
