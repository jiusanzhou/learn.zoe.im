# Lesson 8: HTTP API — OpenAI 兼容服务

## 目标

1. 实现 OpenAI 兼容的 `/v1/completions` 和 `/v1/chat/completions`
2. 支持流式 (SSE) 和非流式响应
3. 把前面所有组件串成一个可用的 HTTP 服务

## SGLang HTTP Server

```
FastAPI App
├── /v1/completions        → OpenAIServingCompletion
├── /v1/chat/completions   → OpenAIServingChat
├── /v1/models             → 模型列表
├── /health                → 健康检查
└── /generate              → SGLang 原生接口
```

核心流程:
```
HTTP Request
  → Pydantic 验证 (ChatCompletionRequest)
  → 转换为 GenerateReqInput
  → TokenizerManager.generate_request()
  → 等待结果 / SSE 流式返回
  → 包装成 OpenAI 格式响应
```

## 代码

见 `mini_engine/server.py`
