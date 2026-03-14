"""
Lesson 8: HTTP API — OpenAI 兼容服务

把前面所有组件串成一个 HTTP 服务，兼容 OpenAI API 格式。

实现：
- POST /v1/completions        — 文本补全
- POST /v1/chat/completions   — 对话补全
- GET  /v1/models             — 模型列表
- GET  /health                — 健康检查

支持流式 (SSE) 和非流式响应。

对标 SGLang:
- entrypoints/http_server.py → FastAPI 路由
- entrypoints/openai/protocol.py → 请求/响应数据结构
- entrypoints/openai/serving_chat.py → 对话补全
- entrypoints/openai/serving_completions.py → 文本补全
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, asdict
from typing import AsyncGenerator, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from mini_engine.engine import MiniEngine, SamplingParams


# ============================================================
# 第一部分：OpenAI 协议数据结构
# ============================================================
# 对标 SGLang: entrypoints/openai/protocol.py


class CompletionRequest(BaseModel):
    """对标 OpenAI: POST /v1/completions"""
    model: str = "gpt2"
    prompt: Union[str, List[str]] = ""
    max_tokens: int = Field(default=64, alias="max_tokens")
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    """对标 OpenAI: POST /v1/chat/completions"""
    model: str = "gpt2"
    messages: List[ChatMessage]
    max_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    stream: bool = False
    stop: Optional[Union[str, List[str]]] = None


class CompletionChoice(BaseModel):
    index: int = 0
    text: str = ""
    finish_reason: Optional[str] = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: Optional[str] = None


class DeltaMessage(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionStreamChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionStreamChoice]


# ============================================================
# 第二部分：构建 HTTP Server
# ============================================================

def create_app(engine: MiniEngine) -> FastAPI:
    """创建 FastAPI 应用。
    
    对标 SGLang: http_server.py 中的路由注册。
    """
    app = FastAPI(title="Mini Inference Engine", version="0.1.0")
    model_name = "gpt2"
    
    # --- Health ---
    
    @app.get("/health")
    async def health():
        return {"status": "ok"}
    
    # --- Models ---
    
    @app.get("/v1/models")
    async def list_models():
        """对标 OpenAI: GET /v1/models"""
        return {
            "object": "list",
            "data": [{
                "id": model_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mini-engine",
            }]
        }
    
    # --- Completions ---
    
    @app.post("/v1/completions")
    async def completions(request: CompletionRequest):
        """文本补全。对标 SGLang: OpenAIServingCompletion"""
        
        params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_tokens,
        )
        
        prompts = request.prompt if isinstance(request.prompt, list) else [request.prompt]
        
        if request.stream:
            return StreamingResponse(
                _stream_completions(engine, prompts[0], params, model_name),
                media_type="text/event-stream",
            )
        
        # 非流式
        results = []
        for prompt in prompts:
            text = engine.generate(prompt, params)
            results.append(text)
        
        req_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        
        # 获取 token 统计
        finished = list(engine.finished_requests.values())
        last = finished[-1] if finished else None
        
        return CompletionResponse(
            id=req_id,
            created=int(time.time()),
            model=model_name,
            choices=[
                CompletionChoice(
                    index=i,
                    text=text,
                    finish_reason="stop" if last and last.finished_reason == "eos" else "length",
                )
                for i, text in enumerate(results)
            ],
            usage=UsageInfo(
                prompt_tokens=len(last.input_ids) if last else 0,
                completion_tokens=last.num_generated if last else 0,
                total_tokens=(len(last.input_ids) + last.num_generated) if last else 0,
            ),
        )
    
    # --- Chat Completions ---
    
    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """对话补全。对标 SGLang: OpenAIServingChat
        
        简化版：把 messages 拼接成单个 prompt。
        真正的实现需要处理 chat template。
        """
        # 简单的 chat template
        prompt = _format_chat_prompt(request.messages)
        
        params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_new_tokens=request.max_tokens,
        )
        
        req_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        
        if request.stream:
            return StreamingResponse(
                _stream_chat_completions(engine, prompt, params, model_name, req_id),
                media_type="text/event-stream",
            )
        
        # 非流式
        text = engine.generate(prompt, params)
        
        finished = list(engine.finished_requests.values())
        last = finished[-1] if finished else None
        
        return ChatCompletionResponse(
            id=req_id,
            created=int(time.time()),
            model=model_name,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=text),
                    finish_reason="stop" if last and last.finished_reason == "eos" else "length",
                )
            ],
            usage=UsageInfo(
                prompt_tokens=len(last.input_ids) if last else 0,
                completion_tokens=last.num_generated if last else 0,
                total_tokens=(len(last.input_ids) + last.num_generated) if last else 0,
            ),
        )
    
    # --- Stats ---
    
    @app.get("/stats")
    async def stats():
        """引擎状态。"""
        return {
            "waiting": len(engine.waiting_queue),
            "running": len(engine.running_requests),
            "finished": len(engine.finished_requests),
            "total_generated": engine.total_generated_tokens,
            "cache_utilization": engine.allocator.utilization(),
        }
    
    return app


# ============================================================
# 第三部分：流式响应 (SSE)
# ============================================================


def _format_chat_prompt(messages: List[ChatMessage]) -> str:
    """简单的 chat template。
    
    真正的推理引擎用 HF tokenizer 的 apply_chat_template()。
    GPT-2 没有 chat template，我们简单拼接。
    """
    parts = []
    for msg in messages:
        if msg.role == "system":
            parts.append(f"System: {msg.content}\n")
        elif msg.role == "user":
            parts.append(f"User: {msg.content}\n")
        elif msg.role == "assistant":
            parts.append(f"Assistant: {msg.content}\n")
    parts.append("Assistant:")
    return "".join(parts)


async def _stream_completions(
    engine: MiniEngine,
    prompt: str,
    params: SamplingParams,
    model_name: str,
) -> AsyncGenerator[str, None]:
    """流式文本补全 — SSE 格式。
    
    对标 SGLang 的 StreamingResponse:
    每生成一个 token 就发一个 SSE event。
    """
    req_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    
    for chunk in engine.generate_stream(prompt, params):
        data = {
            "id": req_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "text": chunk,
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(data)}\n\n"
    
    # 最后一个 event 带 finish_reason
    data = {
        "id": req_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{
            "index": 0,
            "text": "",
            "finish_reason": "stop",
        }],
    }
    yield f"data: {json.dumps(data)}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_chat_completions(
    engine: MiniEngine,
    prompt: str,
    params: SamplingParams,
    model_name: str,
    req_id: str,
) -> AsyncGenerator[str, None]:
    """流式对话补全 — SSE 格式。"""
    
    # 第一个 chunk: role
    data = ChatCompletionStreamResponse(
        id=req_id,
        created=int(time.time()),
        model=model_name,
        choices=[ChatCompletionStreamChoice(
            delta=DeltaMessage(role="assistant", content=""),
        )],
    )
    yield f"data: {data.model_dump_json()}\n\n"
    
    # 内容 chunks
    for chunk in engine.generate_stream(prompt, params):
        data = ChatCompletionStreamResponse(
            id=req_id,
            created=int(time.time()),
            model=model_name,
            choices=[ChatCompletionStreamChoice(
                delta=DeltaMessage(content=chunk),
            )],
        )
        yield f"data: {data.model_dump_json()}\n\n"
    
    # 结束
    data = ChatCompletionStreamResponse(
        id=req_id,
        created=int(time.time()),
        model=model_name,
        choices=[ChatCompletionStreamChoice(
            delta=DeltaMessage(),
            finish_reason="stop",
        )],
    )
    yield f"data: {data.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


# ============================================================
# 第四部分：启动服务
# ============================================================

def main():
    import uvicorn
    
    print("=" * 60)
    print("Lesson 8: HTTP API — OpenAI Compatible Server")
    print("=" * 60)
    
    engine = MiniEngine(model_name="gpt2", max_cache_slots=2048)
    app = create_app(engine)
    
    print("\nServer ready!")
    print("Endpoints:")
    print("  GET  /health")
    print("  GET  /v1/models")
    print("  POST /v1/completions")
    print("  POST /v1/chat/completions")
    print("  GET  /stats")
    print("\nExample:")
    print('  curl http://localhost:8000/v1/completions \\')
    print('    -H "Content-Type: application/json" \\')
    print('    -d \'{"prompt": "Hello, world", "max_tokens": 20}\'')
    
    uvicorn.run(app, host="0.0.0.0", port=8000)


# ============================================================
# 第五部分：测试（不启动服务器）
# ============================================================

def test_api():
    """不启动 HTTP 服务器，直接测试 API 逻辑。"""
    from fastapi.testclient import TestClient
    
    print("=" * 60)
    print("Lesson 8: HTTP API Tests")
    print("=" * 60)
    
    engine = MiniEngine(model_name="gpt2", max_cache_slots=2048)
    app = create_app(engine)
    client = TestClient(app)
    
    # Test 1: Health
    print("\n--- Test 1: Health ---")
    r = client.get("/health")
    print(f"  Status: {r.status_code}")
    print(f"  Body: {r.json()}")
    
    # Test 2: Models
    print("\n--- Test 2: Models ---")
    r = client.get("/v1/models")
    print(f"  Status: {r.status_code}")
    models = r.json()
    print(f"  Models: {[m['id'] for m in models['data']]}")
    
    # Test 3: Completions (non-stream)
    print("\n--- Test 3: Completions ---")
    r = client.post("/v1/completions", json={
        "prompt": "The future of AI is",
        "max_tokens": 20,
        "temperature": 0,
    })
    print(f"  Status: {r.status_code}")
    data = r.json()
    print(f"  Text: {data['choices'][0]['text']}")
    print(f"  Usage: {data['usage']}")
    
    # Test 4: Chat Completions (non-stream)
    print("\n--- Test 4: Chat Completions ---")
    r = client.post("/v1/chat/completions", json={
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is Python?"},
        ],
        "max_tokens": 25,
        "temperature": 0,
    })
    print(f"  Status: {r.status_code}")
    data = r.json()
    print(f"  Reply: {data['choices'][0]['message']['content']}")
    
    # Test 5: Streaming Completions
    print("\n--- Test 5: Streaming Completions ---")
    r = client.post("/v1/completions", json={
        "prompt": "Once upon a time",
        "max_tokens": 15,
        "temperature": 0.7,
        "stream": True,
    })
    print(f"  Status: {r.status_code}")
    print(f"  Stream: ", end="")
    for line in r.iter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            text = chunk["choices"][0]["text"]
            print(text, end="", flush=True)
    print()
    
    # Test 6: Streaming Chat
    print("\n--- Test 6: Streaming Chat ---")
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 15,
        "stream": True,
    })
    print(f"  Stream: ", end="")
    for line in r.iter_lines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunk = json.loads(line[6:])
            delta = chunk["choices"][0]["delta"]
            if "content" in delta and delta["content"]:
                print(delta["content"], end="", flush=True)
    print()
    
    # Test 7: Stats
    print("\n--- Test 7: Stats ---")
    r = client.get("/stats")
    print(f"  {r.json()}")
    
    print("\n" + "=" * 60)
    print("Lesson 8 完成！")
    print("=" * 60)
    print("""
要点总结：
1. OpenAI 兼容 API: /v1/completions, /v1/chat/completions
2. SSE 流式输出: 每个 token 一个 event，实时返回
3. Chat Template: messages → prompt 的转换
4. 请求/响应严格遵循 OpenAI 格式（可直接用 openai SDK）
5. FastAPI + uvicorn 提供异步 HTTP 服务

SGLang 的进阶：
→ apply_chat_template(): 用 tokenizer 的 chat template
→ 多模态支持: 图片/视频/音频输入
→ Tool calling: function call 协议
→ Batch API: 异步批量处理
→ Anthropic/Ollama 兼容: 多种 API 格式

下一步（Lesson 9）：
→ 整合所有组件，构建完整可运行的推理服务
""")


if __name__ == "__main__":
    import sys
    if "--serve" in sys.argv:
        main()
    else:
        test_api()
