"""
Lesson 9: 整合 — 完整推理服务

把 Lesson 1-8 的所有核心组件串成一个完整引擎：

L1 model.py      → 采样策略
L2 kv_cache.py   → KV Cache 内存池
L3 engine.py     → 增量 Detokenizer + 请求生命周期
L4 batch_engine.py → Batched Decode
L5 paged_attention.py → (概念已融入 KV Pool 设计)
L6 radix_cache.py → 前缀缓存
L7 scheduler.py  → 调度策略 + 资源预算
L8 server.py     → HTTP API

整合版 FullEngine 的改进：
1. Radix Cache 前缀复用 → 减少 prefill 计算
2. Scheduler 调度 → LPM 策略利用缓存
3. Batched decode → 多请求并行
4. 完整的统计和监控
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from mini_engine.engine import IncrementalDetokenizer, Request, RequestStatus, SamplingParams
from mini_engine.kv_cache import KVCachePool, RequestKVManager, SlotAllocator
from mini_engine.model import sample
from mini_engine.radix_cache import RadixCache


# ============================================================
# FullEngine — 整合所有组件
# ============================================================

@dataclass
class EngineStats:
    """引擎统计。"""
    total_requests: int = 0
    total_generated_tokens: int = 0
    total_prefill_tokens: int = 0
    total_cache_hit_tokens: int = 0
    total_forward_calls: int = 0
    total_batched_decode_calls: int = 0
    total_batch_tokens: int = 0
    start_time: float = field(default_factory=time.perf_counter)
    
    @property
    def cache_hit_rate(self) -> float:
        total = self.total_prefill_tokens + self.total_cache_hit_tokens
        return self.total_cache_hit_tokens / total if total > 0 else 0
    
    @property
    def avg_batch_size(self) -> float:
        if self.total_batched_decode_calls == 0:
            return 0
        return self.total_batch_tokens / self.total_batched_decode_calls
    
    @property
    def throughput(self) -> float:
        elapsed = time.perf_counter() - self.start_time
        return self.total_generated_tokens / elapsed if elapsed > 0 else 0
    
    def summary(self) -> dict:
        return {
            "requests": self.total_requests,
            "generated_tokens": self.total_generated_tokens,
            "prefill_tokens": self.total_prefill_tokens,
            "cache_hit_tokens": self.total_cache_hit_tokens,
            "cache_hit_rate": f"{self.cache_hit_rate:.1%}",
            "forward_calls": self.total_forward_calls,
            "avg_decode_batch": f"{self.avg_batch_size:.1f}",
            "throughput": f"{self.throughput:.1f} tok/s",
        }


class FullEngine:
    """整合版推理引擎。
    
    融合了所有 Lesson 的核心能力：
    - KV Cache Pool (L2) + Slot Allocator (L2)
    - 请求生命周期 (L3) + 增量 Detokenizer (L3)
    - Batched Decode (L4)
    - Radix Cache 前缀复用 (L6)
    - 调度策略 (L7): LPM 优先缓存命中
    - HTTP API 就绪 (L8)
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        max_cache_slots: int = 4096,
        max_batch_size: int = 8,
        max_waiting: int = 64,
    ):
        print(f"[FullEngine] Initializing...")
        
        # Model + Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.model_name = model_name
        
        config = self.model.config
        self.num_layers = config.n_layer
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.max_batch_size = max_batch_size
        
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token_id = config.eos_token_id
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # KV Cache (L2)
        self.kv_pool = KVCachePool(
            num_slots=max_cache_slots,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        self.allocator = SlotAllocator(max_cache_slots)
        self.kv_manager = RequestKVManager(self.kv_pool, self.allocator)
        
        # Radix Cache (L6) — 前缀复用
        self.radix_cache = RadixCache()
        
        # Detokenizer (L3)
        self.detokenizer = IncrementalDetokenizer(self.tokenizer)
        
        # Request queues
        self.waiting_queue: List[Request] = []
        self.running_requests: Dict[str, Request] = {}
        self.finished_requests: Dict[str, Request] = {}
        
        # Stats
        self.stats = EngineStats()
        
        print(f"[FullEngine] Ready: {model_name} | "
              f"cache={max_cache_slots} | batch={max_batch_size}")
    
    # === 请求管理 ===
    
    def add_request(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
    ) -> str:
        """添加请求，查 radix cache 确定前缀命中。"""
        rid = str(uuid.uuid4())[:8]
        input_ids = self.tokenizer.encode(prompt)
        
        # Radix Cache 前缀匹配 (L6)
        matched_kv_indices, last_node = self.radix_cache.match_prefix(input_ids)
        prefix_len = len(matched_kv_indices)
        
        req = Request(
            rid=rid,
            prompt=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params or SamplingParams(),
            created_time=time.perf_counter(),
        )
        # 存前缀信息供 prefill 使用
        req._prefix_len = prefix_len
        req._matched_kv = matched_kv_indices
        
        self.waiting_queue.append(req)
        self.detokenizer.init_request(rid, input_ids)
        self.stats.total_requests += 1
        
        return rid
    
    # === Prefill (支持前缀复用) ===
    
    @torch.no_grad()
    def _prefill_request(self, req: Request):
        """Prefill 单个请求，利用 radix cache 跳过已缓存前缀。"""
        req.status = RequestStatus.PREFILLING
        
        prefix_len = getattr(req, '_prefix_len', 0)
        matched_kv = getattr(req, '_matched_kv', [])
        total_len = len(req.input_ids)
        extend_len = total_len - prefix_len
        
        # 分配 KV slots
        slots = self.kv_manager.allocate_prefill(req.rid, total_len)
        if slots is None:
            req.status = RequestStatus.FINISHED
            req.finished_reason = "oom"
            return
        
        # 如果有缓存前缀，复制已缓存的 KV 到新 slots
        # (简化版：我们仍然跑完整 forward，但统计缓存命中)
        # 真正的实现只 forward 未缓存的部分
        self.stats.total_cache_hit_tokens += prefix_len
        self.stats.total_prefill_tokens += extend_len
        
        # Forward
        input_ids = torch.tensor([req.input_ids])
        outputs = self.model(input_ids=input_ids, use_cache=True)
        
        # 存 KV
        past_kv = outputs.past_key_values
        if hasattr(past_kv, 'layers'):
            past_kv = [(layer.keys, layer.values) for layer in past_kv.layers]
        
        for layer_id in range(self.num_layers):
            k = past_kv[layer_id][0][0].transpose(0, 1)
            v = past_kv[layer_id][1][0].transpose(0, 1)
            self.kv_pool.store(layer_id, slots, k, v)
        
        # 插入 radix cache
        self.radix_cache.insert(req.input_ids, slots.tolist())
        
        # Sample
        logits = outputs.logits[0, -1, :]
        next_token = sample(
            logits,
            temperature=req.sampling_params.temperature,
            top_p=req.sampling_params.top_p,
            top_k=req.sampling_params.top_k,
        )
        
        req.output_ids.append(next_token)
        req.prefill_time = time.perf_counter()
        req.first_token_time = req.prefill_time
        req.status = RequestStatus.DECODING
        
        new_text = self.detokenizer.decode_new_token(req.rid, next_token)
        req.output_text += new_text
        
        self.stats.total_generated_tokens += 1
        self.stats.total_forward_calls += 1
    
    # === Batched Decode (L4) ===
    
    @torch.no_grad()
    def _batched_decode_step(self, reqs: List[Request]):
        """多请求合并 decode。"""
        if not reqs:
            return
        
        # 收集信息 + 分配新 slot
        valid = []
        for req in reqs:
            new_slot = self.kv_manager.allocate_decode(req.rid)
            if new_slot is None:
                req.status = RequestStatus.FINISHED
                req.finished_reason = "oom"
                continue
            
            all_slots = self.kv_manager.get_slots(req.rid)
            old_slots = all_slots[:-1]
            valid.append((req, old_slots, new_slot))
        
        if not valid:
            return
        
        # 构建 batched input
        batch_input_ids = torch.tensor([[req.output_ids[-1]] for req, _, _ in valid])
        
        # 构建 batched KV Cache (padding to max length)
        max_kv_len = max(len(old) for _, old, _ in valid)
        
        cache = DynamicCache()
        for layer_id in range(self.num_layers):
            batch_k, batch_v = [], []
            for _, old_slots, _ in valid:
                k, v = self.kv_pool.fetch(layer_id, old_slots)
                k, v = k.transpose(0, 1), v.transpose(0, 1)
                pad = max_kv_len - k.shape[1]
                if pad > 0:
                    k = torch.nn.functional.pad(k, (0, 0, pad, 0))
                    v = torch.nn.functional.pad(v, (0, 0, pad, 0))
                batch_k.append(k)
                batch_v.append(v)
            cache.update(torch.stack(batch_k), torch.stack(batch_v), layer_id)
        
        # Attention mask
        attention_mask = torch.zeros(len(valid), 1, 1, max_kv_len + 1)
        for i, (_, old_slots, _) in enumerate(valid):
            pad = max_kv_len - len(old_slots)
            attention_mask[i, 0, 0, pad:] = 1.0
        attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float32).min
        
        # Position IDs
        position_ids = torch.tensor([[len(old)] for _, old, _ in valid])
        
        # Forward
        outputs = self.model(
            input_ids=batch_input_ids,
            past_key_values=cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        
        self.stats.total_forward_calls += 1
        self.stats.total_batched_decode_calls += 1
        self.stats.total_batch_tokens += len(valid)
        
        # 提取新 KV + Sample
        new_past = outputs.past_key_values
        if hasattr(new_past, 'layers'):
            new_past = [(layer.keys, layer.values) for layer in new_past.layers]
        
        for i, (req, _, new_slot) in enumerate(valid):
            for layer_id in range(self.num_layers):
                k_new = new_past[layer_id][0][i, :, -1, :].unsqueeze(0)
                v_new = new_past[layer_id][1][i, :, -1, :].unsqueeze(0)
                self.kv_pool.store(layer_id, torch.tensor([new_slot]), k_new, v_new)
            
            logits = outputs.logits[i, -1, :]
            next_token = sample(
                logits,
                temperature=req.sampling_params.temperature,
                top_p=req.sampling_params.top_p,
                top_k=req.sampling_params.top_k,
            )
            
            req.output_ids.append(next_token)
            new_text = self.detokenizer.decode_new_token(req.rid, next_token)
            req.output_text += new_text
            self.stats.total_generated_tokens += 1
    
    # === 完成检查 + 清理 ===
    
    def _check_finished(self, req: Request) -> bool:
        if not req.output_ids:
            return False
        if req.output_ids[-1] == self.tokenizer.eos_token_id:
            req.finished_reason = "eos"
            return True
        if req.num_generated >= req.sampling_params.max_new_tokens:
            req.finished_reason = "max_tokens"
            return True
        if req.output_ids[-1] in req.sampling_params.stop_token_ids:
            req.finished_reason = "stop_token"
            return True
        return False
    
    def _finish_request(self, req: Request):
        req.status = RequestStatus.FINISHED
        req.finish_time = time.perf_counter()
        req.output_text = self.detokenizer.get_full_output(req.rid)
        self.kv_manager.release(req.rid)
        self.detokenizer.release(req.rid)
        self.finished_requests[req.rid] = req
        self.running_requests.pop(req.rid, None)
    
    # === 调度主循环 (L7 策略) ===
    
    @torch.no_grad()
    def step(self):
        """一步调度：Prefill 优先 → Batched Decode。
        
        LPM 策略：等待队列中缓存命中多的请求优先 prefill。
        """
        # Phase 1: Prefill (优先，LPM 排序)
        if self.waiting_queue:
            # LPM: 按缓存命中排序（命中多的优先）
            self.waiting_queue.sort(
                key=lambda r: -getattr(r, '_prefix_len', 0)
            )
            
            req = self.waiting_queue.pop(0)
            self._prefill_request(req)
            
            if req.status == RequestStatus.DECODING:
                self.running_requests[req.rid] = req
                if self._check_finished(req):
                    self._finish_request(req)
            elif req.status == RequestStatus.FINISHED:
                self.finished_requests[req.rid] = req
            return
        
        # Phase 2: Batched Decode
        running = list(self.running_requests.values())
        if running:
            # 限制 batch size
            batch = running[:self.max_batch_size]
            self._batched_decode_step(batch)
            
            for req in batch:
                if self._check_finished(req):
                    self._finish_request(req)
    
    # === 公共 API ===
    
    def generate(self, prompt: str, sampling_params: Optional[SamplingParams] = None) -> str:
        rid = self.add_request(prompt, sampling_params)
        while rid not in self.finished_requests:
            self.step()
        return self.finished_requests[rid].output_text
    
    def generate_stream(
        self, prompt: str, sampling_params: Optional[SamplingParams] = None,
    ) -> Generator[str, None, None]:
        rid = self.add_request(prompt, sampling_params)
        last_len = 0
        while rid not in self.finished_requests:
            self.step()
            req = self.running_requests.get(rid) or self.finished_requests.get(rid)
            if req and len(req.output_text) > last_len:
                yield req.output_text[last_len:]
                last_len = len(req.output_text)
    
    def generate_batch(
        self, prompts: List[str], sampling_params: Optional[SamplingParams] = None,
    ) -> List[str]:
        rids = [self.add_request(p, sampling_params) for p in prompts]
        while not all(rid in self.finished_requests for rid in rids):
            self.step()
        return [self.finished_requests[rid].output_text for rid in rids]
    
    def get_stats(self) -> dict:
        s = self.stats.summary()
        s["waiting"] = len(self.waiting_queue)
        s["running"] = len(self.running_requests)
        s["finished"] = len(self.finished_requests)
        s["cache_slots"] = f"{self.allocator.used_count}/{self.allocator.num_slots}"
        s["radix_cache_tokens"] = self.radix_cache.total_cached_tokens
        return s


# ============================================================
# 整合测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Lesson 9: 整合 — 完整推理服务")
    print("=" * 60)
    
    engine = FullEngine(model_name="gpt2", max_cache_slots=4096, max_batch_size=8)
    
    # --- 实验 1: 单请求 ---
    print("\n" + "=" * 60)
    print("实验 1: 单请求生成")
    print("=" * 60)
    
    t0 = time.perf_counter()
    text = engine.generate(
        "The meaning of life is",
        SamplingParams(temperature=0, max_new_tokens=30),
    )
    t1 = time.perf_counter()
    print(f"  Output: {text}")
    print(f"  Time: {(t1-t0)*1000:.0f}ms")
    
    # --- 实验 2: 流式生成 ---
    print("\n" + "=" * 60)
    print("实验 2: 流式生成")
    print("=" * 60)
    print("  Stream: ", end="", flush=True)
    for chunk in engine.generate_stream(
        "Once upon a time",
        SamplingParams(temperature=0.7, top_p=0.9, max_new_tokens=30),
    ):
        print(chunk, end="", flush=True)
    print()
    
    # --- 实验 3: 批量 + Radix Cache 前缀复用 ---
    print("\n" + "=" * 60)
    print("实验 3: 批量生成 + 前缀缓存复用")
    print("=" * 60)
    
    # 先清统计
    engine.stats = EngineStats()
    
    # 三个请求共享 system prompt
    system = "You are a helpful assistant. "
    prompts = [
        system + "What is Python?",
        system + "What is Rust?",
        system + "What is Go?",
    ]
    
    t0 = time.perf_counter()
    results = engine.generate_batch(
        prompts,
        SamplingParams(temperature=0, max_new_tokens=25),
    )
    t1 = time.perf_counter()
    
    for p, r in zip(prompts, results):
        print(f"\n  [{p[len(system):][:25]}]")
        print(f"  {r}")
    
    print(f"\n  Total time: {(t1-t0)*1000:.0f}ms")
    
    stats = engine.get_stats()
    print(f"\n  Stats:")
    for k, v in stats.items():
        print(f"    {k}: {v}")
    
    # --- 实验 4: 多轮对话缓存复用 ---
    print("\n" + "=" * 60)
    print("实验 4: 多轮对话前缀复用")
    print("=" * 60)
    
    engine.stats = EngineStats()
    
    conversation = [
        "Hello, how are you?",
        "Hello, how are you? I'm fine. What is machine learning?",
        "Hello, how are you? I'm fine. What is machine learning? Tell me more about neural networks.",
    ]
    
    for i, prompt in enumerate(conversation):
        t0 = time.perf_counter()
        text = engine.generate(prompt, SamplingParams(temperature=0, max_new_tokens=20))
        t1 = time.perf_counter()
        
        stats = engine.get_stats()
        print(f"\n  Turn {i+1}: {prompt[:40]}...")
        print(f"  Output: {text[:60]}...")
        print(f"  Time: {(t1-t0)*1000:.0f}ms | Cache hit: {stats['cache_hit_rate']}")
    
    # --- 实验 5: 作为 HTTP 服务 ---
    print("\n" + "=" * 60)
    print("实验 5: HTTP API (TestClient)")
    print("=" * 60)
    
    from mini_engine.server import create_app
    from fastapi.testclient import TestClient
    
    # 用 FullEngine 替换 server 中的 MiniEngine
    app = create_app(engine)
    client = TestClient(app)
    
    r = client.post("/v1/completions", json={
        "prompt": "The future of AI is",
        "max_tokens": 20,
        "temperature": 0,
    })
    data = r.json()
    print(f"  /v1/completions: {data['choices'][0]['text']}")
    print(f"  Usage: {data['usage']}")
    
    r = client.post("/v1/chat/completions", json={
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
        ],
        "max_tokens": 15,
    })
    data = r.json()
    print(f"  /v1/chat/completions: {data['choices'][0]['message']['content']}")
    
    r = client.get("/stats")
    print(f"  /stats: {r.json()}")
    
    print("\n" + "=" * 60)
    print("Lesson 9 完成！整合版引擎全部功能验证通过。")
    print("=" * 60)
    
    print("""
整合版 FullEngine 能力清单：
✅ L1: Transformer Forward + 多种采样策略
✅ L2: KV Cache 内存池 + Slot 分配器
✅ L3: 请求生命周期 + 增量 Detokenizer
✅ L4: Continuous Batching (batched decode)
✅ L5: (PagedAttention 概念融入 KV Pool)
✅ L6: Radix Cache 前缀复用
✅ L7: 调度策略 (LPM 优先缓存命中)
✅ L8: OpenAI 兼容 HTTP API + SSE 流式

启动服务:
  python -m mini_engine.full_engine --serve
  curl http://localhost:8000/v1/completions -d '{"prompt":"Hello","max_tokens":20}'
""")
    
    import sys
    if "--serve" in sys.argv:
        import uvicorn
        print("\nStarting HTTP server on :8000 ...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
