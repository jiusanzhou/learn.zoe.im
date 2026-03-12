"""
Lesson 4: Continuous Batching

核心改进：多个请求的 decode 合并成一个 batch 一起 forward。
在 Lesson 3 中，我们逐个处理 decode 请求（串行）。
现在把它们拼成一个 batch，一次 forward 同时处理所有请求。

关键挑战：
1. 不同请求的 KV Cache 长度不同 → 需要 attention mask 或 position ids 对齐
2. 请求随时加入/退出 batch → 动态调整 batch 大小
3. Prefill 和 Decode 的 forward 逻辑不同 → 分开处理

对标 SGLang:
- ScheduleBatch: 管理 batch 内请求的所有 tensor
- get_next_batch_to_run(): prefill 优先调度
- run_batch(): batch forward
- Continuous Batching: 请求动态加入/退出
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Generator, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from mini_engine.engine import (
    IncrementalDetokenizer,
    Request,
    RequestStatus,
    SamplingParams,
)
from mini_engine.kv_cache import KVCachePool, RequestKVManager, SlotAllocator
from mini_engine.model import sample


class BatchEngine:
    """支持 Continuous Batching 的推理引擎。
    
    vs Lesson 3 的 MiniEngine:
    - MiniEngine: 逐个 decode（for req in running: forward(req)）
    - BatchEngine: 合并 decode（forward([req1, req2, req3])）
    
    GPU 利用率从 O(1) 提升到 O(batch_size)。
    
    实现方式：
    GPT-2 的 forward 接受 [batch, seq_len] 的 input_ids，
    我们把多个请求的最新 token 拼成 [batch, 1] 一起跑。
    关键是 past_key_values 要正确地按请求组织。
    
    注意：HuggingFace 原生不支持不同长度的 KV Cache batching。
    真正的推理引擎（SGLang/vLLM）用自定义 attention kernel 解决这个问题。
    我们用 padding + attention_mask 模拟。
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        max_cache_slots: int = 4096,
        max_batch_size: int = 8,
    ):
        print(f"Initializing BatchEngine...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        
        config = self.model.config
        self.num_layers = config.n_layer
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.max_batch_size = max_batch_size
        
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token_id = config.eos_token_id
        # GPT-2 没有 pad token，用 eos 代替
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # KV Cache
        self.kv_pool = KVCachePool(
            num_slots=max_cache_slots,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        self.allocator = SlotAllocator(max_cache_slots)
        self.kv_manager = RequestKVManager(self.kv_pool, self.allocator)
        
        # Detokenizer
        self.detokenizer = IncrementalDetokenizer(self.tokenizer)
        
        # Request management
        self.waiting_queue: List[Request] = []
        self.running_requests: Dict[str, Request] = {}
        self.finished_requests: Dict[str, Request] = {}
        
        # Stats
        self.total_generated_tokens = 0
        self.total_forward_calls = 0
        self.total_batch_tokens = 0  # sum of batch sizes across all forward calls
        
        print(f"  Model: {model_name} | Cache: {max_cache_slots} slots | "
              f"Max batch: {max_batch_size}")
    
    def add_request(self, prompt: str, sampling_params: Optional[SamplingParams] = None) -> str:
        rid = str(uuid.uuid4())[:8]
        input_ids = self.tokenizer.encode(prompt)
        
        req = Request(
            rid=rid,
            prompt=prompt,
            input_ids=input_ids,
            sampling_params=sampling_params or SamplingParams(),
            created_time=time.perf_counter(),
        )
        self.waiting_queue.append(req)
        self.detokenizer.init_request(rid, input_ids)
        return rid
    
    # === Prefill（仍然单个处理，因为不同 prompt 长度差异大）===
    
    @torch.no_grad()
    def _prefill_request(self, req: Request):
        """Prefill 单个请求。
        
        SGLang 也是单独 prefill（或 chunked prefill），
        因为 prefill 是 compute-bound，不同长度的 prompt 很难高效 batch。
        """
        req.status = RequestStatus.PREFILLING
        
        slots = self.kv_manager.allocate_prefill(req.rid, len(req.input_ids))
        if slots is None:
            req.status = RequestStatus.FINISHED
            req.finished_reason = "oom"
            return
        
        input_ids = torch.tensor([req.input_ids])
        outputs = self.model(input_ids=input_ids, use_cache=True)
        
        past_kv = outputs.past_key_values
        if hasattr(past_kv, 'layers'):
            past_kv = [(layer.keys, layer.values) for layer in past_kv.layers]
        
        for layer_id in range(self.num_layers):
            k = past_kv[layer_id][0][0].transpose(0, 1)
            v = past_kv[layer_id][1][0].transpose(0, 1)
            self.kv_pool.store(layer_id, slots, k, v)
        
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
        self.total_generated_tokens += 1
        self.total_forward_calls += 1
        self.total_batch_tokens += len(req.input_ids)
    
    # === Batched Decode — 核心改进 ===
    
    @torch.no_grad()
    def _batched_decode_step(self, reqs: List[Request]):
        """批量 decode — 多个请求一次 forward。
        
        这是 Continuous Batching 的核心。
        
        策略：
        - 把每个请求的最新 token 拼成 [batch_size, 1]
        - 把每个请求的 KV Cache pad 到相同长度
        - 用 attention_mask 屏蔽 padding 部分
        - 一次 forward 得到所有请求的 next token logits
        
        注意：这种 padding 方式有性能浪费（padding 部分也参与计算）。
        SGLang/vLLM 用 FlashAttention + variable-length attention 避免这个问题。
        """
        if not reqs:
            return
        
        batch_size = len(reqs)
        
        # 1. 收集每个请求的信息
        new_tokens = []     # 每个请求的最新 token
        kv_lengths = []     # 每个请求的 KV Cache 长度
        req_slots = []      # 每个请求在 pool 中的 slot 列表
        
        for req in reqs:
            new_tokens.append(req.output_ids[-1])
            
            # 分配新 slot
            new_slot = self.kv_manager.allocate_decode(req.rid)
            if new_slot is None:
                req.status = RequestStatus.FINISHED
                req.finished_reason = "oom"
                continue
            
            all_slots = self.kv_manager.get_slots(req.rid)
            old_slots = all_slots[:-1]  # 不含新分配的 slot
            req_slots.append((old_slots, new_slot))
            kv_lengths.append(len(old_slots))
        
        # 过滤掉 OOM 的请求
        valid_reqs = [(req, slots) for req, slots in zip(reqs, req_slots)
                      if req.status != RequestStatus.FINISHED]
        
        if not valid_reqs:
            return
        
        # 2. 构建 batched input
        batch_input_ids = torch.tensor([[req.output_ids[-1]] for req, _ in valid_reqs])
        
        # 3. 构建 batched KV Cache（padding 到最长）
        max_kv_len = max(len(slots[0]) for _, slots in valid_reqs)
        
        cache = DynamicCache()
        for layer_id in range(self.num_layers):
            # 收集每个请求的 K, V 并 pad
            batch_k = []
            batch_v = []
            for _, (old_slots, _) in valid_reqs:
                k, v = self.kv_pool.fetch(layer_id, old_slots)
                # k: [seq_len, heads, dim] → [heads, seq_len, dim]
                k = k.transpose(0, 1)
                v = v.transpose(0, 1)
                
                # Pad to max_kv_len
                pad_len = max_kv_len - k.shape[1]
                if pad_len > 0:
                    k = torch.nn.functional.pad(k, (0, 0, pad_len, 0))  # left pad
                    v = torch.nn.functional.pad(v, (0, 0, pad_len, 0))
                
                batch_k.append(k)
                batch_v.append(v)
            
            # Stack: [batch, heads, max_kv_len, dim]
            batch_k = torch.stack(batch_k)
            batch_v = torch.stack(batch_v)
            cache.update(batch_k, batch_v, layer_id)
        
        # 4. 构建 attention mask（屏蔽 padding）
        attention_mask = torch.zeros(len(valid_reqs), 1, 1, max_kv_len + 1)
        for i, (_, (old_slots, _)) in enumerate(valid_reqs):
            real_len = len(old_slots)
            pad_len = max_kv_len - real_len
            # 有效位置 = 1，padding = 0
            # [pad_len 个 0, real_len 个 1, 1 个 1(当前 token)]
            attention_mask[i, 0, 0, pad_len:] = 1.0
        
        # 转成 causal mask 格式 (0 = attend, -inf = mask)
        attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float32).min
        
        # 5. Position IDs（每个请求的当前位置不同）
        position_ids = torch.tensor([[len(slots[0])] for _, slots in valid_reqs])
        
        # 6. Batched Forward!
        outputs = self.model(
            input_ids=batch_input_ids,
            past_key_values=cache,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )
        
        self.total_forward_calls += 1
        self.total_batch_tokens += len(valid_reqs)
        
        # 7. 提取每个请求的新 KV 并存入 pool
        new_past = outputs.past_key_values
        if hasattr(new_past, 'layers'):
            new_past = [(layer.keys, layer.values) for layer in new_past.layers]
        
        for i, (req, (old_slots, new_slot)) in enumerate(valid_reqs):
            for layer_id in range(self.num_layers):
                k_new = new_past[layer_id][0][i, :, -1, :].unsqueeze(0)  # [1, heads, dim]
                v_new = new_past[layer_id][1][i, :, -1, :].unsqueeze(0)
                self.kv_pool.store(layer_id, torch.tensor([new_slot]), k_new, v_new)
            
            # Sample
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
            self.total_generated_tokens += 1
    
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
    
    # === 主调度循环 ===
    
    @torch.no_grad()
    def step(self):
        """一步调度。
        
        对标 Scheduler.event_loop_normal():
        1. Prefill 优先 — 新请求立即 prefill
        2. Batched decode — 所有 running 请求合并 forward
        """
        # Phase 1: Prefill 等待中的请求（一次一个）
        if self.waiting_queue:
            req = self.waiting_queue.pop(0)
            self._prefill_request(req)
            if req.status == RequestStatus.DECODING:
                self.running_requests[req.rid] = req
                if self._check_finished(req):
                    self._finish_request(req)
            elif req.status == RequestStatus.FINISHED:
                self.finished_requests[req.rid] = req
            return
        
        # Phase 2: Batched decode
        running = list(self.running_requests.values())
        if running:
            self._batched_decode_step(running)
            
            # Check finished
            for req in list(running):
                if self._check_finished(req):
                    self._finish_request(req)
    
    def generate_batch(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[str]:
        """批量生成。"""
        rids = [self.add_request(p, sampling_params) for p in prompts]
        
        while not all(rid in self.finished_requests for rid in rids):
            self.step()
        
        return [self.finished_requests[rid].output_text for rid in rids]
    
    def stats(self) -> dict:
        return {
            "total_tokens": self.total_generated_tokens,
            "total_forward_calls": self.total_forward_calls,
            "total_batch_tokens": self.total_batch_tokens,
            "avg_batch_size": (self.total_batch_tokens / self.total_forward_calls
                              if self.total_forward_calls > 0 else 0),
            "cache_utilization": self.allocator.utilization(),
        }


# ============================================================
# Main: 对比实验
# ============================================================

if __name__ == "__main__":
    from mini_engine.engine import MiniEngine
    
    print("=" * 60)
    print("Lesson 4: Continuous Batching")
    print("=" * 60)
    
    prompts = [
        "The capital of France is",
        "Machine learning is defined as",
        "The best way to learn programming is",
        "In the year 2050, humanity will",
    ]
    params = SamplingParams(temperature=0, max_new_tokens=25)
    
    # 实验 1: Lesson 3 的 MiniEngine（串行 decode）
    print("\n" + "=" * 60)
    print("实验 1: 串行 Decode (MiniEngine from Lesson 3)")
    print("=" * 60)
    
    engine1 = MiniEngine(model_name="gpt2", max_cache_slots=2048)
    
    t0 = time.perf_counter()
    results1 = engine1.generate_batch(prompts, params)
    t1 = time.perf_counter()
    serial_time = t1 - t0
    
    for p, r in zip(prompts, results1):
        print(f"\n  [{p[:30]}...]")
        print(f"  {r}")
    
    print(f"\n  Total time: {serial_time*1000:.0f}ms")
    print(f"  Total tokens: {engine1.total_generated_tokens}")
    print(f"  Throughput: {engine1.total_generated_tokens/serial_time:.1f} tokens/s")
    
    # 实验 2: BatchEngine（batched decode）
    print("\n" + "=" * 60)
    print("实验 2: Batched Decode (BatchEngine)")
    print("=" * 60)
    
    engine2 = BatchEngine(model_name="gpt2", max_cache_slots=2048)
    
    t0 = time.perf_counter()
    results2 = engine2.generate_batch(prompts, params)
    t1 = time.perf_counter()
    batch_time = t1 - t0
    
    for p, r in zip(prompts, results2):
        print(f"\n  [{p[:30]}...]")
        print(f"  {r}")
    
    stats = engine2.stats()
    print(f"\n  Total time: {batch_time*1000:.0f}ms")
    print(f"  Total tokens: {stats['total_tokens']}")
    print(f"  Forward calls: {stats['total_forward_calls']}")
    print(f"  Avg batch size: {stats['avg_batch_size']:.1f}")
    print(f"  Throughput: {stats['total_tokens']/batch_time:.1f} tokens/s")
    
    # 对比
    print("\n" + "=" * 60)
    print("对比")
    print("=" * 60)
    speedup = serial_time / batch_time if batch_time > 0 else 0
    print(f"  Serial:  {serial_time*1000:.0f}ms | "
          f"{engine1.total_generated_tokens/serial_time:.1f} tok/s")
    print(f"  Batched: {batch_time*1000:.0f}ms | "
          f"{stats['total_tokens']/batch_time:.1f} tok/s")
    print(f"  Speedup: {speedup:.2f}x")
    
    # 实验 3: 动态加入请求（展示 Continuous 特性）
    print("\n" + "=" * 60)
    print("实验 3: Continuous Batching（请求动态加入）")
    print("=" * 60)
    
    engine3 = BatchEngine(model_name="gpt2", max_cache_slots=2048)
    
    # 先加 2 个请求
    rid1 = engine3.add_request("Python is great because", params)
    rid2 = engine3.add_request("Rust is fast because", params)
    
    step_count = 0
    while not all(rid in engine3.finished_requests for rid in [rid1, rid2]):
        engine3.step()
        step_count += 1
        
        # 在第 5 步动态加入新请求
        if step_count == 5:
            rid3 = engine3.add_request("Go is simple because", params)
            print(f"  [Step {step_count}] → New request added! (rid={rid3})")
        
        # 打印当前状态
        running = len(engine3.running_requests)
        waiting = len(engine3.waiting_queue)
        finished = len(engine3.finished_requests)
        if step_count <= 10 or step_count % 5 == 0:
            print(f"  [Step {step_count}] waiting={waiting} running={running} finished={finished}")
    
    # 等第三个也完成
    if 'rid3' in dir():
        while rid3 not in engine3.finished_requests:
            engine3.step()
            step_count += 1
    
    print(f"\n  Total steps: {step_count}")
    for rid in [rid1, rid2] + ([rid3] if 'rid3' in dir() else []):
        req = engine3.finished_requests[rid]
        print(f"  [{req.rid}] {req.output_text[:60]}...")
    
    print("\n" + "=" * 60)
    print("Lesson 4 完成！")
    print("=" * 60)
    print("""
要点总结：
1. Continuous Batching: 请求动态加入/退出 batch
2. Batched Decode: 多请求的 decode 合并成一个 forward call
3. Prefill 仍然单独处理（compute-bound，长度差异大）
4. 我们用 padding + attention_mask 模拟 batching
   → SGLang/vLLM 用 FlashAttention 的 variable-length 避免 padding 浪费
5. 吞吐量随 batch size 增长（GPU 并行度提升）

下一步（Lesson 5）：
→ PagedAttention — 不需要连续内存的 KV Cache 管理
""")
