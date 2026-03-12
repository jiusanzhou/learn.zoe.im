"""
Lesson 3: 完整请求链路 — Engine

把 Lesson 1 (model forward + sampling) 和 Lesson 2 (KV Cache) 串起来，
构建一个完整的单线程推理引擎。

组件：
- Request: 请求数据结构 + 生命周期
- IncrementalDetokenizer: 增量解码
- MiniEngine: 串联所有组件的引擎

对标 SGLang:
- io_struct.py → GenerateReqInput / TokenizedGenerateReqInput
- tokenizer_manager.py → tokenize + 请求分发
- scheduler.py → 调度 + forward + sample
- detokenizer_manager.py → 增量 detokenize
- engine.py → 总入口
"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Dict, Generator, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from mini_engine.kv_cache import KVCachePool, RequestKVManager, SlotAllocator
from mini_engine.model import sample


# ============================================================
# 第一部分：请求数据结构
# ============================================================
# 对标 SGLang: managers/io_struct.py + managers/schedule_batch.py


class RequestStatus(Enum):
    """请求状态。对标 SGLang 的隐式状态管理。"""
    WAITING = "waiting"       # 在等待队列
    PREFILLING = "prefilling" # 正在 prefill
    DECODING = "decoding"     # 正在 decode
    FINISHED = "finished"     # 完成


@dataclass
class SamplingParams:
    """采样参数。对标 SGLang: sampling/sampling_params.py"""
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    max_new_tokens: int = 64
    stop_token_ids: List[int] = field(default_factory=list)


@dataclass
class Request:
    """一个推理请求。
    
    对标 SGLang: managers/schedule_batch.py → Req
    SGLang 的 Req 有大量字段（prefix_indices, output_ids, sampling_params 等），
    我们只保留核心。
    """
    rid: str                                # 请求 ID
    prompt: str                             # 原始文本
    input_ids: List[int] = field(default_factory=list)  # tokenized
    output_ids: List[int] = field(default_factory=list) # 已生成的 token ids
    sampling_params: SamplingParams = field(default_factory=SamplingParams)
    status: RequestStatus = RequestStatus.WAITING
    
    # Timing
    created_time: float = 0.0
    prefill_time: float = 0.0
    first_token_time: float = 0.0
    finish_time: float = 0.0
    
    # Output
    output_text: str = ""
    finished_reason: str = ""
    
    @property
    def seq_len(self) -> int:
        """当前总序列长度（input + output）"""
        return len(self.input_ids) + len(self.output_ids)
    
    @property
    def num_generated(self) -> int:
        return len(self.output_ids)


# ============================================================
# 第二部分：增量 Detokenizer
# ============================================================
# 对标 SGLang: managers/detokenizer_manager.py → DecodeStatus


class IncrementalDetokenizer:
    """增量解码器。
    
    SGLang 的 DetokenizerManager 处理的关键问题：
    1. UTF-8 多字节字符可能被切在中间（surrogate pair）
    2. 需要增量输出，不能每次从头 decode
    3. batch decode 效率更高
    
    简化版：用 HF tokenizer 的增量 decode。
    """
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
        # request_id → (all_token_ids, last_decoded_len)
        self._states: Dict[str, Tuple[List[int], int]] = {}
    
    def init_request(self, rid: str, input_ids: List[int]):
        """初始化请求的 decode 状态。"""
        self._states[rid] = (list(input_ids), len(input_ids))
    
    def decode_new_token(self, rid: str, token_id: int) -> str:
        """增量解码一个新 token，返回新增的文本。
        
        SGLang 用 surr_offset / read_offset 处理 surrogate pair：
        - 先 decode 到 read_offset 的文本
        - 再 decode 到 surr_offset 的文本（安全边界）
        - 新文本 = read_text[len(surr_text):]
        
        我们简化：直接对比前后 decode 结果的差异。
        """
        all_ids, last_len = self._states[rid]
        
        # decode 添加新 token 前的文本
        old_text = self.tokenizer.decode(all_ids[last_len-1:last_len] if last_len > 0 else [],
                                          skip_special_tokens=True)
        
        all_ids.append(token_id)
        
        # decode 包含新 token 的文本（多取一个旧 token 处理边界）
        new_text = self.tokenizer.decode(all_ids[last_len-1:] if last_len > 0 else all_ids,
                                          skip_special_tokens=True)
        
        self._states[rid] = (all_ids, last_len)
        
        # 增量部分
        if new_text.startswith(old_text):
            return new_text[len(old_text):]
        
        # fallback：直接 decode 新 token
        return self.tokenizer.decode([token_id], skip_special_tokens=True)
    
    def get_full_output(self, rid: str) -> str:
        """获取完整的输出文本（不含 prompt）。"""
        all_ids, last_len = self._states[rid]
        # 只 decode output 部分
        output_ids = all_ids[last_len:]
        return self.tokenizer.decode(output_ids, skip_special_tokens=True)
    
    def release(self, rid: str):
        """释放请求的 decode 状态。"""
        self._states.pop(rid, None)


# ============================================================
# 第三部分：Mini Engine — 串联一切
# ============================================================
# 对标 SGLang: entrypoints/engine.py + managers/scheduler.py


class MiniEngine:
    """最简推理引擎。
    
    SGLang 的 Engine 是多进程架构（TokenizerManager / Scheduler / Detokenizer 各一个进程）。
    我们的 MiniEngine 把所有逻辑放在单线程里，专注于理解数据流。
    
    核心循环（对标 Scheduler.event_loop_normal）：
    1. 接收请求 → tokenize → 加入等待队列
    2. 选择下一个 batch（prefill 优先）
    3. 执行 forward + sample
    4. 增量 detokenize
    5. 检查完成条件
    """
    
    def __init__(
        self,
        model_name: str = "gpt2",
        max_cache_slots: int = 4096,
    ):
        print(f"Initializing MiniEngine with {model_name}...")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        
        # Model config
        config = self.model.config
        self.num_layers = config.n_layer
        self.num_heads = config.n_head
        self.head_dim = config.n_embd // config.n_head
        
        if self.tokenizer.eos_token_id is None:
            self.tokenizer.eos_token_id = config.eos_token_id
        
        # KV Cache management (from Lesson 2)
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
        
        # Request queues (对标 Scheduler 的 waiting_queue + running_batch)
        self.waiting_queue: List[Request] = []
        self.running_requests: Dict[str, Request] = {}
        self.finished_requests: Dict[str, Request] = {}
        
        # Stats
        self.total_generated_tokens = 0
        
        print(f"Engine ready. Model: {model_name}, Cache: {max_cache_slots} slots")
    
    # --- 请求管理 ---
    
    def add_request(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
    ) -> str:
        """添加请求。对标 TokenizerManager.generate_request()。
        
        1. Tokenize
        2. 创建 Request 对象
        3. 加入等待队列
        """
        rid = str(uuid.uuid4())[:8]
        
        # Tokenize (SGLang 在 TokenizerManager 里做)
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
    
    # --- 核心推理循环 ---
    
    def _prefill_request(self, req: Request):
        """Prefill 一个请求。
        
        对标 Scheduler 中处理 ForwardMode.EXTEND 的逻辑。
        """
        req.status = RequestStatus.PREFILLING
        
        # 分配 KV Cache slots
        slots = self.kv_manager.allocate_prefill(req.rid, len(req.input_ids))
        if slots is None:
            req.status = RequestStatus.FINISHED
            req.finished_reason = "oom"
            return
        
        # Forward
        input_ids = torch.tensor([req.input_ids])
        outputs = self.model(input_ids=input_ids, use_cache=True)
        
        # 存 KV 到 pool
        past_kv = outputs.past_key_values
        if hasattr(past_kv, 'layers'):
            past_kv = [(layer.keys, layer.values) for layer in past_kv.layers]
        
        for layer_id in range(self.num_layers):
            k = past_kv[layer_id][0][0].transpose(0, 1)  # [seq, heads, dim]
            v = past_kv[layer_id][1][0].transpose(0, 1)
            self.kv_pool.store(layer_id, slots, k, v)
        
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
        
        # Detokenize
        new_text = self.detokenizer.decode_new_token(req.rid, next_token)
        req.output_text += new_text
        
        self.total_generated_tokens += 1
    
    def _decode_step(self, req: Request):
        """Decode 一步。
        
        对标 Scheduler 中处理 ForwardMode.DECODE 的逻辑。
        """
        # 分配新 slot
        new_slot = self.kv_manager.allocate_decode(req.rid)
        if new_slot is None:
            req.status = RequestStatus.FINISHED
            req.finished_reason = "oom"
            return
        
        # 从 pool 重建 past_key_values
        all_slots = self.kv_manager.get_slots(req.rid)
        old_slots = all_slots[:-1]
        
        cache = DynamicCache()
        for layer_id in range(self.num_layers):
            k, v = self.kv_pool.fetch(layer_id, old_slots)
            cache.update(
                k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0),
                layer_id,
            )
        
        # Forward (只传最新 token)
        new_input = torch.tensor([[req.output_ids[-1]]])
        outputs = self.model(input_ids=new_input, past_key_values=cache, use_cache=True)
        
        # 存新 KV
        new_past = outputs.past_key_values
        if hasattr(new_past, 'layers'):
            new_past = [(layer.keys, layer.values) for layer in new_past.layers]
        
        for layer_id in range(self.num_layers):
            k_new = new_past[layer_id][0][0, :, -1, :].unsqueeze(0)
            v_new = new_past[layer_id][1][0, :, -1, :].unsqueeze(0)
            self.kv_pool.store(layer_id, torch.tensor([new_slot]), k_new, v_new)
        
        # Sample
        logits = outputs.logits[0, -1, :]
        next_token = sample(
            logits,
            temperature=req.sampling_params.temperature,
            top_p=req.sampling_params.top_p,
            top_k=req.sampling_params.top_k,
        )
        
        req.output_ids.append(next_token)
        
        # Detokenize
        new_text = self.detokenizer.decode_new_token(req.rid, next_token)
        req.output_text += new_text
        
        self.total_generated_tokens += 1
    
    def _check_finished(self, req: Request) -> bool:
        """检查请求是否完成。"""
        # EOS
        if req.output_ids[-1] == self.tokenizer.eos_token_id:
            req.finished_reason = "eos"
            return True
        
        # Max tokens
        if req.num_generated >= req.sampling_params.max_new_tokens:
            req.finished_reason = "max_tokens"
            return True
        
        # Stop token ids
        if req.output_ids[-1] in req.sampling_params.stop_token_ids:
            req.finished_reason = "stop_token"
            return True
        
        return False
    
    def _finish_request(self, req: Request):
        """完成请求，释放资源。"""
        req.status = RequestStatus.FINISHED
        req.finish_time = time.perf_counter()
        req.output_text = self.detokenizer.get_full_output(req.rid)
        
        # 释放资源
        self.kv_manager.release(req.rid)
        self.detokenizer.release(req.rid)
        
        # 移动到已完成
        self.finished_requests[req.rid] = req
        if req.rid in self.running_requests:
            del self.running_requests[req.rid]
    
    # --- 主循环 ---
    
    @torch.no_grad()
    def step(self):
        """执行一步调度。
        
        对标 Scheduler.event_loop_normal() 的单次迭代：
        1. 如果有等待的请求 → prefill（优先）
        2. 否则对 running 请求做 decode
        3. 检查完成条件
        """
        # Prefill 优先（SGLang: get_next_batch_to_run 先检查 new_batch）
        if self.waiting_queue:
            req = self.waiting_queue.pop(0)
            self._prefill_request(req)
            if req.status == RequestStatus.DECODING:
                self.running_requests[req.rid] = req
                # 检查是否第一个 token 就触发结束
                if self._check_finished(req):
                    self._finish_request(req)
            return
        
        # Decode running requests（简化版：逐个处理，SGLang 会 batch）
        finished_rids = []
        for rid, req in list(self.running_requests.items()):
            self._decode_step(req)
            if self._check_finished(req):
                finished_rids.append(rid)
        
        for rid in finished_rids:
            self._finish_request(self.running_requests[rid] if rid in self.running_requests else self.finished_requests[rid])
    
    def generate(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
        stream: bool = False,
    ) -> str:
        """同步生成。
        
        对标 Engine.generate()。
        """
        rid = self.add_request(prompt, sampling_params)
        
        while rid not in self.finished_requests:
            self.step()
        
        return self.finished_requests[rid].output_text
    
    def generate_stream(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
    ) -> Generator[str, None, None]:
        """流式生成 — 每生成一个 token 就 yield。
        
        对标 Engine.generate(stream=True)。
        """
        rid = self.add_request(prompt, sampling_params)
        last_text_len = 0
        
        while rid not in self.finished_requests:
            self.step()
            
            # 找到这个请求（可能在 running 或 finished）
            req = self.running_requests.get(rid) or self.finished_requests.get(rid)
            if req and len(req.output_text) > last_text_len:
                new_chunk = req.output_text[last_text_len:]
                last_text_len = len(req.output_text)
                yield new_chunk
    
    def generate_batch(
        self,
        prompts: List[str],
        sampling_params: Optional[SamplingParams] = None,
    ) -> List[str]:
        """批量生成。所有请求共享 KV Cache Pool。"""
        rids = []
        for prompt in prompts:
            rid = self.add_request(prompt, sampling_params)
            rids.append(rid)
        
        # 跑到所有请求完成
        while not all(rid in self.finished_requests for rid in rids):
            self.step()
        
        return [self.finished_requests[rid].output_text for rid in rids]
    
    def status(self):
        """打印引擎状态。"""
        print(f"  Waiting: {len(self.waiting_queue)}")
        print(f"  Running: {len(self.running_requests)}")
        print(f"  Finished: {len(self.finished_requests)}")
        print(f"  Total tokens generated: {self.total_generated_tokens}")
        print(f"  Cache utilization: {self.allocator.utilization():.1%}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Lesson 3: 完整请求链路")
    print("=" * 60)
    
    engine = MiniEngine(model_name="gpt2", max_cache_slots=2048)
    
    # 实验 1: 单请求同步生成
    print("\n" + "=" * 60)
    print("实验 1: 单请求同步生成")
    print("=" * 60)
    
    t0 = time.perf_counter()
    result = engine.generate(
        "The meaning of life is",
        SamplingParams(temperature=0, max_new_tokens=30),
    )
    t1 = time.perf_counter()
    
    print(f"\n  Output: {result}")
    print(f"  Time: {(t1-t0)*1000:.0f}ms")
    engine.status()
    
    # 实验 2: 流式生成
    print("\n" + "=" * 60)
    print("实验 2: 流式生成")
    print("=" * 60)
    
    print("\n  Stream: ", end="", flush=True)
    for chunk in engine.generate_stream(
        "Once upon a time",
        SamplingParams(temperature=0.7, top_p=0.9, max_new_tokens=40),
    ):
        print(chunk, end="", flush=True)
    print()
    
    # 实验 3: 批量生成
    print("\n" + "=" * 60)
    print("实验 3: 批量生成（3 个请求共享 Cache）")
    print("=" * 60)
    
    prompts = [
        "Python is",
        "Rust is",
        "Go is",
    ]
    
    t0 = time.perf_counter()
    results = engine.generate_batch(
        prompts,
        SamplingParams(temperature=0, max_new_tokens=25),
    )
    t1 = time.perf_counter()
    
    for prompt, result in zip(prompts, results):
        print(f"\n  [{prompt}...]")
        print(f"  {result}")
    
    print(f"\n  Total time: {(t1-t0)*1000:.0f}ms")
    engine.status()
    
    # 实验 4: 请求统计
    print("\n" + "=" * 60)
    print("实验 4: 请求统计")
    print("=" * 60)
    
    for rid, req in engine.finished_requests.items():
        ttft = (req.first_token_time - req.created_time) * 1000
        total = (req.finish_time - req.created_time) * 1000
        tps = req.num_generated / (req.finish_time - req.first_token_time) if req.finish_time > req.first_token_time else 0
        print(f"\n  [{req.rid}] {req.prompt[:30]}...")
        print(f"    Tokens: {len(req.input_ids)} in + {req.num_generated} out")
        print(f"    TTFT: {ttft:.0f}ms | Total: {total:.0f}ms | TPS: {tps:.1f}")
        print(f"    Finished: {req.finished_reason}")
    
    print("\n" + "=" * 60)
    print("Lesson 3 完成！")
    print("=" * 60)
    print("""
要点总结：
1. Engine 是所有组件的粘合层：tokenizer + scheduler + KV cache + detokenizer
2. 请求有明确的生命周期：WAITING → PREFILLING → DECODING → FINISHED
3. Prefill 优先策略：新请求尽快获得第一个 token (低 TTFT)
4. 流式输出：每生成一个 token 就可以返回增量文本
5. 批量处理：多请求共享 KV Cache Pool，逐个 prefill + 交替 decode

SGLang 的进阶（后续实现）：
→ Continuous Batching: 多请求同时 forward（GPU 并行）
→ 异步架构: TokenizerManager / Scheduler / Detokenizer 各跑一个进程
→ ZMQ IPC: 进程间高效通信

下一步（Lesson 4）：
→ Continuous Batching — 真正的批处理推理
""")
