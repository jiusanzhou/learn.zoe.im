"""
Lesson 2: KV Cache — 手写内存管理

学习要点：
1. KV Cache 的内存布局：[num_slots, num_heads, head_dim] per layer
2. Slot 分配器：管理哪些 slot 空闲/已用
3. 手写带 KV Cache 的 Attention
4. Prefill vs Decode 的不同处理

对标 SGLang:
- MHATokenToKVPool: 实际的 KV buffer
- TokenToKVPoolAllocator: slot 分配
- ReqToTokenPool: 请求→token 位置映射
"""

import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, GPT2LMHeadModel

from mini_engine.model import sample


# ============================================================
# 第一部分：KV Cache 内存池
# ============================================================
# 对标 SGLang: mem_cache/memory_pool.py → MHATokenToKVPool


class KVCachePool:
    """KV Cache 内存池 — 预分配固定大小的 KV 存储。
    
    SGLang 的 MHATokenToKVPool 为每层分配独立的 k_buffer 和 v_buffer:
        k_buffer[layer] = tensor[num_slots, num_heads, head_dim]
        v_buffer[layer] = tensor[num_slots, num_heads, head_dim]
    
    我们的简化版：
    - 不做 page 对齐（SGLang 有 page_size 参数）
    - 不做多 GPU / TP 切分
    - 不做 FP8 量化存储
    """
    
    def __init__(
        self,
        num_slots: int,         # 总 slot 数（能缓存多少个 token 的 KV）
        num_layers: int,        # 模型层数
        num_heads: int,         # KV head 数量
        head_dim: int,          # 每个 head 的维度
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        self.num_slots = num_slots
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        
        # 预分配 KV buffer — 关键：一次性分配，避免反复 malloc
        # SGLang: self.k_buffer = [torch.zeros(...) for _ in range(layer_num)]
        self.k_buffer: List[torch.Tensor] = []
        self.v_buffer: List[torch.Tensor] = []
        
        for _ in range(num_layers):
            self.k_buffer.append(
                torch.zeros(num_slots, num_heads, head_dim, dtype=dtype, device=device)
            )
            self.v_buffer.append(
                torch.zeros(num_slots, num_heads, head_dim, dtype=dtype, device=device)
            )
        
        mem_bytes = num_slots * num_heads * head_dim * 2 * num_layers * dtype.itemsize
        print(f"KV Cache Pool: {num_slots} slots × {num_layers} layers × "
              f"{num_heads} heads × {head_dim} dim = {mem_bytes/1024/1024:.1f} MB")
    
    def store(self, layer_id: int, slot_indices: torch.Tensor,
              k: torch.Tensor, v: torch.Tensor):
        """把新计算的 K, V 存入指定 slot。
        
        对标 SGLang: MHATokenToKVPool.set_kv_buffer()
        
        Args:
            layer_id: 层索引
            slot_indices: [seq_len] 要写入的 slot 位置
            k: [seq_len, num_heads, head_dim]
            v: [seq_len, num_heads, head_dim]
        """
        self.k_buffer[layer_id][slot_indices] = k
        self.v_buffer[layer_id][slot_indices] = v
    
    def fetch(self, layer_id: int, slot_indices: torch.Tensor
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """从指定 slot 取出缓存的 K, V。
        
        对标 SGLang: MHATokenToKVPool.get_kv_buffer()
        
        Args:
            layer_id: 层索引
            slot_indices: [seq_len] 要读取的 slot 位置
            
        Returns:
            k: [seq_len, num_heads, head_dim]
            v: [seq_len, num_heads, head_dim]
        """
        return (
            self.k_buffer[layer_id][slot_indices],
            self.v_buffer[layer_id][slot_indices],
        )


# ============================================================
# 第二部分：Slot 分配器
# ============================================================
# 对标 SGLang: mem_cache/allocator.py → TokenToKVPoolAllocator


class SlotAllocator:
    """管理 KV Cache slot 的分配和释放。
    
    SGLang 的 TokenToKVPoolAllocator:
    - 维护 free_pages tensor（空闲 slot 列表）
    - alloc(n) → 分配 n 个连续 slot
    - free(indices) → 释放 slot
    
    我们的简化版用 Python list 管理，SGLang 用 torch.Tensor（GPU 上操作更快）。
    """
    
    def __init__(self, num_slots: int):
        self.num_slots = num_slots
        # slot 0 保留作为 padding（SGLang 也这样做）
        self.free_slots = list(range(1, num_slots))
        self.used_count = 0
    
    def alloc(self, num_tokens: int) -> Optional[torch.Tensor]:
        """分配 num_tokens 个 slot。
        
        返回 slot 索引的 tensor，失败返回 None。
        """
        if num_tokens > len(self.free_slots):
            return None  # OOM
        
        allocated = self.free_slots[:num_tokens]
        self.free_slots = self.free_slots[num_tokens:]
        self.used_count += num_tokens
        return torch.tensor(allocated, dtype=torch.long)
    
    def free(self, slot_indices: torch.Tensor):
        """释放 slot。"""
        freed = slot_indices.tolist()
        self.free_slots.extend(freed)
        self.used_count -= len(freed)
    
    def available(self) -> int:
        return len(self.free_slots)
    
    def utilization(self) -> float:
        return self.used_count / (self.num_slots - 1)  # -1 for padding slot


# ============================================================
# 第三部分：请求级 KV Cache 管理
# ============================================================
# 对标 SGLang: mem_cache/memory_pool.py → ReqToTokenPool


class RequestKVManager:
    """管理每个请求的 KV Cache slot 映射。
    
    每个请求维护它占用的 slot 列表。
    支持：
    - 新请求分配（prefill）
    - 追加 token（decode）
    - 请求完成后释放
    """
    
    def __init__(self, kv_pool: KVCachePool, allocator: SlotAllocator):
        self.kv_pool = kv_pool
        self.allocator = allocator
        # request_id → slot indices (tensor)
        self.request_slots: Dict[str, torch.Tensor] = {}
    
    def allocate_prefill(self, request_id: str, num_tokens: int) -> Optional[torch.Tensor]:
        """为 prefill 阶段分配 slot。"""
        slots = self.allocator.alloc(num_tokens)
        if slots is None:
            return None
        self.request_slots[request_id] = slots
        return slots
    
    def allocate_decode(self, request_id: str) -> Optional[int]:
        """为 decode 阶段分配 1 个新 slot。"""
        new_slot = self.allocator.alloc(1)
        if new_slot is None:
            return None
        # 追加到请求的 slot 列表
        self.request_slots[request_id] = torch.cat([
            self.request_slots[request_id], new_slot
        ])
        return new_slot.item()
    
    def get_slots(self, request_id: str) -> torch.Tensor:
        """获取请求的所有 slot。"""
        return self.request_slots[request_id]
    
    def release(self, request_id: str):
        """释放请求的所有 slot。"""
        if request_id in self.request_slots:
            self.allocator.free(self.request_slots[request_id])
            del self.request_slots[request_id]
    
    def status(self):
        """打印当前状态。"""
        print(f"  Active requests: {len(self.request_slots)}")
        print(f"  Used slots: {self.allocator.used_count}")
        print(f"  Free slots: {self.allocator.available()}")
        print(f"  Utilization: {self.allocator.utilization():.1%}")


# ============================================================
# 第四部分：带手写 KV Cache 的推理
# ============================================================

def extract_kv_from_hf_model(
    model: GPT2LMHeadModel,
    input_ids: torch.Tensor,
) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
    """用 HF 模型跑一次 forward，提取每层的 KV。
    
    HuggingFace past_key_values 格式:
    - transformers <5: tuple of (key, value) per layer
    - transformers ≥5: DynamicCache 对象
      - cache.layers[i].keys / .values: [batch, num_heads, seq_len, head_dim]
    
    统一转成 list of (key, value) tuples。
    """
    outputs = model(input_ids=input_ids, use_cache=True)
    logits = outputs.logits
    past_kv = outputs.past_key_values
    
    # 兼容 transformers ≥ 5.x DynamicCache
    if hasattr(past_kv, 'layers'):
        past_kv = [(layer.keys, layer.values) for layer in past_kv.layers]
    
    return logits, past_kv


@torch.no_grad()
def generate_with_manual_kv_cache(
    model: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 30,
    temperature: float = 0.0,
    max_cache_slots: int = 1024,
) -> str:
    """使用我们手写的 KV Cache 管理器进行推理。
    
    流程：
    1. 创建 KV Cache Pool 和 Allocator
    2. Prefill: 跑完整 prompt，把 KV 存入我们的 pool
    3. Decode: 每步只跑 1 个 token，从 pool 取旧 KV + 存新 KV
    
    注意：我们用 HF 模型内部的 past_key_values 来做实际计算，
    但用我们的 pool 来管理存储。这样可以验证内存管理逻辑的正确性。
    真正的推理引擎（如 SGLang）会完全自己管理 KV 存储。
    """
    # 获取模型参数
    config = model.config
    num_layers = config.n_layer
    num_heads = config.n_head
    head_dim = config.n_embd // config.n_head
    
    # 创建我们的 KV Cache 管理器
    kv_pool = KVCachePool(
        num_slots=max_cache_slots,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
        dtype=torch.float32,
        device="cpu",
    )
    allocator = SlotAllocator(max_cache_slots)
    manager = RequestKVManager(kv_pool, allocator)
    
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    seq_len = input_ids.shape[1]
    generated_ids = input_ids[0].tolist()
    
    print(f"Prompt: {prompt!r} ({seq_len} tokens)")
    print(f"Cache pool: {max_cache_slots} slots")
    
    # === Prefill ===
    print("\n--- Prefill ---")
    
    # 分配 slot
    slots = manager.allocate_prefill("req-0", seq_len)
    if slots is None:
        raise RuntimeError("OOM during prefill")
    print(f"  Allocated slots: {slots.tolist()}")
    
    # 跑 forward，获取 KV
    t0 = time.perf_counter()
    logits, past_kv = extract_kv_from_hf_model(model, input_ids)
    t1 = time.perf_counter()
    
    # 把 KV 存入我们的 pool
    for layer_id in range(num_layers):
        k = past_kv[layer_id][0][0]  # [num_heads, seq_len, head_dim]
        v = past_kv[layer_id][1][0]  # [num_heads, seq_len, head_dim]
        # 转置为 [seq_len, num_heads, head_dim]
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        kv_pool.store(layer_id, slots, k, v)
    
    # 采样第一个 token
    next_token_id = sample(logits[0, -1, :], temperature=temperature)
    generated_ids.append(next_token_id)
    
    token_text = tokenizer.decode([next_token_id])
    print(f"  Prefill time: {(t1-t0)*1000:.1f}ms")
    print(f"  First token: {next_token_id} ({token_text!r})")
    manager.status()
    
    # === Decode ===
    print("\n--- Decode ---")
    total_decode_time = 0
    
    # 构建 HF 格式的 past_key_values 从我们的 pool
    def build_past_kv_from_pool(req_slots: torch.Tensor):
        """从我们的 pool 重建 HF 格式的 past_key_values。"""
        cache = DynamicCache()
        for layer_id in range(num_layers):
            k, v = kv_pool.fetch(layer_id, req_slots)
            # [seq_len, num_heads, head_dim] → [1, num_heads, seq_len, head_dim]
            k = k.transpose(0, 1).unsqueeze(0)
            v = v.transpose(0, 1).unsqueeze(0)
            cache.update(k, v, layer_id)
        return cache
    
    for step in range(max_new_tokens - 1):
        # 为新 token 分配 slot
        new_slot = manager.allocate_decode("req-0")
        if new_slot is None:
            print("  [OOM - Cache full!]")
            break
        
        # 从 pool 取出所有已缓存的 KV
        all_slots = manager.get_slots("req-0")
        old_slots = all_slots[:-1]  # 不含新 slot
        past_kv_rebuilt = build_past_kv_from_pool(old_slots)
        
        # 只跑新 token 的 forward
        new_input = torch.tensor([[generated_ids[-1]]])
        
        t0 = time.perf_counter()
        outputs = model(input_ids=new_input, past_key_values=past_kv_rebuilt, use_cache=True)
        t1 = time.perf_counter()
        
        decode_time = t1 - t0
        total_decode_time += decode_time
        
        # 存新 token 的 KV 到 pool
        new_past_kv = outputs.past_key_values
        # 兼容 DynamicCache
        if hasattr(new_past_kv, 'layers'):
            new_past_kv = [(layer.keys, layer.values) for layer in new_past_kv.layers]
        for layer_id in range(num_layers):
            # new KV 是 past_kv + new 的拼接，只取最后一个
            k_new = new_past_kv[layer_id][0][0, :, -1, :]  # [num_heads, head_dim]
            v_new = new_past_kv[layer_id][1][0, :, -1, :]  # [num_heads, head_dim]
            kv_pool.store(layer_id, torch.tensor([new_slot]),
                         k_new.unsqueeze(0), v_new.unsqueeze(0))
        
        # 采样
        next_token_id = sample(outputs.logits[0, -1, :], temperature=temperature)
        generated_ids.append(next_token_id)
        
        token_text = tokenizer.decode([next_token_id])
        print(f"  Step {step+2}: token={next_token_id:5d} | "
              f"text={token_text!r:15s} | "
              f"slot={new_slot:3d} | "
              f"cached={len(all_slots):3d} | "
              f"decode={decode_time*1000:.1f}ms")
        
        if next_token_id == tokenizer.eos_token_id:
            break
    
    print(f"\n  Total decode time: {total_decode_time*1000:.1f}ms")
    print(f"  Avg per decode step: {total_decode_time/(max_new_tokens-1)*1000:.1f}ms")
    
    # 释放
    print("\n--- Cleanup ---")
    manager.release("req-0")
    manager.status()
    
    return tokenizer.decode(generated_ids, skip_special_tokens=True)


# ============================================================
# 第五部分：多请求并发 — 展示内存管理的价值
# ============================================================

@torch.no_grad()
def demo_concurrent_requests(
    model: GPT2LMHeadModel,
    tokenizer: AutoTokenizer,
    prompts: List[str],
    max_new_tokens: int = 20,
    max_cache_slots: int = 2048,
):
    """演示多请求共享同一个 KV Cache Pool。
    
    这就是推理引擎的核心场景：多个请求共享 GPU 显存。
    - 请求 A 用 slot 1-10
    - 请求 B 用 slot 11-25
    - 请求 A 完成后，释放 slot 1-10，给新请求用
    """
    config = model.config
    num_layers = config.n_layer
    num_heads = config.n_head
    head_dim = config.n_embd // config.n_head
    
    kv_pool = KVCachePool(
        num_slots=max_cache_slots,
        num_layers=num_layers,
        num_heads=num_heads,
        head_dim=head_dim,
    )
    allocator = SlotAllocator(max_cache_slots)
    manager = RequestKVManager(kv_pool, allocator)
    
    print(f"\n{'='*60}")
    print(f"Multi-request demo: {len(prompts)} requests, {max_cache_slots} slots")
    print(f"{'='*60}")
    
    # Prefill all requests
    request_states = {}
    for i, prompt in enumerate(prompts):
        req_id = f"req-{i}"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        seq_len = input_ids.shape[1]
        
        slots = manager.allocate_prefill(req_id, seq_len)
        if slots is None:
            print(f"  {req_id}: OOM during prefill!")
            continue
        
        logits, past_kv = extract_kv_from_hf_model(model, input_ids)
        
        for layer_id in range(num_layers):
            k = past_kv[layer_id][0][0].transpose(0, 1)
            v = past_kv[layer_id][1][0].transpose(0, 1)
            kv_pool.store(layer_id, slots, k, v)
        
        next_token = sample(logits[0, -1, :], temperature=0)
        
        request_states[req_id] = {
            "generated_ids": input_ids[0].tolist() + [next_token],
            "prompt": prompt,
        }
        
        print(f"  {req_id}: prefilled {seq_len} tokens, "
              f"slots {slots[0].item()}-{slots[-1].item()}")
    
    manager.status()
    
    # Decode round-robin (简化版，真正的调度器会 batch 处理)
    print("\n--- Decode (round-robin) ---")
    for step in range(max_new_tokens):
        for req_id in list(request_states.keys()):
            state = request_states[req_id]
            
            new_slot = manager.allocate_decode(req_id)
            if new_slot is None:
                print(f"  {req_id}: OOM at step {step}!")
                manager.release(req_id)
                del request_states[req_id]
                continue
            
            all_slots = manager.get_slots(req_id)
            old_slots = all_slots[:-1]
            
            past = DynamicCache()
            for layer_id in range(num_layers):
                k, v = kv_pool.fetch(layer_id, old_slots)
                past.update(
                    k.transpose(0, 1).unsqueeze(0),
                    v.transpose(0, 1).unsqueeze(0),
                    layer_id,
                )
            
            new_input = torch.tensor([[state["generated_ids"][-1]]])
            outputs = model(input_ids=new_input, past_key_values=past, use_cache=True)
            
            decode_past_kv = outputs.past_key_values
            if hasattr(decode_past_kv, 'layers'):
                decode_past_kv = [(layer.keys, layer.values) for layer in decode_past_kv.layers]
            for layer_id in range(num_layers):
                k_new = decode_past_kv[layer_id][0][0, :, -1, :].unsqueeze(0)
                v_new = decode_past_kv[layer_id][1][0, :, -1, :].unsqueeze(0)
                kv_pool.store(layer_id, torch.tensor([new_slot]), k_new, v_new)
            
            next_token = sample(outputs.logits[0, -1, :], temperature=0)
            state["generated_ids"].append(next_token)
            
            if next_token == tokenizer.eos_token_id:
                print(f"  {req_id}: EOS at step {step}")
    
    # Print results and cleanup
    print("\n--- Results ---")
    for req_id, state in request_states.items():
        text = tokenizer.decode(state["generated_ids"], skip_special_tokens=True)
        num_slots = len(manager.get_slots(req_id))
        print(f"\n  [{req_id}] ({num_slots} slots)")
        print(f"  {text}")
        manager.release(req_id)
    
    print("\n--- After cleanup ---")
    manager.status()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Lesson 2: KV Cache — 手写内存管理")
    print("=" * 60)
    
    print("\nLoading GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    
    config = model.config
    print(f"Model: GPT-2 ({config.n_layer} layers, "
          f"{config.n_head} heads, dim={config.n_embd})")
    print(f"KV per token per layer: "
          f"2 × {config.n_head} × {config.n_embd // config.n_head} × 4 bytes = "
          f"{2 * config.n_head * (config.n_embd // config.n_head) * 4} bytes")
    print(f"KV per token all layers: "
          f"{2 * config.n_embd * config.n_layer * 4 / 1024:.1f} KB")
    
    # 实验 1: 单请求，手写 KV Cache
    print("\n" + "=" * 60)
    print("实验 1: 手写 KV Cache 管理器")
    print("=" * 60)
    
    text = generate_with_manual_kv_cache(
        model, tokenizer,
        prompt="The future of artificial intelligence is",
        max_new_tokens=30,
        temperature=0,
        max_cache_slots=256,
    )
    print(f"\nGenerated: {text}")
    
    # 实验 2: 多请求并发
    print("\n" + "=" * 60)
    print("实验 2: 多请求共享 KV Cache Pool")
    print("=" * 60)
    
    demo_concurrent_requests(
        model, tokenizer,
        prompts=[
            "Paris is the capital of",
            "Machine learning is a field of",
            "The best programming language is",
        ],
        max_new_tokens=20,
        max_cache_slots=512,
    )
    
    print("\n" + "=" * 60)
    print("Lesson 2 完成！")
    print("=" * 60)
    print("""
要点总结：
1. KV Cache Pool: 预分配固定大小，避免反复 malloc
2. Slot Allocator: 管理 slot 的分配/释放，类似内存分配器
3. Request Manager: 跟踪每个请求占用的 slot
4. 多请求共享同一个 Pool — 这就是推理引擎管理显存的核心
5. 请求完成后释放 slot，给新请求复用

SGLang 的进阶设计（我们后续实现）：
→ Radix Tree: 不同请求共享相同前缀的 KV Cache（Lesson 6）
→ Paged Allocation: slot 不需要连续（已体现）
→ GPU Tensor 管理: 用 torch.Tensor 而非 Python list（性能）

下一步（Lesson 3）：
→ Tokenizer + Detokenizer + 完整请求链路
""")
