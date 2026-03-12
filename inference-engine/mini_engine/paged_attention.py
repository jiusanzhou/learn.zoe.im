"""
Lesson 5: PagedAttention

核心改进：按 page 粒度管理 KV Cache，而非预分配整块连续内存。

类比：
- 传统 KV Cache = malloc(max_seq_len) — 预分配，浪费严重
- PagedAttention = mmap/虚拟内存 — 按需分配 page，物理不连续

对标 SGLang:
- PagedTokenToKVPoolAllocator: page 级分配器
- page_size: 每个 page 包含多少 token 的 KV（默认 1）
- alloc_extend/alloc_decode: 智能分配，page 用完才申请新 page
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ============================================================
# 第一部分：Paged KV Cache
# ============================================================


class PagedKVPool:
    """Paged KV Cache 内存池。
    
    与 Lesson 2 的 KVCachePool 区别：
    - KVCachePool: slot 粒度（1 slot = 1 token）
    - PagedKVPool: page 粒度（1 page = page_size tokens）
    
    内存布局:
      k_buffer[layer][page_id * page_size + offset, heads, dim]
    
    物理上是连续 tensor，逻辑上按 page 管理。
    """
    
    def __init__(
        self,
        num_pages: int,
        page_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        self.num_pages = num_pages
        self.page_size = page_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_slots = num_pages * page_size
        
        # 物理存储
        self.k_buffer = [
            torch.zeros(self.total_slots, num_heads, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.v_buffer = [
            torch.zeros(self.total_slots, num_heads, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        
        mem_mb = self.total_slots * num_heads * head_dim * 2 * num_layers * 4 / 1024 / 1024
        print(f"PagedKVPool: {num_pages} pages × {page_size} tokens/page = "
              f"{self.total_slots} slots | {mem_mb:.1f} MB")
    
    def page_to_slots(self, page_id: int) -> torch.Tensor:
        """Page ID → 物理 slot 索引。"""
        start = page_id * self.page_size
        return torch.arange(start, start + self.page_size)
    
    def store(self, layer_id: int, slot_indices: torch.Tensor,
              k: torch.Tensor, v: torch.Tensor):
        self.k_buffer[layer_id][slot_indices] = k
        self.v_buffer[layer_id][slot_indices] = v
    
    def fetch(self, layer_id: int, slot_indices: torch.Tensor):
        return (
            self.k_buffer[layer_id][slot_indices],
            self.v_buffer[layer_id][slot_indices],
        )


class PageAllocator:
    """Page 级分配器。
    
    对标 SGLang: PagedTokenToKVPoolAllocator
    
    管理 page 的分配和释放，而非单个 slot。
    """
    
    def __init__(self, num_pages: int, page_size: int):
        self.num_pages = num_pages
        self.page_size = page_size
        self.free_pages = list(range(num_pages))
        self.used_pages = 0
    
    def alloc_pages(self, num_pages: int) -> Optional[List[int]]:
        """分配 n 个 page。"""
        if num_pages > len(self.free_pages):
            return None
        allocated = self.free_pages[:num_pages]
        self.free_pages = self.free_pages[num_pages:]
        self.used_pages += num_pages
        return allocated
    
    def free_pages_list(self, page_ids: List[int]):
        """释放 page。"""
        self.free_pages.extend(page_ids)
        self.used_pages -= len(page_ids)
    
    def available(self) -> int:
        return len(self.free_pages)
    
    def utilization(self) -> float:
        return self.used_pages / self.num_pages


# ============================================================
# 第二部分：Page Table（请求级 page 映射）
# ============================================================


@dataclass
class PageTable:
    """一个请求的 page table。
    
    记录这个请求的 KV 存在哪些 page 里，以及当前 page 用了多少。
    
    类比操作系统的页表：
    - virtual address (token position) → physical address (page_id * page_size + offset)
    """
    pages: List[int] = field(default_factory=list)       # 已分配的 page ids
    current_offset: int = 0   # 当前 page 内的偏移
    total_tokens: int = 0     # 总 token 数
    
    def get_all_slot_indices(self, page_size: int) -> torch.Tensor:
        """获取所有已用 slot 的物理索引。"""
        if not self.pages:
            return torch.tensor([], dtype=torch.long)
        
        indices = []
        for i, page_id in enumerate(self.pages):
            start = page_id * page_size
            if i < len(self.pages) - 1:
                # 之前的 page 都是满的
                indices.extend(range(start, start + page_size))
            else:
                # 最后一个 page 可能没满
                indices.extend(range(start, start + self.current_offset))
        
        return torch.tensor(indices, dtype=torch.long)
    
    def get_next_slot(self, page_size: int, allocator: PageAllocator) -> Optional[int]:
        """获取下一个可写入的 slot 索引。可能需要分配新 page。
        
        对标 SGLang: alloc_decode() — 检查是否需要新 page。
        """
        if not self.pages or self.current_offset >= page_size:
            # 需要新 page
            new_pages = allocator.alloc_pages(1)
            if new_pages is None:
                return None  # OOM
            self.pages.append(new_pages[0])
            self.current_offset = 0
        
        slot = self.pages[-1] * page_size + self.current_offset
        self.current_offset += 1
        self.total_tokens += 1
        return slot
    
    def alloc_slots(self, num_tokens: int, page_size: int, 
                    allocator: PageAllocator) -> Optional[torch.Tensor]:
        """分配多个 slot（用于 prefill）。
        
        对标 SGLang: alloc_extend()
        """
        slots = []
        for _ in range(num_tokens):
            slot = self.get_next_slot(page_size, allocator)
            if slot is None:
                return None
            slots.append(slot)
        return torch.tensor(slots, dtype=torch.long)


class PagedRequestManager:
    """管理所有请求的 page table。"""
    
    def __init__(self, pool: PagedKVPool, allocator: PageAllocator):
        self.pool = pool
        self.allocator = allocator
        self.page_size = pool.page_size
        self.tables: Dict[str, PageTable] = {}
    
    def create(self, rid: str) -> PageTable:
        table = PageTable()
        self.tables[rid] = table
        return table
    
    def get(self, rid: str) -> PageTable:
        return self.tables[rid]
    
    def release(self, rid: str):
        if rid in self.tables:
            table = self.tables[rid]
            if table.pages:
                self.allocator.free_pages_list(table.pages)
            del self.tables[rid]
    
    def status(self):
        total_tokens = sum(t.total_tokens for t in self.tables.values())
        total_pages = sum(len(t.pages) for t in self.tables.values())
        waste = total_pages * self.page_size - total_tokens if total_pages > 0 else 0
        
        print(f"  Requests: {len(self.tables)}")
        print(f"  Pages used: {total_pages} / {self.allocator.num_pages}")
        print(f"  Tokens stored: {total_tokens}")
        print(f"  Internal fragmentation: {waste} slots "
              f"({waste/(total_pages * self.page_size)*100:.1f}%)" if total_pages > 0 else "")
        print(f"  Page utilization: {self.allocator.utilization():.1%}")


# ============================================================
# 第三部分：Paged Attention 计算
# ============================================================


def paged_attention(
    query: torch.Tensor,         # [batch, num_heads, 1, head_dim] (decode: 1 query)
    kv_pool: PagedKVPool,
    page_tables: List[PageTable],
    layer_id: int,
) -> torch.Tensor:
    """手写 Paged Attention 计算。
    
    不需要把 KV 拷贝成连续 tensor！
    直接按 page table 索引从 pool 里取 KV 计算 attention。
    
    真正的推理引擎用 CUDA kernel 做这件事（FlashAttention + paged）。
    我们用纯 PyTorch 模拟。
    
    Args:
        query: [batch, heads, 1, dim]  (decode 时每请求只有 1 个 query)
        kv_pool: KV Cache 内存池
        page_tables: 每个请求的 page table
        layer_id: 当前层
    
    Returns:
        output: [batch, heads, 1, dim]
    """
    batch_size = len(page_tables)
    num_heads = query.shape[1]
    head_dim = query.shape[3]
    page_size = kv_pool.page_size
    
    outputs = []
    
    for b in range(batch_size):
        table = page_tables[b]
        slot_indices = table.get_all_slot_indices(page_size)
        
        if len(slot_indices) == 0:
            outputs.append(torch.zeros(1, num_heads, 1, head_dim))
            continue
        
        # 从 pool 直接按 slot 索引取 K, V（不需要连续！）
        k, v = kv_pool.fetch(layer_id, slot_indices)
        # k: [seq_len, heads, dim] → [1, heads, seq_len, dim]
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        
        # Standard attention: softmax(Q @ K^T / sqrt(d)) @ V
        q = query[b:b+1]  # [1, heads, 1, dim]
        
        scale = 1.0 / math.sqrt(head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale  # [1, heads, 1, seq_len]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # [1, heads, 1, dim]
        
        outputs.append(attn_output)
    
    return torch.cat(outputs, dim=0)  # [batch, heads, 1, dim]


# ============================================================
# 第四部分：演示
# ============================================================

def demo_paged_vs_contiguous():
    """对比 Paged 和 Contiguous 内存管理。"""
    
    print("=" * 60)
    print("实验 1: Page 粒度对内存利用率的影响")
    print("=" * 60)
    
    num_heads = 12
    head_dim = 64
    num_layers = 1  # 简化
    
    # 模拟 5 个请求，不同长度
    request_lengths = [7, 23, 4, 15, 31]
    
    for page_size in [1, 4, 8, 16]:
        total_tokens = sum(request_lengths)
        # 每个请求需要多少 page
        pages_per_req = [math.ceil(l / page_size) for l in request_lengths]
        total_pages = sum(pages_per_req)
        total_allocated = total_pages * page_size
        waste = total_allocated - total_tokens
        
        print(f"\n  page_size={page_size}:")
        print(f"    Pages: {total_pages} | "
              f"Allocated: {total_allocated} slots | "
              f"Used: {total_tokens} | "
              f"Waste: {waste} ({waste/total_allocated*100:.1f}%)")
        for i, (length, pages) in enumerate(zip(request_lengths, pages_per_req)):
            waste_i = pages * page_size - length
            print(f"    Req {i}: {length} tokens → {pages} pages "
                  f"(waste {waste_i})")


def demo_paged_attention():
    """演示 Paged Attention 计算。"""
    
    print("\n" + "=" * 60)
    print("实验 2: Paged Attention 计算")
    print("=" * 60)
    
    num_heads = 4
    head_dim = 8
    num_layers = 2
    page_size = 4
    num_pages = 32
    
    # 创建 pool 和 allocator
    pool = PagedKVPool(num_pages, page_size, num_layers, num_heads, head_dim)
    allocator = PageAllocator(num_pages, page_size)
    manager = PagedRequestManager(pool, allocator)
    
    # 创建两个请求，不同长度
    req_a = manager.create("A")
    req_b = manager.create("B")
    
    # 模拟 prefill: 请求 A 有 6 个 token，请求 B 有 3 个 token
    torch.manual_seed(42)
    
    # Req A: 6 tokens → 2 pages (page_size=4: page0 满, page1 用 2)
    slots_a = req_a.alloc_slots(6, page_size, allocator)
    print(f"\n  Req A: 6 tokens")
    print(f"    Pages: {req_a.pages}")
    print(f"    Slots: {slots_a.tolist()}")
    
    for layer_id in range(num_layers):
        k = torch.randn(6, num_heads, head_dim)
        v = torch.randn(6, num_heads, head_dim)
        pool.store(layer_id, slots_a, k, v)
    
    # Req B: 3 tokens → 1 page
    slots_b = req_b.alloc_slots(3, page_size, allocator)
    print(f"\n  Req B: 3 tokens")
    print(f"    Pages: {req_b.pages}")
    print(f"    Slots: {slots_b.tolist()}")
    
    for layer_id in range(num_layers):
        k = torch.randn(3, num_heads, head_dim)
        v = torch.randn(3, num_heads, head_dim)
        pool.store(layer_id, slots_b, k, v)
    
    manager.status()
    
    # 模拟 decode: 两个请求同时 decode
    print("\n  --- Batched Decode with Paged Attention ---")
    
    # 新 token 的 query
    query = torch.randn(2, num_heads, 1, head_dim)
    
    for layer_id in range(num_layers):
        output = paged_attention(query, pool, [req_a, req_b], layer_id)
        print(f"    Layer {layer_id}: output shape = {output.shape}")
    
    # 添加新 token 到各请求
    for req, name in [(req_a, "A"), (req_b, "B")]:
        new_slot = req.get_next_slot(page_size, allocator)
        for layer_id in range(num_layers):
            k_new = torch.randn(1, num_heads, head_dim)
            v_new = torch.randn(1, num_heads, head_dim)
            pool.store(layer_id, torch.tensor([new_slot]), k_new, v_new)
        print(f"    Req {name}: new slot={new_slot}, "
              f"pages={req.pages}, tokens={req.total_tokens}")
    
    manager.status()
    
    # 释放请求 B
    print("\n  --- Release Req B ---")
    manager.release("B")
    manager.status()
    
    # 新请求 C 可以复用 B 释放的 page
    print("\n  --- New Req C (reuses released pages) ---")
    req_c = manager.create("C")
    slots_c = req_c.alloc_slots(5, page_size, allocator)
    print(f"    Req C: 5 tokens, pages={req_c.pages}, slots={slots_c.tolist()}")
    manager.status()


def demo_memory_efficiency():
    """对比预分配 vs paged 的内存效率。"""
    
    print("\n" + "=" * 60)
    print("实验 3: 内存效率对比")
    print("=" * 60)
    
    max_seq_len = 2048
    num_requests = 100
    
    # 模拟真实场景：请求长度服从对数正态分布
    torch.manual_seed(42)
    actual_lengths = torch.clamp(
        torch.exp(torch.randn(num_requests) * 0.8 + 4).int(),
        min=1, max=max_seq_len
    ).tolist()
    
    total_tokens = sum(actual_lengths)
    avg_len = total_tokens / num_requests
    
    # 方案 1: 预分配（每请求分配 max_seq_len）
    preallocated = num_requests * max_seq_len
    prealloc_waste = preallocated - total_tokens
    
    # 方案 2: Paged（page_size=16）
    page_size = 16
    pages_needed = sum(math.ceil(l / page_size) for l in actual_lengths)
    paged_allocated = pages_needed * page_size
    paged_waste = paged_allocated - total_tokens
    
    # 方案 3: 逐 token 分配（page_size=1，即 Lesson 2 的方式）
    per_token = total_tokens  # 零浪费
    
    print(f"\n  {num_requests} requests, avg length = {avg_len:.0f}")
    print(f"  Total tokens needed: {total_tokens}")
    print(f"\n  {'Method':<25} {'Allocated':>10} {'Waste':>10} {'Util':>8}")
    print(f"  {'-'*55}")
    print(f"  {'Pre-alloc (max_seq)':<25} {preallocated:>10} {prealloc_waste:>10} "
          f"{total_tokens/preallocated*100:>7.1f}%")
    print(f"  {'Paged (page_size=16)':<25} {paged_allocated:>10} {paged_waste:>10} "
          f"{total_tokens/paged_allocated*100:>7.1f}%")
    print(f"  {'Per-token (page_size=1)':<25} {per_token:>10} {'0':>10} "
          f"{'100.0':>7}%")
    
    print(f"\n  Paged vs Pre-alloc: {preallocated/paged_allocated:.1f}x less memory")
    print(f"  Paged internal fragmentation: {paged_waste/paged_allocated*100:.1f}%")


if __name__ == "__main__":
    print("=" * 60)
    print("Lesson 5: PagedAttention")
    print("=" * 60)
    
    demo_paged_vs_contiguous()
    demo_paged_attention()
    demo_memory_efficiency()
    
    print("\n" + "=" * 60)
    print("Lesson 5 完成！")
    print("=" * 60)
    print("""
要点总结：
1. PagedAttention = 虚拟内存思想应用于 KV Cache
2. Page Table 映射：virtual (token position) → physical (page_id * page_size + offset)
3. 按需分配 page，只有最后一个 page 可能有碎片
4. Attention kernel 直接按 page table 索引取 KV，不需要连续内存
5. 内存利用率从 ~30% (预分配) 提升到 ~95%+ (paged)

SGLang 的更进一步:
→ page_size=1 时退化为 token-level 分配（零碎片，但管理开销大）
→ Radix Tree 前缀缓存: 不同请求的公共前缀共享 page（Lesson 6）
→ FlashAttention + Paged: CUDA kernel 级别的高效实现

下一步（Lesson 6）：
→ Radix Tree Cache — SGLang 的核心创新
""")
