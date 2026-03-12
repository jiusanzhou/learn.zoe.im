"""
Lesson 7: Scheduler — 调度策略

调度器是推理引擎的大脑，决定：
1. 哪些请求应该 prefill？（PrefillAdder）
2. 哪些请求应该 decode？（running batch）
3. 资源不够时怎么办？（驱逐/抢占）
4. 请求按什么顺序处理？（调度策略）

对标 SGLang:
- scheduler.py → Scheduler.event_loop_normal()
- schedule_policy.py → SchedulePolicy + PrefillAdder
- schedule_batch.py → ScheduleBatch
"""

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional, Tuple

from mini_engine.radix_cache import RadixCache


# ============================================================
# 第一部分：调度策略
# ============================================================


class ScheduleStrategy(Enum):
    """调度策略。对标 SGLang: CacheAwarePolicy / CacheAgnosticPolicy"""
    FCFS = "fcfs"               # 先来先服务
    LPM = "lpm"                 # 最长前缀匹配优先
    SHORTEST_FIRST = "shortest" # 短请求优先（减少 HoL blocking）
    PRIORITY = "priority"       # 用户自定义优先级


@dataclass
class ScheduleRequest:
    """调度器中的请求表示。"""
    rid: str
    input_ids: List[int]
    prefix_len: int = 0           # 缓存命中的前缀长度
    extend_len: int = 0           # 需要新 prefill 的 token 数
    max_new_tokens: int = 64
    priority: int = 0             # 用户优先级（越小越高）
    arrival_time: float = 0.0
    output_len: int = 0           # 已生成的 token 数
    
    @property
    def total_tokens(self) -> int:
        """当前总 token 占用（用于预算计算）。"""
        return len(self.input_ids) + self.output_len


class SchedulePolicy:
    """调度策略实现。
    
    对标 SGLang: schedule_policy.py → SchedulePolicy.calc_priority()
    """
    
    @staticmethod
    def sort_fcfs(queue: List[ScheduleRequest]):
        """先来先服务 — 按到达时间排序。"""
        queue.sort(key=lambda r: r.arrival_time)
    
    @staticmethod
    def sort_lpm(queue: List[ScheduleRequest]):
        """最长前缀匹配 — 优先调度缓存命中最多的请求。
        
        SGLang 默认策略。好处：
        - 缓存命中多的请求 prefill 更快（更少的新 token）
        - 相同前缀的请求一起调度，增加缓存利用率
        """
        queue.sort(key=lambda r: -r.prefix_len)
    
    @staticmethod
    def sort_shortest(queue: List[ScheduleRequest]):
        """短请求优先 — 减少 Head-of-Line blocking。"""
        queue.sort(key=lambda r: r.extend_len)
    
    @staticmethod
    def sort_priority(queue: List[ScheduleRequest]):
        """优先级调度 — 优先级高的先处理。"""
        queue.sort(key=lambda r: (r.priority, r.arrival_time))


# ============================================================
# 第二部分：资源预算管理
# ============================================================


@dataclass
class ResourceBudget:
    """资源预算。控制 prefill 一次能处理多少 token。
    
    对标 SGLang: PrefillAdder 中的各种 rem_* 字段。
    
    为什么需要预算？
    - KV Cache 总量有限，不能把所有等待请求一次性 prefill
    - 需要为 decode 请求预留空间（它们每步生成 1 个 token）
    - Prefill 的 token 太多会导致延迟峰值
    """
    total_kv_slots: int           # KV Cache 总容量
    used_kv_slots: int = 0        # 已使用的 slot
    max_prefill_tokens: int = 512 # 单次 prefill 的最大 token 数
    new_token_ratio: float = 0.3  # 为未来 decode 预留的比例
    max_running_requests: int = 32
    running_requests: int = 0
    
    @property
    def available_slots(self) -> int:
        """可用 KV slot 数（扣除预留）。"""
        reserved = int(self.used_kv_slots * self.new_token_ratio)
        return self.total_kv_slots - self.used_kv_slots - reserved
    
    @property
    def can_add_request(self) -> bool:
        return (self.running_requests < self.max_running_requests and
                self.available_slots > 0)
    
    def try_add(self, num_tokens: int, max_new_tokens: int) -> bool:
        """尝试为请求分配预算。
        
        对标 SGLang: PrefillAdder._update_prefill_budget()
        """
        # 需要的总空间 = prefill token + 预计的 decode token
        estimated_total = num_tokens + int(max_new_tokens * self.new_token_ratio)
        
        if estimated_total > self.available_slots:
            return False
        if self.running_requests >= self.max_running_requests:
            return False
        
        self.used_kv_slots += num_tokens
        self.running_requests += 1
        return True
    
    def release(self, num_tokens: int):
        """释放预算。"""
        self.used_kv_slots -= num_tokens
        self.running_requests -= 1


# ============================================================
# 第三部分：Scheduler
# ============================================================


class AddResult(Enum):
    ADDED = auto()
    NO_BUDGET = auto()
    SKIPPED = auto()


@dataclass
class ScheduleBatch:
    """一个调度批次。
    
    对标 SGLang: schedule_batch.py → ScheduleBatch
    """
    prefill_reqs: List[ScheduleRequest] = field(default_factory=list)
    decode_reqs: List[ScheduleRequest] = field(default_factory=list)
    
    @property
    def is_empty(self) -> bool:
        return not self.prefill_reqs and not self.decode_reqs
    
    @property
    def total_tokens(self) -> int:
        """batch 内总 token 数（用于计算吞吐）。"""
        prefill_tokens = sum(r.extend_len for r in self.prefill_reqs)
        decode_tokens = len(self.decode_reqs)
        return prefill_tokens + decode_tokens


class Scheduler:
    """调度器。
    
    核心职责：
    1. 管理等待队列和运行中的请求
    2. 根据策略决定下一个 batch
    3. 资源预算控制
    4. Prefill 优先策略
    
    对标 SGLang: managers/scheduler.py → Scheduler
    """
    
    def __init__(
        self,
        total_kv_slots: int = 4096,
        max_prefill_tokens: int = 512,
        max_running_requests: int = 32,
        max_batch_size: int = 8,
        new_token_ratio: float = 0.3,
        strategy: ScheduleStrategy = ScheduleStrategy.FCFS,
        radix_cache: Optional[RadixCache] = None,
    ):
        self.strategy = strategy
        self.max_batch_size = max_batch_size
        self.radix_cache = radix_cache
        
        self.budget = ResourceBudget(
            total_kv_slots=total_kv_slots,
            max_prefill_tokens=max_prefill_tokens,
            new_token_ratio=new_token_ratio,
            max_running_requests=max_running_requests,
        )
        
        # Queues
        self.waiting_queue: List[ScheduleRequest] = []
        self.running_reqs: Dict[str, ScheduleRequest] = {}
        self.finished_rids: List[str] = []
        
        # Stats
        self.total_prefill_tokens = 0
        self.total_decode_steps = 0
        self.total_cache_hits = 0
        self.total_batches = 0
    
    def add_request(self, req: ScheduleRequest):
        """添加新请求到等待队列。"""
        req.arrival_time = time.monotonic()
        
        # 如果有 radix cache，先匹配前缀
        if self.radix_cache and self.strategy == ScheduleStrategy.LPM:
            matched_kv, _ = self.radix_cache.match_prefix(req.input_ids)
            req.prefix_len = len(matched_kv)
        
        req.extend_len = len(req.input_ids) - req.prefix_len
        self.waiting_queue.append(req)
    
    def finish_request(self, rid: str):
        """标记请求完成。"""
        if rid in self.running_reqs:
            req = self.running_reqs.pop(rid)
            self.budget.release(req.total_tokens)
            self.finished_rids.append(rid)
    
    def get_next_batch(self) -> ScheduleBatch:
        """获取下一个调度批次。
        
        对标 SGLang: Scheduler.get_next_batch_to_run()
        
        逻辑：
        1. 排序等待队列（根据策略）
        2. 尝试从等待队列中选请求做 prefill（PrefillAdder 逻辑）
        3. 收集所有 running 请求做 decode
        4. Prefill 优先：有新 prefill 时先处理 prefill
        """
        batch = ScheduleBatch()
        
        # 1. 排序等待队列
        self._sort_waiting_queue()
        
        # 2. 选择 prefill 请求
        prefill_token_budget = self.budget.max_prefill_tokens
        to_remove = []
        
        for i, req in enumerate(self.waiting_queue):
            if not self.budget.can_add_request:
                break
            
            if req.extend_len > prefill_token_budget:
                # Chunked prefill 的场景（我们简化为跳过）
                continue
            
            if self.budget.try_add(len(req.input_ids), req.max_new_tokens):
                batch.prefill_reqs.append(req)
                self.running_reqs[req.rid] = req
                to_remove.append(i)
                
                prefill_token_budget -= req.extend_len
                self.total_prefill_tokens += req.extend_len
                self.total_cache_hits += req.prefix_len
                
                if len(batch.prefill_reqs) >= self.max_batch_size:
                    break
        
        # 移除已调度的请求
        for i in reversed(to_remove):
            self.waiting_queue.pop(i)
        
        # 3. 如果有 prefill，优先处理 prefill（SGLang 的策略）
        if batch.prefill_reqs:
            self.total_batches += 1
            return batch
        
        # 4. 没有新 prefill 时，做 decode
        decode_reqs = list(self.running_reqs.values())
        if decode_reqs:
            batch.decode_reqs = decode_reqs[:self.max_batch_size]
            self.total_decode_steps += 1
            self.total_batches += 1
        
        return batch
    
    def _sort_waiting_queue(self):
        """根据策略排序等待队列。"""
        if self.strategy == ScheduleStrategy.FCFS:
            SchedulePolicy.sort_fcfs(self.waiting_queue)
        elif self.strategy == ScheduleStrategy.LPM:
            SchedulePolicy.sort_lpm(self.waiting_queue)
        elif self.strategy == ScheduleStrategy.SHORTEST_FIRST:
            SchedulePolicy.sort_shortest(self.waiting_queue)
        elif self.strategy == ScheduleStrategy.PRIORITY:
            SchedulePolicy.sort_priority(self.waiting_queue)
    
    def stats(self) -> dict:
        return {
            "waiting": len(self.waiting_queue),
            "running": len(self.running_reqs),
            "finished": len(self.finished_rids),
            "total_batches": self.total_batches,
            "total_prefill_tokens": self.total_prefill_tokens,
            "total_cache_hits": self.total_cache_hits,
            "cache_hit_rate": (self.total_cache_hits / 
                              (self.total_prefill_tokens + self.total_cache_hits)
                              if self.total_prefill_tokens + self.total_cache_hits > 0 else 0),
            "budget_used": f"{self.budget.used_kv_slots}/{self.budget.total_kv_slots}",
            "budget_utilization": self.budget.used_kv_slots / self.budget.total_kv_slots,
        }


# ============================================================
# 第四部分：演示
# ============================================================

def demo_fcfs():
    """FCFS 调度演示。"""
    print("=" * 60)
    print("实验 1: FCFS 调度")
    print("=" * 60)
    
    scheduler = Scheduler(
        total_kv_slots=200,
        max_prefill_tokens=100,
        max_running_requests=4,
        strategy=ScheduleStrategy.FCFS,
    )
    
    # 添加 6 个请求，但 max_running = 4
    for i in range(6):
        scheduler.add_request(ScheduleRequest(
            rid=f"req-{i}",
            input_ids=list(range(20 + i * 5)),
            max_new_tokens=30,
        ))
    
    print(f"  Added 6 requests, max_running=4")
    
    # 模拟调度循环
    for step in range(8):
        batch = scheduler.get_next_batch()
        
        if batch.prefill_reqs:
            rids = [r.rid for r in batch.prefill_reqs]
            tokens = sum(r.extend_len for r in batch.prefill_reqs)
            print(f"  Step {step}: PREFILL {rids} ({tokens} tokens)")
            
            # 模拟 prefill 后请求变成 running
            for req in batch.prefill_reqs:
                req.output_len = 1
        
        elif batch.decode_reqs:
            rids = [r.rid for r in batch.decode_reqs]
            print(f"  Step {step}: DECODE {rids} (batch={len(rids)})")
            
            # 模拟 decode
            for req in batch.decode_reqs:
                req.output_len += 1
                # 模拟部分请求完成
                if req.output_len >= 5:
                    scheduler.finish_request(req.rid)
        else:
            print(f"  Step {step}: IDLE")
    
    s = scheduler.stats()
    print(f"\n  Stats: waiting={s['waiting']} running={s['running']} "
          f"finished={s['finished']} batches={s['total_batches']}")


def demo_lpm():
    """LPM (最长前缀匹配) 调度演示。"""
    print("\n" + "=" * 60)
    print("实验 2: LPM 调度 + Radix Cache")
    print("=" * 60)
    
    cache = RadixCache()
    
    # 预先缓存一个 system prompt
    system = list(range(1, 51))  # 50 tokens
    cache.insert(system, list(range(100, 150)))
    print(f"  Pre-cached: system prompt ({len(system)} tokens)")
    
    scheduler = Scheduler(
        total_kv_slots=500,
        max_prefill_tokens=200,
        max_running_requests=8,
        strategy=ScheduleStrategy.LPM,
        radix_cache=cache,
    )
    
    # 添加请求：有些能命中缓存，有些不能
    requests = [
        ("A", system + [100, 101, 102]),          # 50 token prefix hit
        ("B", system + [200, 201]),                # 50 token prefix hit
        ("C", list(range(500, 530))),              # 0 prefix hit
        ("D", system + [100, 101, 102, 103, 104]), # 50 token prefix hit
        ("E", list(range(600, 660))),              # 0 prefix hit
    ]
    
    for name, ids in requests:
        scheduler.add_request(ScheduleRequest(
            rid=f"req-{name}",
            input_ids=ids,
            max_new_tokens=20,
        ))
        prefix = scheduler.waiting_queue[-1].prefix_len
        extend = scheduler.waiting_queue[-1].extend_len
        print(f"  {name}: {len(ids)} tokens, prefix_hit={prefix}, extend={extend}")
    
    # 调度一次 — LPM 应该优先调度缓存命中多的请求
    batch = scheduler.get_next_batch()
    
    print(f"\n  LPM scheduling order (prefill):")
    for req in batch.prefill_reqs:
        print(f"    {req.rid}: prefix={req.prefix_len}, extend={req.extend_len}")
    
    s = scheduler.stats()
    print(f"\n  Cache hit rate: {s['cache_hit_rate']:.1%}")
    print(f"  Total prefill tokens: {s['total_prefill_tokens']}")
    print(f"  Total cache hits: {s['total_cache_hits']}")


def demo_budget():
    """资源预算管理演示。"""
    print("\n" + "=" * 60)
    print("实验 3: 资源预算管理")
    print("=" * 60)
    
    scheduler = Scheduler(
        total_kv_slots=100,        # 很小的 cache
        max_prefill_tokens=50,
        max_running_requests=3,
        new_token_ratio=0.3,
        strategy=ScheduleStrategy.FCFS,
    )
    
    # 添加很多请求，超过容量
    for i in range(10):
        scheduler.add_request(ScheduleRequest(
            rid=f"req-{i}",
            input_ids=list(range(15)),  # 每个 15 tokens
            max_new_tokens=20,
        ))
    
    print(f"  Added 10 requests (each 15 tokens), cache=100 slots")
    
    # 尝试调度
    for step in range(5):
        batch = scheduler.get_next_batch()
        
        if batch.prefill_reqs:
            rids = [r.rid for r in batch.prefill_reqs]
            print(f"  Step {step}: PREFILL {rids}")
            for req in batch.prefill_reqs:
                req.output_len = 1
        elif batch.decode_reqs:
            rids = [r.rid for r in batch.decode_reqs]
            print(f"  Step {step}: DECODE {rids}")
            for req in batch.decode_reqs:
                req.output_len += 1
        else:
            print(f"  Step {step}: IDLE (budget exhausted!)")
        
        s = scheduler.stats()
        print(f"    Budget: {s['budget_used']} ({s['budget_utilization']:.0%}), "
              f"waiting={s['waiting']}, running={s['running']}")


def demo_strategy_comparison():
    """不同策略对比。"""
    print("\n" + "=" * 60)
    print("实验 4: 策略对比")
    print("=" * 60)
    
    cache = RadixCache()
    system = list(range(1, 101))  # 100 tokens
    cache.insert(system, list(range(1000, 1100)))
    
    requests = [
        ("short-no-cache", list(range(500, 510)), 0),     # 10 tokens, no cache
        ("long-cached", system + list(range(200, 230)), 0), # 130 tokens, 100 cached
        ("short-cached", system + list(range(300, 305)), 0), # 105 tokens, 100 cached
        ("priority-high", list(range(400, 420)), -1),        # 20 tokens, high priority
    ]
    
    for strategy in [ScheduleStrategy.FCFS, ScheduleStrategy.LPM, 
                     ScheduleStrategy.SHORTEST_FIRST, ScheduleStrategy.PRIORITY]:
        scheduler = Scheduler(
            total_kv_slots=1000,
            max_prefill_tokens=500,
            max_running_requests=8,
            strategy=strategy,
            radix_cache=cache if strategy == ScheduleStrategy.LPM else None,
        )
        
        for name, ids, priority in requests:
            scheduler.add_request(ScheduleRequest(
                rid=name, input_ids=ids, max_new_tokens=20, priority=priority,
            ))
        
        batch = scheduler.get_next_batch()
        order = [r.rid for r in batch.prefill_reqs]
        
        print(f"\n  {strategy.value:>12}: {' → '.join(order)}")


if __name__ == "__main__":
    print("=" * 60)
    print("Lesson 7: Scheduler")
    print("=" * 60)
    
    demo_fcfs()
    demo_lpm()
    demo_budget()
    demo_strategy_comparison()
    
    print("\n" + "=" * 60)
    print("Lesson 7 完成！")
    print("=" * 60)
    print("""
要点总结：
1. 调度器的核心决策：谁先 prefill，谁一起 decode
2. Prefill 优先：新请求尽快获得首 token（低 TTFT）
3. LPM 策略：缓存命中多的请求优先 → 减少 prefill 计算
4. 资源预算：为 decode 预留空间，避免 OOM
5. 不同策略适合不同场景：
   - FCFS: 公平，通用
   - LPM: 有共享前缀时最优
   - Shortest: 低延迟场景
   - Priority: 需要 QoS 时

SGLang 的进阶：
→ DFS_WEIGHT: 同子树请求一起调度，最大化缓存复用
→ Chunked Prefill: 大 prompt 分块处理，和 decode 交替
→ Preemption: OOM 时踢出低优先级请求（retract_decode）
→ new_token_ratio 动态调整: 根据实际使用情况自适应

下一步（Lesson 8）：
→ HTTP API — OpenAI 兼容的 HTTP 服务
""")
