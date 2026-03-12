"""
Lesson 6: Radix Tree Cache — SGLang 的核心创新

Radix Tree（基数树/压缩前缀树）让不同请求共享相同前缀的 KV Cache。

核心思想：
- 所有请求的 token 序列构成一棵前缀树
- 相同前缀只存一份 KV Cache
- 新请求先匹配最长前缀 → 复用已有 KV → 只 prefill 差异部分

对标 SGLang: mem_cache/radix_cache.py
- TreeNode: 树节点，存 key + value(KV indices) + children
- RadixCache: 前缀匹配/插入/驱逐
- match_prefix(): O(L) 前缀匹配（L = 序列长度）
- _split_node(): 匹配到节点中间时分裂
- evict(): LRU 驱逐释放内存
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ============================================================
# 第一部分：树节点
# ============================================================

@dataclass
class RadixNode:
    """Radix Tree 节点。
    
    对标 SGLang: TreeNode
    
    每个节点存储一段 token 序列（key）和对应的 KV Cache 索引（value）。
    Radix Tree 的特点是把公共前缀压缩到一个节点中。
    """
    token_ids: List[int] = field(default_factory=list)    # 这个节点代表的 token 序列
    kv_indices: Optional[List[int]] = None                 # KV Cache 中的 slot 索引
    children: Dict[int, 'RadixNode'] = field(default_factory=dict)  # 子节点，key 是第一个 token
    parent: Optional['RadixNode'] = None
    
    # 引用计数：正在被请求使用的节点不能被驱逐
    # 对标 SGLang: lock_ref
    ref_count: int = 0
    
    # LRU 驱逐用
    last_access_time: float = 0.0
    
    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)
    
    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def __repr__(self):
        tokens_preview = self.token_ids[:5]
        suffix = "..." if len(self.token_ids) > 5 else ""
        return (f"RadixNode(tokens={tokens_preview}{suffix}, "
                f"len={self.num_tokens}, children={len(self.children)}, "
                f"ref={self.ref_count})")


# ============================================================
# 第二部分：Radix Cache
# ============================================================

class RadixCache:
    """基于 Radix Tree 的 KV Cache 前缀共享。
    
    核心操作：
    1. match_prefix(token_ids) → 找最长匹配前缀
    2. insert(token_ids, kv_indices) → 插入新序列
    3. evict(num_tokens) → LRU 驱逐释放空间
    
    对标 SGLang: RadixCache
    """
    
    def __init__(self):
        self.root = RadixNode()
        self.root.ref_count = 1  # root 永远不被驱逐
        self.total_cached_tokens = 0
    
    def match_prefix(self, token_ids: List[int]) -> Tuple[List[int], RadixNode]:
        """匹配最长缓存前缀。
        
        返回: (匹配到的 KV indices, 最后匹配的节点)
        
        对标 SGLang: RadixCache.match_prefix() → _match_prefix_helper()
        """
        node = self.root
        matched_kv_indices = []
        pos = 0
        
        while pos < len(token_ids):
            first_token = token_ids[pos]
            
            if first_token not in node.children:
                break
            
            child = node.children[first_token]
            child.last_access_time = time.monotonic()
            
            # 检查 child 的 token_ids 与输入的匹配长度
            match_len = 0
            for i in range(min(child.num_tokens, len(token_ids) - pos)):
                if child.token_ids[i] != token_ids[pos + i]:
                    break
                match_len += 1
            
            if match_len == 0:
                break
            
            if match_len < child.num_tokens:
                # 匹配到节点中间 → 需要分裂
                # 对标 SGLang: _split_node()
                self._split_node(child, match_len)
                # 分裂后 child 的前 match_len 个 token 变成了新的中间节点
                new_node = node.children[first_token]
                if new_node.kv_indices:
                    matched_kv_indices.extend(new_node.kv_indices)
                node = new_node
                pos += match_len
                break
            else:
                # 完全匹配这个节点
                if child.kv_indices:
                    matched_kv_indices.extend(child.kv_indices)
                node = child
                pos += child.num_tokens
        
        return matched_kv_indices, node
    
    def insert(self, token_ids: List[int], kv_indices: List[int]) -> int:
        """插入一个序列的 KV Cache 到树中。
        
        返回: 已存在的前缀长度（这部分不需要重复存储）
        
        对标 SGLang: RadixCache.insert() → _insert_helper()
        """
        node = self.root
        pos = 0
        
        # 先沿着已有路径走
        while pos < len(token_ids):
            first_token = token_ids[pos]
            
            if first_token not in node.children:
                break
            
            child = node.children[first_token]
            
            match_len = 0
            for i in range(min(child.num_tokens, len(token_ids) - pos)):
                if child.token_ids[i] != token_ids[pos + i]:
                    break
                match_len += 1
            
            if match_len < child.num_tokens:
                # 部分匹配 → 分裂
                self._split_node(child, match_len)
                node = node.children[first_token]
                pos += match_len
                break
            else:
                node = child
                pos += child.num_tokens
        
        prefix_len = pos  # 已有前缀的长度
        
        # 插入剩余部分
        if pos < len(token_ids):
            new_node = RadixNode(
                token_ids=token_ids[pos:],
                kv_indices=kv_indices[pos:],
                parent=node,
                last_access_time=time.monotonic(),
            )
            first_token = token_ids[pos]
            node.children[first_token] = new_node
            self.total_cached_tokens += len(token_ids) - pos
        
        return prefix_len
    
    def _split_node(self, node: RadixNode, split_pos: int):
        """在 split_pos 处分裂节点。
        
        对标 SGLang: _split_node()
        
        Before:  parent → [ABCDE]
        After:   parent → [AB] → [CDE]
                          (new)   (old, shortened)
        """
        # 创建新的中间节点（前半部分）
        new_node = RadixNode(
            token_ids=node.token_ids[:split_pos],
            kv_indices=node.kv_indices[:split_pos] if node.kv_indices else None,
            parent=node.parent,
            ref_count=node.ref_count,
            last_access_time=node.last_access_time,
        )
        
        # 缩短原节点（后半部分）
        remaining_first_token = node.token_ids[split_pos]
        node.token_ids = node.token_ids[split_pos:]
        if node.kv_indices:
            node.kv_indices = node.kv_indices[split_pos:]
        node.parent = new_node
        
        # 重新连接
        new_node.children[remaining_first_token] = node
        
        # 更新父节点的 children
        if new_node.parent:
            first_token = new_node.token_ids[0]
            new_node.parent.children[first_token] = new_node
    
    def evict_lru(self, num_tokens: int) -> int:
        """LRU 驱逐：释放最久未使用的叶子节点。
        
        对标 SGLang: RadixCache.evict()
        
        只驱逐 ref_count == 0 的叶子节点。
        驱逐后如果父节点变成叶子且 ref_count == 0，也可以被驱逐。
        """
        evicted = 0
        
        while evicted < num_tokens:
            # 找所有可驱逐的叶子（ref_count == 0）
            leaves = self._get_evictable_leaves()
            if not leaves:
                break
            
            # 按 last_access_time 排序（LRU）
            leaves.sort(key=lambda n: n.last_access_time)
            
            # 驱逐最老的叶子
            victim = leaves[0]
            evicted += victim.num_tokens
            self.total_cached_tokens -= victim.num_tokens
            
            # 从父节点中删除
            if victim.parent:
                first_token = victim.token_ids[0]
                victim.parent.children.pop(first_token, None)
            
            # 检查父节点是否可以合并（只有一个子节点时）
            # SGLang 不做这个优化，我们也跳过
        
        return evicted
    
    def _get_evictable_leaves(self) -> List[RadixNode]:
        """获取所有可驱逐的叶子节点。"""
        leaves = []
        
        def dfs(node):
            if node.is_leaf and node.ref_count == 0 and node != self.root:
                leaves.append(node)
            for child in node.children.values():
                dfs(child)
        
        dfs(self.root)
        return leaves
    
    def inc_ref(self, node: RadixNode):
        """增加引用计数（请求开始使用）。
        
        对标 SGLang: inc_lock_ref() — 沿路径向上增加所有祖先的 ref
        """
        while node != self.root:
            node.ref_count += 1
            node = node.parent
    
    def dec_ref(self, node: RadixNode):
        """减少引用计数（请求完成）。
        
        对标 SGLang: dec_lock_ref()
        """
        while node != self.root:
            node.ref_count -= 1
            node = node.parent
    
    def pretty_print(self, max_depth: int = 5):
        """可视化树结构。"""
        def _print(node, depth, prefix=""):
            if depth > max_depth:
                return
            
            if node == self.root:
                print(f"ROOT (cached: {self.total_cached_tokens} tokens)")
            else:
                tokens_str = str(node.token_ids[:8])
                if len(node.token_ids) > 8:
                    tokens_str = tokens_str[:-1] + ", ...]"
                ref_str = f" 🔒×{node.ref_count}" if node.ref_count > 0 else ""
                print(f"{prefix}├── [{node.num_tokens} tokens] {tokens_str}{ref_str}")
            
            children = list(node.children.values())
            for i, child in enumerate(children):
                child_prefix = prefix + ("│   " if i < len(children) - 1 else "    ")
                _print(child, depth + 1, child_prefix)
        
        _print(self.root, 0)


# ============================================================
# 第三部分：演示
# ============================================================

def demo_prefix_sharing():
    """演示前缀共享。"""
    
    print("=" * 60)
    print("实验 1: 前缀共享")
    print("=" * 60)
    
    cache = RadixCache()
    
    # 模拟 tokenized 序列（用小数字表示 token id）
    # System prompt: [1, 2, 3, 4, 5]
    # Req A: system + [10, 11, 12]
    # Req B: system + [20, 21, 22]
    # Req C: system + [10, 11, 30]  ← 和 A 共享更长前缀
    
    system = [1, 2, 3, 4, 5]
    req_a = system + [10, 11, 12]
    req_b = system + [20, 21, 22]
    req_c = system + [10, 11, 30]
    
    # KV indices（模拟 KV Cache 中的 slot 位置）
    kv_a = list(range(100, 100 + len(req_a)))
    kv_b = list(range(200, 200 + len(req_b)))
    kv_c = list(range(300, 300 + len(req_c)))
    
    # 插入 Req A
    print("\n  --- Insert Req A ---")
    prefix_len = cache.insert(req_a, kv_a)
    print(f"  Tokens: {req_a}")
    print(f"  Prefix reuse: {prefix_len} tokens")
    cache.pretty_print()
    
    # 插入 Req B — 共享 system prompt
    print("\n  --- Insert Req B ---")
    prefix_len = cache.insert(req_b, kv_b)
    print(f"  Tokens: {req_b}")
    print(f"  Prefix reuse: {prefix_len} tokens (system prompt!)")
    cache.pretty_print()
    
    # 插入 Req C — 和 A 共享更长前缀
    print("\n  --- Insert Req C ---")
    prefix_len = cache.insert(req_c, kv_c)
    print(f"  Tokens: {req_c}")
    print(f"  Prefix reuse: {prefix_len} tokens")
    cache.pretty_print()
    
    # 匹配新请求的前缀
    print("\n  --- Match prefix for new request ---")
    new_req = system + [10, 11, 12, 40, 41]  # 和 A 共享完整前缀
    matched_kv, last_node = cache.match_prefix(new_req)
    print(f"  New request: {new_req}")
    print(f"  Matched KV indices: {matched_kv}")
    print(f"  Matched tokens: {len(matched_kv)} / {len(new_req)}")
    print(f"  → Only need to prefill {len(new_req) - len(matched_kv)} new tokens!")


def demo_savings():
    """演示缓存节省的计算量。"""
    
    print("\n" + "=" * 60)
    print("实验 2: 缓存节省的计算量")
    print("=" * 60)
    
    cache = RadixCache()
    
    # 模拟 ChatGPT 场景：所有对话共享 system prompt
    system_prompt = list(range(1, 201))  # 200 token 的 system prompt
    
    # 10 个用户的不同问题
    user_queries = [
        list(range(1000 + i * 100, 1000 + i * 100 + 30 + i * 5))
        for i in range(10)
    ]
    
    total_tokens_without_cache = 0
    total_tokens_with_cache = 0
    total_prefill_saved = 0
    
    for i, query in enumerate(user_queries):
        full_seq = system_prompt + query
        kv_indices = list(range(i * 1000, i * 1000 + len(full_seq)))
        
        # 匹配前缀
        matched_kv, last_node = cache.match_prefix(full_seq)
        prefix_len = len(matched_kv)
        new_tokens = len(full_seq) - prefix_len
        
        # 插入
        cache.insert(full_seq, kv_indices)
        
        total_tokens_without_cache += len(full_seq)
        total_tokens_with_cache += new_tokens
        total_prefill_saved += prefix_len
        
        if i < 3 or i == 9:
            print(f"  Req {i}: {len(full_seq)} tokens, "
                  f"prefix reuse={prefix_len}, new={new_tokens}")
    
    print(f"\n  Total tokens (no cache):   {total_tokens_without_cache}")
    print(f"  Total tokens (with cache): {total_tokens_with_cache}")
    print(f"  Saved prefill tokens:      {total_prefill_saved}")
    print(f"  Reduction: {total_prefill_saved/total_tokens_without_cache*100:.1f}%")
    print(f"  Cached in tree: {cache.total_cached_tokens} tokens")


def demo_eviction():
    """演示 LRU 驱逐。"""
    
    print("\n" + "=" * 60)
    print("实验 3: LRU 驱逐")
    print("=" * 60)
    
    cache = RadixCache()
    
    # 插入几个序列
    sequences = [
        ([1, 2, 3, 4, 5], "A"),
        ([1, 2, 3, 6, 7], "B"),  # 和 A 共享前缀 [1,2,3]
        ([10, 20, 30], "C"),
    ]
    
    for seq, name in sequences:
        kv = list(range(len(seq)))
        cache.insert(seq, kv)
        print(f"  Inserted {name}: {seq}")
    
    print(f"\n  Cached: {cache.total_cached_tokens} tokens")
    cache.pretty_print()
    
    # 锁定 A 的路径（模拟正在处理的请求）
    print("\n  --- Lock Req A's path ---")
    _, node_a = cache.match_prefix([1, 2, 3, 4, 5])
    cache.inc_ref(node_a)
    cache.pretty_print()
    
    # 尝试驱逐
    print("\n  --- Evict 5 tokens ---")
    evicted = cache.evict_lru(5)
    print(f"  Evicted: {evicted} tokens")
    print(f"  (Locked nodes are protected!)")
    cache.pretty_print()
    
    # 释放锁
    print("\n  --- Unlock Req A ---")
    cache.dec_ref(node_a)
    
    # 再次驱逐
    print("\n  --- Evict 10 tokens ---")
    evicted = cache.evict_lru(10)
    print(f"  Evicted: {evicted} tokens")
    cache.pretty_print()


def demo_multi_turn_conversation():
    """演示多轮对话的前缀复用。"""
    
    print("\n" + "=" * 60)
    print("实验 4: 多轮对话前缀复用")
    print("=" * 60)
    
    cache = RadixCache()
    
    # 模拟多轮对话：每轮都在上一轮基础上追加
    turns = [
        [1, 2, 3, 4, 5],                           # Turn 1: system
        [1, 2, 3, 4, 5, 10, 11, 12],               # Turn 2: system + user1
        [1, 2, 3, 4, 5, 10, 11, 12, 20, 21],       # Turn 3: + assistant1
        [1, 2, 3, 4, 5, 10, 11, 12, 20, 21, 30, 31, 32],  # Turn 4: + user2
    ]
    
    for i, turn in enumerate(turns):
        matched_kv, _ = cache.match_prefix(turn)
        prefix_len = len(matched_kv)
        new_tokens = len(turn) - prefix_len
        
        kv = list(range(i * 100, i * 100 + len(turn)))
        cache.insert(turn, kv)
        
        print(f"  Turn {i+1}: {len(turn)} tokens → "
              f"reuse {prefix_len}, new {new_tokens}")
    
    print(f"\n  Final tree:")
    cache.pretty_print()
    
    print(f"\n  多轮对话中，每轮只需要 prefill 新增的 token！")
    print(f"  Turn 4 只需 prefill 3 个新 token，而非全部 13 个。")


if __name__ == "__main__":
    print("=" * 60)
    print("Lesson 6: Radix Tree Cache")
    print("=" * 60)
    
    demo_prefix_sharing()
    demo_savings()
    demo_eviction()
    demo_multi_turn_conversation()
    
    print("\n" + "=" * 60)
    print("Lesson 6 完成！")
    print("=" * 60)
    print("""
要点总结：
1. Radix Tree 让不同请求共享公共前缀的 KV Cache
2. match_prefix(): O(L) 找最长缓存前缀
3. insert(): 自动处理节点分裂（公共前缀提取）
4. 引用计数保护正在使用的节点不被驱逐
5. LRU 驱逐从叶子节点开始，向上传播

实际效果（ChatGPT 场景）：
→ 10 个请求共享 200 token 的 system prompt
→ 节省 ~80% 的 prefill 计算！
→ 多轮对话每轮只 prefill 新增内容

SGLang 的进阶:
→ 多种驱逐策略: LRU/LFU/FIFO/SLRU
→ page_size 对齐: 树的边界对齐到 page
→ 分层缓存 (HiRadixCache): GPU → CPU → 磁盘

下一步（Lesson 7）：
→ Scheduler — 调度策略的完整实现
""")
