"""
Lesson 10: Speculative Decoding — 投机采样加速

核心算法：
1. Draft: 用小模型（或 n-gram）快速猜 K 个 token
2. Verify: 用大模型一次 forward 验证所有猜测
3. Accept: 从左到右，接受概率正确的 token，第一个错误处重新采样

数学保证：
- 接受概率 = min(1, p_target(x) / p_draft(x))
- 被拒绝时从修正分布采样: (p_target - p_draft)+ / sum
- 最终输出分布 = p_target（无损！）

对标 SGLang:
- speculative/eagle_worker.py → draft() + verify()
- speculative/ngram_worker.py → n-gram based drafting
"""

import time
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 第一部分：标准 Speculative Decoding 算法
# ============================================================


@torch.no_grad()
def speculative_decode_step(
    target_model: AutoModelForCausalLM,
    draft_model: AutoModelForCausalLM,
    input_ids: torch.Tensor,       # [1, seq_len]
    target_past: Optional[object],  # target model KV cache
    draft_past: Optional[object],   # draft model KV cache
    num_speculative: int = 4,       # 投机步数 K
    temperature: float = 1.0,
) -> Tuple[List[int], int, object, object]:
    """一步 Speculative Decoding。
    
    Returns:
        accepted_tokens: 被接受的 token 列表 (1 ~ K+1 个)
        num_accepted: 接受的 token 数
        new_target_past: 更新后的 target KV cache
        new_draft_past: 更新后的 draft KV cache
    """
    
    # === Phase 1: Draft — 用小模型快速生成 K 个 token ===
    draft_tokens = []
    draft_probs = []
    current_input = input_ids[:, -1:]  # 只取最后一个 token
    current_draft_past = draft_past
    
    for _ in range(num_speculative):
        if current_draft_past is None:
            draft_out = draft_model(input_ids=input_ids, use_cache=True)
        else:
            draft_out = draft_model(
                input_ids=current_input,
                past_key_values=current_draft_past,
                use_cache=True,
            )
        
        current_draft_past = draft_out.past_key_values
        logits = draft_out.logits[:, -1, :]
        
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            token = torch.multinomial(probs, 1).item()
        else:
            probs = F.softmax(logits, dim=-1)
            token = logits.argmax(-1).item()
        
        draft_tokens.append(token)
        draft_probs.append(probs[0].clone())  # [vocab_size]
        current_input = torch.tensor([[token]])
    
    # === Phase 2: Verify — 用大模型一次 forward 检查所有 draft token ===
    # 关键：把所有 draft token 一起传给 target model，一次 forward！
    
    verify_input = torch.tensor([draft_tokens]).unsqueeze(0) if len(draft_tokens) > 0 else None
    
    # 构建验证序列：[last_real_token, draft_1, draft_2, ..., draft_K]
    if target_past is None:
        # 首次：传完整序列 + draft tokens
        full_input = torch.cat([input_ids, torch.tensor([draft_tokens])], dim=1)
        target_out = target_model(input_ids=full_input, use_cache=True)
        # target logits 从原序列最后一个位置开始
        target_logits = target_out.logits[:, -(num_speculative + 1):, :]
    else:
        # 增量：只传 [last_token, draft_1, ..., draft_K]
        verify_ids = torch.tensor([[input_ids[0, -1].item()] + draft_tokens])
        target_out = target_model(
            input_ids=verify_ids,
            past_key_values=target_past,
            use_cache=True,
        )
        target_logits = target_out.logits  # [1, K+1, vocab]
    
    new_target_past = target_out.past_key_values
    
    # === Phase 3: Accept/Reject — 逐个检查 draft token ===
    accepted_tokens = []
    
    for i in range(num_speculative):
        if temperature > 0:
            target_prob = F.softmax(target_logits[0, i, :] / temperature, dim=-1)
        else:
            target_prob = F.softmax(target_logits[0, i, :], dim=-1)
        
        draft_token = draft_tokens[i]
        draft_prob = draft_probs[i]
        
        # 接受概率 = min(1, p_target / p_draft)
        p_target = target_prob[draft_token].item()
        p_draft = draft_prob[draft_token].item()
        
        if p_draft == 0:
            # Draft 概率为 0 但 target 不为 0 → 总是接受
            accept_prob = 1.0
        else:
            accept_prob = min(1.0, p_target / p_draft)
        
        if temperature == 0:
            # Greedy: target 和 draft 一致就接受
            target_token = target_logits[0, i, :].argmax().item()
            if draft_token == target_token:
                accepted_tokens.append(draft_token)
            else:
                # 拒绝，使用 target 的 greedy 选择
                accepted_tokens.append(target_token)
                break
        else:
            # Stochastic: 按概率接受
            if torch.rand(1).item() < accept_prob:
                accepted_tokens.append(draft_token)
            else:
                # 拒绝 → 从修正分布采样
                # p_corrected = max(0, p_target - p_draft) / Z
                corrected = torch.clamp(target_prob - draft_prob, min=0)
                corrected_sum = corrected.sum()
                if corrected_sum > 0:
                    corrected = corrected / corrected_sum
                    new_token = torch.multinomial(corrected, 1).item()
                else:
                    new_token = torch.multinomial(target_prob, 1).item()
                accepted_tokens.append(new_token)
                break
    else:
        # 所有 K 个 draft token 都被接受！bonus: 从 target 采样第 K+1 个
        if temperature > 0:
            bonus_probs = F.softmax(target_logits[0, -1, :] / temperature, dim=-1)
            bonus_token = torch.multinomial(bonus_probs, 1).item()
        else:
            bonus_token = target_logits[0, -1, :].argmax().item()
        accepted_tokens.append(bonus_token)
    
    return accepted_tokens, len(accepted_tokens), new_target_past, current_draft_past


# ============================================================
# 第二部分：N-gram Speculative Decoding（无需 draft model）
# ============================================================


class NGramDrafter:
    """N-gram 猜测器。
    
    对标 SGLang: speculative/ngram_worker.py
    
    用已生成的 token 历史建立 n-gram 表，
    根据上下文猜测下一个 token。
    不需要额外的 draft model！
    """
    
    def __init__(self, n: int = 3):
        self.n = n
        # (t1, t2, ..., tn) → {next_token: count}
        self.table = {}
    
    def update(self, tokens: List[int]):
        """用新 token 更新 n-gram 表。"""
        for i in range(len(tokens) - self.n):
            context = tuple(tokens[i:i + self.n])
            next_token = tokens[i + self.n]
            if context not in self.table:
                self.table[context] = {}
            self.table[context][next_token] = self.table[context].get(next_token, 0) + 1
    
    def draft(self, context: List[int], num_tokens: int) -> List[int]:
        """根据上下文猜测接下来的 token。"""
        drafted = []
        ctx = list(context[-self.n:])
        
        for _ in range(num_tokens):
            key = tuple(ctx[-self.n:])
            if key in self.table:
                # 取最常见的 next token
                candidates = self.table[key]
                next_token = max(candidates, key=candidates.get)
                drafted.append(next_token)
                ctx.append(next_token)
            else:
                break  # 没有匹配的 n-gram
        
        return drafted


# ============================================================
# 第三部分：完整的 Speculative Generation
# ============================================================


@torch.no_grad()
def generate_speculative(
    target_model: AutoModelForCausalLM,
    draft_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    num_speculative: int = 4,
    temperature: float = 0.0,
) -> Tuple[str, dict]:
    """完整的 Speculative Decoding 生成。"""
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated = input_ids[0].tolist()
    
    target_past = None
    draft_past = None
    
    total_draft = 0
    total_accepted = 0
    total_steps = 0
    total_target_forwards = 0
    
    t0 = time.perf_counter()
    
    while len(generated) - len(input_ids[0]) < max_new_tokens:
        current_input = torch.tensor([generated])
        
        accepted, num_acc, target_past, draft_past = speculative_decode_step(
            target_model, draft_model,
            current_input, target_past, draft_past,
            num_speculative=num_speculative,
            temperature=temperature,
        )
        
        generated.extend(accepted)
        total_draft += num_speculative
        total_accepted += num_acc
        total_steps += 1
        total_target_forwards += 1
        
        # 检查 EOS
        if tokenizer.eos_token_id in accepted:
            break
        
        # KV cache 需要截断到实际接受的长度
        # (简化: 我们每步都重算，不复用 past)
        target_past = None
        draft_past = None
    
    t1 = time.perf_counter()
    
    output = tokenizer.decode(generated, skip_special_tokens=True)
    num_generated = len(generated) - len(input_ids[0])
    
    stats = {
        "num_generated": num_generated,
        "total_steps": total_steps,
        "total_draft": total_draft,
        "total_accepted": total_accepted,
        "acceptance_rate": total_accepted / total_draft if total_draft > 0 else 0,
        "tokens_per_step": total_accepted / total_steps if total_steps > 0 else 0,
        "time_ms": (t1 - t0) * 1000,
        "tokens_per_sec": num_generated / (t1 - t0) if t1 > t0 else 0,
    }
    
    return output, stats


@torch.no_grad()
def generate_standard(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.0,
) -> Tuple[str, dict]:
    """标准 autoregressive 生成（baseline 对比用）。"""
    
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated = input_ids[0].tolist()
    
    past = None
    t0 = time.perf_counter()
    
    for step in range(max_new_tokens):
        if past is None:
            out = model(input_ids=torch.tensor([generated]), use_cache=True)
        else:
            out = model(
                input_ids=torch.tensor([[generated[-1]]]),
                past_key_values=past,
                use_cache=True,
            )
        past = out.past_key_values
        
        logits = out.logits[:, -1, :]
        if temperature > 0:
            probs = F.softmax(logits / temperature, dim=-1)
            token = torch.multinomial(probs, 1).item()
        else:
            token = logits.argmax(-1).item()
        
        generated.append(token)
        if token == tokenizer.eos_token_id:
            break
    
    t1 = time.perf_counter()
    output = tokenizer.decode(generated, skip_special_tokens=True)
    num_generated = len(generated) - len(input_ids[0])
    
    return output, {
        "num_generated": num_generated,
        "total_steps": num_generated,
        "time_ms": (t1 - t0) * 1000,
        "tokens_per_sec": num_generated / (t1 - t0) if t1 > t0 else 0,
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Lesson 10: Speculative Decoding")
    print("=" * 60)
    
    # 加载 target model (GPT-2 medium, 345M) 和 draft model (GPT-2 small, 124M)
    # 注意：理想情况下 draft model 应该比 target 小很多
    # GPT-2 系列刚好有不同大小的版本
    print("\nLoading models...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # 用 GPT-2 (124M) 同时当 target 和 draft（演示算法）
    # 真实场景: target = 70B, draft = 7B
    target_model = AutoModelForCausalLM.from_pretrained("gpt2")
    draft_model = AutoModelForCausalLM.from_pretrained("gpt2")
    target_model.eval()
    draft_model.eval()
    
    prompt = "The future of artificial intelligence is"
    max_tokens = 30
    
    # --- 实验 1: 标准生成 (Baseline) ---
    print("\n" + "=" * 60)
    print("实验 1: 标准 Autoregressive (Baseline)")
    print("=" * 60)
    
    text_std, stats_std = generate_standard(
        target_model, tokenizer, prompt, max_tokens, temperature=0,
    )
    print(f"  Output: {text_std}")
    print(f"  Steps: {stats_std['total_steps']}")
    print(f"  Time: {stats_std['time_ms']:.0f}ms")
    print(f"  Speed: {stats_std['tokens_per_sec']:.1f} tok/s")
    
    # --- 实验 2: Speculative Decoding ---
    print("\n" + "=" * 60)
    print("实验 2: Speculative Decoding (K=4)")
    print("=" * 60)
    
    text_spec, stats_spec = generate_speculative(
        target_model, draft_model, tokenizer, prompt, max_tokens,
        num_speculative=4, temperature=0,
    )
    print(f"  Output: {text_spec}")
    print(f"  Steps: {stats_spec['total_steps']} (vs {stats_std['total_steps']} standard)")
    print(f"  Acceptance rate: {stats_spec['acceptance_rate']:.1%}")
    print(f"  Tokens per step: {stats_spec['tokens_per_step']:.1f}")
    print(f"  Time: {stats_spec['time_ms']:.0f}ms")
    print(f"  Speed: {stats_spec['tokens_per_sec']:.1f} tok/s")
    
    # --- 实验 3: 不同 K 值对比 ---
    print("\n" + "=" * 60)
    print("实验 3: 不同投机步数 K 的影响")
    print("=" * 60)
    
    print(f"\n  {'K':>3} | {'Steps':>5} | {'Accept%':>7} | {'Tok/Step':>8} | {'Time':>8} | {'Tok/s':>6}")
    print(f"  {'-'*50}")
    
    for k in [1, 2, 4, 6, 8]:
        _, stats = generate_speculative(
            target_model, draft_model, tokenizer, prompt, max_tokens,
            num_speculative=k, temperature=0,
        )
        print(f"  {k:>3} | {stats['total_steps']:>5} | "
              f"{stats['acceptance_rate']:>6.1%} | "
              f"{stats['tokens_per_step']:>8.1f} | "
              f"{stats['time_ms']:>6.0f}ms | "
              f"{stats['tokens_per_sec']:>5.1f}")
    
    # --- 实验 4: N-gram Drafting ---
    print("\n" + "=" * 60)
    print("实验 4: N-gram Drafting（无需 draft model）")
    print("=" * 60)
    
    drafter = NGramDrafter(n=2)
    
    # 用一些文本建立 n-gram 表
    sample_text = tokenizer.encode(
        "The future of artificial intelligence is a topic that has been "
        "discussed for decades. The future of AI will shape the world. "
        "The future of technology is bright and promising."
    )
    drafter.update(sample_text)
    
    # 测试猜测
    context = tokenizer.encode("The future of")
    drafted = drafter.draft(context, 5)
    drafted_text = tokenizer.decode(drafted) if drafted else "(no match)"
    print(f"  Context: 'The future of'")
    print(f"  N-gram draft: {drafted} → '{drafted_text}'")
    print(f"  N-gram table size: {len(drafter.table)} entries")
    
    # --- 实验 5: 输出一致性验证 ---
    print("\n" + "=" * 60)
    print("实验 5: 输出一致性验证 (Greedy)")
    print("=" * 60)
    
    prompts = [
        "Python is a",
        "The capital of France",
        "Machine learning",
    ]
    
    all_match = True
    for p in prompts:
        text_std, _ = generate_standard(target_model, tokenizer, p, 20, temperature=0)
        text_spec, _ = generate_speculative(
            target_model, draft_model, tokenizer, p, 20,
            num_speculative=4, temperature=0,
        )
        match = text_std == text_spec
        all_match = all_match and match
        status = "✅" if match else "❌"
        print(f"  {status} '{p[:25]}...'")
        if not match:
            print(f"    Standard: {text_std[:60]}")
            print(f"    Speculative: {text_spec[:60]}")
    
    print(f"\n  All outputs match: {'✅ Yes' if all_match else '❌ No'}")
    
    print("\n" + "=" * 60)
    print("Lesson 10 完成！")
    print("=" * 60)
    print("""
要点总结：
1. Draft-then-Verify: 小模型猜，大模型验，一次 forward 验证多个 token
2. 数学无损: 接受/拒绝概率保证输出分布与 target model 完全一致
3. 加速比取决于: draft/target 一致性（acceptance rate）× 投机步数 K
4. 同模型时 acceptance rate = 100%（greedy），不同模型时通常 70-90%
5. N-gram drafting 不需要额外模型，适合重复性强的文本

SGLang 的进阶:
→ EAGLE: 用 target 的 hidden state 训练轻量 draft head（更高 acceptance）
→ Tree-based verification: 同时验证多条候选路径
→ CUDA Graph + Speculative: 进一步减少 kernel launch 开销

实际加速效果:
→ Llama-70B + Llama-7B: ~2-3x 加速
→ EAGLE on Llama-3-70B: ~3-4x 加速
""")
