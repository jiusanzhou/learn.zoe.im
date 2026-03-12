"""
Lesson 1: Transformer 前向推理 — 从零开始

用 GPT-2 (124M) 演示完整的 autoregressive 生成过程。
不用 model.generate()，手写每一步。

学习要点：
1. 模型加载和权重结构
2. 前向推理：input_ids → logits
3. 采样策略：greedy / temperature / top-p / top-k
4. Autoregressive 循环：逐 token 生成
5. 感受没有 KV Cache 的痛点
"""

import time
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 第一部分：采样策略
# ============================================================
# 对标 SGLang: srt/sampling/sampling_batch_info.py
# SGLang 把采样参数 batch 化处理（向量化 temperature/top_p/top_k）
# 我们先实现单请求版本，理解原理


def sample_greedy(logits: torch.Tensor) -> int:
    """贪心采样 — 直接取概率最大的 token。
    
    确定性输出，适合代码生成、事实性回答。
    """
    return logits.argmax(dim=-1).item()


def sample_temperature(logits: torch.Tensor, temperature: float = 1.0) -> int:
    """温度采样 — 控制随机性。
    
    temperature < 1: 更确定（分布更尖锐）
    temperature = 1: 原始分布
    temperature > 1: 更随机（分布更平坦）
    """
    if temperature == 0:
        return sample_greedy(logits)
    
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def sample_top_p(logits: torch.Tensor, top_p: float = 0.9, temperature: float = 1.0) -> int:
    """Top-p (Nucleus) 采样 — 从累积概率 ≤ p 的最小集合中采样。
    
    自适应地选择候选 token 数量：
    - 分布集中时，候选少（快速收敛）
    - 分布分散时，候选多（保持多样性）
    """
    if temperature != 1.0:
        logits = logits / temperature
    
    # 按概率降序排列
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # 找到累积概率超过 top_p 的位置，把后面的 token 过滤掉
    sorted_indices_to_remove = cumulative_probs > top_p
    # 保证至少保留一个 token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False
    
    # 把被过滤的 token 的 logits 设为 -inf
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    logits[indices_to_remove] = float('-inf')
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def sample_top_k(logits: torch.Tensor, top_k: int = 50, temperature: float = 1.0) -> int:
    """Top-k 采样 — 只从得分最高的 k 个 token 中选择。"""
    if temperature != 1.0:
        logits = logits / temperature
    
    # 找到 top-k 之外的 token，设为 -inf
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = float('-inf')
    
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def sample(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
) -> int:
    """统一采样入口。"""
    if temperature == 0:
        return sample_greedy(logits)
    if top_k > 0:
        return sample_top_k(logits, top_k=top_k, temperature=temperature)
    if top_p < 1.0:
        return sample_top_p(logits, top_p=top_p, temperature=temperature)
    return sample_temperature(logits, temperature=temperature)


# ============================================================
# 第二部分：前向推理（无 KV Cache）
# ============================================================
# 对标 SGLang: models/llama.py LlamaForCausalLM.forward()
# 我们直接用 HuggingFace 模型的 forward()，但不用 generate()
# 重点是理解 autoregressive 循环


@torch.no_grad()
def generate_without_kv_cache(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
) -> str:
    """不使用 KV Cache 的自回归生成。
    
    每一步都重新计算整个序列的 attention。
    这是最朴素的实现，用来感受 KV Cache 的必要性。
    
    对标 SGLang 流程：
    1. tokenize → input_ids
    2. model.forward(input_ids) → logits
    3. sample(logits[-1]) → next_token
    4. 拼接 next_token，回到 2
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated_ids = input_ids[0].tolist()
    
    print(f"Prompt: {prompt!r}")
    print(f"Input tokens: {len(generated_ids)}")
    print(f"Generating (no KV Cache)...")
    print("-" * 50)
    
    total_forward_time = 0
    
    for step in range(max_new_tokens):
        # ⚠️ 每次都传入完整序列 — 这就是没有 KV Cache 的代价
        # 序列长度 N → attention 复杂度 O(N²)
        # 而且之前的 KV 每次都重新算，纯浪费
        step_input = torch.tensor([generated_ids])
        
        t0 = time.perf_counter()
        outputs = model(input_ids=step_input, use_cache=False)  # 显式关闭 KV Cache
        t1 = time.perf_counter()
        
        forward_time = t1 - t0
        total_forward_time += forward_time
        
        # 只取最后一个位置的 logits（因为我们只需要预测下一个 token）
        next_token_logits = outputs.logits[0, -1, :]
        
        # 采样
        next_token_id = sample(
            next_token_logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        
        generated_ids.append(next_token_id)
        
        # 解码当前 token 并打印
        token_text = tokenizer.decode([next_token_id])
        print(f"  Step {step+1}: token={next_token_id:5d} | "
              f"text={token_text!r:15s} | "
              f"seq_len={len(generated_ids):3d} | "
              f"forward={forward_time*1000:.1f}ms")
        
        # 检查是否生成了 EOS
        if next_token_id == tokenizer.eos_token_id:
            print("  [EOS reached]")
            break
    
    print("-" * 50)
    print(f"Total forward time: {total_forward_time*1000:.1f}ms")
    print(f"Avg per step: {total_forward_time/max_new_tokens*1000:.1f}ms")
    
    # 解码完整输出
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return full_text


# ============================================================
# 第三部分：对比 — 有 KV Cache 的版本
# ============================================================
# 预览 Lesson 2 的内容，先用 HuggingFace 内置的 KV Cache 对比性能


@torch.no_grad()
def generate_with_kv_cache(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
) -> str:
    """使用 KV Cache 的自回归生成。
    
    第一步（prefill）：处理完整 prompt，缓存所有 KV。
    后续步骤（decode）：只传入新 token，复用之前的 KV Cache。
    
    对标 SGLang:
    - Prefill: ForwardMode.PREFILL / EXTEND
    - Decode: ForwardMode.DECODE
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    generated_ids = input_ids[0].tolist()
    
    print(f"Prompt: {prompt!r}")
    print(f"Input tokens: {len(generated_ids)}")
    print(f"Generating (with KV Cache)...")
    print("-" * 50)
    
    past_key_values = None
    total_forward_time = 0
    
    for step in range(max_new_tokens):
        if past_key_values is None:
            # Prefill: 第一次传入完整序列
            step_input = torch.tensor([generated_ids])
        else:
            # Decode: 只传入最新的 token
            step_input = torch.tensor([[generated_ids[-1]]])
        
        t0 = time.perf_counter()
        outputs = model(input_ids=step_input, past_key_values=past_key_values, use_cache=True)
        t1 = time.perf_counter()
        
        forward_time = t1 - t0
        total_forward_time += forward_time
        
        # 更新 KV Cache
        past_key_values = outputs.past_key_values
        
        # 采样
        next_token_logits = outputs.logits[0, -1, :]
        next_token_id = sample(
            next_token_logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        
        generated_ids.append(next_token_id)
        
        token_text = tokenizer.decode([next_token_id])
        mode = "prefill" if step == 0 else "decode "
        print(f"  Step {step+1} [{mode}]: token={next_token_id:5d} | "
              f"text={token_text!r:15s} | "
              f"input_len={step_input.shape[1]:3d} | "
              f"forward={forward_time*1000:.1f}ms")
        
        if next_token_id == tokenizer.eos_token_id:
            print("  [EOS reached]")
            break
    
    print("-" * 50)
    print(f"Total forward time: {total_forward_time*1000:.1f}ms")
    print(f"Avg per step: {total_forward_time/max_new_tokens*1000:.1f}ms")
    
    full_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return full_text


# ============================================================
# Main: 运行对比实验
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Lesson 1: Transformer 前向推理")
    print("=" * 60)
    
    # 加载 GPT-2 (124M params) — 足够小可以在 CPU 跑
    print("\nLoading GPT-2...")
    model_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    
    # 打印模型结构概览
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name}")
    print(f"Parameters: {total_params:,} ({total_params/1e6:.0f}M)")
    print(f"Layers: {model.config.n_layer}")
    print(f"Hidden size: {model.config.n_embd}")
    print(f"Attention heads: {model.config.n_head}")
    print(f"Vocab size: {model.config.vocab_size}")
    
    prompt = "The future of artificial intelligence is"
    max_tokens = 30
    
    # 实验 1: 无 KV Cache
    print("\n" + "=" * 60)
    print("实验 1: 无 KV Cache（每步重算整个序列）")
    print("=" * 60)
    text1 = generate_without_kv_cache(
        model, tokenizer, prompt,
        max_new_tokens=max_tokens,
        temperature=0,  # greedy，便于对比
    )
    print(f"\nGenerated: {text1}")
    
    # 实验 2: 有 KV Cache
    print("\n" + "=" * 60)
    print("实验 2: 有 KV Cache（decode 阶段只传新 token）")
    print("=" * 60)
    text2 = generate_with_kv_cache(
        model, tokenizer, prompt,
        max_new_tokens=max_tokens,
        temperature=0,  # greedy，便于对比
    )
    print(f"\nGenerated: {text2}")
    
    # 实验 3: 不同采样策略
    print("\n" + "=" * 60)
    print("实验 3: 采样策略对比（同一个 prompt，不同策略）")
    print("=" * 60)
    
    strategies = [
        ("Greedy (temperature=0)", dict(temperature=0)),
        ("Temperature=0.7", dict(temperature=0.7)),
        ("Temperature=1.5", dict(temperature=1.5)),
        ("Top-p=0.9", dict(temperature=1.0, top_p=0.9)),
        ("Top-k=50", dict(temperature=1.0, top_k=50)),
    ]
    
    for name, params in strategies:
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        generated = input_ids[0].tolist()
        past = None
        
        for _ in range(max_tokens):
            if past is None:
                step_input = torch.tensor([generated])
            else:
                step_input = torch.tensor([[generated[-1]]])
            
            outputs = model(input_ids=step_input, past_key_values=past, use_cache=True)
            past = outputs.past_key_values
            
            next_id = sample(outputs.logits[0, -1, :], **params)
            generated.append(next_id)
            
            if next_id == tokenizer.eos_token_id:
                break
        
        text = tokenizer.decode(generated, skip_special_tokens=True)
        print(f"\n  [{name}]")
        print(f"  {text}")
    
    print("\n" + "=" * 60)
    print("Lesson 1 完成！")
    print("=" * 60)
    print("""
要点总结：
1. LLM 是自回归模型，每次只生成一个 token
2. 没有 KV Cache → 每步重算整个序列 → O(N²) 且重复计算
3. 有 KV Cache → decode 阶段只算新 token → 大幅加速
4. 采样策略决定输出的多样性和质量

下一步（Lesson 2）：
→ 手写 KV Cache 管理，理解它的内存布局和生命周期
""")
