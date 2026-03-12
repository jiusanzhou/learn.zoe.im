# Lesson 1: Transformer 前向推理 — 从零开始

## 目标

不用任何框架的 `model.generate()`，手动实现 autoregressive 文本生成：
1. 加载预训练模型权重
2. 手写前向推理循环
3. 手写采样策略
4. 理解 KV Cache 的作用（本节先不实现，感受痛点）

## 背景知识

### Autoregressive 生成

LLM 是 **自回归** 模型 — 每次只生成一个 token，然后把它拼回输入，再生成下一个：

```
输入: "The capital of France is"
第1步: model(["The", "capital", "of", "France", "is"]) → "Paris"
第2步: model(["The", "capital", "of", "France", "is", "Paris"]) → ","
第3步: model(["The", "capital", "of", "France", "is", "Paris", ","]) → "a"
...
```

**关键问题**：每一步都要重新算整个序列！序列越长越慢。这就是 KV Cache 要解决的问题（Lesson 2）。

### Transformer Decoder 结构

```
Input IDs → Embedding → [DecoderLayer × N] → LayerNorm → LM Head → Logits
                              │
                    ┌─────────┴──────────┐
                    │   DecoderLayer      │
                    │  ┌───────────────┐  │
                    │  │ LayerNorm     │  │
                    │  │ Self-Attention│  │  ← RoPE 位置编码
                    │  │ + Residual    │  │
                    │  ├───────────────┤  │
                    │  │ LayerNorm     │  │
                    │  │ FFN (MLP)     │  │  ← SiLU(gate * up) * down
                    │  │ + Residual    │  │
                    │  └───────────────┘  │
                    └────────────────────┘
```

### 采样策略

模型输出的是 logits（每个 token 的得分），需要通过采样变成 token：

- **Greedy**: 直接取 argmax（确定性，但无创意）
- **Temperature**: logits / temperature，温度越高越随机
- **Top-p (Nucleus)**: 只从累积概率 ≤ p 的 token 中采样
- **Top-k**: 只从得分最高的 k 个 token 中采样

## SGLang 对照

| 我们做的 | SGLang 对应 |
|---------|------------|
| 加载模型 | `model_loader/` + `models/llama.py` |
| 前向推理 | `LlamaForCausalLM.forward()` |
| Attention | `RadixAttention`（我们先用原生 attention） |
| MLP | `LlamaMLP` — gate_up + SiLU + down |
| 采样 | `sampling/sampling_batch_info.py` |

SGLang 里 `LlamaForCausalLM.forward()` 做的事：
1. `embed_tokens(input_ids)` → 词嵌入
2. 逐层跑 `LlamaDecoderLayer`（attention + MLP + residual）
3. `norm` → 最后的 LayerNorm
4. `logits_processor` → 把 hidden states 映射到 vocab logits

## 代码

见 `mini_engine/model.py` — 我们用 GPT-2 (124M) 作为第一个模型，足够小可以在 CPU 跑。

## 运行

```bash
cd /Users/zoe/projects/learn.zoe.im/inference-engine
python -m mini_engine.model
```
