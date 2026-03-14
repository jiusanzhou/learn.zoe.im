# Lesson 11: Quantization — 量化推理

## 核心问题

LLM 的参数量巨大，内存是瓶颈：
```
Llama-3-70B (FP16): 70B × 2 bytes = 140 GB → 需要 2× A100 80GB
Llama-3-70B (INT4):  70B × 0.5 bytes = 35 GB → 1× A100 就够！
```

## 量化类型

| 方法 | 精度 | 压缩比 | 精度损失 | 代表 |
|------|------|--------|---------|------|
| FP32 | 32-bit | 1x | 无 | 训练原始 |
| FP16/BF16 | 16-bit | 2x | 极小 | 标准推理 |
| FP8 | 8-bit | 4x | 小 | H100 原生 |
| INT8 | 8-bit | 4x | 小 | W8A8, SmoothQuant |
| INT4 | 4-bit | 8x | 中等 | GPTQ, AWQ |

## SGLang 支持

SGLang 在 `layers/quantization/` 下支持：
- FP8 (W8A8): `fp8.py` — H100/Ada 原生支持
- INT8 (W8A8): `w8a8_int8.py` — SmoothQuant
- GPTQ: `gptq.py` — 训练后量化
- AWQ: `awq.py` — 激活感知量化
- BitsAndBytes: `bitsandbytes.py` — NF4
- KV Cache 量化: `kv_cache.py` — FP8 KV

## 代码

见 `mini_engine/quantization.py`
