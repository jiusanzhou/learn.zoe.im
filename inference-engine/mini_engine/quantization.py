"""
Lesson 11: Quantization — 量化推理

量化 = 用更少的 bit 表示权重/激活值，换取更小的内存和更快的推理。

三种量化方式：
1. Weight-only: 只量化权重，推理时反量化回 FP16 计算
2. Weight + Activation (W8A8): 权重和激活值都量化，用 INT8 矩阵乘
3. KV Cache 量化: 把 KV Cache 存为 FP8/INT8，减少显存占用

对标 SGLang:
- layers/quantization/fp8.py → FP8 量化
- layers/quantization/w8a8_int8.py → INT8 量化
- layers/quantization/gptq.py → GPTQ (INT4 weight-only)
- layers/quantization/awq.py → AWQ (INT4 weight-only)
- layers/quantization/kv_cache.py → KV Cache FP8
"""

import math
import time
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# 第一部分：量化基础 — 手写量化/反量化
# ============================================================


def quantize_symmetric_int8(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """对称 INT8 量化。
    
    公式: q = round(x / scale)
    scale = max(|x|) / 127
    
    对标 SGLang: layers/quantization/int8_utils.py
    """
    abs_max = tensor.abs().max()
    scale = abs_max / 127.0
    if scale == 0:
        scale = torch.tensor(1.0)
    quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
    return quantized, scale


def dequantize_symmetric_int8(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """INT8 反量化。"""
    return quantized.float() * scale


def quantize_per_channel_int8(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-channel INT8 量化（每个输出通道独立 scale）。
    
    更精确：不同通道的值域可能差很多。
    weight: [out_features, in_features]
    """
    abs_max = weight.abs().max(dim=1, keepdim=True)[0]  # [out, 1]
    scale = abs_max / 127.0
    scale = scale.clamp(min=1e-10)
    quantized = torch.round(weight / scale).clamp(-128, 127).to(torch.int8)
    return quantized, scale.squeeze()


def quantize_per_group_int4(
    weight: torch.Tensor, group_size: int = 128
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-group INT4 量化 (GPTQ/AWQ 风格)。
    
    每 group_size 个元素共享一个 scale。
    INT4 范围: [-8, 7]
    
    weight: [out_features, in_features]
    """
    out_features, in_features = weight.shape
    assert in_features % group_size == 0
    
    # Reshape to groups
    weight_groups = weight.reshape(out_features, -1, group_size)  # [out, groups, gs]
    
    abs_max = weight_groups.abs().max(dim=2, keepdim=True)[0]
    scale = abs_max / 7.0
    scale = scale.clamp(min=1e-10)
    
    quantized = torch.round(weight_groups / scale).clamp(-8, 7).to(torch.int8)
    
    return quantized.reshape(out_features, in_features), scale.squeeze()


def dequantize_per_group_int4(
    quantized: torch.Tensor, scale: torch.Tensor, group_size: int = 128
) -> torch.Tensor:
    """INT4 反量化。"""
    out_features, in_features = quantized.shape
    q_groups = quantized.reshape(out_features, -1, group_size).float()
    return (q_groups * scale.unsqueeze(-1)).reshape(out_features, in_features)


# ============================================================
# 第二部分：量化 Linear 层
# ============================================================


class QuantizedLinearINT8(nn.Module):
    """INT8 Weight-only Quantized Linear。
    
    权重存为 INT8，推理时反量化回 FP32 再做矩阵乘。
    真正的实现用 INT8 矩阵乘 kernel（cutlass_scaled_mm）。
    """
    
    def __init__(self, original: nn.Linear):
        super().__init__()
        weight = original.weight.data  # [out, in]
        
        self.quantized_weight, self.scale = quantize_per_channel_int8(weight)
        self.bias = original.bias
        
        # 统计
        self.original_bytes = weight.numel() * weight.element_size()
        self.quantized_bytes = (self.quantized_weight.numel() * 1 +  # int8 = 1 byte
                                self.scale.numel() * 4)  # float32 scale
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 反量化 + 矩阵乘
        weight_fp = dequantize_symmetric_int8(
            self.quantized_weight,
            self.scale.unsqueeze(1),  # [out, 1]
        )
        out = F.linear(x, weight_fp, self.bias)
        return out


class QuantizedLinearINT4(nn.Module):
    """INT4 Weight-only Quantized Linear (GPTQ 风格)。"""
    
    def __init__(self, original: nn.Linear, group_size: int = 128):
        super().__init__()
        weight = original.weight.data
        self.group_size = group_size
        
        # Pad if needed
        in_features = weight.shape[1]
        if in_features % group_size != 0:
            pad = group_size - (in_features % group_size)
            weight = F.pad(weight, (0, pad))
        
        self.quantized_weight, self.scale = quantize_per_group_int4(weight, group_size)
        self.bias = original.bias
        self.in_features = in_features
        
        self.original_bytes = original.weight.numel() * original.weight.element_size()
        # INT4 实际存储可以 2 个值打包成 1 byte
        self.quantized_bytes = (self.quantized_weight.numel() // 2 +
                                self.scale.numel() * 4)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight_fp = dequantize_per_group_int4(
            self.quantized_weight, self.scale, self.group_size
        )
        weight_fp = weight_fp[:, :self.in_features]
        out = F.linear(x, weight_fp, self.bias)
        return out


# ============================================================
# 第三部分：量化整个模型
# ============================================================


def quantize_model_int8(model: AutoModelForCausalLM) -> Tuple[AutoModelForCausalLM, dict]:
    """将模型的所有 Linear 层替换为 INT8 量化版本。"""
    stats = {"original_bytes": 0, "quantized_bytes": 0, "layers_quantized": 0}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            q_linear = QuantizedLinearINT8(module)
            stats["original_bytes"] += q_linear.original_bytes
            stats["quantized_bytes"] += q_linear.quantized_bytes
            stats["layers_quantized"] += 1
            
            # Replace module
            parts = name.split(".")
            parent = model
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], q_linear)
    
    stats["compression"] = stats["original_bytes"] / stats["quantized_bytes"] if stats["quantized_bytes"] > 0 else 0
    return model, stats


def quantize_model_int4(model: AutoModelForCausalLM, group_size: int = 128) -> Tuple[AutoModelForCausalLM, dict]:
    """将模型的所有 Linear 层替换为 INT4 量化版本。"""
    stats = {"original_bytes": 0, "quantized_bytes": 0, "layers_quantized": 0}
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            try:
                q_linear = QuantizedLinearINT4(module, group_size)
                stats["original_bytes"] += q_linear.original_bytes
                stats["quantized_bytes"] += q_linear.quantized_bytes
                stats["layers_quantized"] += 1
                
                parts = name.split(".")
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                setattr(parent, parts[-1], q_linear)
            except Exception:
                pass  # Skip layers that can't be quantized (e.g., odd dimensions)
    
    stats["compression"] = stats["original_bytes"] / stats["quantized_bytes"] if stats["quantized_bytes"] > 0 else 0
    return model, stats


# ============================================================
# 第四部分：量化误差分析
# ============================================================


def analyze_quantization_error(weight: torch.Tensor, group_size: int = 128):
    """分析不同量化精度的误差。"""
    
    results = {}
    
    # INT8 per-tensor
    q8, s8 = quantize_symmetric_int8(weight)
    dq8 = dequantize_symmetric_int8(q8, s8)
    results["int8_tensor"] = {
        "mse": F.mse_loss(dq8, weight).item(),
        "max_error": (dq8 - weight).abs().max().item(),
        "relative_error": ((dq8 - weight).abs() / (weight.abs() + 1e-10)).mean().item(),
    }
    
    # INT8 per-channel
    q8c, s8c = quantize_per_channel_int8(weight)
    dq8c = dequantize_symmetric_int8(q8c, s8c.unsqueeze(1))
    results["int8_channel"] = {
        "mse": F.mse_loss(dq8c, weight).item(),
        "max_error": (dq8c - weight).abs().max().item(),
        "relative_error": ((dq8c - weight).abs() / (weight.abs() + 1e-10)).mean().item(),
    }
    
    # INT4 per-group
    if weight.shape[1] % group_size == 0:
        q4, s4 = quantize_per_group_int4(weight, group_size)
        dq4 = dequantize_per_group_int4(q4, s4, group_size)
        results["int4_group"] = {
            "mse": F.mse_loss(dq4, weight).item(),
            "max_error": (dq4 - weight).abs().max().item(),
            "relative_error": ((dq4 - weight).abs() / (weight.abs() + 1e-10)).mean().item(),
        }
    
    return results


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Lesson 11: Quantization")
    print("=" * 60)
    
    # --- 实验 1: 量化误差分析 ---
    print("\n" + "=" * 60)
    print("实验 1: 量化误差分析")
    print("=" * 60)
    
    # 用 GPT-2 的实际权重
    print("\nLoading GPT-2...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model_fp32 = AutoModelForCausalLM.from_pretrained("gpt2")
    model_fp32.eval()
    
    # 取一个 Linear 层的权重分析
    sample_weight = model_fp32.transformer.h[0].mlp.c_fc.weight.data
    print(f"  Weight shape: {sample_weight.shape}")
    print(f"  Weight range: [{sample_weight.min():.4f}, {sample_weight.max():.4f}]")
    print(f"  Weight std: {sample_weight.std():.4f}")
    
    errors = analyze_quantization_error(sample_weight)
    
    print(f"\n  {'Method':<18} {'MSE':>12} {'Max Error':>12} {'Rel Error':>12}")
    print(f"  {'-'*55}")
    for method, err in errors.items():
        print(f"  {method:<18} {err['mse']:>12.2e} {err['max_error']:>12.4f} "
              f"{err['relative_error']:>11.2%}")
    
    # --- 实验 2: INT8 量化模型 ---
    print("\n" + "=" * 60)
    print("实验 2: INT8 量化 GPT-2")
    print("=" * 60)
    
    model_int8 = AutoModelForCausalLM.from_pretrained("gpt2")
    model_int8.eval()
    model_int8, stats8 = quantize_model_int8(model_int8)
    
    print(f"  Layers quantized: {stats8['layers_quantized']}")
    print(f"  Original: {stats8['original_bytes']/1024/1024:.1f} MB")
    print(f"  INT8: {stats8['quantized_bytes']/1024/1024:.1f} MB")
    print(f"  Compression: {stats8['compression']:.1f}x")
    
    # --- 实验 3: INT4 量化模型 ---
    print("\n" + "=" * 60)
    print("实验 3: INT4 量化 GPT-2")
    print("=" * 60)
    
    model_int4 = AutoModelForCausalLM.from_pretrained("gpt2")
    model_int4.eval()
    model_int4, stats4 = quantize_model_int4(model_int4, group_size=128)
    
    print(f"  Layers quantized: {stats4['layers_quantized']}")
    print(f"  Original: {stats4['original_bytes']/1024/1024:.1f} MB")
    print(f"  INT4: {stats4['quantized_bytes']/1024/1024:.1f} MB")
    print(f"  Compression: {stats4['compression']:.1f}x")
    
    # --- 实验 4: 输出质量对比 ---
    print("\n" + "=" * 60)
    print("实验 4: 输出质量对比 (FP32 vs INT8 vs INT4)")
    print("=" * 60)
    
    prompt = "The capital of France is"
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    for name, model in [("FP32", model_fp32), ("INT8", model_int8), ("INT4", model_int4)]:
        generated = input_ids[0].tolist()
        past = None
        
        t0 = time.perf_counter()
        for _ in range(20):
            if past is None:
                out = model(input_ids=torch.tensor([generated]), use_cache=True)
            else:
                out = model(input_ids=torch.tensor([[generated[-1]]]), 
                           past_key_values=past, use_cache=True)
            past = out.past_key_values
            token = out.logits[:, -1, :].argmax(-1).item()
            generated.append(token)
        t1 = time.perf_counter()
        
        text = tokenizer.decode(generated, skip_special_tokens=True)
        print(f"\n  [{name}] ({(t1-t0)*1000:.0f}ms)")
        print(f"  {text}")
    
    # --- 实验 5: Perplexity 对比 ---
    print("\n" + "=" * 60)
    print("实验 5: 困惑度 (Perplexity) 对比")
    print("=" * 60)
    
    test_text = "The quick brown fox jumps over the lazy dog. Machine learning is transforming the world of technology."
    test_ids = tokenizer.encode(test_text, return_tensors="pt")
    
    for name, model in [("FP32", model_fp32), ("INT8", model_int8), ("INT4", model_int4)]:
        with torch.no_grad():
            outputs = model(input_ids=test_ids, labels=test_ids)
            loss = outputs.loss.item()
            ppl = math.exp(loss)
        print(f"  {name}: loss={loss:.4f}, perplexity={ppl:.2f}")
    
    # --- 实验 6: 内存对比总结 ---
    print("\n" + "=" * 60)
    print("实验 6: 内存占用总结")
    print("=" * 60)
    
    total_params = sum(p.numel() for p in model_fp32.parameters())
    
    print(f"\n  GPT-2: {total_params/1e6:.0f}M parameters")
    print(f"\n  {'Precision':<12} {'Size':>10} {'Compression':>12} {'PPL Impact':>12}")
    print(f"  {'-'*48}")
    print(f"  {'FP32':<12} {total_params*4/1024/1024:>8.1f} MB {'1.0x':>12} {'baseline':>12}")
    print(f"  {'FP16':<12} {total_params*2/1024/1024:>8.1f} MB {'2.0x':>12} {'~0%':>12}")
    print(f"  {'INT8':<12} {stats8['quantized_bytes']/1024/1024:>8.1f} MB "
          f"{stats8['compression']:>11.1f}x {'<1%':>12}")
    print(f"  {'INT4':<12} {stats4['quantized_bytes']/1024/1024:>8.1f} MB "
          f"{stats4['compression']:>11.1f}x {'1-3%':>12}")
    
    print("\n" + "=" * 60)
    print("Lesson 11 完成！")
    print("=" * 60)
    print("""
要点总结：
1. 量化 = 用更少 bit 表示模型参数，换取内存和速度
2. Per-channel/per-group 比 per-tensor 更精确（粒度更细）
3. INT8 几乎无损（MSE ~1e-7），INT4 有轻微退化
4. Weight-only 量化最简单：存 INT8/INT4，推理时反量化回 FP
5. 实际部署：INT8 是甜蜜点（4x 压缩 + 几乎无损）

SGLang 的实际做法：
→ FP8 W8A8: H100/Ada 原生支持，无需反量化（硬件直接算）
→ GPTQ/AWQ: 训练后量化 INT4，有校准数据集
→ KV Cache FP8: 大幅减少 KV Cache 显存（tokens 翻倍）
→ MoE INT4+FP8: 混合精度推理

实际效果（Llama-3-70B）:
→ FP16: 140GB → 2× A100
→ INT8: 70GB → 1× A100
→ INT4: 35GB → 1× A100 还有余
""")
