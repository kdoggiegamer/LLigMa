# LLigMa 3.0 Architecture

## Overview

LLigMa 3.0 is built on a hybrid architecture combining insights from DeepSeek-V3 and Kimi K2/K3, optimized for a 2B parameter footprint with strong reasoning capabilities.

## Key Components

### 1. Multi-Head Latent Attention (MLA)

Inspired by DeepSeek's MLA mechanism:
- **Latent KV Compression**: KV states compressed into low-rank latent vectors (kv_lora_rank=512)
- **Query Compression**: Optional LoRA compression for queries (q_lora_rank=1536)
- **Decoupled RoPE**: Separate rotary embeddings for positional information
- **Massive KV Cache Reduction**: Up to 93% reduction compared to standard MHA

### 2. DeepSeek-style Fine-Grained MoE

- **64 Routed Experts**: Each expert is a small SwiGLU FFN (intermediate_size=1408)
- **2 Shared Experts**: Always active, capture common knowledge
- **Top-6 Routing**: Sparse activation for compute efficiency
- **Routed Scaling**: Factor of 2.5 to balance expert contributions
- **Load Balancing**: Implicit through routing with normalization

### 3. RoPE with YaRN Scaling

Kimi-inspired long-context capabilities:
- **Base Theta**: 10000 with YaRN interpolation
- **128K Context**: Extended from 4K training context using YaRN
- **Frequency Interpolation**: Different scaling for high/low frequency dimensions
- **Beta Parameters**: Fast (32) and slow (1) correction ranges

### 4. Modern Transformer Components

- **RMSNorm**: Pre-normalization for stability
- **SwiGLU Activation**: More expressive than ReLU/GELU
- **GQA**: 16 query heads, 4 KV heads
- **RoPE**: Rotary positional embeddings
- **Deep Residual Connections**: 28 layers for depth

## Parameter Distribution

For 2B active parameters:
- Embeddings: ~262M (vocab_size * hidden_size)
- Attention (MLA): ~150M per layer * 28 = ~4.2B (but compressed)
- MoE: Majority of parameters in routed/shared experts
- Active per token: ~2B, Total: ~12-16B

## Reasoning Design

- **Long CoT Support**: Optimized for extended thinking sequences
- **Native Thinking Mode**: Built-in support for step-by-step reasoning
- **Context Efficiency**: MLA enables long reasoning chains within memory limits
- **Hybrid Architecture**: First layer dense, rest MoE for flexibility

## Comparison to Inspirations

| Feature | DeepSeek-V3 | Kimi K3 | LLigMa 3.0 |
|---|---|---|---|
| MLA | ✓ | Partial | ✓ |
| Fine-grained MoE | ✓ | ✓ | ✓ |
| Long Context (128K+) | ✓ | ✓ (128K+) | ✓ |
| Reasoning Focus | ✓ | ✓ | ✓ |
| 2B Scale | - | - | ✓ (optimized) |
| Local Deployable | Partial | No | ✓ |
| YaRN Scaling | ✓ | ✓ | ✓ |
