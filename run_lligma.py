#!/usr/bin/env python3
from lligma.model.config import LLigMaConfig
from lligma.model.lligma import LLigMaForCausalLM
from lligma.tokenizer import LLigMaTokenizer
from lligma.inference.generate import LLigMaGenerator
import torch

if __name__ == "__main__":
    print("=== LLigMa 3.0 (2B) ===")
    config = LLigMaConfig()
    model = LLigMaForCausalLM(config)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {params/1e9:.2f}B parameters")
    print(f"Active/MoE: {config.num_experts} routed experts + {config.num_shared_experts} shared, top-{config.num_experts_per_tok}")
    print(f"Attention: MLA (kv_lora_rank={config.kv_lora_rank}), {config.num_attention_heads}Q/{config.num_key_value_heads}KV heads")
    print(f"Context: {config.max_position_embeddings} (YaRN)")
    print(f"Layers: {config.num_hidden_layers}")
    
    tokenizer = LLigMaTokenizer.from_pretrained('lligma-3.0-2b')
    generator = LLigMaGenerator(model, tokenizer)
    
    prompt = "Explain quantum entanglement step by step."
    print(f"\nPrompt: {prompt}")
    response = generator.generate_with_reasoning(prompt, max_new_tokens=50)
    print(f"\nResponse: {response}")
    print("\n=== Done ===")
