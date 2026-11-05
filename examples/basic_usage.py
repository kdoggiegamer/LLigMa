#!/usr/bin/env python3
"""
Basic usage example for LLigMa 3.0
"""
from lligma import LLigMaForCausalLM, LLigMaConfig
from lligma.inference.generate import LLigMaGenerator


def main():
    # Create model from config
    config = LLigMaConfig()
    model = LLigMaForCausalLM(config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B parameters")
    
    # For inference with trained weights
    # tokenizer = LLigMaTokenizer.from_pretrained("path/to/lligma-3.0-2b")
    # generator = LLigMaGenerator(model, tokenizer)
    # response = generator.generate_with_reasoning("Explain quantum entanglement")
    # print(response)


if __name__ == "__main__":
    main()
