import torch
from typing import Optional, List, Union
from transformers import GenerationConfig


class LLigMaGenerator:
    """Generation utilities for LLigMa with reasoning support."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]],
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        use_cache: bool = True,
        enable_thinking: bool = True,
        thinking_budget: Optional[int] = None,
        **kwargs
    ):
        """
        Generate text with optional thinking/reasoning mode.
        Inspired by DeepSeek's reasoning format and Kimi's approach.
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            use_cache=use_cache,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs
        )
        
        outputs = self.model.generate(**inputs, generation_config=generation_config)
        
        responses = []
        for i, output in enumerate(outputs):
            response = self.tokenizer.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            responses.append(response)
        
        return responses[0] if len(responses) == 1 else responses
    
    def generate_with_reasoning(self, prompt: str, max_reasoning_tokens: int = 4096, **kwargs):
        """Generate with explicit reasoning/thinking mode."""
        reasoning_prompt = f"{prompt}\n\nLet us think step by step.\n"
        kwargs['enable_thinking'] = True
        return self.generate(reasoning_prompt, max_new_tokens=max_reasoning_tokens, **kwargs)
