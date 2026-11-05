import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
    
    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Expert(nn.Module):
    """Individual expert in MoE."""
    def __init__(self, config):
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
        self.w2 = nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False)
    
    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoEGate(nn.Module):
    """Router for Mixture of Experts."""
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.routed_scaling_factor = config.routed_scaling_factor
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Linear(config.hidden_size, config.num_experts, bias=False)
    
    def forward(self, hidden_states):
        if hidden_states.ndim == 3:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        elif hidden_states.ndim != 2:
            raise ValueError("MoE gate expects [tokens, hidden] or [batch, sequence, hidden]")
        logits = self.weight(hidden_states)
        scores = F.softmax(logits, dim=-1, dtype=torch.float32).to(logits.dtype)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        
        if self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator
        
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_weight, topk_idx


class DeepSeekMoE(nn.Module):
    """
    DeepSeek-style fine-grained Mixture of Experts.
    Combines routed experts and shared experts.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.num_shared_experts = config.num_shared_experts
        
        # Router
        self.gate = MoEGate(config)
        
        # Routed experts
        self.experts = nn.ModuleList([Expert(config) for _ in range(self.num_experts)])
        
        # Shared experts (always active)
        if self.num_shared_experts > 0:
            self.shared_experts = SwiGLU(
                config.hidden_size,
                config.moe_intermediate_size * self.num_shared_experts,
                bias=False
            )
    
    def forward(self, hidden_states):
        identity = hidden_states
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        
        # Get routing
        topk_weight, topk_idx = self.gate(hidden_states)
        
        # Compute shared expert output
        if self.num_shared_experts > 0:
            shared_output = self.shared_experts(hidden_states)
        else:
            shared_output = 0
        
        # Compute routed expert output
        final_hidden_states = torch.zeros_like(hidden_states)
        
        # Process each expert
        expert_mask = torch.nn.functional.one_hot(topk_idx, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx >= self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            expert_out = self.experts[expert_idx](current_state)
            final_hidden_states[token_idx] += topk_weight[token_idx, top_k_pos, None] * expert_out
        
        final_hidden_states = final_hidden_states + shared_output
        final_hidden_states = final_hidden_states.view(bsz, seq_len, h)
        return final_hidden_states
