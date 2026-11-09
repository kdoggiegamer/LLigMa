import torch
import torch.nn as nn
from typing import Optional, Tuple
import math


class RoPE(nn.Module):
    """Rotary Positional Embeddings with YaRN scaling for long context."""
    
    def __init__(self, dim, max_position_embeddings=131072, base=10000.0, 
                 scaling_config=None, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scaling_config = scaling_config or {}
        
        # Compute inverse frequencies with scaling
        if self.scaling_config and self.scaling_config.get("type") == "yarn":
            self._compute_yarn_freqs(device)
        else:
            self._compute_freqs(device)
        
        self.register_buffer("cos_cached", None, persistent=False)
        self.register_buffer("sin_cached", None, persistent=False)
    
    def _compute_freqs(self, device=None):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def _compute_yarn_freqs(self, device=None):
        """YaRN (Yet another RoPE extensioN) scaling."""
        config = self.scaling_config
        factor = config.get("factor", 32.0)
        beta_fast = config.get("beta_fast", 32)
        beta_slow = config.get("beta_slow", 1)
        original_max = config.get("original_max_position_embeddings", 4096)
        
        # Use original scaling for lower frequencies, interpolation for higher
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (factor * pos_freqs)
        
        # Find correction range
        low, high = self._yarn_find_correction_range(beta_fast, beta_slow, self.dim, self.base, original_max)
        inv_freq_mask = (1 - self._yarn_linear_ramp_mask(low, high, self.dim // 2).to(device)) * factor
        inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_extrapolation * inv_freq_mask
        
        self.register_buffer("inv_freq", inv_freq)
    
    def _yarn_find_correction_dim(self, num_rotations, dim, base=10000, max_pos=4096):
        return (dim * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))
    
    def _yarn_find_correction_range(self, low_rot, high_rot, dim, base=10000, max_pos=4096):
        low = math.floor(self._yarn_find_correction_dim(low_rot, dim, base, max_pos))
        high = math.ceil(self._yarn_find_correction_dim(high_rot, dim, base, max_pos))
        return max(low, 0), min(high, dim // 2 - 1)
    
    def _yarn_linear_ramp_mask(self, low, high, dim):
        if low == high:
            high += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - low) / (high - low)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func
    
    def rotations(self, position_ids, device=None):
        """Cosines and sines for explicit positions.

        Returns tensors of shape [batch, 1, seq_len, dim] so they broadcast over
        attention heads. Positions may be arbitrary and non-contiguous: a cached
        step needs the real prefix length, and a document-wise layout restarts at
        zero for every document.
        """
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)
        positions = position_ids.to(device=device or position_ids.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('bs,j->bsj', positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().unsqueeze(1), emb.sin().unsqueeze(1)

    def forward(self, x, seq_len=None, position_ids=None):
        if position_ids is not None:
            return self.rotations(position_ids, device=x.device)

        if seq_len > self.max_position_embeddings:
            # Extend if needed
            t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        else:
            t = torch.arange(self.max_position_embeddings, device=x.device, dtype=self.inv_freq.dtype)
        
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]
        return cos_cached, sin_cached


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None):
    # q, k: [batch, num_heads, seq_len, head_dim]
    cos = cos[:, :, :q.shape[2], :]
    sin = sin[:, :, :q.shape[2], :]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
