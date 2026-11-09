import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from .rope import RoPE, apply_rotary_pos_emb


class FusedRMSNorm(nn.RMSNorm):
    """Same module as torch.nn.RMSNorm, except the fp32 master weight is
    down-cast to the activation dtype at call time.

    torch.nn.RMSNorm passes its stored fp32 weight straight into the op: under
    fp16 autocast the dtypes mismatch and "Cannot dispatch to fused
    implementation" falls back to the generic kernel. Here the [n] weight is
    cast per call so the fused kernel dispatches; the reduction itself stays
    fp32 in kernel, so numerics are preserved within fp16 rounding. Same state
    dict keys, same checkpoint format.
    """

    def forward(self, x):
        # torch <2.x stored the epsilon as `variance_epsilon`, newer as `eps`.
        eps = getattr(self, "eps", None)
        if eps is None:
            eps = getattr(self, "variance_epsilon")
        return torch.rms_norm(x, self.normalized_shape, self.weight.to(x.dtype), eps)


class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (MLA) - DeepSeek inspired.
    Compresses KV cache into latent representation for efficiency.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        
        self.scale = self.qk_head_dim ** -0.5
        
        # Query projection with LoRA
        if config.q_lora_rank > 0:
            self.q_a_proj = nn.Linear(self.hidden_size, config.q_lora_rank, bias=False)
            self.q_a_layernorm = FusedRMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
            self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        
        # Key-Value projection with LoRA (latent compression)
        self.kv_a_proj_with_mqa = nn.Linear(self.hidden_size, 
                                             config.kv_lora_rank + self.qk_rope_head_dim, 
                                             bias=False)
        self.kv_a_layernorm = FusedRMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.kv_b_proj = nn.Linear(config.kv_lora_rank, 
                                   self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), 
                                   bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        
        # RoPE
        self.rope = RoPE(
            dim=self.qk_rope_head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
            scaling_config=config.rope_scaling
        )
        
        self.attn_dropout = config.attention_dropout if hasattr(config, 'attention_dropout') else 0.0
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        
        # Query
        if hasattr(self, 'q_a_proj'):
            q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        else:
            q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, self.num_heads, self.qk_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        
        # Compressed KV
        kv_compressed = self.kv_a_proj_with_mqa(hidden_states)
        kv, k_pe = torch.split(kv_compressed, [self.config.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_a_layernorm(kv)
        kv = self.kv_b_proj(kv)
        kv = kv.view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim).transpose(1, 2)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        
        # Apply RoPE at the real positions. During cached decoding the new token
        # sits at `past length`, so rotating it at position 0 would break the
        # match with the keys already cached at their own positions.
        if position_ids is None:
            cos, sin = self.rope(q_pe, seq_len=q_len)
        else:
            cos, sin = self.rope(q_pe, position_ids=position_ids)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)
        
        # Concatenate RoPE and non-RoPE parts
        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe.expand(-1, self.num_heads, -1, -1)], dim=-1)
        
        # Handle KV cache
        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=2)
            v = torch.cat([past_key_value[1], v], dim=2)
        
        past_key_value = (k, v) if use_cache else None
        
        dropout_p = self.attn_dropout if self.training else 0.0

        if output_attentions:
            # The caller wants the score matrix, so it has to be materialised.
            attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scale
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            if dropout_p > 0:
                attn_weights = nn.functional.dropout(attn_weights, p=dropout_p, training=True)
            
            attn_output = torch.matmul(attn_weights, v)
        else:
            # scaled_dot_product_attention fuses the score matmul, the softmax
            # and the value matmul into one kernel and never materialises the
            # [batch, heads, seq, seq] score matrix. The scale is the same one
            # applied above: SDPA multiplies by 1/sqrt(q.shape[-1]).
            attn_weights = None
            if attention_mask is not None:
                mask = attention_mask
                if mask.is_floating_point() and mask.dtype != q.dtype:
                    mask = mask.to(q.dtype)
                attn_output = nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, dropout_p=dropout_p, is_causal=False
                )
            else:
                # No padding and no cache means the mask is exactly causal.
                # `is_causal` uses an upper-left alignment, which only matches
                # our mask when queries and keys have the same length, i.e. when
                # there is no cache (past == 0). A cached step (q_len 1 against
                # a longer key axis) must take the explicit-mask branch above.
                attn_output = nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=dropout_p,
                    is_causal=q_len == k.shape[2] and q_len > 1,
                )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.v_head_dim)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights, past_key_value
