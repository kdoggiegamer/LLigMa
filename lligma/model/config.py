from typing import Optional
from transformers import PretrainedConfig


class LLigMaConfig(PretrainedConfig):
    model_type = "lligma"
    architectures = ["LLigMaForCausalLM"]
    
    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 2048,
        intermediate_size: int = 5632,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        head_dim: int = 128,
        use_mla: bool = True,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_rope_head_dim: int = 64,
        qk_nope_head_dim: int = 64,
        v_head_dim: int = 128,
        rope_theta: float = 10000.0,
        max_position_embeddings: int = 131072,
        rope_scaling: Optional[dict] = None,
        use_moe: bool = True,
        num_experts: int = 64,
        num_shared_experts: int = 2,
        num_experts_per_tok: int = 6,
        moe_intermediate_size: int = 1408,
        moe_layer_freq: int = 1,
        first_k_dense_replace: int = 1,
        norm_topk_prob: bool = True,
        routed_scaling_factor: float = 2.5,
        rms_norm_eps: float = 1e-6,
        use_rmsnorm: bool = True,
        hidden_act: str = "swiglu",
        initializer_range: float = 0.02,
        rope_scaling_factor: float = 1.0,
        use_cache: bool = True,
        tie_word_embeddings: bool = False,
        use_sliding_window: bool = False,
        sliding_window: int = 4096,
        quantization_config: Optional[dict] = None,
        enable_thinking_mode: bool = True,
        max_thinking_tokens: int = 8192,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.use_mla = use_mla
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.rope_theta = float(rope_theta)
        self.max_position_embeddings = max_position_embeddings
        self.rope_scaling = rope_scaling
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.moe_layer_freq = moe_layer_freq
        self.first_k_dense_replace = first_k_dense_replace
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor
        # YAML 1.1 does not resolve `1e-6` as a float (it needs a dot), so these
        # values can arrive as strings from a config file. Coerce them here.
        self.rms_norm_eps = float(rms_norm_eps)
        self.use_rmsnorm = use_rmsnorm
        self.hidden_act = hidden_act
        self.initializer_range = float(initializer_range)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.rope_scaling_factor = float(rope_scaling_factor)
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        if quantization_config is not None:
            self.quantization_config = quantization_config
        self.enable_thinking_mode = enable_thinking_mode
        self.max_thinking_tokens = max_thinking_tokens
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
    
    @property
    def num_moe_layers(self) -> int:
        return self.num_hidden_layers - self.first_k_dense_replace
    
    @property
    def hidden_size_per_head(self) -> int:
        return self.head_dim


# Preset configurations
LLIGMA_3_0_2B = LLigMaConfig()
LLIGMA_3_0_2B_FLASH = LLigMaConfig(
    num_hidden_layers=24,
    use_sliding_window=True,
)
LLIGMA_3_0_2B_PRO = LLigMaConfig(
    num_hidden_layers=32,
    max_position_embeddings=262144,
)
