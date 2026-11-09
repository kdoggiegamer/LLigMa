import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from transformers import PreTrainedModel, GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .config import LLigMaConfig
from .ml_attention import MultiHeadLatentAttention
from .moe import DeepSeekMoE, SwiGLU


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
    
    def forward(self, hidden_states):
        # The fp32 master weight is down-cast to the activation dtype at call
        # time: matching dtypes is what lets torch.rms_norm use its fused CUDA
        # kernel (fp16 weight + fp16 input emitted "Mismatch dtype ... Cannot
        # dispatch to fused implementation"). The [n] cast is negligible next
        # to the [B,S,n] activation and the kernel reduces in fp32, so numerics
        # are preserved within fp16 rounding. Measured on a T4: -7.3% step time
        # end to end (lligma-rms-bench, verdict.json in its output).
        return torch.rms_norm(hidden_states, hidden_states.shape[-1:],
                              self.weight.to(hidden_states.dtype), self.eps)


class LLigMaDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        
        # Attention
        self.self_attn = MultiHeadLatentAttention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # MLP/MoE
        if config.use_moe and layer_idx >= config.first_k_dense_replace:
            self.mlp = DeepSeekMoE(config)
        else:
            self.mlp = SwiGLU(config.hidden_size, config.intermediate_size, bias=False)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states
        
        # MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        
        return outputs


class LLigMaPreTrainedModel(PreTrainedModel):
    config_class = LLigMaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LLigMaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"


class LLigMaModel(LLigMaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id if hasattr(config, 'pad_token_id') else 0
        self.vocab_size = config.vocab_size
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([LLigMaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.gradient_checkpointing = False
        self.post_init()
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        hidden_states = inputs_embeds
        
        # Prepare past_key_values. transformers builds a DynamicCache when
        # use_cache is on, while this model stores legacy (key, value) tuples:
        # normalise on the way in so both callers work.
        if past_key_values is not None:
            if hasattr(past_key_values, "to_legacy_cache"):
                past_key_values = past_key_values.to_legacy_cache()
            # An empty DynamicCache converts to one (None, None) pair per layer,
            # which is non-empty but carries no keys: treat it as no cache.
            if not past_key_values or past_key_values[0][0] is None:
                past_key_values = None
        
        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, input_ids.shape[1] + past_key_values_length,
                dtype=torch.long, device=device
            ).unsqueeze(0)
        
        # Causal mask. When the caller supplies no padding mask there is nothing
        # to combine: the layers use the fused causal kernel instead of a
        # materialised [batch, 1, seq, seq] additive mask. Passing an all-ones
        # mask here was semantically empty work.
        if attention_mask is not None:
            combined_attention_mask = self._prepare_decoder_attention_mask(
                attention_mask, input_ids.shape if input_ids is not None else (inputs_embeds.shape[0], inputs_embeds.shape[1]),
                inputs_embeds, past_key_values_length
            )
        else:
            combined_attention_mask = None
        
        next_decoder_cache = () if use_cache else None
        
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    combined_attention_mask,
                    position_ids,
                    past_key_value,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=combined_attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)
        
        hidden_states = self.norm(hidden_states)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, next_decoder_cache] if v is not None)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=None,
            attentions=None,
        )
    
    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # Causal mask with one row per QUERY token and one column per key. When a
        # cache is present the query rows are the new tokens only, so query i sits
        # at global position past_key_values_length + i and may see every key up to
        # and including itself. Using seq_length_with_past rows here would add
        # phantom queries and let broadcasting drift the attention shape.
        batch_size, seq_length = input_shape
        seq_length_with_past = seq_length + past_key_values_length
        device = inputs_embeds.device
        
        query_positions = torch.arange(past_key_values_length, seq_length_with_past, device=device)
        key_positions = torch.arange(seq_length_with_past, device=device)
        mask = torch.full(
            (seq_length, seq_length_with_past), torch.finfo(inputs_embeds.dtype).min,
            device=device, dtype=inputs_embeds.dtype,
        )
        mask.masked_fill_(key_positions[None, :] <= query_positions[:, None], 0.0)
        
        if attention_mask is not None:
            mask = mask[None, None, :, :].expand(batch_size, 1, seq_length, seq_length_with_past)
            inverted_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=inputs_embeds.dtype)
            inverted_mask = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(inputs_embeds.dtype).min)
            mask = mask + inverted_mask
            return mask
        return mask[None, None, :, :].expand(batch_size, 1, seq_length, seq_length_with_past)


class LLigMaForCausalLM(LLigMaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config):
        super().__init__(config)
        self.model = LLigMaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, value):
        self.lm_head = value
    
    def set_decoder(self, decoder):
        self.model = decoder
    
    def get_decoder(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else True
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        # transformers passes a (possibly empty) Cache object, which is not None
        # even on the first step. Truncating on `is not None` would drop the
        # prompt and leave one query against the whole mask.
        past_length = 0
        if past_key_values is not None:
            if hasattr(past_key_values, "get_seq_length"):
                past_length = past_key_values.get_seq_length()
            elif len(past_key_values) and past_key_values[0][0] is not None:
                past_length = past_key_values[0][0].shape[2]
        
        if past_length > 0:
            input_ids = input_ids[:, -1:]
        
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_length > 0:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}
        
        model_inputs.update({
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask,
        })
        return model_inputs
