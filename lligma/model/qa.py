"""Extractive reading comprehension on top of LLigMaModel.

Why a span head and not generation: the answer is returned as a slice of the
retrieved paragraph, so the text the student reads is always copied from the
source. A 0.4B model cannot be trusted to produce a fact, but it can learn to
point at one.

The decoder stays causal, which keeps the pretrained weights usable as a
language model. A causal reader can only see the prefix of the text, so the
question must come *before* the context: at the position where the answer
starts, the model has already read the question and the context up to that
point.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from .config import LLigMaConfig
from .lligma import LLigMaModel, LLigMaPreTrainedModel


class LLigMaForQuestionAnswering(LLigMaPreTrainedModel):
    config_class = LLigMaConfig
    base_model_prefix = "model"

    def __init__(self, config: LLigMaConfig):
        super().__init__(config)
        self.model = LLigMaModel(config)
        # Two logits per position: starts here / ends here. ~2k parameters on
        # the 0.4B model, so nothing about the checkpoint layout moves.
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask,
                             use_cache=False, return_dict=True)
        logits = self.qa_outputs(outputs.last_hidden_state)
        start_logits = logits[..., 0]
        end_logits = logits[..., 1]

        loss = None
        if start_positions is not None and end_positions is not None:
            # Padded positions must not win the argmax, and they must not
            # contribute to the loss either: the gold spans are never there.
            if attention_mask is not None:
                pad = attention_mask.to(start_logits.dtype) - 1.0
                start_logits = start_logits + pad * 1e4
                end_logits = end_logits + pad * 1e4
            loss = 0.5 * (F.cross_entropy(start_logits, start_positions)
                          + F.cross_entropy(end_logits, end_positions))

        return QuestionAnsweringModelOutput(
            loss=loss, start_logits=start_logits, end_logits=end_logits,
        )

    @torch.no_grad()
    def best_span(self, input_ids, attention_mask, max_answer_len: int = 30
                  ) -> Tuple[int, int, float]:
        """Highest-scoring (start, end) with start <= end and a length cap.

        A span logit pair is only meaningful as a pair: the joint score is the
        sum, so the search is over legal pairs, not over two independent argmaxes.
        """
        out = self(input_ids=input_ids, attention_mask=attention_mask)
        start = out.start_logits[0].float()
        end = out.end_logits[0].float()
        length = start.shape[0]
        # Joint score of every legal pair in one tensor, so the search is
        # vectorised instead of one argmax per start position.
        scores = start[:, None] + end[None, :]
        positions = torch.arange(length, device=scores.device)
        legal = (positions[None, :] >= positions[:, None]) & \
                (positions[None, :] < positions[:, None] + max_answer_len)
        scores = scores.masked_fill(~legal, -float("inf"))
        flat = int(torch.argmax(scores))
        s, e = divmod(flat, length)
        return s, e, float(scores[s, e])
