"""Cached incremental decoding must reproduce the full-sequence forward exactly.

A cached step feeds a single token and reuses the keys/values of the prefix, so
two things have to be right for the logits to agree with a single forward pass
over the whole prefix:

* the rotary embedding must rotate the new token at its real position (the
  prefix length), not at position 0;
* the causal mask must have one row per *query* token, aligned with the keys
  already in the cache.

If either drifts, the two paths produce different logits. This test measures the
drift instead of trusting that the shapes line up.

Run with pytest, or directly: python tests/test_incremental_decoding.py
"""

from pathlib import Path
import sys

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lligma.model.config import LLigMaConfig  # noqa: E402
from lligma.model.lligma import LLigMaForCausalLM  # noqa: E402

# float32 logits from the same weights, computed with different reduction
# orders, differ in the last bits. 1e-5 absolute is tight enough to catch a
# wrong position or a misaligned mask (which shift logits by O(1)).
ATOL = 1e-5
RTOL = 1e-4

PROMPT = 8


def build_model(seed=0):
    with (ROOT / "configs" / "lligma_3_0_small.yaml").open(encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    values.pop("model_type", None)
    # A small vocabulary keeps the test fast; the architecture stays the real one.
    values["vocab_size"] = 256
    values["max_position_embeddings"] = 64
    values["use_cache"] = True
    torch.manual_seed(seed)
    model = LLigMaForCausalLM(LLigMaConfig(**values))
    model.eval()
    return model


def full_forward_logits(model, token_ids):
    attention_mask = torch.ones_like(token_ids)
    with torch.no_grad():
        return model(input_ids=token_ids, attention_mask=attention_mask, use_cache=False).logits


def cached_step_logits(model, token_ids):
    """Feed the prompt one token at a time, reusing the cache at every step."""
    logits, past = [], None
    with torch.no_grad():
        for step in range(token_ids.shape[1]):
            outputs = model(
                input_ids=token_ids[:, step:step + 1],
                attention_mask=torch.ones(1, step + 1, dtype=torch.long),
                past_key_values=past,
                use_cache=True,
            )
            past = outputs.past_key_values
            logits.append(outputs.logits[:, -1])
    return torch.stack(logits, dim=1)


def test_cached_decoding_matches_full_forward():
    model = build_model()
    token_ids = torch.arange(1, PROMPT + 1).unsqueeze(0)

    reference = full_forward_logits(model, token_ids)
    incremental = cached_step_logits(model, token_ids)

    difference = (reference - incremental).abs()
    per_position = [round(value, 8) for value in difference[0].max(dim=-1).values.tolist()]
    print(f"max |full - cached| per position: {per_position}")
    print(f"max |full - cached| overall: {difference.max().item():.3e}")

    torch.testing.assert_close(incremental, reference, atol=ATOL, rtol=RTOL)


def test_generate_matches_uncached_reference():
    model = build_model()
    prompt = torch.arange(1, PROMPT + 1).unsqueeze(0)
    new_tokens = 5

    # Reference: greedy decoding that recomputes the whole prefix every step,
    # i.e. no cache at all.
    reference = prompt.clone()
    with torch.no_grad():
        for _ in range(new_tokens):
            logits = full_forward_logits(model, reference)
            reference = torch.cat([reference, logits[:, -1].argmax(dim=-1, keepdim=True)], dim=1)

    generated = model.generate(
        input_ids=prompt,
        attention_mask=torch.ones_like(prompt),
        max_new_tokens=new_tokens,
        do_sample=False,
        use_cache=True,
        pad_token_id=0,
        eos_token_id=None,
    )

    print(f"reference tokens: {reference[0, PROMPT:].tolist()}")
    print(f"generated tokens: {generated[0, PROMPT:].tolist()}")
    assert torch.equal(generated, reference), "generate() diverged from the uncached greedy reference"


if __name__ == "__main__":
    test_cached_decoding_matches_full_forward()
    test_generate_matches_uncached_reference()
    print("OK: cached decoding matches the full forward")
