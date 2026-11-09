"""Hugging Face tokenizer adapter used by LLigMa inference examples."""


class LLigMaTokenizer:
    """Load a real Hugging Face tokenizer; no synthetic token IDs are generated."""

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError("Install the project requirements to load a tokenizer") from exc
        return AutoTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)
