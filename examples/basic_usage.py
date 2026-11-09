#!/usr/bin/env python3
"""Basic usage example for LLigMa 3.0.

Il default e' il preset di debug, non ``LLigMaConfig()``. La configurazione
predefinita e' quella sperimentale da ~16B parametri (gli esperti routati sono
la maggior parte dei pesi): istanziarla su un portatile significa allocare
decine di GB. Il preset di debug ha la stessa architettura — MLA, MoE, RoPE —
con dimensioni da smoke test, quindi l'esempio gira in un secondo.

    python examples/basic_usage.py
    python examples/basic_usage.py --config configs/lligma_3_0_small.yaml
"""
import argparse
import sys
from pathlib import Path

import yaml

# Lanciato come `python examples/basic_usage.py` sys.path[0] e' examples/,
# quindi il pacchetto lligma non sarebbe importabile senza questo bootstrap.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lligma import LLigMaConfig, LLigMaForCausalLM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "lligma_3_0_debug.yaml"


def load_config(path: Path) -> LLigMaConfig:
    values = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # `model_type` e' un campo di Hugging Face, non un argomento del costruttore.
    values.pop("model_type", None)
    return LLigMaConfig(**values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    config = load_config(args.config)
    model = LLigMaForCausalLM(config)

    parameters = sum(p.numel() for p in model.parameters())
    print(f"Model created with {parameters / 1e6:.2f}M parameters ({args.config.name})")

    # For inference with trained weights
    # tokenizer = LLigMaTokenizer.from_pretrained("path/to/lligma-small")
    # generator = LLigMaGenerator(model, tokenizer)
    # response = generator.generate_with_reasoning("Explain quantum entanglement")
    # print(response)


if __name__ == "__main__":
    main()
