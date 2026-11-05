"""LLigMa 3.0.

Gli export dell'architettura sono pigri (PEP 562). Il motivo e' concreto: il
sotto-pacchetto ``lligma.tutor`` deve poter girare su una macchina che non ha
PyTorch ne' transformers installati (un laboratorio, un portatile della scuola,
la macchina che costruisce l'indice). Con un import anticipato di ``.model``,
qualunque ``import lligma.tutor`` fallirebbe per una dipendenza che il tutor non
usa nemmeno.
"""

from typing import Any

__version__ = "3.0.0"

_MODEL_EXPORTS = {
    "LLigMaConfig": "config",
    "LLigMaForCausalLM": "lligma",
    "LLigMaModel": "lligma",
    "LLigMaPreTrainedModel": "lligma",
    "LLIGMA_3_0_2B": "config",
    "LLIGMA_3_0_2B_FLASH": "config",
    "LLIGMA_3_0_2B_PRO": "config",
}

__all__ = ["__version__", *_MODEL_EXPORTS]


def __getattr__(name: str) -> Any:
    if name in _MODEL_EXPORTS:
        from importlib import import_module

        module = import_module(f".model.{_MODEL_EXPORTS[name]}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(list(globals()) + __all__))
