"""Deriva fra schede spedite e indice spedito.

``index/tutor-index.json`` si rigenera a mano:

    python scripts/build_tutor_index.py

Se una scuola modifica o aggiunge una scheda in ``data/tutor/`` e dimentica quel
comando, il tutor continua a rispondere con la conoscenza vecchia: nessuna
eccezione, nessun log, solo risposte basate su un testo che non c'e' piu'. E' un
guasto silenzioso, e questo test lo intercetta senza avviare il server.

Il confronto e' sui chunk (documento, ordinale, testo) e sui parametri di
costruzione. ``built_at`` e' escluso di proposito: e' un timestamp e cambierebbe
a ogni ricostruzione anche con le stesse schede.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lligma.tutor.config import TutorConfig  # noqa: E402
from lligma.tutor.index import BM25Index  # noqa: E402


def _builder():
    """Il modulo vero di build, non una copia della sua logica.

    Una copia delle regole di raccolta (quali file sono schede, in che ordine,
    con quale deduplica) divergerebbe dallo script al primo cambio e il test
    comincerebbe a controllare se stesso invece dell'indice.
    """
    spec = importlib.util.spec_from_file_location(
        "build_tutor_index", ROOT / "scripts" / "build_tutor_index.py"
    )
    assert spec is not None and spec.loader is not None, "script di build non caricabile"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _documents(builder) -> list:
    """Le schede spedite, raccolte e filtrate come fa lo script."""
    documents = builder.collect(ROOT / "data" / "tutor", "generale")
    seen: set[str] = set()
    unique = []
    for document in documents:
        if not document.text.strip() or document.id in seen:
            continue
        seen.add(document.id)
        unique.append(document)
    return unique


def _fingerprint(index: BM25Index) -> list[tuple[str, int, str]]:
    return [(chunk.document_id, chunk.ordinal, chunk.text) for chunk in index.chunks]


def test_indice_ricostruibile_dalle_schede() -> None:
    builder = _builder()
    config = TutorConfig.from_file(ROOT / "configs" / "tutor.yaml")
    built = BM25Index.build(
        _documents(builder),
        chunk_words=config.index.chunk_words,
        overlap_words=config.index.overlap_words,
        title_boost=config.retrieval.title_boost,
    )
    shipped = BM25Index.load(
        config.index.path, title_boost=config.retrieval.title_boost
    )

    # Un indice costruito con altri parametri ha gli stessi testi ma confini
    # diversi: senza questo confronto il test passerebbe e le citazioni
    # mostrerebbero frasi tagliate in un altro punto.
    assert shipped.metadata.get("chunk_words") == config.index.chunk_words
    assert shipped.metadata.get("overlap_words") == config.index.overlap_words

    expected = _fingerprint(built)
    actual = _fingerprint(shipped)
    if expected != actual:
        pytest.fail(
            "index/tutor-index.json non corrisponde a data/tutor/ "
            f"(ricostruiti {len(expected)} chunk, spediti {len(actual)}): "
            "rigenera con `python scripts/build_tutor_index.py`",
            pytrace=False,
        )
