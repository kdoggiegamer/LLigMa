"""Regressione della sezione 6: il fallback rilassato usa la query espansa.

Il terzo tentativo di ``TutorService.retrieve`` (materia probabile, una sola
parola in comune, soglia alta) e' il ripiego dopo che i primi due hanno fallito.
Deve cercare la stessa query espansa dei primi due: altrimenti una regola di
espansione che sostituisce un termine assente dal corpus aiuta il match stretto
e sparisce proprio nel tentativo che deve salvarlo.

Il test usa il codice reale con un backend fittizio: ``retrieve`` non genera
nulla, quindi non servono ne' modello ne' rete. Non tocca ``server.py`` ne'
``service.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lligma.tutor.config import TutorConfig  # noqa: E402
from lligma.tutor.index import BM25Index  # noqa: E402
from lligma.tutor.prompt import guess_subject  # noqa: E402
from lligma.tutor.service import TutorService  # noqa: E402

RELAXED_CASE = "l'involucro cellulare"
EXPECTED_DOCUMENT = "bio.membrana"


def _service() -> TutorService:
    config = TutorConfig.from_file(ROOT / "configs" / "tutor.yaml")
    return TutorService(config, backend=object())


def _questions(path: Path) -> list[str]:
    questions: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            question = (json.loads(line).get("question") or "").strip()
            if question:
                questions.append(question)
    return questions


def test_relaxed_attempt_uses_the_expanded_query() -> None:
    service = _service()
    calls: list[str] = []
    original = BM25Index.search

    def traced(self, query, **kwargs):
        hits = original(self, query, **kwargs)
        calls.append(query)
        return hits

    BM25Index.search = traced
    try:
        reached = 0
        for name in ("eval.jsonl", "eval-paraphrase.jsonl"):
            for question in _questions(ROOT / "data" / "tutor" / name):
                calls.clear()
                service.retrieve(question)
                # Tre chiamate = i due tentativi stretti sono falliti e si e'
                # entrati nel ripiego rilassato.
                if len(calls) >= 3:
                    reached += 1
                    assert calls[-1] == calls[0], (
                        "il fallback rilassato cerca una query diversa dal primo "
                        f"tentativo per {question!r}: {calls[0]!r} vs {calls[-1]!r}"
                    )
    finally:
        BM25Index.search = original

    assert reached >= 1, "nessuna domanda raggiunge il ramo rilassato: test cieco"


def test_relaxed_expanded_query_finds_the_scheda_first() -> None:
    service = _service()
    config = service.config
    expanded = service._search_question(RELAXED_CASE)
    assert expanded != RELAXED_CASE, "la regola di espansione attesa non e' attiva"

    relaxed = service.index.search(
        expanded,
        top_k=config.retrieval.top_k,
        subject=guess_subject(RELAXED_CASE),
        min_score=config.retrieval.min_score * 1.5,
        min_terms=1,
    )
    assert relaxed, "nessun passaggio dal fallback rilassato"
    assert relaxed[0].chunk.document_id == EXPECTED_DOCUMENT, [
        hit.chunk.document_id for hit in relaxed
    ]
