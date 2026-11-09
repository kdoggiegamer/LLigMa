#!/usr/bin/env python3
"""Regressione sul terzo tentativo "rilassato" di ``TutorService.retrieve``.

Punto §6 del registro condiviso: i primi due tentativi cercano
``search_question`` (la domanda con i sinonimi espansi), il terzo cerca
``question`` (la domanda originale). La domanda aperta era: questo ramo puo'
pescare la scheda sbagliata?

Misura del 2026-11-06 (albero dimostrato pristine: i sette hash ``FROZEN``
coerenti e nessuna directory ``.lligma-mutation-*`` viva prima e dopo il run):

* il ramo rilassato e' raggiunto da **2/34** domande di ``eval.jsonl`` e
  **0/39** di ``eval-paraphrase.jsonl``; su un corpus allargato di 196 domande
  (i due set, 117 varianti costruite dalle regole del dizionario, 6 domande
  fuori ambito) lo raggiunge **una volta sola**, su "Che cosa e' l'humus e perche'
  e' importante?";
* su quel caso la forma espansa e quella non espansa danno la **stessa** fonte
  (``agro.humus``); sulle due domande che lo raggiungono nei set spediti non c'e'
  nessuna risposta diversa, nessun danno, nessun guadagno, nessuna regressione di
  astensione;
* il ramo e' raggiunto **solo** quando la forma espansa ha gia' fallito la
  soglia nei primi due tentativi. Non e' quindi un percorso che perde i sinonimi:
  e' l'ultimo tentativo, volutamente a un solo termine in comune, e arriva dopo
  che l'espansione ha gia' avuto la sua occasione.

Conclusione della misura: **nessun danno end-to-end**. Il fix (terzo tentativo
sulla query espansa) e' stato poi applicato da Buffy-B3 con l'aggiornamento di
``FROZEN`` per ``service.py``, ed e' coerente con la misura: cambia una riga e
non sposta nessuna metrica. Il test della query espansa e' in
``tests/test_tutor_expansion_relaxed.py`` (dell'altra sessione); qui restano le
tre invarianti che quel file non copre, e che valgono **qualunque forma** usi il
ripiego: il ramo deve restare raggiungibile (altrimenti la misura non significa
piu' niente), non deve rispondere da una scheda diversa da quella attesa, e deve
intervenire solo dopo che i tentativi precedenti sono finiti sotto soglia.

Eseguibile sia da pytest sia direttamente::

    python -m pytest tests/test_tutor_relaxed_branch.py -q
    python tests/test_tutor_relaxed_branch.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_tutor_index import collect  # noqa: E402
from lligma.tutor.backends import ExtractiveBackend  # noqa: E402
from lligma.tutor.config import TutorConfig  # noqa: E402
from lligma.tutor.index import BM25Index  # noqa: E402
from lligma.tutor.service import TutorService  # noqa: E402

# La domanda dei set spediti che arriva davvero al terzo tentativo.
HUMUS = "Che cosa è l'humus e perché è importante?"
# Il ramo rilassato deve restare un'eccezione, non diventare il percorso normale.
MAX_REACHED = 5


class TracingIndex:
    """Proxy trasparente che registra quali tentativi hanno risposto.

    ``rewrite`` riscrive la sola query del ramo rilassato (``min_score > 0``):
    cosi' si confronta la forma attuale con quella espansa senza modificare
    ``service.py``.
    """

    def __init__(self, inner, rewrite=None):
        self.inner = inner
        self.rewrite = rewrite
        self.calls: list[dict] = []

    def __len__(self):
        return len(self.inner)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def search(self, query, **kwargs):
        relaxed = kwargs.get("min_score", 0.0) > 0.0
        effective = self.rewrite(query) if (relaxed and self.rewrite) else query
        hits = self.inner.search(effective, **kwargs)
        self.calls.append(
            {
                "relaxed": relaxed,
                "query": query,
                "effective": effective,
                "ids": [hit.chunk.document_id for hit in hits],
                "top_score": hits[0].score if hits else None,
            }
        )
        return hits


def build_index() -> BM25Index:
    documents = collect(ROOT / "data" / "tutor", "generale")
    return BM25Index.build(documents, chunk_words=110, overlap_words=28)


def make_service(rewrite=None) -> tuple[TutorService, TracingIndex]:
    config = TutorConfig.from_file(ROOT / "configs" / "tutor.yaml")
    config.generation.backend = "extractive"
    service = TutorService(
        config,
        index=build_index(),
        backend=ExtractiveBackend(max_sentences=5, min_overlap=2),
    )
    proxy = TracingIndex(service.index, rewrite=rewrite)
    service.index = proxy
    return service, proxy


def load_eval(name: str) -> list[dict]:
    path = ROOT / "data" / "tutor" / name
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def scan(service: TutorService, proxy: TracingIndex, items: list[dict]) -> list[dict]:
    """Restituisce i casi che raggiungono il ramo rilassato, con la risposta."""
    reached: list[dict] = []
    for item in items:
        proxy.calls.clear()
        answer = service.ask(
            item["question"], subject=item.get("subject"), use_cache=False
        )
        if not any(call["relaxed"] for call in proxy.calls):
            continue
        reached.append({"item": item, "answer": answer, "calls": list(proxy.calls)})
    return reached


def test_relaxed_branch_is_reachable_on_a_shipped_question():
    """Se il ramo sparisce, la misura del §6 non descrive piu' nulla."""
    service, proxy = make_service()
    proxy.calls.clear()
    answer = service.ask(HUMUS, use_cache=False)

    relaxed = [call for call in proxy.calls if call["relaxed"]]
    assert relaxed, (
        "il ramo rilassato di retrieve() non viene piu' raggiunto da nessuna "
        "domanda spedita: la misura del punto §6 va rifatta prima di fidarsi "
        "delle conclusioni in docs/AGENT-SYNC.md"
    )
    assert len(relaxed) == 1, [call["query"] for call in relaxed]
    assert not answer.abstained
    assert answer.sources[0].document_id == "agro.humus"


def test_relaxed_branch_answers_only_the_expected_scheda():
    """Nessuna risposta da una scheda diversa da quella attesa, e resta raro."""
    service, proxy = make_service()
    items = load_eval("eval.jsonl") + load_eval("eval-paraphrase.jsonl")
    reached = scan(service, proxy, items)

    assert reached, "nessuna domanda raggiunge piu' il ramo rilassato"
    assert len(reached) <= MAX_REACHED, (
        f"{len(reached)} domande raggiungono il ramo rilassato (limite "
        f"{MAX_REACHED}): e' diventato un percorso normale, non l'ultimo "
        "tentativo; la soglia e il ramo vanno rimisurati"
    )

    wrong = [
        (row["item"]["question"], row["item"].get("expect_doc"), row["answer"].sources[0].document_id)
        for row in reached
        if row["item"].get("expect_doc")
        and not row["answer"].abstained
        and row["answer"].sources
        and row["answer"].sources[0].document_id != row["item"]["expect_doc"]
    ]
    assert not wrong, f"il ramo rilassato ha risposto dalla scheda sbagliata: {wrong}"


def test_relaxed_branch_runs_only_after_the_expanded_attempts_failed():
    """Il ramo rilassato e' l'ultima chance, non una scorciatoia sui sinonimi."""
    service, proxy = make_service()
    config = service.config.retrieval
    items = load_eval("eval.jsonl") + load_eval("eval-paraphrase.jsonl")

    for row in scan(service, proxy, items):
        before = [call for call in row["calls"] if not call["relaxed"]]
        assert before, f"nessun tentativo prima del rilassato: {row['item']['question']}"
        for call in before:
            top = call["top_score"] or 0.0
            assert top < config.min_score, (
                f"il ramo rilassato e' stato raggiunto con un tentativo gia' "
                f"sopra soglia ({top:.2f} >= {config.min_score}) su "
                f"{row['item']['question']}"
            )


def _run_all() -> int:
    from lligma.tutor.console import setup_utf8

    setup_utf8()
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
            print(f"ok   {test.__name__}")
        except AssertionError as error:
            failures += 1
            print(f"FAIL {test.__name__}: {error}")
    print(f"\n{len(tests) - failures}/{len(tests)} ok")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
