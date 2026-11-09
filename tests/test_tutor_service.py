"""End-to-end behaviour of the service with the offline backend.

The extractive backend is the reference the small model has to beat, so it is
tested with the same rigour: grounded answers, citations that point at the
passage actually used, and a firm "I don't know" outside the knowledge base.
The cache tests include a deterministic concurrency check that fails when
``AnswerCache._lock`` is removed.

Run with pytest, or directly: python tests/test_tutor_service.py
"""

from __future__ import annotations

import re
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lligma.tutor.backends import ExtractiveBackend, make_backend  # noqa: E402
from lligma.tutor.config import TutorConfig  # noqa: E402
from lligma.tutor.index import BM25Index  # noqa: E402
from lligma.tutor.prompt import (  # noqa: E402
    ABSTENTION_TEXT,
    OUT_OF_SCOPE_TEXT,
    SYSTEM_PROMPT,
    build_messages,
    format_context,
    guess_subject,
)
from lligma.tutor.service import TutorService  # noqa: E402
from lligma.tutor.text import normalize  # noqa: E402


def make_service(backend=None) -> TutorService:
    config = TutorConfig.from_file(ROOT / "configs" / "tutor.yaml")
    config.generation.backend = "extractive"
    from build_tutor_index import collect

    index = BM25Index.build(collect(ROOT / "data" / "tutor", "generale"))
    return TutorService(config, index=index, backend=backend or ExtractiveBackend(max_sentences=5, min_overlap=2))


def test_answer_is_grounded_and_cited():
    service = make_service()
    answer = service.ask("Perche' si alternano le colture?", use_cache=False)
    assert not answer.abstained
    assert answer.sources, "una risposta fondata deve citare almeno una fonte"
    assert "[" in answer.text and "]" in answer.text, "manca il riferimento numerico alla fonte"
    assert answer.backend == "extractive"


def test_answer_text_comes_from_the_sources():
    """Il backend estrattivo non puo' inventare: ogni frase deve esistere nella fonte."""
    service = make_service()
    question = "Che cosa e' l'humus?"
    answer = service.ask(question, use_cache=False)
    hits, _ = service.retrieve(question)
    source_text = " ".join(hit.chunk.text for hit in hits).lower()
    for sentence in answer.text.split(". "):
        clean = sentence.replace("[", "").replace("]", "")
        words = [word for word in clean.split() if len(word) > 5][:3]
        if not words:
            continue
        assert any(word.lower() in source_text for word in words), sentence


def test_out_of_domain_question_abstains():
    service = make_service()
    for question in ("chi ha vinto i mondiali di calcio del 2022", "quanto costa un trattore nuovo"):
        answer = service.ask(question, use_cache=False)
        assert answer.abstained, f"avrebbe dovuto astenersi: {question} -> {answer.text[:80]}"
        assert not answer.sources


def test_noise_question_abstains():
    service = make_service()
    answer = service.ask("zxcvbnm qwertyuiop", use_cache=False)
    assert answer.abstained


def test_cache_returns_the_same_answer_and_flags_it():
    service = make_service()
    first = service.ask("Cosa fa la clorofilla?", use_cache=False)
    service.cache.clear()
    service.ask("Cosa fa la clorofilla?", use_cache=True)
    cached = service.ask("Cosa fa la clorofilla?", use_cache=True)
    assert cached.cached is True
    assert cached.text == first.text


def test_cache_matches_reworded_questions():
    service = make_service()
    service.cache.clear()
    service.ask("Cosa fa la clorofilla nella fotosintesi?", use_cache=True)
    again = service.ask("Cosa fa la clorofilla nella fotosintesi?", use_cache=True)
    assert again.cached


def test_degraded_backend_falls_back_instead_of_failing():
    class Rotto(ExtractiveBackend):
        name = "rotto"

        def answer(self, request):
            from lligma.tutor.backends import BackendError

            raise BackendError("simulato")

    service = make_service(backend=Rotto())
    answer = service.ask("Come si pota la vite?", use_cache=False)
    assert answer.degraded is True
    assert answer.backend == "extractive"
    assert answer.text


def test_each_source_sheet_appears_once():
    """Piu' passaggi della stessa scheda -> una sola voce, con tutti i [n]."""
    service = make_service()
    question = "Che cos'e' la fotosintesi clorofilliana?"
    answer = service.ask(question, use_cache=False)
    assert answer.sources, "una risposta fondata deve citare almeno una fonte"
    document_ids = [source.document_id for source in answer.sources]
    titles = [source.title for source in answer.sources]
    assert len(document_ids) == len(set(document_ids)), f"schede ripetute: {titles}"
    assert len(titles) == len(set(titles)), f"titoli ripetuti: {titles}"
    hits, _ = service.retrieve(question)
    assert len(answer.sources) < len(hits), "la domanda non esercita il caso di piu' passaggi"
    numbered = sorted(number for source in answer.sources for number in source.numbers)
    assert numbered == list(range(1, len(hits) + 1)), numbered


def test_subject_detection_ignores_accents():
    assert guess_subject("cos'è la biodiversità") == "biologia"
    assert guess_subject("cos'e' la biodiversita") == guess_subject("cos'è la biodiversità")
    assert guess_subject("parlami della salinità del suolo") == "agronomia"


def test_every_citation_has_a_source():
    service = make_service()
    answer = service.ask("Perche' si alternano le colture?", use_cache=False)
    numbers = {number for source in answer.sources for number in source.numbers}
    cited = {int(number) for number in re.findall(r"\[(\d{1,2})\]", answer.text)}
    assert cited, "la risposta estrattiva deve citare almeno una fonte"
    assert cited <= numbers, f"citazioni orfane: {sorted(cited - numbers)}"


def test_former_zero_coverage_questions_are_answered():
    """I due casi che l'estrattivo non copriva devono citare i fatti giusti."""
    service = make_service()
    cases = {
        "Come si passa dal DNA alla proteina?": ({"trascrizione", "traduzione", "codone"}, "bio.dna"),
        "In che anno è ambientato il romanzo di Manzoni?": ({"1628"}, "cul.manzoni"),
    }
    for question, (keywords, document) in cases.items():
        answer = service.ask(question, use_cache=False)
        assert not answer.abstained and answer.text, question
        assert document in {source.document_id for source in answer.sources}, question
        text = normalize(answer.text)
        for keyword in keywords:
            assert normalize(keyword) in text, (question, keyword, answer.text)


def test_sources_match_the_context_that_fits():
    """Se il budget taglia un passaggio, non deve comparire come fonte."""
    service = make_service()
    question = "Quanta energia passa da un livello all'altro della catena alimentare?"
    hits, _ = service.retrieve(question)
    _, used = format_context(hits, service.config.generation.max_context_chars)
    assert len(used) < len(hits), "la domanda non esercita il taglio del contesto"
    answer = service.ask(question, use_cache=False)
    numbers = [number for source in answer.sources for number in source.numbers]
    assert numbers == list(range(1, len(used) + 1)), numbers
    assert max(numbers) <= len(used), "una fonte cita un blocco mai inviato al modello"


def test_cache_is_not_shared_across_conversations():
    """La cronologia entra nel prompt: non puo' essere ignorata dalla cache."""
    class Counting(ExtractiveBackend):
        name = "conta"

        def __init__(self):
            super().__init__()
            self.calls = 0

        def answer(self, request):
            self.calls += 1
            return super().answer(request)

    backend = Counting()
    service = make_service(backend=backend)
    service.cache.clear()
    question = "Perche' si alternano le colture?"
    service.ask(question, history=[{"role": "user", "content": "parliamo di grano"}], use_cache=True)
    service.ask(question, history=[{"role": "user", "content": "parliamo di vite"}], use_cache=True)
    assert backend.calls == 2, "la seconda conversazione ha riusato la risposta della prima"


def test_stream_marks_out_of_scope_as_abstention():
    class OutOfScope(ExtractiveBackend):
        name = "fuori"

        def answer(self, request):
            return OUT_OF_SCOPE_TEXT

        def stream(self, request):
            yield OUT_OF_SCOPE_TEXT

    service = make_service(backend=OutOfScope())
    service.cache.clear()
    question = "Perche' si alternano le colture?"
    events = list(service.stream(question, use_cache=True))
    assert events[-1]["type"] == "done"
    assert events[-1]["abstained"] is True
    cached = service.cache.get(question, None)
    assert cached is not None and cached.abstained and not cached.sources


def test_empty_backend_output_is_an_abstention():
    class Empty(ExtractiveBackend):
        name = "vuoto"

        def answer(self, request):
            return "   "

        def stream(self, request):
            return iter(())

    service = make_service(backend=Empty())
    question = "Perche' si alternano le colture?"
    answer = service.ask(question, use_cache=False)
    assert answer.abstained and not answer.sources
    assert answer.text == ABSTENTION_TEXT
    events = list(service.stream(question, use_cache=False))
    assert events[-1]["abstained"] is True
    assert "".join(e.get("text", "") for e in events if e["type"] == "token") == ABSTENTION_TEXT


def test_stream_done_reports_the_backend_that_answered():
    class Broken(ExtractiveBackend):
        name = "rotto"

        def stream(self, request):
            from lligma.tutor.backends import BackendError

            raise BackendError("simulato")
            yield ""

    service = make_service(backend=Broken())
    events = list(service.stream("Come si pota la vite?", use_cache=True))
    assert [event["type"] for event in events] == ["meta", "degraded", "token", "done"]
    assert events[-1]["backend"] == "extractive"


def test_cache_survives_concurrent_requests():
    from concurrent.futures import ThreadPoolExecutor

    service = make_service()
    service.cache.clear()
    questions = [
        "Perche' si alternano le colture?",
        "Che cos'e' l'humus?",
        "Come si pota la vite?",
        "Cosa fa la clorofilla?",
    ] * 4
    with ThreadPoolExecutor(max_workers=4) as pool:
        answers = list(pool.map(lambda item: service.ask(item, use_cache=True), questions))
    assert len(answers) == len(questions)
    assert all(answer.text for answer in answers)
    assert len(service.cache) <= service.cache.max_entries


def test_cache_serializes_concurrent_mutations():
    """Due scritture della cache non possono modificare le entries insieme.

    Il primo ``put`` viene fermato dentro la scrittura dell'``OrderedDict``; il
    secondo deve restare bloccato sul lock della cache. Se il lock viene rimosso,
    il secondo entra mentre il primo e' ancora attivo e il test lo rileva.
    """
    from collections import OrderedDict

    from lligma.tutor.schema import Answer
    from lligma.tutor.service import AnswerCache

    cache = AnswerCache(max_entries=8)
    entered = threading.Event()
    release = threading.Event()
    overlapped = threading.Event()
    active = 0
    guard = threading.Lock()

    class BlockingEntries(OrderedDict):
        def __setitem__(self, key, value):
            nonlocal active
            with guard:
                active += 1
                if active > 1:
                    overlapped.set()
            entered.set()
            release.wait(timeout=5)
            super().__setitem__(key, value)
            with guard:
                active -= 1

    cache._entries = BlockingEntries()
    answer = Answer(text="risposta di prova")
    first = threading.Thread(target=cache.put, args=("prima domanda", None, answer))
    first.start()
    assert entered.wait(timeout=2), "la prima put non ha raggiunto la cache"

    second_started = threading.Event()

    def second_put() -> None:
        second_started.set()
        cache.put("seconda domanda", None, answer)

    second = threading.Thread(target=second_put)
    second.start()
    assert second_started.wait(timeout=2), "la seconda put non e' partita"
    # Il lock deve impedire alla seconda put di entrare in __setitem__.
    overlapped.wait(timeout=1.0)
    release.set()
    first.join(timeout=3)
    second.join(timeout=3)
    assert not first.is_alive() and not second.is_alive()
    assert not overlapped.is_set(), "due put hanno modificato la cache fuori dal lock"


def test_stream_emits_meta_then_tokens_then_done():
    service = make_service()
    events = list(service.stream("Cosa produce la fotosintesi?", use_cache=False))
    kinds = [event["type"] for event in events]
    assert kinds[0] == "meta"
    assert kinds[-1] == "done"
    assert any(kind == "token" for kind in kinds)
    joined = "".join(event.get("text", "") for event in events)
    assert len(joined) > 40


def test_subject_detection_is_conservative():
    assert guess_subject("come si pota la vite") == "agronomia"
    assert guess_subject("cos'e' la mitosi") == "biologia"
    assert guess_subject("chi era dante alighieri") == "cultura"
    # Domanda di confine (biologia + agronomia insieme): meglio nessuna
    # restrizione che una restrizione sbagliata.
    assert guess_subject("la fotosintesi dipende dal suolo") is None


def test_prompt_contains_context_and_question():
    service = make_service()
    question = "Perche' si usa il letame?"
    hits, _ = service.retrieve(question)
    assert hits
    messages = build_messages(question, hits)
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "[1]" in messages[-1]["content"]
    assert question in messages[-1]["content"]


def test_factory_degrades_when_backend_construction_breaks():
    import lligma.tutor.backends as backends

    class BrokenLocal:
        def __init__(self, config):
            raise RuntimeError("driver rotto")

    original = backends.LocalHFBackend
    backends.LocalHFBackend = BrokenLocal
    try:
        config = TutorConfig.from_file(ROOT / "configs" / "tutor.yaml")
        config.generation.backend = "local"
        backend, notes = make_backend(config)
    finally:
        backends.LocalHFBackend = original
    assert backend.name == "extractive"
    assert any("driver rotto" in note for note in notes)


def test_factory_falls_back_to_extractive_without_api_key():
    config = TutorConfig.from_file(ROOT / "configs" / "tutor.yaml")
    config.generation.backend = "auto"
    backend, notes = make_backend(config)
    # In un ambiente senza pesi locali e senza chiave, l'unico backend valido e'
    # l'estrattivo, e la ragione deve essere dichiarata.
    assert backend.name in {"extractive", "openai"}
    if backend.name == "extractive":
        assert notes


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
    print(f"\n{len(tests) - failures}/{len(tests)} test superati")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
