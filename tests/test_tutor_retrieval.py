"""Retrieval, segmentation and abstention must behave on real Italian text.

These are the load-bearing behaviours of the tutor: if chunking cuts a fact in
half, or BM25 ranks the wrong scheda first, no amount of model quality recovers
it. The tests run on the shipped knowledge base, not on toy strings, because a
retriever that works on three English sentences proves nothing here. The save
test also checks that an interrupted index replacement leaves the previous
index intact and cleans up its temporary file.

Run with pytest, or directly: python tests/test_tutor_retrieval.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lligma.tutor import index as index_module  # noqa: E402
from lligma.tutor.backends import ExtractiveBackend  # noqa: E402
from lligma.tutor.config import TutorConfig  # noqa: E402
from lligma.tutor.expansion import QueryExpander  # noqa: E402
from lligma.tutor.index import BM25Index, Document  # noqa: E402
from lligma.tutor.service import TutorService  # noqa: E402
from lligma.tutor.text import chunk_text, normalize, split_sentences, stem_tokens  # noqa: E402


def load_documents() -> list[Document]:
    from build_tutor_index import collect

    return collect(ROOT / "data" / "tutor", "generale")


def build_index() -> BM25Index:
    return BM25Index.build(load_documents(), chunk_words=110, overlap_words=28)


def load_paraphrases() -> list[dict]:
    path = ROOT / "data" / "tutor" / "eval-paraphrase.jsonl"
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def make_paraphrase_service(expanded: bool) -> TutorService:
    config = TutorConfig.from_file(ROOT / "configs" / "tutor.yaml")
    config.generation.backend = "extractive"
    config.retrieval.expansion.enabled = expanded
    return TutorService(
        config,
        index=build_index(),
        backend=ExtractiveBackend(max_sentences=5, min_overlap=2),
    )


def test_normalize_removes_accents_and_punctuation():
    assert normalize("Perché l'Acqua è H2O?") == "perche l acqua e h2o"
    assert normalize("  più   spazi  ") == "piu spazi"


def test_stemming_aligns_singular_and_plural():
    # La riduzione minima deve far coincidere le forme flesse italiane.
    assert stem_tokens("mela")[0] == stem_tokens("mele")[0]
    assert stem_tokens("suolo")[0] == stem_tokens("suoli")[0]


def test_stemming_aligns_verb_and_noun_families():
    # "pota", "potare" e "potatura" devono cercare la stessa scheda: uno
    # studente scrive il verbo, la dispensa usa il nome.
    assert stem_tokens("pota")[0] == stem_tokens("potatura")[0] == stem_tokens("potare")[0]
    assert stem_tokens("concima")[0] == stem_tokens("concimazione")[0]


def test_stopwords_are_removed_but_negations_survive():
    tokens = stem_tokens("il suolo non trattiene l'acqua")
    assert "il" not in tokens and "l" not in tokens
    assert "non" in tokens, "la negazione non va rimossa: cambia il significato"


def test_sentence_split_respects_abbreviations():
    text = "Fra gli organuli ci sono i mitocondri, per es. quelli del muscolo. Sono importanti."
    sentences = split_sentences(text)
    assert len(sentences) == 2, sentences


def test_chunking_covers_all_text_and_respects_size():
    text = " ".join(f"Frase numero {index} con un contenuto qualsiasi." for index in range(60))
    pieces = chunk_text(text, max_words=40, overlap_words=10)
    assert len(pieces) > 1
    for piece in pieces:
        assert len(piece.split()) <= 40
    # Nessuna frase deve sparire: la prima e l'ultima devono esserci.
    assert "Frase numero 0" in pieces[0]
    assert "Frase numero 59" in pieces[-1]


def test_chunking_overlaps_neighbouring_pieces():
    text = " ".join(f"Frase numero {index} con un contenuto qualsiasi." for index in range(60))
    pieces = chunk_text(text, max_words=40, overlap_words=12)
    first_words = set(pieces[0].split())
    second_words = set(pieces[1].split())
    assert len(first_words & second_words) >= 6, "senza overlap un fatto a cavallo va perso"


def test_index_ranks_the_right_scheda_first():
    index = build_index()
    cases = {
        "perche' si alternano le colture": "agro.rotazione",
        "quanta energia passa al livello trofico successivo": "bio.ecosistemi",
        "chi ha scritto la divina commedia": "cul.dante",
        "come si chiama il fiume piu' lungo d'italia": "cul.geografia",
        "a cosa serve il letame": "agro.organici",
    }
    failures = []
    for question, expected in cases.items():
        hits = index.search(question, top_k=3)
        found = [hit.chunk.document_id for hit in hits]
        if expected not in found:
            failures.append((question, expected, found))
    assert not failures, failures


def test_index_filters_by_subject():
    index = build_index()
    hits = index.search("terreno", top_k=5, subject="agronomia")
    assert hits, "il terreno e' un tema centrale di agronomia"
    assert all(hit.chunk.subject == "agronomia" for hit in hits)


def test_index_returns_nothing_for_noise():
    index = build_index()
    assert index.search("qwertyuiop zxcvbnm") == []


def test_min_terms_rejects_single_word_coincidences():
    """Una parola in comune non basta: "calcola" non rende pertinente la zootecnia."""
    index = build_index()
    weak = index.search("come si calcola l'integrale di una funzione", top_k=3, min_terms=1)
    strict = index.search("come si calcola l'integrale di una funzione", top_k=3, min_terms=2)
    assert weak, "con un solo termine la ricerca trova comunque qualcosa"
    assert strict == [], "con due termini richiesti la coincidenza deve sparire"


def test_shipped_index_matches_the_schede():
    """L'indice versionato deve corrispondere alle schede su disco.

    Se si modifica il corpus senza rigenerarlo, il tutor continua a rispondere
    dal vecchio indice: un errore silenzioso, gia' successo una volta.
    """
    shipped = BM25Index.load(ROOT / "index" / "tutor-index.json")
    rebuilt = build_index()
    assert [(c.document_id, c.ordinal, c.text) for c in shipped.chunks] == [
        (c.document_id, c.ordinal, c.text) for c in rebuilt.chunks
    ], "indice stantio: esegui python scripts/build_tutor_index.py"
    assert shipped.metadata["documents"] == rebuilt.metadata["documents"]


def test_index_save_round_trips_after_atomic_replace():
    index = build_index()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tutor-index.json"
        index.save(path)
        loaded = BM25Index.load(path)
        assert len(loaded) == len(index)
        assert loaded.metadata["chunks"] == len(index)
        assert loaded.chunks[0].text == index.chunks[0].text


def test_index_save_keeps_the_old_file_if_replace_fails():
    """Una scrittura interrotta non deve distruggere l'indice funzionante."""
    index = build_index()
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tutor-index.json"
        path.write_text("old-index", encoding="utf-8")
        with mock.patch.object(
            index_module.os, "replace", side_effect=OSError("replace simulato")
        ):
            try:
                index.save(path)
            except OSError as error:
                assert str(error) == "replace simulato", error
            else:
                raise AssertionError("save avrebbe dovuto propagare l'errore")
        assert path.read_text(encoding="utf-8") == "old-index"
        assert [item.name for item in Path(directory).iterdir()] == ["tutor-index.json"]


def test_paraphrase_block_fails_without_expansion():
    """Il blocco nuovo deve poter fallire: senza sinonimi il recall non e' pieno."""
    service = make_paraphrase_service(expanded=False)
    items = load_paraphrases()
    missed = []
    for item in items:
        hits, _ = service.retrieve(item["question"], item.get("subject"))
        found = {hit.chunk.document_id for hit in hits}
        if item["expect_doc"] not in found:
            missed.append(item["question"])
    assert missed, "il blocco non discrimina: tutte le parafrasi passano gia' senza espansione"
    assert len(missed) == len(items), (
        "il blocco deve fallire interamente senza espansione; "
        f"gia' trovate: {sorted(set(item['question'] for item in items) - set(missed))}"
    )


def test_paraphrase_block_reaches_full_recall_with_expansion():
    """Con l'espansione la scheda attesa deve essere la prima, non un hit qualsiasi."""
    service = make_paraphrase_service(expanded=True)
    failures = []
    for item in load_paraphrases():
        question = item["question"]
        hits, _ = service.retrieve(question, item.get("subject"))
        found = [hit.chunk.document_id for hit in hits]
        if not found or found[0] != item["expect_doc"]:
            failures.append((question, item["expect_doc"], found[:3]))
            continue
        answer = service.ask(question, subject=item.get("subject"), use_cache=False)
        if answer.abstained:
            failures.append((question, "astensione inattesa", []))
            continue
        source_ids = [source.document_id for source in answer.sources]
        if not source_ids or source_ids[0] != item["expect_doc"]:
            failures.append((question, "risposta non ancorata alla scheda attesa", source_ids[:3]))
            continue
        text = normalize(answer.text)
        missing = [word for word in item.get("must_include", []) if normalize(word) not in text]
        if missing:
            failures.append((question, f"mancano {missing}", answer.text[:120]))
    assert not failures, failures


def test_evaluate_compare_reports_both_sets_in_one_run():
    """Il report unico deve mostrare prima/dopo per entrambi i set."""
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "evaluate_tutor.py"), "--compare"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=120,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "eval.jsonl" in output, output
    assert "eval-paraphrase.jsonl" in output, output
    assert "30/30" in output, output
    assert "0/39" in output, output
    assert "39/39" in output, output
    assert "PASS" in output, output


def test_expansion_default_is_on_and_cache_uses_the_original_question():
    """Il default distribuito attiva il dizionario: spento, le parafrasi sono 0/39."""
    config = TutorConfig.from_file(ROOT / "configs" / "tutor.yaml")
    assert config.retrieval.expansion.enabled is True

    disabled = make_paraphrase_service(expanded=False)
    assert disabled.retrieve("Che cosa è lo stallatico?", "agronomia")[0] == []

    enabled = make_paraphrase_service(expanded=True)
    question = "Che cosa è lo stallatico?"
    answer = enabled.ask(question, subject="agronomia", use_cache=True)
    assert not answer.abstained
    cached = enabled.cache.get(question, "agronomia")
    assert cached is not None and cached.cached
    assert cached.text == answer.text


def test_expansion_preserves_out_of_scope_abstention():
    service = make_paraphrase_service(expanded=True)
    for question in (
        "Chi ha vinto i campionati mondiali di calcio del 2022?",
        "Come si calcola l'integrale di una funzione razionale?",
        "Quanto costa un trattore nuovo di 120 cavalli?",
        "Qual è la ricetta originale della carbonara?",
    ):
        assert service.ask(question, use_cache=False).abstained, question


def test_expander_caps_the_retrieval_query_at_800_characters():
    expander = QueryExpander.from_mapping({"stallatico": "letame"})
    expanded = expander.expand("stallatico " * 200)
    assert expanded.startswith("letame")
    assert len(expanded) <= 800
    assert not expanded.endswith(" ")


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
