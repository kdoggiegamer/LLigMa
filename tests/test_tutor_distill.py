"""The parts of the distillation script that can be tested without a network.

The dataset builder is mostly a parser and a filter: it must survive a teacher
that wraps JSON in prose, return nothing instead of crashing on garbage, and
reject answers that are actually about something else. Those are the pieces that
decide whether the student is trained on grounded data or on noise, so they are
tested directly.

The script imports torch-free modules only, so this runs on any machine.

Run with pytest, or directly: python tests/test_tutor_distill.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from distill_tutor import extract_pairs, valid_pair  # noqa: E402

PASSAGE = (
    "La rotazione colturale alterna nello stesso appezzamento colture con esigenze "
    "diverse. Le leguminose vivono in simbiosi con batteri del genere Rhizobium che "
    "fissano l'azoto dell'aria nelle radici."
)


def test_extracts_a_plain_json_array():
    raw = '[{"question": "Che cos\'e\' la rotazione?", "answer": "E\' l\'alternanza di colture."}]'
    pairs = extract_pairs(raw)
    assert len(pairs) == 1
    assert pairs[0]["question"].startswith("Che cos")


def test_extracts_json_wrapped_in_prose_or_fences():
    raw = 'Ecco le coppie:\n```json\n[{"question": "A?", "answer": "B."}]\n```\nSpero vadano bene.'
    assert len(extract_pairs(raw)) == 1


def test_garbage_returns_an_empty_list_instead_of_crashing():
    assert extract_pairs("Non posso aiutarti con questa richiesta.") == []
    assert extract_pairs('[{"question": "senza chiusura"') == []


def test_accepts_a_grounded_pair():
    assert valid_pair(
        "Perche' la rotazione aiuta il terreno?",
        "La rotazione alterna colture diverse e le leguminose fissano l'azoto grazie ai batteri Rhizobium.",
        PASSAGE,
    )


def test_rejects_an_answer_about_something_else():
    assert not valid_pair(
        "Perche' la rotazione aiuta il terreno?",
        "Il Parmigiano Reggiano si produce in Emilia con latte di vacca e lunghe stagionature.",
        PASSAGE,
    )


def test_rejects_an_abstention_and_a_non_question():
    assert not valid_pair("Cosa?", "Non ho informazioni sufficienti per rispondere a questa domanda.", PASSAGE)
    assert not valid_pair("La rotazione colturale", "La rotazione alterna colture diverse nel tempo.", PASSAGE)


def test_rejects_a_too_short_answer():
    assert not valid_pair("Perche' ruotare?", "Perche' si deve.", PASSAGE)


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
