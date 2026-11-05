"""Cache isolation and consistent SSE metadata.

The service cache is shared by all browser sessions. A caller that mutates
``Answer.sources`` after a ``put`` must not change the response another student
receives, and a cached abstention must go through the same SSE path as a cached
answer. These tests keep that behavior visible without changing the production
lock invariants already exercised by ``mutation_cache_lock.py``.

Run with pytest, or directly: python tests/test_tutor_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from lligma.tutor.prompt import ABSTENTION_TEXT, subject_label  # noqa: E402
from lligma.tutor.schema import Answer, Source  # noqa: E402
from lligma.tutor.service import AnswerCache  # noqa: E402
from test_tutor_service import make_service  # noqa: E402


def test_cache_copies_sources_on_put_and_get():
    cache = AnswerCache()
    source = Source(
        title="Il suolo",
        subject="Agronomia",
        source="Dispensa",
        quote="citazione originale",
        document_id="agro.suolo",
        numbers=[1],
    )
    answer = Answer(text="risposta originale", sources=[source])
    cache.put("Che cosa e' il suolo?", None, answer)

    source.quote = "mutata dopo il put"
    cached = cache.get("Che cosa e' il suolo?", None)
    assert cached is not None
    assert cached.sources[0].quote == "citazione originale"

    cached.sources[0].quote = "mutata dopo il get"
    again = cache.get("Che cosa e' il suolo?", None)
    assert again is not None
    assert again.sources[0].quote == "citazione originale"


def test_stream_caches_and_reuses_abstention():
    service = make_service()
    service.cache.clear()
    question = "zxcvbnm qwertyuiop"

    first = list(service.stream(question, use_cache=True))
    assert first[0]["type"] == "meta"
    assert first[0]["cached"] is False
    assert first[0]["abstained"] is True
    assert first[-1]["type"] == "done"
    assert first[-1]["abstained"] is True
    assert "".join(event.get("text", "") for event in first) == ABSTENTION_TEXT

    second = list(service.stream(question, use_cache=True))
    assert second[0]["cached"] is True
    assert second[-1]["cached"] is True
    assert second[-1]["abstained"] is True
    assert "".join(event.get("text", "") for event in second) == ABSTENTION_TEXT


def test_stream_cached_meta_keeps_the_subject_label():
    service = make_service()
    service.cache.clear()
    question = "Come si pota la vite?"

    first = list(service.stream(question, use_cache=True))
    assert first[0]["cached"] is False
    assert first[0]["subject"] == "agronomia"
    assert first[0]["subject_label"] == subject_label("agronomia")

    second = list(service.stream(question, use_cache=True))
    assert second[0]["cached"] is True
    assert second[0]["subject"] == first[0]["subject"]
    assert second[0]["subject_label"] == first[0]["subject_label"]
    assert second[0]["sources"] == first[0]["sources"]


def _run_all() -> int:
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
