"""Contratto di formato dell'indice: rifiutare un v1, e ricostruire identico.

I termini ridotti vivono dentro ``index/tutor-index.json``. Un indice costruito
con un vocabolario diverso si caricherebbe senza errori e cercherebbe con chiavi
che non corrispondono a quelle nuove: risposte sbagliate, nessun messaggio. Per
questo il formato e' versionato e ogni consumatore (service, server, script di
valutazione, test di deriva) passa da ``BM25Index.load``, che rifiuta in modo
esplicito tutto cio' che non e' la versione attesa.

Seconda proprieta' verificata qui: la costruzione e' riproducibile.
``built_at`` e' l'unico campo che dipende dall'orologio; con
``SOURCE_DATE_EPOCH`` fissato due costruzioni devono produrre lo stesso file
byte per byte.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lligma.tutor.index import BM25Index, Document, INDEX_VERSION  # noqa: E402

SHIPPED = ROOT / "index" / "tutor-index.json"
EPOCH = 1800000000


def _documents() -> list[Document]:
    return [
        Document(
            id="demo.suolo",
            title="Il suolo agrario",
            text=(
                "Il suolo agrario e' lo strato superficiale nel quale le piante "
                "affondano le radici. La tessitura descrive la proporzione fra "
                "sabbia, limo e argilla. Un suolo franco e' il piu' adatto "
                "all'agricoltura perche' trattiene acqua e nutrienti."
            ),
            subject="agronomia",
        )
    ]


def test_legacy_v1_index_is_rejected_with_an_explicit_message(tmp_path):
    legacy = {"version": 1, "metadata": {"documents": 1}, "chunks": []}
    path = tmp_path / "tutor-index.json"
    path.write_text(json.dumps(legacy), encoding="utf-8")

    with pytest.raises(ValueError) as error:
        BM25Index.load(path)

    message = str(error.value)
    assert "v1" in message, message
    assert f"atteso v{INDEX_VERSION}" in message, message
    assert "ricostruiscilo" in message, message


def test_index_without_version_is_rejected(tmp_path):
    path = tmp_path / "tutor-index.json"
    path.write_text(json.dumps({"metadata": {}, "chunks": []}), encoding="utf-8")

    with pytest.raises(ValueError) as error:
        BM25Index.load(path)

    assert "ricostruiscilo" in str(error.value)


def test_shipped_index_is_the_current_version():
    payload = json.loads(SHIPPED.read_text(encoding="utf-8"))
    assert payload["version"] == INDEX_VERSION

    loaded = BM25Index.load(SHIPPED)
    assert loaded.metadata.get("version") == INDEX_VERSION
    assert len(loaded) > 0


def test_rebuild_is_byte_identical_with_a_fixed_timestamp(tmp_path, monkeypatch):
    monkeypatch.setenv("SOURCE_DATE_EPOCH", str(EPOCH))
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    for path in (first, second):
        BM25Index.build(_documents(), chunk_words=40, overlap_words=10).save(path)

    assert first.read_bytes() == second.read_bytes()
    expected = datetime.fromtimestamp(EPOCH, timezone.utc).isoformat(timespec="seconds")
    metadata = json.loads(first.read_text(encoding="utf-8"))["metadata"]
    assert metadata["built_at"] == expected
