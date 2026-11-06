#!/usr/bin/env python3
"""Costruisce l'indice del tutor dalle schede di conoscenza.

    python scripts/build_tutor_index.py --data-dir data/tutor --probe "cos'e' l'humus"

Legge ``.jsonl`` (un documento per riga: id, title, subject, text, source),
``.json`` (lista di documenti), ``.txt`` e ``.md``. Per i file di testo il titolo
e' la prima riga esplicita (``# Titolo``) oppure il nome del file, e la materia
viene dalla sottocartella o da ``--subject``.

Il file prodotto e' JSON leggibile: si puo' aprire e capire cosa sa il tutor.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lligma.tutor.console import setup_utf8  # noqa: E402
from lligma.tutor.index import BM25Index, Document  # noqa: E402

SUPPORTED = {".jsonl", ".json", ".txt", ".md"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/tutor")
    parser.add_argument("--output", default="index/tutor-index.json")
    parser.add_argument("--chunk-words", type=int, default=110)
    parser.add_argument("--overlap-words", type=int, default=28)
    parser.add_argument("--title-boost", type=float, default=1.5)
    parser.add_argument("--subject", default=None, help="Materia per .txt/.md senza metadati")
    parser.add_argument("--probe", nargs="*", default=[], help="Domande di prova dopo la costruzione")
    return parser.parse_args()


def documents_from_jsonl(path: Path) -> list[Document]:
    documents: list[Document] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(f"{path}:{number}: JSON non valido ({error})") from error
            documents.append(Document.from_dict(payload))
    return documents


def documents_from_json(path: Path) -> list[Document]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("documents") or [payload]
    return [Document.from_dict(item) for item in payload]


def documents_from_text(path: Path, default_subject: str) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    title = path.stem.replace("_", " ").replace("-", " ").strip().title()
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            break
        if stripped:
            break
    subject = path.parent.name if path.parent.name not in {"tutor", "data", ""} else default_subject
    return [
        Document(
            id=f"{path.parent.name}/{path.stem}",
            title=title,
            text=text,
            subject=subject.lower() if subject else "generale",
            source=path.name,
        )
    ]


def collect(data_dir: Path, default_subject: str) -> list[Document]:
    if not data_dir.is_dir():
        raise SystemExit(f"cartella dati non trovata: {data_dir}")
    documents: list[Document] = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED or not path.is_file():
            continue
        # Nella stessa cartella convivono schede, set di valutazione e dataset di
        # distillazione: solo i primi sono conoscenza indicizzabile.
        if path.stem.lower().startswith(("eval", "distill")):
            print(f"  salto {path.name} (non e' una scheda di conoscenza)")
            continue
        if path.suffix.lower() == ".jsonl":
            found = documents_from_jsonl(path)
        elif path.suffix.lower() == ".json":
            found = documents_from_json(path)
        else:
            found = documents_from_text(path, default_subject)
        documents.extend(found)
        print(f"  {path.relative_to(data_dir)}: {len(found)} schede")
    return documents


def main() -> int:
    setup_utf8()
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = Path(__file__).resolve().parents[1] / data_dir

    print(f"Leggo le schede da {data_dir}")
    documents = collect(data_dir, args.subject or "generale")
    if not documents:
        raise SystemExit("nessuna scheda trovata: l'indice sarebbe vuoto")

    seen: set[str] = set()
    unique: list[Document] = []
    for document in documents:
        if not document.text.strip():
            print(f"  salto scheda vuota: {document.id}")
            continue
        if document.id in seen:
            print(f"  salto id duplicato: {document.id}")
            continue
        seen.add(document.id)
        unique.append(document)

    index = BM25Index.build(
        unique,
        chunk_words=args.chunk_words,
        overlap_words=args.overlap_words,
        title_boost=args.title_boost,
    )
    output = Path(args.output)
    if not output.is_absolute():
        output = Path(__file__).resolve().parents[1] / output
    index.save(output)

    print()
    print(f"Schede      : {len(unique)}")
    print(f"Passaggi    : {len(index)}")
    print(f"Vocabolario : {len(index.idf):,} termini")
    print(f"Lunghezza media: {index.average_length:.1f} termini per passaggio")
    print(f"Materie     : {', '.join(index.metadata['subjects'])}")
    print(f"Costruito   : {index.metadata['built_at']}")
    print(f"Salvato in  : {output} ({output.stat().st_size / 1024:.0f} KB)")

    for question in args.probe:
        print(f"\n> {question}")
        hits = index.search(question, top_k=3)
        if not hits:
            print("  nessun risultato")
        for hit in hits:
            print(f"  {hit.score:6.2f}  [{hit.chunk.subject}] {hit.title}")
            print(f"          {hit.text[:120]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
