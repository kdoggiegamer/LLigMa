"""Indice BM25 in puro Python, salvabile su disco come JSON.

Perche' BM25 e non embedding densi: su un corpus di poche migliaia di passaggi
scritti nella stessa lingua della domanda, BM25 e' accurato quanto un modello di
embedding, costa zero parametri, zero GPU e pochi millisecondi. Un embedding
aggiungerebbe una dipendenza da 100 MB per guadagnare poco: la scelta "piccolo e
veloce" vale anche per il retriever, non solo per il generatore.

Il formato su disco e' deliberatamente leggibile (JSON): una scuola puo'
ispezionare cosa sa il tutor senza strumenti particolari.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .text import chunk_text, stem_tokens

INDEX_VERSION = 1
BM25_K1 = 1.5
BM25_B = 0.75


@dataclass
class Document:
    """Un'unita' di conoscenza: una scheda, un capitolo, una voce."""

    id: str
    title: str
    text: str
    subject: str
    source: str = ""

    @classmethod
    def from_dict(cls, payload: dict) -> "Document":
        missing = {"title", "text"} - set(payload)
        if missing:
            raise ValueError(f"documento senza campi obbligatori {sorted(missing)}: {payload!r}")
        return cls(
            id=str(payload.get("id") or payload["title"]).strip(),
            title=str(payload["title"]).strip(),
            text=str(payload["text"]).strip(),
            subject=str(payload.get("subject", "generale")).strip().lower(),
            source=str(payload.get("source", "")).strip(),
        )


@dataclass
class Chunk:
    """Passaggio indicizzato, con le frequenze dei termini gia' calcolate."""

    document_id: str
    title: str
    subject: str
    source: str
    text: str
    ordinal: int
    terms: dict[str, int] = field(default_factory=dict)
    length: int = 0
    title_terms: list[str] = field(default_factory=list)

    @property
    def chunk_id(self) -> str:
        return f"{self.document_id}#{self.ordinal}"

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "subject": self.subject,
            "source": self.source,
            "text": self.text,
            "ordinal": self.ordinal,
            "terms": self.terms,
            "length": self.length,
            "title_terms": self.title_terms,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "Chunk":
        return cls(
            document_id=payload["document_id"],
            title=payload["title"],
            subject=payload["subject"],
            source=payload.get("source", ""),
            text=payload["text"],
            ordinal=int(payload["ordinal"]),
            terms={str(key): int(value) for key, value in (payload.get("terms") or {}).items()},
            length=int(payload.get("length") or 0),
            title_terms=list(payload.get("title_terms") or []),
        )


@dataclass
class Hit:
    """Risultato di ricerca: punteggio e passaggio."""

    score: float
    chunk: Chunk

    @property
    def title(self) -> str:
        return self.chunk.title

    @property
    def text(self) -> str:
        return self.chunk.text


class BM25Index:
    """Indice invertito con scoring Okapi BM25.

    Il punteggio somma, per ogni termine della domanda, ``idf * tf_normalizzato``.
    I termini che compaiono nel titolo ricevono un bonus: in un manuale il titolo
    e' gia' una risposta compressa alla domanda "di cosa parla questo passaggio".
    """

    def __init__(
        self,
        chunks: list[Chunk],
        metadata: dict | None = None,
        title_boost: float = 1.5,
    ):
        if not chunks:
            raise ValueError("indice vuoto: servono almeno un passaggio")
        self.chunks = chunks
        self.metadata = metadata or {}
        self.title_boost = title_boost
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        document_frequency: Counter[str] = Counter()

        for index, chunk in enumerate(chunks):
            for term, frequency in chunk.terms.items():
                self.postings[term][index] = frequency
            for term in chunk.terms:
                document_frequency[term] += 1

        self.document_frequency = dict(document_frequency)
        total = sum(chunk.length for chunk in chunks)
        self.average_length = total / len(chunks) if chunks else 0.0
        self.idf = {
            term: math.log(1.0 + (len(chunks) - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    # -- costruzione ---------------------------------------------------------

    @classmethod
    def build(
        cls,
        documents: Iterable[Document],
        chunk_words: int = 110,
        overlap_words: int = 28,
        title_boost: float = 1.5,
    ) -> "BM25Index":
        chunks: list[Chunk] = []
        document_list = list(documents)
        for document in document_list:
            pieces = chunk_text(document.text, chunk_words, overlap_words)
            for ordinal, piece in enumerate(pieces):
                tokens = stem_tokens(piece)
                if not tokens:
                    continue
                chunks.append(
                    Chunk(
                        document_id=document.id,
                        title=document.title,
                        subject=document.subject,
                        source=document.source,
                        text=piece,
                        ordinal=ordinal,
                        terms=dict(Counter(tokens)),
                        length=len(tokens),
                        title_terms=sorted(set(stem_tokens(document.title))),
                    )
                )
        metadata = {
            "version": INDEX_VERSION,
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "documents": len(document_list),
            "chunks": len(chunks),
            "subjects": sorted({document.subject for document in document_list}),
            "chunk_words": chunk_words,
            "overlap_words": overlap_words,
        }
        return cls(chunks, metadata, title_boost=title_boost)

    # -- ricerca -------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        subject: str | None = None,
        min_score: float = 0.0,
        min_terms: int = 1,
    ) -> list[Hit]:
        terms = stem_tokens(query)
        if not terms:
            return []
        # Servono `min_terms` termini in comune, e mai piu' dei termini distinti
        # che la domanda contiene: una domanda di una sola parola non puo'
        # pretendere due riscontri. Il filtro e' volutamente sui termini grezzi e
        # non solo su quelli rari: le parole "comuni" di una domanda italiana
        # ("perche'", "colture", "anni") sono spesso proprio quelle che
        # confermano che il passaggio parla di quello e non d'altro.
        required = min(max(1, min_terms), len(set(terms)))

        scores: dict[int, float] = defaultdict(float)
        matched: dict[int, int] = defaultdict(int)
        for term, query_frequency in Counter(terms).items():
            idf = self.idf.get(term)
            if idf is None:
                continue
            posting = self.postings.get(term, {})
            for chunk_index, frequency in posting.items():
                length = self.chunks[chunk_index].length or 1
                denominator = frequency + BM25_K1 * (
                    1.0 - BM25_B + BM25_B * length / (self.average_length or 1.0)
                )
                scores[chunk_index] += query_frequency * idf * (frequency * (BM25_K1 + 1.0)) / denominator
                matched[chunk_index] += 1

        wanted = (subject or "").strip().lower()
        hits: list[Hit] = []
        seen: set[str] = set()
        for chunk_index, score in scores.items():
            chunk = self.chunks[chunk_index]
            if wanted and chunk.subject != wanted:
                continue
            if self.title_boost and any(term in chunk.title_terms for term in set(terms)):
                score += self.title_boost * max((self.idf.get(term, 0.0) for term in set(terms)), default=0.0)
            if matched.get(chunk_index, 0) < required:
                continue
            if score < min_score:
                continue
            # Passaggi quasi identici (overlap fra chunk) non vanno mostrati due volte.
            fingerprint = chunk.text[:80].lower()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            hits.append(Hit(score=round(score, 4), chunk=chunk))

        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]

    def subject_counts(self) -> dict[str, int]:
        counts: Counter[str] = Counter(chunk.subject for chunk in self.chunks)
        return dict(counts)

    # -- persistenza ---------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": INDEX_VERSION,
            "metadata": self.metadata,
            "chunks": [chunk.to_dict() for chunk in self.chunks],
        }
        serialized = json.dumps(payload, ensure_ascii=False)
        try:
            mode = path.stat().st_mode & 0o777
        except FileNotFoundError:
            mode = 0o644
        # Un indice scritto a meta' e' peggio di un indice vecchio: la scuola
        # resta senza tutor finche' non se ne accorge. Scrivi su un file
        # temporaneo e sostituisci solo a contenuto completo e sincronizzato.
        descriptor, temporary_name = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(serialized)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, mode)
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return path

    @classmethod
    def load(cls, path: str | Path, title_boost: float = 1.5) -> "BM25Index":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Indice non trovato: {path}. Costruiscilo con "
                f"`python scripts/build_tutor_index.py`."
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = payload.get("version")
        if version != INDEX_VERSION:
            raise ValueError(
                f"Indice in formato v{version}, atteso v{INDEX_VERSION}: ricostruiscilo."
            )
        chunks = [Chunk.from_dict(item) for item in payload["chunks"]]
        return cls(chunks, payload.get("metadata") or {}, title_boost=title_boost)

    def __len__(self) -> int:
        return len(self.chunks)
