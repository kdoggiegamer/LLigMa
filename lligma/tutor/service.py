"""Orchestrazione: domanda -> retrieval -> guardia -> generazione -> citazioni.

Qui vive la logica che rende il tutor utilizzabile in classe:

* **guardia di astensione**: se il miglior passaggio e' sotto soglia, non si
  genera nulla. Un modello piccolo lasciato libero con contesto debole produce
  la cosa peggiore possibile in una scuola: una risposta sicura e sbagliata.
* **cache semantica**: due studenti che chiedono la stessa cosa con parole
  leggermente diverse non devono pagare due generazioni. Le domande in una classe
  si ripetono moltissimo, quindi questa e' la singola ottimizzazione di latenza
  con il miglior rapporto beneficio/rischio.
* **degradazione controllata**: se il backend generativo fallisce si risponde con
  l'estrattivo, dichiarandolo, invece di mostrare un errore a uno studente.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Iterator

from .backends import AnswerRequest, Backend, BackendError, make_backend
from .config import TutorConfig
from .expansion import QueryExpander
from .index import BM25Index, Hit
from .prompt import (
    ABSTENTION_TEXT,
    build_messages,
    format_context,
    guess_subject,
    is_abstention_text,
    sources_from_hits,
    subject_label,
)
from .schema import Answer, Source
from .text import content_words

# Soglia di similarita' (Jaccard sulle parole) sopra la quale due domande sono
# considerate la stessa. 0,9 e' alto di proposito: "cos'e' il suolo" e "cos'e' il
# suolo agrario" sono la stessa domanda, "come si concima il grano" e "come si
# pota il grano" non lo sono.
CACHE_SIMILARITY = 0.9


def _copy_answer(answer: Answer) -> Answer:
    """Copia indipendente di una risposta e delle sue fonti.

    La cache non deve condividere oggetti mutabili con il chiamante: una
    modifica accidentale a ``answer.sources`` non puo' cambiare una risposta
    gia' memorizzata e servita a un altro studente.
    """
    return Answer(
        text=answer.text,
        sources=[Source(**source.to_dict()) for source in answer.sources],
        backend=answer.backend,
        abstained=answer.abstained,
        cached=answer.cached,
        degraded=answer.degraded,
        latency_ms=answer.latency_ms,
        retrieved=answer.retrieved,
        subject=answer.subject,
    )


class AnswerCache:
    """Cache LRU in memoria con confronto fuzzy delle domande."""

    def __init__(self, max_entries: int = 512, similarity: float = CACHE_SIMILARITY):
        self.max_entries = max_entries
        self.similarity = similarity
        # La chiave include il filtro richiesto dall'utente, non la materia
        # indovinata: una domanda posta senza filtro deve ritrovare la propria
        # risposta anche se il rilevamento automatico aveva scelto una materia.
        self._entries: OrderedDict[str, tuple[set[str], str | None, Answer]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        # Il server e' ThreadingHTTPServer: due studenti possono chiedere nello
        # stesso istante, quindi anche la cache deve essere protetta.
        self._lock = threading.Lock()

    def get(self, question: str, subject: str | None) -> Answer | None:
        words = content_words(question)
        if not words:
            return None
        wanted = subject or None
        with self._lock:
            for key, (cached_words, cached_subject, answer) in list(self._entries.items()):
                if cached_subject != wanted:
                    continue
                union = cached_words | words
                if not union:
                    continue
                overlap = len(cached_words & words) / len(union)
                if overlap >= self.similarity:
                    self._entries.move_to_end(key)
                    self.hits += 1
                    cached = _copy_answer(answer)
                    cached.cached = True
                    return cached
            self.misses += 1
            return None

    def put(self, question: str, subject: str | None, answer: Answer) -> None:
        words = content_words(question)
        if not words:
            return
        key = f"{subject or '*'}|{question.strip().lower()}"
        snapshot = _copy_answer(answer)
        with self._lock:
            self._entries[key] = (words, subject or None, snapshot)
            self._entries.move_to_end(key)
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class TutorService:
    """Facade usata dal server, dagli script di valutazione e dai test."""

    def __init__(
        self,
        config: TutorConfig,
        index: BM25Index | None = None,
        backend: Backend | None = None,
        cache: AnswerCache | None = None,
    ):
        self.config = config
        self.index = index or BM25Index.load(
            config.index.path, title_boost=config.retrieval.title_boost
        )
        notes: list[str] = []
        if backend is None:
            self.backend, notes = make_backend(config)
        else:
            self.backend = backend
        self.startup_notes = notes
        self.cache = cache if cache is not None else AnswerCache()
        self.expander = QueryExpander.from_mapping(
            config.retrieval.expansion.synonyms
        )

    def _search_question(self, question: str) -> str:
        """Applica i sinonimi solo alla ricerca, mai alla domanda o alla cache."""
        expansion = self.config.retrieval.expansion
        if not expansion.enabled:
            return question
        return self.expander.expand(question)

    # -- costruzione ---------------------------------------------------------

    @classmethod
    def from_config(
        cls, path: str | Path, backend: Backend | None = None
    ) -> "TutorService":
        return cls(TutorConfig.from_file(path), backend=backend)

    # -- retrieval -----------------------------------------------------------

    def retrieve(self, question: str, subject: str | None = None) -> tuple[list[Hit], str | None]:
        """Recupera passaggi, provando prima la materia probabile.

        Se il filtro per materia non produce risultati utili si riprova su tutto
        l'indice: un filtro sbagliato non deve trasformarsi in un "non lo so".
        """
        config = self.config.retrieval
        guessed = (subject or "").strip().lower() or guess_subject(question)
        min_score = config.min_score
        search_question = self._search_question(question)
        # Se una regola ha sostituito un termine assente dal corpus, il termine
        # originale non puo' contribuire al conteggio dei riscontri: la prova
        # minima scende a uno, mentre la soglia assoluta del punteggio resta.
        # Senza espansione il valore resta esattamente quello configurato.
        min_terms = 1 if search_question != question else config.min_terms

        hits = self.index.search(
            search_question, top_k=config.top_k, subject=guessed, min_score=0.0,
            min_terms=min_terms,
        )
        if hits and hits[0].score >= min_score:
            return [hit for hit in hits if hit.score >= min_score * 0.5], guessed

        all_hits = self.index.search(
            search_question, top_k=config.top_k, min_score=0.0, min_terms=min_terms
        )
        if all_hits and all_hits[0].score >= min_score:
            return [hit for hit in all_hits if hit.score >= min_score * 0.5], None

        # Ultimo tentativo, volutamente stretto: una sola parola in comune basta
        # solo se la domanda stessa si colloca in una materia riconosciuta e il
        # passaggio e' di quella materia, con un punteggio ben sopra soglia. Cosi'
        # "che cos'e' l'humus e perche' conta" trova la scheda sull'humus, mentre
        # una domanda fuori ambito, che non appartiene a nessuna delle tre
        # materie, non si aggrappa a una parola qualsiasi pur di rispondere.
        if guessed:
            relaxed = self.index.search(
                question,
                top_k=config.top_k,
                subject=guessed,
                min_score=min_score * 1.5,
                min_terms=1,
            )
            if relaxed:
                return relaxed, guessed

        return [], guessed

    # -- risposta ------------------------------------------------------------

    def _fit_context(self, hits: list[Hit]) -> list[Hit]:
        """Tiene solo i passaggi che entrano davvero nel contesto del modello.

        La numerazione delle fonti deve coincidere con i blocchi ``[n]`` che il
        generatore vede: se il budget taglia il quinto passaggio, la quinta
        fonte non va mostrata come se fosse stata usata.
        """
        _, used = format_context(hits, self.config.generation.max_context_chars)
        return used

    def _request(self, question: str, hits: list[Hit], history: list[dict] | None) -> AnswerRequest:
        return AnswerRequest(
            question=question,
            hits=hits,
            history=history or [],
            max_context_chars=self.config.generation.max_context_chars,
            max_sentences=self.config.generation.max_sentences,
            min_sentence_overlap=self.config.generation.min_sentence_overlap,
            # L'estrattivo deve selezionare le frasi con gli stessi termini che
            # hanno portato agli hit; il modello continua invece a vedere la
            # domanda originale, senza sinonimi iniettati.
            match_query=self._search_question(question),
        )

    def ask(
        self,
        question: str,
        subject: str | None = None,
        history: list[dict] | None = None,
        use_cache: bool = True,
    ) -> Answer:
        question = (question or "").strip()
        if not question:
            return Answer(text="Scrivi una domanda per iniziare.", backend=self.backend.name)
        if len(question) > 800:
            question = question[:800]

        # La cronologia entra nel prompt: due domande identiche in conversazioni
        # diverse non sono la stessa richiesta e non possono condividere la cache.
        cacheable = use_cache and not history
        if cacheable:
            cached = self.cache.get(question, subject)
            if cached is not None:
                return cached

        started = time.perf_counter()
        hits, guessed = self.retrieve(question, subject)

        if not hits:
            answer = Answer(
                text=ABSTENTION_TEXT,
                backend=self.backend.name,
                abstained=True,
                subject=guessed,
            )
            answer.latency_ms = round((time.perf_counter() - started) * 1000, 1)
            if cacheable:
                self.cache.put(question, subject, answer)
            return answer

        hits = self._fit_context(hits)
        request = self._request(question, hits, history)
        degraded = False
        text = ""
        backend_name = self.backend.name
        try:
            text = self.backend.answer(request).strip()
        except BackendError:
            degraded = True
            from .backends import ExtractiveBackend

            fallback = ExtractiveBackend(
                max_sentences=self.config.generation.max_sentences,
                min_overlap=self.config.generation.min_sentence_overlap,
            )
            text = fallback.answer(request).strip()
            backend_name = fallback.name

        text = text or ABSTENTION_TEXT
        abstained = is_abstention_text(text)
        answer = Answer(
            text=text,
            sources=[] if abstained else sources_from_hits(hits),
            backend=backend_name,
            abstained=abstained,
            degraded=degraded,
            retrieved=len(hits),
            subject=guessed,
        )
        answer.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        if cacheable and not degraded:
            self.cache.put(question, subject, answer)
        return answer

    def stream(
        self,
        question: str,
        subject: str | None = None,
        history: list[dict] | None = None,
        use_cache: bool = True,
    ) -> Iterator[dict]:
        """Eventi incrementali per la risposta in tempo reale.

        Il primo evento sono sempre le fonti: lo studente vede subito su cosa si
        basera' la risposta, e se le fonti sono palesemente sbagliate puo'
        interrompere la lettura prima che il modello finisca di scrivere.
        """
        question = (question or "").strip()
        if len(question) > 800:
            question = question[:800]
        started = time.perf_counter()

        cacheable = use_cache and not history
        if cacheable:
            cached = self.cache.get(question, subject)
            if cached is not None:
                yield {"type": "meta", "sources": [s.to_dict() for s in cached.sources],
                       "backend": cached.backend, "cached": True, "abstained": cached.abstained,
                       "subject": cached.subject,
                       "subject_label": subject_label(cached.subject) if cached.subject else None}
                yield {"type": "token", "text": cached.text}
                yield {"type": "done", "latency_ms": cached.latency_ms,
                       "backend": cached.backend, "cached": True, "abstained": cached.abstained}
                return

        hits, guessed = self.retrieve(question, subject)
        if not hits:
            elapsed = round((time.perf_counter() - started) * 1000, 1)
            answer = Answer(
                text=ABSTENTION_TEXT,
                backend=self.backend.name,
                abstained=True,
                latency_ms=elapsed,
                subject=guessed,
            )
            # Anche un'astensione e' una risposta stabile: ripeterla non deve
            # rieseguire il retrieval, soprattutto per le domande fuori ambito.
            if cacheable:
                self.cache.put(question, subject, answer)
            yield {"type": "meta", "sources": [], "backend": self.backend.name,
                   "cached": False, "abstained": True, "subject": guessed,
                   "subject_label": subject_label(guessed) if guessed else None}
            yield {"type": "token", "text": ABSTENTION_TEXT}
            yield {"type": "done", "latency_ms": elapsed,
                   "backend": self.backend.name, "cached": False, "abstained": True}
            return

        hits = self._fit_context(hits)
        request = self._request(question, hits, history)
        sources = sources_from_hits(hits)
        yield {
            "type": "meta",
            "sources": [source.to_dict() for source in sources],
            "backend": self.backend.name,
            "cached": False,
            "abstained": False,
            "subject": guessed,
            "subject_label": subject_label(guessed) if guessed else None,
        }

        pieces: list[str] = []
        degraded = False
        backend_name = self.backend.name
        stream_fn = getattr(self.backend, "stream", None)
        try:
            if stream_fn is None:
                text = self.backend.answer(request)
                pieces.append(text)
                yield {"type": "token", "text": text}
            else:
                for piece in stream_fn(request):
                    if piece:
                        pieces.append(piece)
                        yield {"type": "token", "text": piece}
        except BackendError:
            degraded = True
            from .backends import ExtractiveBackend

            text = ExtractiveBackend(
                max_sentences=self.config.generation.max_sentences,
                min_overlap=self.config.generation.min_sentence_overlap,
            ).answer(request)
            pieces = [text]
            backend_name = "extractive"
            yield {"type": "degraded", "message": "Modello non disponibile: risposta estrattiva."}
            yield {"type": "token", "text": text}

        full_text = "".join(pieces).strip()
        if not full_text:
            full_text = ABSTENTION_TEXT
            pieces = [ABSTENTION_TEXT]
            yield {"type": "token", "text": ABSTENTION_TEXT}
        elapsed = round((time.perf_counter() - started) * 1000, 1)
        abstained = is_abstention_text(full_text)

        if cacheable and not degraded:
            self.cache.put(
                question,
                subject,
                Answer(
                    text=full_text,
                    sources=[] if abstained else sources,
                    backend=backend_name,
                    abstained=abstained,
                    retrieved=len(hits),
                    subject=guessed,
                ),
            )
        yield {"type": "done", "latency_ms": elapsed, "backend": backend_name,
               "cached": False, "abstained": abstained}

    # -- diagnostica ---------------------------------------------------------

    def health(self) -> dict:
        return {
            "status": "ok",
            "backend": self.backend.name,
            "chunks": len(self.index),
            "documents": self.index.metadata.get("documents"),
            "subjects": self.index.metadata.get("subjects", []),
            "built_at": self.index.metadata.get("built_at"),
            "cache_hit_rate": round(self.cache.hit_rate, 3),
            "notes": self.startup_notes,
        }
