"""I tre modi di generare una risposta, dietro la stessa interfaccia.

* ``ExtractiveBackend``  nessun modello: sceglie e ordina frasi gia' presenti nei
  passaggi recuperati. E' il comportamento di riferimento: se il tutor risponde
  bene anche cosi', il modello piccolo sta aggiungendo valore; se risponde meglio
  del modello, il modello sta solo peggiorando il testo.
* ``OpenAIBackend``      il "docente" via API. Non gira a scuola: serve a
  costruire il dataset e come metro di paragone.
* ``LocalHFBackend``     lo "studente" distillato, caricato in locale.

La regola comune: qualunque backend riceve solo i passaggi recuperati e non puo'
attingere ad altro. La differenza fra i tre e' la qualita' della prosa, non
l'origine dei fatti.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Iterator, Protocol

from .config import LocalConfig, OpenAIConfig, TutorConfig
from .index import Hit
from .prompt import ABSTENTION_TEXT, build_messages, format_context
from .text import normalize, split_sentences, stem_tokens

# Domande che chiedono una data: la risposta deve contenere un anno, anche se la
# frase che lo contiene non condivide parole con la domanda.
_DATE_WORDS = frozenset({"anno", "anni", "quando", "data", "date", "secolo", "secoli"})
_YEAR = re.compile(r"\b(1\d{3}|20\d{2})\b")


class BackendError(RuntimeError):
    """Errore recuperabile: il service passa al backend estrattivo."""


@dataclass
class AnswerRequest:
    question: str
    hits: list[Hit]
    history: list[dict] = field(default_factory=list)
    max_context_chars: int = 3600
    max_sentences: int = 5
    min_sentence_overlap: int = 2
    # Query usata per scegliere le frasi nel backend estrattivo. Puo' contenere
    # i sinonimi espansi, mentre ``question`` resta quella mostrata al modello e
    # alla cache. Vuota = usa la domanda originale.
    match_query: str = ""

    @property
    def context(self) -> str:
        return format_context(self.hits, self.max_context_chars)[0]


class Backend(Protocol):
    name: str

    def available(self) -> bool: ...

    def answer(self, request: AnswerRequest) -> str: ...


class ExtractiveBackend:
    """Risponde incollando le frasi piu' pertinenti dei passaggi recuperati.

    Zero parametri, latenza nell'ordine del millisecondo e, soprattutto, zero
    possibilita' di allucinazione: ogni parola della risposta esiste gia' nella
    fonte e viene citata. Non e' un modello linguistico, ed e' esattamente questo
    il punto: e' il pavimento di qualita' sotto il quale il modello piccolo non
    deve mai scendere.
    """

    name = "extractive"

    def __init__(self, max_sentences: int = 5, min_overlap: int = 2):
        self.max_sentences = max_sentences
        self.min_overlap = max(1, min_overlap)

    def available(self) -> bool:
        return True

    def answer(self, request: AnswerRequest) -> str:
        query_terms = set(stem_tokens(request.match_query or request.question))
        if not query_terms or not request.hits:
            return ABSTENTION_TEXT
        # Con una domanda di una sola parola ("humus", "fotosintesi") pretendere
        # due termini in comune con la frase renderebbe il caso irrisolvibile.
        required = min(self.min_overlap, len(query_terms))
        budget = max(1, request.max_sentences)
        # Una frase oltre il budget: completa il concetto senza trasformare la
        # risposta in un riassunto dell'intero passaggio.
        limit = budget + 1
        date_question = bool(_DATE_WORDS & set(normalize(request.question).split()))
        top_score = max(hit.score for hit in request.hits)
        per_hit: list[tuple[int, list[str]]] = []
        for rank, hit in enumerate(request.hits):
            sentences = [
                sentence for sentence in split_sentences(hit.text)
                if len(sentence.split()) >= 4
            ]
            per_hit.append((rank, sentences))
        sentences_by_rank = dict(per_hit)

        # La pertinenza l'ha gia' decisa il retrieval: qui si scelgono le frasi
        # migliori e si completa il contesto con le adiacenti. Una domanda che
        # chiede un anno non puo' ricevere una risposta senza anni: in quel caso
        # entra anche una frase senza parole in comune, presa pero' solo dai
        # passaggi che il retriever ha giudicato quasi altrettanto pertinenti.
        #
        # La stessa frase compare spesso in due chunk consecutivi (overlap). La
        # teniamo una volta sola nella risposta, ma ricordiamo tutte le sue
        # occorrenze: la continuazione del discorso puo' stare nell'occorrenza
        # successiva, non in quella scelta per il punteggio.
        occurrences: dict[str, list[tuple[int, int]]] = {}
        for rank, sentences in per_hit:
            for position, sentence in enumerate(sentences):
                occurrences.setdefault(sentence, []).append((rank, position))

        ranked: list[tuple[float, int, int, int, str]] = []
        seen: set[str] = set()
        for rank, sentences in per_hit:
            date_eligible = date_question and request.hits[rank].score >= 0.5 * top_score
            for position, sentence in enumerate(sentences):
                if sentence in seen:
                    continue
                overlap = len(query_terms & set(stem_tokens(sentence)))
                has_date = date_eligible and bool(_YEAR.search(sentence))
                if overlap < 1 and not has_date:
                    continue
                seen.add(sentence)
                # ``required`` non e' piu' una soglia rigida: una frase con una
                # sola parola in comune puo' essere l'unico ponte verso la
                # continuazione. A parita' di punteggio e nello stesso passaggio
                # vince pero' quella che raggiunge la sovrapposizione minima.
                # Sulle domande di data il tie-break non si applica: la frase
                # con l'anno puo' avere zero parole in comune e deve comunque
                # restare davanti a una frase generica dello stesso punteggio.
                score = (
                    overlap
                    + (0.5 if rank == 0 else 0.0)
                    + (2.0 if has_date else 0.0)
                )
                tie_break = 0 if (not date_question and overlap >= required) else 1
                ranked.append((-score, rank, tie_break, position, sentence))
        if not ranked:
            return ABSTENTION_TEXT

        ranked.sort()
        # Le frasi si scelgono per punteggio, poi si riordinano per posizione,
        # perche' la risposta deve leggersi in ordine.
        chosen: dict[str, tuple[int, int]] = {}
        for _, rank, _, position, sentence in ranked[: max(1, budget - 1)]:
            chosen.setdefault(sentence, (rank, position))

        def add(sentence: str, rank: int, position: int) -> bool:
            if len(chosen) >= limit or sentence in chosen:
                return False
            chosen[sentence] = (rank, position)
            return True

        # Prima la continuazione di confine: la frase scelta chiude un chunk ed
        # e' anche la prima del chunk successivo. La frase seguente di quel
        # chunk e' il naturale completamento (DNA -> trascrizione -> traduzione),
        # e va preferita ai vicini generici del primo chunk.
        for sentence, (rank, _) in list(chosen.items()):
            if len(chosen) >= limit:
                break
            for other_rank, other_position in occurrences.get(sentence, []):
                if other_rank != rank and other_position == 0:
                    neighbours = sentences_by_rank.get(other_rank, [])
                    if len(neighbours) > 1:
                        add(neighbours[1], other_rank, 1)

        # Poi la frase precedente e quella successiva, partendo dai passaggi
        # meno rappresentati: la definizione sta spesso prima, la conseguenza
        # dopo, e il passaggio appena entrato va completato.
        for sentence, (rank, position) in sorted(
            chosen.items(),
            key=lambda item: (
                sum(1 for other, _ in chosen.values() if other == item[1][0]),
                item[1],
            ),
        ):
            if len(chosen) >= limit:
                break
            for other_rank, other_position in occurrences.get(sentence, []):
                sentences = sentences_by_rank.get(other_rank, [])
                for neighbour in (other_position - 1, other_position + 1):
                    if len(chosen) >= limit:
                        break
                    if 0 <= neighbour < len(sentences):
                        add(sentences[neighbour], other_rank, neighbour)

        ordered = sorted(chosen.items(), key=lambda item: item[1])
        return " ".join(
            f"{sentence.rstrip()} [{rank + 1}]" for sentence, (rank, _) in ordered
        )

    def stream(self, request: AnswerRequest) -> Iterator[str]:
        yield self.answer(request)


class OpenAIBackend:
    """Client minimale per un endpoint OpenAI-compatibile (chat completions).

    Usa solo ``urllib``: aggiungere un SDK per due chiamate HTTP non si giustifica
    e ogni dipendenza in piu' e' una cosa da aggiornare in una scuola.
    """

    name = "openai"

    def __init__(self, config: OpenAIConfig):
        self.config = config
        self.api_key = os.environ.get(config.api_key_env, "").strip()

    def available(self) -> bool:
        return bool(self.api_key and self.config.base_url and self.config.model)

    def _payload(self, request: AnswerRequest, stream: bool) -> dict:
        return {
            "model": self.config.model,
            "messages": build_messages(
                request.question,
                request.hits,
                history=request.history,
                max_context_chars=request.max_context_chars,
            ),
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": stream,
        }

    def _open(self, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        http_request = urllib.request.Request(
            self.config.chat_url,
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream" if payload.get("stream") else "application/json",
            },
            method="POST",
        )
        try:
            return urllib.request.urlopen(http_request, timeout=self.config.timeout)
        except urllib.error.HTTPError as error:
            detail = ""
            try:
                detail = error.read().decode("utf-8", "replace")[:300]
            except Exception:  # pragma: no cover - solo diagnostica
                pass
            raise BackendError(f"HTTP {error.code} da {self.config.chat_url}: {detail or error.reason}")
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            raise BackendError(f"API non raggiungibile ({self.config.chat_url}): {error}")

    def answer(self, request: AnswerRequest) -> str:
        if not self.available():
            raise BackendError(
                f"variabile {self.config.api_key_env} assente: nessuna chiave per il backend API"
            )
        with self._open(self._payload(request, False)) as response:
            raw = response.read().decode("utf-8", "replace")
        try:
            payload = json.loads(raw)
            text = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise BackendError(f"risposta API inattesa: {raw[:300]!r}") from error
        text = (text or "").strip()
        if not text:
            raise BackendError("l'API ha restituito una risposta vuota")
        return text

    def stream(self, request: AnswerRequest) -> Iterator[str]:
        if not self.available():
            raise BackendError(f"variabile {self.config.api_key_env} assente")
        with self._open(self._payload(request, True)) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", "replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                chunk = line[5:].strip()
                if chunk == "[DONE]":
                    break
                try:
                    payload = json.loads(chunk)
                except json.JSONDecodeError:
                    continue
                choices = payload.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if piece:
                    yield piece


class LocalHFBackend:
    """Il modello studente in locale, opzionalmente in 4 bit.

    Gli import sono dentro il costruttore perche' ``transformers`` e ``peft`` non
    devono essere obbligatori: il tutor gira anche su una macchina che ha solo
    l'API o solo il backend estrattivo.

    4 bit (bitsandbytes, NF4) dimezza ulteriormente un modello gia' piccolo: un
    0,6B passa da ~1,2 GB a ~0,4 GB di pesi, che e' la differenza fra dover
    avere una GPU e poter girare su un portatile del laboratorio.
    """

    name = "local"

    def __init__(self, config: LocalConfig):
        self.config = config
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:  # pragma: no cover - dipende dall'ambiente
            raise BackendError(
                "backend locale non disponibile: installa torch e transformers "
                "(vedi docs/TUTOR.md)"
            ) from error

        self.torch = torch
        path = str(config.model_path)
        if not os.path.isdir(path):
            raise BackendError(f"modello locale non trovato in {path}")

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(config.dtype, torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(path)
        kwargs: dict = {}
        if config.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_use_double_quant=True,
                )
                kwargs["device_map"] = "auto"
            except ImportError:
                kwargs["torch_dtype"] = dtype
        else:
            kwargs["torch_dtype"] = dtype

        self.model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
        if "device_map" not in kwargs:
            device = "cuda" if (config.device != "cpu" and torch.cuda.is_available()) else "cpu"
            self.model.to(device)
        self.model.eval()

    def available(self) -> bool:
        return True

    def _generate(self, request: AnswerRequest, stream: bool):
        messages = build_messages(
            request.question,
            request.hits,
            history=request.history,
            max_context_chars=request.max_context_chars,
        )
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
        except TypeError:
            # Template senza il flag "thinking": alcuni modelli piccoli lo ignorano.
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        generation = dict(
            max_new_tokens=self.config.max_new_tokens,
            do_sample=self.config.temperature > 0,
            temperature=max(self.config.temperature, 1e-5),
            top_p=0.9,
            repetition_penalty=1.05,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if not stream:
            with self.torch.no_grad():
                output = self.model.generate(**inputs, **generation)
            new_tokens = output[0][inputs["input_ids"].shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        from transformers import TextIteratorStreamer
        from threading import Thread

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = Thread(
            target=self.model.generate,
            kwargs={**inputs, **generation, "streamer": streamer},
            daemon=True,
        )
        thread.start()
        return streamer

    def answer(self, request: AnswerRequest) -> str:
        text = self._generate(request, stream=False)
        if not text:
            raise BackendError("il modello locale non ha prodotto testo")
        return text

    def stream(self, request: AnswerRequest) -> Iterator[str]:
        streamer = self._generate(request, stream=True)
        for piece in streamer:
            if piece:
                yield piece


def make_backend(
    config: TutorConfig, requested: str | None = None
) -> tuple[Backend, list[str]]:
    """Sceglie il backend e restituisce anche le note di degradazione.

    Ordine di ``auto``: studente locale (privacy e offline, che per una scuola
    contano), poi docente via API, infine estrattivo. Ogni passo verso il basso
    viene registrato in ``notes`` cosi' l'interfaccia puo' dire allo studente
    perche' sta ricevendo una risposta diversa dal solito.
    """
    choice = (requested or config.generation.backend or "auto").strip().lower()
    notes: list[str] = []
    extractive = ExtractiveBackend(
        max_sentences=config.generation.max_sentences,
        min_overlap=config.generation.min_sentence_overlap,
    )

    if choice == "extractive":
        return extractive, notes

    candidates: list[str] = []
    if choice == "auto":
        candidates = ["local", "openai"]
    elif choice in {"local", "openai"}:
        candidates = [choice]
    else:
        notes.append(f"backend '{choice}' sconosciuto: uso l'estrattivo")
        return extractive, notes

    for name in candidates:
        try:
            if name == "local":
                return LocalHFBackend(config.local), notes
            if name == "openai":
                backend = OpenAIBackend(config.openai)
                if backend.available():
                    return backend, notes
                notes.append(
                    f"backend API non configurato ({config.openai.api_key_env} assente)"
                )
        except Exception as error:  # noqa: BLE001 - il factory deve degradare, non crashare
            # Un backend locale puo' fallire con OSError/RuntimeError (pesi
            # corrotti, driver CUDA): per la scuola e' comunque "non
            # disponibile", non un errore da mostrare allo studente.
            notes.append(f"{name}: {error}")

    notes.append("risposta generata dal backend estrattivo")
    return extractive, notes
