"""Configurazione di LLigMa Tutor.

Un solo file YAML descrive indice, retrieval, generazione e server. I segreti non
stanno mai nel YAML: ogni stringa supporta ``${VAR}`` o ``${VAR:default}`` ed e'
espansa da variabili d'ambiente al caricamento. Cosi' il repository puo' essere
versionato senza chiavi e una scuola puo' cambiare modello senza toccare il codice.

I percorsi relativi nel YAML sono risolti rispetto alla radice del progetto (la
cartella che contiene ``lligma/``), non rispetto alla cwd: lo stesso comando
funziona lanciato da ovunque.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .env import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z0-9_]+)(?::([^}]*))?\}")


def _expand(value: Any) -> Any:
    """Espande ${VAR} ovunque dentro la struttura YAML."""
    if isinstance(value, str):
        return _ENV_PATTERN.sub(
            lambda match: os.environ.get(match.group(1), match.group(2) or ""), value
        )
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def _resolve(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)


@dataclass
class IndexConfig:
    """Dove vive l'indice e come vengono tagliati i documenti."""

    path: Path = Path("index/tutor-index.json")
    chunk_words: int = 110
    overlap_words: int = 28


@dataclass
class ExpansionConfig:
    """Sinonimi applicati alla sola query di retrieval.

    Il default di questa dataclass resta spento: chi costruisce una
    ``TutorConfig`` a mano non deve ricevere un dizionario che non ha chiesto.
    La configurazione distribuita (``configs/tutor.yaml``) invece lo attiva,
    perche' con il dizionario spento il set di parafrasi misura 0/39 e una
    domanda in ambito riscritta con un sinonimo viene rifiutata. Le chiavi
    possono essere parole o frasi; i valori sono una stringa o una lista di
    stringhe. La domanda mostrata, la cache e il limite di 800 caratteri non
    vengono toccati dall'espansione.
    """

    enabled: bool = False
    synonyms: dict[str, str | list[str]] = field(default_factory=dict)


@dataclass
class RetrievalConfig:
    """Parametri di ricerca.

    ``min_score`` e' la soglia sotto la quale il tutor dichiara di non sapere:
    con BM25 e' un valore assoluto che dipende dalla dimensione del corpus, per
    questo lo script di valutazione riporta la distribuzione dei punteggi e il
    valore va ricalibrato se il corpus cresce molto.
    """

    top_k: int = 5
    min_score: float = 2.5
    title_boost: float = 1.5
    # Termini distinti che la domanda e il passaggio devono avere in comune.
    min_terms: int = 2
    expansion: ExpansionConfig = field(default_factory=ExpansionConfig)

    def __post_init__(self) -> None:
        # YAML consegna un dizionario: la sezione deve diventare un oggetto
        # tipizzato, senza obbligare ogni chiamante a costruirlo a mano.
        if isinstance(self.expansion, dict):
            self.expansion = ExpansionConfig(**self.expansion)


@dataclass
class GenerationConfig:
    backend: str = "auto"  # auto | extractive | openai | local
    max_context_chars: int = 3600
    max_sentences: int = 5
    min_sentence_overlap: int = 2
    history_turns: int = 3


@dataclass
class OpenAIConfig:
    """Backend "docente": un modello grande usato via API.

    Serve a due cose: generare il dataset di distillazione e fare da riferimento
    quando una risposta deve essere per forza corretta. Non e' il modello che
    gira a scuola.
    """

    base_url: str = "${OPENAI_BASE_URL:https://freebuff.com/api/v1/paid}"
    model: str = "${TUTOR_TEACHER_MODEL:gpt-4o-mini}"
    api_key_env: str = "OPENAI_API_KEY"
    temperature: float = 0.2
    max_tokens: int = 450
    timeout: float = 60.0

    @property
    def chat_url(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"


@dataclass
class LocalConfig:
    """Backend "studente": il modello piccolo addestrato con la distillazione."""

    model_path: Path = Path("outputs/tutor-ft")
    load_in_4bit: bool = True
    max_new_tokens: int = 320
    device: str = "auto"
    dtype: str = "bfloat16"
    temperature: float = 0.2


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    web_dir: Path = Path("web/tutor")
    # Se vuoto, il bind loopback accetta solo 127.0.0.1, localhost e ::1.
    # Un reverse proxy con un nome pubblico deve elencarlo qui.
    allowed_hosts: list[str] = field(default_factory=list)


@dataclass
class TutorConfig:
    index: IndexConfig = field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    local: LocalConfig = field(default_factory=LocalConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    project_root: Path = PROJECT_ROOT

    @classmethod
    def from_file(cls, path: str | Path) -> "TutorConfig":
        path = _resolve(path)
        # Prima di espandere ${VAR}: senza questo, la chiave scritta in .env non
        # arriverebbe mai al backend API e il servizio sembrerebbe non configurato.
        load_dotenv()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw = _expand(raw)

        def section(key: str, klass, **extra) -> Any:
            values = dict(raw.get(key) or {})
            values.update(extra)
            return klass(**values)

        config = cls(
            index=section("index", IndexConfig),
            retrieval=section("retrieval", RetrievalConfig),
            generation=section("generation", GenerationConfig),
            openai=section("openai", OpenAIConfig),
            local=section("local", LocalConfig),
            server=section("server", ServerConfig),
        )
        config.index.path = _resolve(config.index.path)
        config.local.model_path = _resolve(config.local.model_path)
        config.server.web_dir = _resolve(config.server.web_dir)
        return config

    def describe(self) -> str:
        return (
            f"indice={self.index.path.name} top_k={self.retrieval.top_k} "
            f"soglia={self.retrieval.min_score} backend={self.generation.backend} "
            f"modello_locale={self.local.model_path}"
        )
