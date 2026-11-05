"""Strutture condivise fra service, server e interfaccia web."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Source:
    """Una scheda citabile mostrata allo studente sotto la risposta.

    Una scheda puo' aver fornito piu' passaggi: in quel caso compare una volta
    sola, con la citazione del passaggio migliore, e ``numbers`` elenca i
    riferimenti ``[n]`` che il testo usa per quella scheda.
    """

    title: str
    subject: str
    source: str
    quote: str
    score: float = 0.0
    document_id: str = ""
    numbers: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Answer:
    """Risposta completa di metadati.

    ``abstained`` e' un campo di prima classe, non un caso di errore: in una
    scuola una risposta "non lo so" e' preferibile a una risposta inventata, e
    l'interfaccia la mostra in modo diverso.
    """

    text: str
    sources: list[Source] = field(default_factory=list)
    backend: str = "extractive"
    abstained: bool = False
    cached: bool = False
    degraded: bool = False
    latency_ms: float = 0.0
    retrieved: int = 0
    subject: str | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["sources"] = [source.to_dict() for source in self.sources]
        return payload
