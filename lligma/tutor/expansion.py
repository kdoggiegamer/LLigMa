"""Espansione controllata delle query con un dizionario di sinonimi.

Il dizionario non vive nel codice: arriva da ``configs/tutor.yaml``, dove e'
attivo per default. L'espansione tocca solo la stringa usata dal retriever:
la domanda mostrata, la chiave della cache e il limite di 800 caratteri restano
quelli originali. Le regole possono essere di una parola (``stallatico`` ->
``letame``) o di piu' parole (``corso d'acqua`` -> ``fiume``); le piu' lunghe
hanno la precedenza, cosi' una frase non viene spezzata da una regola piu'
corta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from .text import normalize

MAX_QUERY_CHARS = 800


class ExpansionError(ValueError):
    """Configurazione del dizionario non valida."""


def _target_words(value: str | Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        values: Sequence[str] = [value]
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        raise ExpansionError(
            "ogni sinonimo deve essere una stringa o una lista di stringhe"
        )

    words: list[str] = []
    for item in values:
        if not isinstance(item, str):
            raise ExpansionError("i sinonimi devono essere stringhe")
        normalized = normalize(item)
        if not normalized:
            raise ExpansionError("un sinonimo non puo' essere vuoto")
        words.extend(normalized.split())
    return tuple(words)


@dataclass(frozen=True)
class QueryExpander:
    """Applica un dizionario di sinonimi alla sola query di retrieval."""

    rules: Mapping[tuple[str, ...], tuple[str, ...]]
    max_phrase_words: int = 1

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, str | Sequence[str]] | None
    ) -> "QueryExpander":
        rules: dict[tuple[str, ...], tuple[str, ...]] = {}
        for raw_source, raw_targets in (mapping or {}).items():
            if not isinstance(raw_source, str):
                raise ExpansionError("le chiavi del dizionario devono essere stringhe")
            source = tuple(normalize(raw_source).split())
            if not source:
                raise ExpansionError("una chiave del dizionario non puo' essere vuota")
            targets = _target_words(raw_targets)
            if not targets:
                raise ExpansionError(f"sinonimo vuoto per {raw_source!r}")
            if source == targets:
                continue
            rules[source] = targets
        longest = max((len(source) for source in rules), default=1)
        return cls(rules=rules, max_phrase_words=longest)

    def expand(self, query: str, max_chars: int = MAX_QUERY_CHARS) -> str:
        """Sostituisce i sinonimi noti e restituisce la query espansa.

        Se non c'e' alcuna regola applicabile restituisce la query originale,
        non una sua normalizzazione: il comportamento senza espansione deve
        restare esattamente quello di prima. La lunghezza e' limitata anche
        quando il dizionario e' configurato dall'utente, cosi' il retriever non
        riceve mai una query oltre il limite del servizio.
        """
        words = normalize(query).split()
        if not words or not self.rules:
            return query

        output: list[str] = []
        index = 0
        changed = False
        while index < len(words):
            matched = False
            for size in range(min(self.max_phrase_words, len(words) - index), 0, -1):
                key = tuple(words[index : index + size])
                target = self.rules.get(key)
                if target is None:
                    continue
                output.extend(target)
                index += size
                changed = True
                matched = True
                break
            if not matched:
                output.append(words[index])
                index += 1

        if not changed:
            return query

        expanded = " ".join(output)
        if len(expanded) <= max_chars:
            return expanded
        # Taglia su un confine di parola: una query troncata a meta' parola
        # sarebbe un termine che nessun documento contiene.
        shortened = expanded[:max_chars].rsplit(" ", 1)[0]
        return shortened or query[:max_chars]