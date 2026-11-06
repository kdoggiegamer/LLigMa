"""Testo italiano: normalizzazione, stemming leggero, frasi e passaggi.

Niente dipendenze: un tokenizer vero (BPE) serve ad addestrare il modello, non a
costruire un indice BM25. Qui bastano poche regole, a patto che siano le regole
giuste per l'italiano.

La scelta discutibile e' lo stemming: invece di uno stemmer completo (Snowball)
si tolgono un suffisso e la vocale finale. In italiano la flessione e' quasi
tutta vocalica (suolo/suoli, foglia/foglie, mela/mele) e bastano poche regole
per allineare singolare, plurale e famiglia di parole. I suffissi sono divisi in
due classi con soglie diverse: i derivativi pretendono una radice di almeno tre
lettere, i flessionali anche di due, cosi' `arare` e `aratura` cadono sulla
stessa chiave senza che `unica` diventi `un`.
"""

from __future__ import annotations

import re
import unicodedata

# Parole funzionali: rumore per BM25 e per l'overlap delle frasi.
# "non", "piu'", "meno", "senza", "mai" restano: cambiano il significato.
_RAW_STOPWORDS = """
a ad al alla alle allo agli ai anche che chi ci come con cosa cui da dal dalla dalle
dallo dagli dai de del della delle dello degli dei di dove e ed era erano essere fa
fra gli ha hanno ho i il in io la le lei lo loro lui ma mi mia mie miei mio ne nei nel
nella nelle nello negli nell noi o ogni per puo quale quali quando quanto quel quella
quelle quelli quello questa queste questi questo qui se sei si sia siamo siete sono su
sua sue suo sui sul sulla sulle sullo te ti tra tu tua tue tuo un una uno vi voi
vostro vostra stato stata stati state essere viene vengono del dell alla alla
"""


def normalize(text: str) -> str:
    """Minuscolo, senza accenti, senza punteggiatura, spazi singoli."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    lowered = stripped.lower()
    cleaned = re.sub(r"[^0-9a-z]+", " ", lowered)
    return " ".join(cleaned.split())


STOPWORDS = frozenset(normalize(" ".join(_RAW_STOPWORDS.split())).split())

# Acronimi brevi che non vanno scartati solo perche' hanno meno di tre lettere.
_SHORT_KEEP = frozenset({"ph", "dna", "rna", "atp", "adp", "kg", "cm", "mm", "ue"})

_VOWELS = frozenset("aeiou")


# Suffissi dell'italiano, ciascuno con la lunghezza minima che deve restare
# alla radice. Servono per far coincidere parole della stessa famiglia che uno
# studente usa come sinonimi: pota/potare/potatura, concima/concimare/
# concimazione, fertile/fertilita'.
#
# Sono due classi, e la distinzione non e' un dettaglio:
#
# * **derivativi** (`azione`, `atura`, `mente`, `ico`, ...) trasformano una
#   parola in un'altra. Si tolgono solo se resta una radice di almeno tre
#   lettere: con due, `unica` diventerebbe `un`, che e' anche `unire`.
# * **flessionali** (`are`, `ere`, `ire`, `ano`, `ono`, `ando`, ...) sono la
#   stessa parola in un'altra forma, quindi la radice puo' essere di due
#   lettere. E' cio' che tiene insieme `arare` e `aratura`: senza, il verbo
#   restava `arar` mentre il nome diventava `ara`, e il retriever perdeva la
#   domanda piu' naturale che uno studente possa fare su una lavorazione.
#
# Ordine: dal suffisso piu' lungo al piu' corto, cosi' `atura` vince su `tura`
# e `avano` su `ano`.
# Nota su cosa NON c'e' in questa lista: `zione`/`zioni` e `ale`/`ali` sono
# stati tolti dopo aver visto `funzione` diventare `fun` e `integrale` diventare
# `integr`: accorciavano tanto da far combaciare parole che non c'entrano nulla.
_SUFFIXES: tuple[tuple[str, int], ...] = (
    # 6 lettere
    ("azione", 3), ("azioni", 3), ("amento", 3), ("amenti", 3),
    ("imento", 3), ("imenti", 3),
    # 5 lettere: i deverbali `atura`/`ature` e le forme verbali `avano` ecc.
    # stanno qui, prima dei suffissi piu' corti che li contengono (`tura`, `ano`).
    ("atura", 2), ("ature", 2),
    ("mente", 3), ("menti", 3), ("aggio", 3),
    ("avano", 3), ("evano", 3), ("ivano", 3),
    # 4 lettere
    ("aggi", 3), ("enza", 3), ("enze", 3), ("anza", 3), ("anze", 3),
    ("ismo", 3), ("ismi", 3), ("tura", 3), ("ture", 3), ("iche", 3),
    ("ando", 3), ("endo", 3), ("iamo", 3), ("iate", 3),
    # 3 lettere: derivativi, infiniti (radice di due lettere ammessa) e
    # flessionali del verbo.
    ("ato", 3), ("ata", 3), ("ati", 3), ("ate", 3),
    ("ito", 3), ("ita", 3), ("iti", 3), ("ite", 3),
    ("ico", 3), ("ica", 3), ("ici", 3),
    ("ivo", 3), ("iva", 3), ("ivi", 3), ("ive", 3),
    ("oso", 3), ("osa", 3), ("ana", 3), ("ane", 3), ("ani", 3),
    ("are", 2), ("ere", 2), ("ire", 2),
    ("ano", 3), ("ono", 3),
)


def stem(word: str) -> str:
    """Riduzione minima: via il suffisso (derivativo o flessionale) e la vocale.

    Non e' uno stemmer: e' una normalizzazione. Snowball sarebbe piu' preciso e
    richiederebbe una tabella di regole ordinate e un caso di studio per ogni
    eccezione; qui basta che "potatura", "potare" e "pota" cadano sulla stessa
    chiave, che "arare" e "aratura" facciano lo stesso, e che "vite" non
    diventi "vit" solo quando non serve.
    """
    if len(word) <= 3:
        return word
    for suffix, min_root in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= min_root:
            word = word[: -len(suffix)]
            break
    if len(word) >= 4 and word[-1] in _VOWELS:
        return word[:-1]
    return word


def stem_tokens(text: str) -> list[str]:
    """Token normalizzati e ridotti, senza parole funzionali.

    I duplicati sono voluti: chi chiama decide se contarne le occorrenze (BM25)
    oppure usarli come insieme (overlap di una frase).
    """
    tokens: list[str] = []
    for word in normalize(text).split():
        if word in _SHORT_KEEP:
            tokens.append(word)
            continue
        if len(word) < 3 or word in STOPWORDS:
            continue
        tokens.append(stem(word))
    return tokens


def content_words(text: str) -> set[str]:
    """Insieme di parole grezze: serve al confronto fuzzy della cache."""
    return {word for word in normalize(text).split() if len(word) > 2 and word not in STOPWORDS}


_ABBREVIATIONS = frozenset(
    {"ecc", "es", "n", "art", "cap", "sec", "cfr", "ed", "vol", "pag", "prof", "dott", "fig", "ca"}
)

# Fine frase seguita da una maiuscola (anche accentata) o da una cifra.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])[\"'\u00bb)\]]*\s+(?=[A-Z\u00c0-\u00dd0-9\u00ab\"(])")


def _ends_with_abbreviation(sentence: str) -> bool:
    last = normalize(sentence.rsplit(" ", 1)[-1]).strip(".")
    return last in _ABBREVIATIONS


def split_sentences(text: str) -> list[str]:
    """Segmentazione conservativa: meglio una frase lunga che due monche."""
    prepared = re.sub(r"\s+", " ", (text or "").replace("\r", " ")).strip()
    if not prepared:
        return []
    sentences: list[str] = []
    for piece in _SENTENCE_SPLIT.split(prepared):
        piece = piece.strip()
        if not piece:
            continue
        # "es. " non chiude una frase: riattacca il pezzo precedente.
        if sentences and _ends_with_abbreviation(sentences[-1]):
            sentences[-1] = f"{sentences[-1]} {piece}"
        else:
            sentences.append(piece)
    return sentences


def chunk_text(text: str, max_words: int = 110, overlap_words: int = 28) -> list[str]:
    """Passaggi da ~max_words parole, tagliati sui confini di frase.

    La sovrapposizione esiste perche' un fatto puo' cadere a cavallo di due
    passaggi: senza overlap la frase che contiene la risposta finirebbe divisa e
    ne' il primo ne' il secondo passaggio sarebbero citabili.
    """
    if max_words < 20:
        raise ValueError("max_words deve essere almeno 20")
    if overlap_words >= max_words:
        raise ValueError("overlap_words deve essere minore di max_words")

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0

    for sentence in sentences:
        words = sentence.split()
        if len(words) > max_words:
            # Frase singola troppo lunga: va spezzata a forza, non c'e' scelta.
            if current:
                chunks.append(" ".join(current))
                current, current_words = [], 0
            for start in range(0, len(words), max_words):
                chunks.append(" ".join(words[start:start + max_words]))
            continue

        if current and current_words + len(words) > max_words:
            chunks.append(" ".join(current))
            kept: list[str] = []
            kept_words = 0
            for previous in reversed(current):
                size = len(previous.split())
                if kept_words + size > overlap_words:
                    break
                kept.insert(0, previous)
                kept_words += size
            current, current_words = kept, kept_words

        current.append(sentence)
        current_words += len(words)

    if current:
        chunks.append(" ".join(current))

    return [chunk for chunk in chunks if chunk.strip()]
