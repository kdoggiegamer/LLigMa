"""Testo italiano: normalizzazione, stemming leggero, frasi e passaggi.

Niente dipendenze: un tokenizer vero (BPE) serve ad addestrare il modello, non a
costruire un indice BM25. Qui bastano poche regole, a patto che siano le regole
giuste per l'italiano.

La scelta discutibile e' lo stemming: invece di uno stemmer completo (Snowball)
si toglie solo la vocale finale delle parole lunghe. In italiano la flessione e'
quasi tutta vocalica (suolo/suoli, foglia/foglie, mela/mele) e questo trucco da
tre righe allinea singolare e plurale nella grande maggioranza dei casi, senza
accorciare parole che non vanno accorciate.
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


# Suffissi derivativi dell'italiano. Servono per far coincidere parole della
# stessa famiglia che uno studente usa come sinonimi: pota/potare/potatura,
# concima/concimare/concimazione, fertile/fertilita'. Senza questo passo il
# retriever perde domande del tutto legittime solo perche' il testo usa il nome
# e lo studente il verbo. Sono ordinati dal piu' lungo: "azione" prima di "zione".
# Nota su cosa NON c'e' in questa lista: "zione"/"zioni" e "ale"/"ali" sono
# stati tolti dopo aver visto "funzione" diventare "fun" e "integrale" diventare
# "integr": accorciavano tanto da far combaciare parole che non c'entrano nulla.
# Un suffisso va tolto solo se il resto resta riconoscibile.
_SUFFIXES = (
    "azione", "azioni", "amento", "amenti", "imento", "imenti", "atura", "ature",
    "mente", "menti", "aggio", "aggi", "enza", "enze",
    "anza", "anze", "ismo", "ismi", "tura", "ture", "iche", "are", "ere",
    "ire", "ato", "ata", "ati", "ate", "ito", "ita", "iti", "ite",
    "ico", "ica", "ici", "ivo", "iva", "ivi", "ive", "oso", "osa",
)


def stem(word: str) -> str:
    """Riduzione minima: via il suffisso derivativo e la vocale finale.

    Non e' uno stemmer: e' una normalizzazione. Snowball sarebbe piu' preciso e
    richiederebbe una tabella di regole ordinate e un caso di studio per ogni
    eccezione; qui basta che "potatura", "potare" e "pota" cadano sulla stessa
    chiave, e che "vite" non diventi "vit" solo quando non serve.
    """
    if len(word) <= 3:
        return word
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
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
