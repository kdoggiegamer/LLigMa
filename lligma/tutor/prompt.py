"""Contratto di risposta e costruzione del prompt.

Tutto il comportamento "da scuola" del tutor vive in questo file: il livello
linguistico, l'obbligo di citare, il divieto di inventare e la regola di astensione.
E' volutamente un testo unico e leggibile: se una scuola vuole cambiare il tono
dell'assistente, cambia queste righe e nient'altro.
"""

from __future__ import annotations

from .index import Hit
from .schema import Source
from .text import normalize

SUBJECT_LABELS = {
    "agronomia": "Agronomia",
    "biologia": "Biologia",
    "cultura": "Cultura generale",
}

SUBJECT_KEYWORDS = {
    "agronomia": """
        suolo terreno aratura rotazione concime concimazione fertilizzante letame
        irrigazione vigneto vite potatura innesto olivo ulivo frumento grano mais
        erba medica foraggio pascolo trattore semina raccolto potassio fosforo
        azoto nitrati humus argilla sabbia ph salinita drenaggio siccita
        fitofarmaco fungicida diserbo parassita afide peronospora oidio ruggine
        bovino vacca latte caseificio zootecnia razione stalla apicoltura
    """.split(),
    "biologia": """
        cellula cellulare dna rna proteina enzima mitosi meiosi cromosoma gene
        genetica mendel alleli mutazione fotosintesi clorofilla cloroplasto
        respirazione mitocondrio atp membrana osmosi batterio virus lievito
        apparato organo tessuto sangue neurone ormone auxina giberellina
        ecosistema catena alimentare biodiversita evoluzione darwin specie
        linneo tassonomia
    """.split(),
    "cultura": """
        dante manzoni leopard petrarca boccaccio divina commedia promessi sposi
        risorgimento garibaldi cavour costituzione parlamento senato deputati
        governo presidente repubblica regione provincia comune unione europea
        trattato guerra storia rinascimento medioevo roma latino letteratura
        poesia romanzo geografia alpi appennini po sicilia sardegna unesco
        paesaggio agrario
    """.split(),
}

ABSTENTION_TEXT = (
    "Non ho trovato informazioni sufficienti nel materiale disponibile per "
    "rispondere con sicurezza. Posso provare con una domanda piu' specifica, "
    "oppure puoi verificare sul libro di testo o chiedere al tuo insegnante."
)

SYSTEM_PROMPT = """Sei LLigMa Tutor, l'assistente didattico di un istituto agrario italiano.
Parli con studenti delle superiori (14-19 anni), nei laboratori e in classe.

Regole, in ordine di importanza:
1. Rispondi SOLO con informazioni contenute nel CONTESTO fornito. Non usare
   conoscenze esterne, nemmeno se le ritieni corrette.
2. Se il contesto non basta a rispondere, dillo chiaramente e suggerisci come
   cercare meglio. Non riempire i vuoti con ipotesi plausibili.
3. Cita ogni affermazione con il numero della fonte fra parentesi quadre, per
   esempio [1]. Se una frase unisce due fonti, cita entrambe: [1][2].
4. Non inventare mai numeri, dosi, date, nomi propri o riferimenti normativi che
   non compaiono nel contesto. Per i valori agronomici (dosi, profondita',
   epoche) riporta esattamente quelli scritti.
5. Livello linguistico: italiano chiaro e corretto, frasi brevi. I termini
   tecnici si usano, ma vanno spiegati la prima volta.
6. Lunghezza: al massimo 5 frasi, salvo che lo studente chieda un
   approfondimento o un elenco.
7. Le domande fuori dai tre ambiti del tutor (agronomia, biologia, cultura
   generale italiana) vanno dichiarate fuori ambito, senza tentare risposte.
8. Non sei un medico ne' un consulente aziendale: per decisioni su trattamenti
   fitosanitari, salute o soldi rimanda all'insegnante o al tecnico."""

OUT_OF_SCOPE_TEXT = (
    "Questa domanda esce dai tre ambiti che seguo: agronomia, biologia e cultura "
    "generale italiana. Posso aiutarti su uno di questi argomenti."
)


def is_abstention_text(text: str) -> bool:
    """Vero se il testo e' un rifiuto, fuori ambito o vuoto.

    Un backend che restituisce stringa vuota non ha risposto: trattarlo come
    astensione evita di mostrare fonti per un contenuto che non esiste.
    """
    stripped = (text or "").strip()
    if not stripped:
        return True
    return stripped.startswith(ABSTENTION_TEXT[:40]) or stripped == OUT_OF_SCOPE_TEXT


def subject_label(key: str | None) -> str:
    return SUBJECT_LABELS.get((key or "").strip().lower(), (key or "Generale").title())


def guess_subject(question: str, weights: dict[str, float] | None = None) -> str | None:
    """Materia probabile della domanda, dal lessico della domanda stessa.

    Non e' un classificatore: e' un suggerimento usato per restringere la ricerca
    quando i segnali sono netti. Se due materie si contendono la domanda la
    funzione restituisce ``None`` e si cerca su tutto l'indice, che e' il
    comportamento giusto per le domande di confine (il suolo e' agronomia ma
    anche biologia).
    """
    # Normalizzare gli accenti e' essenziale: "biodiversità" e "biodiversita"
    # devono produrre gli stessi token, altrimenti le parole chiave scritte
    # senza accento non trovano mai la domanda.
    tokens = set(normalize(question).split())
    scores = {
        subject: len(tokens & set(keywords))
        for subject, keywords in SUBJECT_KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return None
    tied = [subject for subject, score in scores.items() if score == scores[best]]
    if len(tied) > 1:
        return None
    return best


def format_context(hits: list[Hit], max_chars: int = 3600) -> tuple[str, list[Hit]]:
    """Blocchi numerati, tagliati al budget di caratteri.

    Un passaggio tagliato a meta' frase e' peggio di un passaggio in meno: il
    budget si consuma solo per intero, e appena il prossimo non ci sta si smette.
    """
    blocks: list[str] = []
    used: list[Hit] = []
    consumed = 0
    for position, hit in enumerate(hits, start=1):
        header = f"[{position}] {subject_label(hit.chunk.subject)} - {hit.title}"
        if hit.chunk.source:
            header += f" (fonte: {hit.chunk.source})"
        block = f"{header}\n{hit.text}"
        if consumed + len(block) > max_chars and used:
            break
        blocks.append(block)
        used.append(hit)
        consumed += len(block) + 2
    return "\n\n".join(blocks), used


def build_messages(
    question: str,
    hits: list[Hit],
    history: list[dict] | None = None,
    max_context_chars: int = 3600,
    history_turns: int = 3,
) -> list[dict]:
    """Messaggi in formato OpenAI/chat-template, con la domanda per ultima."""
    context, _ = format_context(hits, max_context_chars)
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in (history or [])[-2 * history_turns:]:
        role = turn.get("role")
        content = (turn.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            messages.append({"role": role, "content": content[:800]})

    messages.append(
        {
            "role": "user",
            "content": (
                f"CONTESTO:\n{context}\n\n"
                f"DOMANDA DELLO STUDENTE: {question.strip()}\n\n"
                "Rispondi in italiano, usando solo il contesto e citando le fonti con [n]."
            ),
        }
    )
    return messages


def sources_from_hits(hits: list[Hit], quote_chars: int = 220) -> list[Source]:
    """Fonti mostrate in interfaccia, una voce per scheda.

    Piu' passaggi della stessa scheda non devono produrre piu' voci: gli hit
    arrivano ordinati per punteggio, quindi si tiene la prima citazione (la
    migliore) e si accumulano in ``numbers`` tutti i riferimenti ``[n]`` a quella
    scheda. Cosi' l'elenco resta corto senza perdere il legame con il testo.
    """
    sources: list[Source] = []
    by_document: dict[str, Source] = {}
    for position, hit in enumerate(hits, start=1):
        key = hit.chunk.document_id or hit.chunk.title
        existing = by_document.get(key)
        if existing is not None:
            existing.numbers.append(position)
            continue
        text = hit.chunk.text.strip()
        quote = text if len(text) <= quote_chars else text[:quote_chars].rsplit(" ", 1)[0] + "..."
        source = Source(
            title=hit.chunk.title,
            subject=subject_label(hit.chunk.subject),
            source=hit.chunk.source,
            quote=quote,
            score=hit.score,
            document_id=hit.chunk.document_id,
            numbers=[position],
        )
        by_document[key] = source
        sources.append(source)
    return sources
