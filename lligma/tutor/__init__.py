"""LLigMa Tutor: assistente didattico italiano per istituti agrari.

Il pacchetto e' diviso in cinque strati, in ordine di dipendenza:

* ``text``      normalizzazione, stemming leggero e segmentazione in frasi;
* ``index``     indice BM25 su passaggi (nessuna dipendenza esterna);
* ``prompt``    contratto di risposta e costruzione del prompt groundato;
* ``backends``  tre modi di generare: estrattivo, API del docente, modello locale;
* ``service``   orchestratore: retrieval -> guardia -> generazione -> citazioni.

Il punto architetturale e' che la conoscenza non sta nei pesi. Un modello da
0,3-0,6 miliardi di parametri non puo' memorizzare agronomia, biologia e cultura
italiana in modo affidabile, ma puo' scrivere bene una risposta a partire da
passaggi recuperati. Togliere i fatti dal modello e' cio' che permette di
tenerlo piccolo e veloce senza renderlo inutile.
"""

from .config import (
    GenerationConfig,
    IndexConfig,
    LocalConfig,
    OpenAIConfig,
    RetrievalConfig,
    ServerConfig,
    TutorConfig,
)
from .index import BM25Index, Chunk, Document, Hit
from .schema import Answer, Source
from .service import TutorService

__version__ = "1.0.0"

__all__ = [
    "Answer",
    "BM25Index",
    "Chunk",
    "Document",
    "GenerationConfig",
    "Hit",
    "IndexConfig",
    "LocalConfig",
    "OpenAIConfig",
    "RetrievalConfig",
    "ServerConfig",
    "Source",
    "TutorConfig",
    "TutorService",
]
