#!/usr/bin/env python3
"""Costruisce il dataset di distillazione: il docente genera, lo studente impara.

    python scripts/distill_tutor.py --per-chunk 2 --out data/tutor/distill.jsonl
    python scripts/distill_tutor.py --list-models

Ogni esempio nasce da un passaggio reale delle schede: il docente riceve il
passaggio, non l'intero corpus, e deve scrivere domande la cui risposta sia
contenuta li'. Questo e' il punto che rende il dataset utile: le risposte sono
ancorate a una fonte verificabile, non alla memoria del docente. Se il docente
inventa un dato che non c'e' nel passaggio, il filtro lo scarta.

La chiave API si legge da ``OPENAI_API_KEY``. Non va mai scritta nel repository.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lligma.tutor.console import setup_utf8  # noqa: E402
from lligma.tutor.env import load_dotenv  # noqa: E402
from lligma.tutor.text import chunk_text, normalize  # noqa: E402
from build_tutor_index import collect  # noqa: E402

TEACHER_PROMPT = """Sei un docente di un istituto agrario italiano che prepara materiale
di studio. Ricevi un passaggio tratto dalle dispense del corso.

Scrivi {count} coppie domanda-risposta in italiano su quel passaggio.

Regole:
- La domanda deve essere quella che farebbe davvero uno studente di 16 anni, in
  linguaggio naturale, e deve potersi rispondere SOLO con il passaggio.
- La risposta deve essere autosufficiente: 2-4 frasi, in italiano corretto,
  con i termini tecnici usati nel passaggio.
- Non aggiungere informazioni che non siano nel passaggio. Se il passaggio non
  contiene abbastanza contenuto per {count} domande, restituisci meno coppie.
- Non inventare numeri, date o nomi assenti dal passaggio.
- Rispondi SOLO con un array JSON, senza testo attorno, in questa forma:
  [{{"question": "...", "answer": "..."}}]

PASSAGGIO:
{passage}"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/tutor")
    parser.add_argument("--out", default="data/tutor/distill.jsonl")
    parser.add_argument("--per-chunk", type=int, default=2)
    parser.add_argument("--limit-chunks", type=int, default=0)
    parser.add_argument("--subject", default=None)
    parser.add_argument("--model", default=os.environ.get("TUTOR_TEACHER_MODEL", "gpt-4o-mini"))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://freebuff.com/api/v1/paid"))
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--sleep", type=float, default=0.0, help="Pausa fra chiamate (secondi)")
    parser.add_argument("--dry-run", action="store_true", help="Mostra i prompt senza chiamare l'API")
    parser.add_argument("--list-models", action="store_true")
    return parser.parse_args()


def api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "OPENAI_API_KEY non impostata.\n"
            "  Windows PowerShell : $env:OPENAI_API_KEY = \"...\"\n"
            "  bash               : export OPENAI_API_KEY=...\n"
            "La chiave non deve essere scritta nel repository."
        )
    return key


def post_chat(base_url: str, key: str, payload: dict, timeout: float = 120.0) -> str:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:300]
        raise SystemExit(f"HTTP {error.code}: {detail}") from error
    return data["choices"][0]["message"]["content"]


def list_models(base_url: str, key: str) -> int:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.loads(response.read().decode("utf-8", "replace"))
    for item in payload.get("data", []):
        print(item.get("id"))
    return 0


def extract_pairs(raw: str) -> list[dict]:
    """Estrae l'array JSON anche se il docente lo avvolge in testo o in ```."""
    candidate = raw.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", candidate, re.S)
    if fence:
        candidate = fence.group(1).strip()
    start, end = candidate.find("["), candidate.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        payload = json.loads(candidate[start:end + 1])
    except json.JSONDecodeError:
        return []
    return [item for item in payload if isinstance(item, dict)]


def valid_pair(question: str, answer: str, passage: str) -> bool:
    """Filtro di qualita': scarta le coppie che non reggono.

    Il controllo sull'ancoraggio e' volutamente leggero, perche' parafrasare e'
    normale: si limita a pretendere che almeno due parole di contenuto della
    risposta compaiano nel passaggio. Non e' una prova di correttezza, e' un
    filtro contro le risposte che parlano d'altro.
    """
    if not 12 <= len(question) <= 200 or not question.rstrip().endswith("?"):
        return False
    if not 40 <= len(answer) <= 900:
        return False
    lowered = answer.lower()
    if "non ho" in lowered and "informazion" in lowered:
        return False
    answer_words = {word for word in normalize(answer).split() if len(word) > 4}
    passage_words = set(normalize(passage).split())
    if len(answer_words & passage_words) < 2:
        return False
    return True


def main() -> int:
    setup_utf8()
    load_dotenv()
    args = parse_args()
    base_url = args.base_url
    key = api_key()
    if args.list_models:
        return list_models(base_url, key)

    data_dir = ROOT / args.data_dir
    documents = collect(data_dir, "generale")
    if args.subject:
        documents = [document for document in documents if document.subject == args.subject]
    if not documents:
        raise SystemExit("nessuna scheda selezionata")

    jobs: list[tuple[object, str]] = []
    for document in documents:
        for piece in chunk_text(document.text, 110, 28):
            if len(piece.split()) >= 40:
                jobs.append((document, piece))
    if args.limit_chunks:
        jobs = jobs[: args.limit_chunks]

    print(f"Schede: {len(documents)} | passaggi da trasformare: {len(jobs)}")
    print(f"Modello docente: {args.model} via {base_url}")

    output = ROOT / args.out
    output.parent.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.add(normalize(json.loads(line)["question"]))
        print(f"Riprendo: {len(seen)} domande gia' presenti in {output.name}")

    written = skipped = failed = 0
    started = time.perf_counter()

    with output.open("a", encoding="utf-8") as handle:
        for number, (document, passage) in enumerate(jobs, start=1):
            prompt = TEACHER_PROMPT.format(count=args.per_chunk, passage=passage)
            if args.dry_run:
                print(f"\n--- {document.id} ---\n{prompt[:600]}...")
                continue

            payload = {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
            }
            try:
                raw = post_chat(base_url, key, payload)
            except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as error:
                failed += 1
                print(f"[{number}/{len(jobs)}] {document.id}: errore ({error})")
                continue

            pairs = extract_pairs(raw)
            if not pairs:
                failed += 1
                print(f"[{number}/{len(jobs)}] {document.id}: nessuna coppia JSON valida")
                continue

            for pair in pairs:
                question = str(pair.get("question", "")).strip()
                answer = str(pair.get("answer", "")).strip()
                key_question = normalize(question)
                if key_question in seen:
                    skipped += 1
                    continue
                if not valid_pair(question, answer, passage):
                    skipped += 1
                    continue
                seen.add(key_question)
                handle.write(
                    json.dumps(
                        {
                            "question": question,
                            "answer": answer,
                            "subject": document.subject,
                            "source_title": document.title,
                            "source_id": document.id,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                written += 1
            handle.flush()
            print(f"[{number}/{len(jobs)}] {document.id}: +{len(pairs) - skipped} coppie")
            if args.sleep:
                time.sleep(args.sleep)

    elapsed = time.perf_counter() - started
    print()
    print(f"Scritte    : {written}")
    print(f"Scartate   : {skipped}")
    print(f"Errori     : {failed}")
    print(f"Tempo      : {elapsed:.0f}s")
    print(f"File       : {output}")
    print("\nProssimo passo: python scripts/finetune_tutor.py --data data/tutor/distill.jsonl")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
