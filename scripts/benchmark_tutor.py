#!/usr/bin/env python3
"""Misura la latenza reale del tutor, in processo oppure via HTTP.

    python scripts/benchmark_tutor.py --repeat 2
    python scripts/benchmark_tutor.py --url http://127.0.0.1:8080

Le metriche che contano per uno studente sono due: quanto aspetta prima di
vedere le prime parole (TTFT) e quanto aspetta per la risposta completa. La
latenza media da sola non basta: se una risposta su cinque e' lentissima, la
classe intera se ne accorge.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lligma.tutor.config import TutorConfig  # noqa: E402
from lligma.tutor.console import setup_utf8  # noqa: E402
from lligma.tutor.service import TutorService  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/tutor.yaml")
    parser.add_argument("--questions", default="data/tutor/eval.jsonl")
    parser.add_argument("--backend", default=None)
    parser.add_argument("--repeat", type=int, default=1, help="Quante volte ripetere ogni domanda")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--url", default=None, help="Misura un server gia' avviato invece del processo locale")
    parser.add_argument("--no-cache", action="store_true", help="Disattiva la cache (misura il caso peggiore)")
    return parser.parse_args()


def percentiles(values: list[float]) -> dict:
    ordered = sorted(values)
    def at(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]
    return {
        "mean": statistics.mean(ordered),
        "p50": statistics.median(ordered),
        "p95": at(0.95),
        "max": ordered[-1],
    }


def load_questions(path: Path, limit: int) -> list[dict]:
    items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    items = [item for item in items if not item.get("should_abstain")]
    return items[:limit] if limit else items


def bench_http(url: str, questions: list[dict], repeat: int) -> list[float]:
    latencies: list[float] = []
    for _ in range(repeat):
        for item in questions:
            body = json.dumps({"question": item["question"], "subject": item.get("subject")}).encode()
            request = urllib.request.Request(
                url.rstrip("/") + "/api/ask", data=body,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            started = time.perf_counter()
            with urllib.request.urlopen(request, timeout=120) as response:
                response.read()
            latencies.append((time.perf_counter() - started) * 1000)
    return latencies


def main() -> int:
    setup_utf8()
    args = parse_args()
    questions = load_questions(ROOT / args.questions, args.limit)
    if not questions:
        raise SystemExit("nessuna domanda da misurare")

    if args.url:
        latencies = bench_http(args.url, questions, args.repeat)
        print(f"Modalita' HTTP: {args.url}")
        backend = "http"
        caches = None
    else:
        config = TutorConfig.from_file(args.config)
        if args.backend:
            config.generation.backend = args.backend
        service = TutorService(config)
        backend = service.backend.name

        # Warm-up: la prima richiesta paga l'import, l'allocazione dei buffer e
        # la lettura del file dell'indice. Non e' la latenza che vede lo studente.
        service.ask(questions[0]["question"], use_cache=False)

        latencies = []
        for _ in range(args.repeat):
            for item in questions:
                answer = service.ask(item["question"], subject=item.get("subject"),
                                     use_cache=not args.no_cache)
                latencies.append(answer.latency_ms)
        caches = service.cache

    stats = percentiles(latencies)
    print()
    print(f"Backend        : {backend}")
    print(f"Richieste      : {len(latencies)} su {len(questions)} domande x {args.repeat}")
    print(f"Latenza media  : {stats['mean']:.1f} ms")
    print(f"p50            : {stats['p50']:.1f} ms")
    print(f"p95            : {stats['p95']:.1f} ms")
    print(f"massima        : {stats['max']:.1f} ms")
    if caches is not None:
        print(f"Cache hit rate : {caches.hit_rate:.0%}")
    budget = 800
    print(f"\nSotto la soglia di {budget} ms: "
          f"{100 * sum(1 for value in latencies if value <= budget) / len(latencies):.0f}% delle risposte")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
