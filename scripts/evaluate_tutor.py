#!/usr/bin/env python3
"""Misura il tutor su un set di domande italiane.

    python scripts/evaluate_tutor.py
    python scripts/evaluate_tutor.py --scores      # distribuzione dei punteggi BM25
    python scripts/evaluate_tutor.py --eval data/tutor/eval-paraphrase.jsonl --expand
    python scripts/evaluate_tutor.py --compare     # prima/dopo su entrambi i set

Tre metriche, scelte perche' misurano cose diverse e servono tutte:

* **recall@3** e' il passaggio giusto fra i primi tre? Se no, nessun modello
  puo' rispondere bene: e' il tetto del sistema.
* **copertura** e' quante delle parole chiave attese compaiono nella risposta.
  Grezza, ma onesta: misura se il contenuto minimo c'e'.
* **astensione** e' il tutor tace quando deve tacere? E' la metrica piu'
  importante in una scuola, ed e' l'unica che un modello grande non migliora.

``--scores`` stampa i punteggi del primo risultato sulle domande in-dominio e
fuori-dominio: e' cosi' che si calibra ``retrieval.min_score`` invece di tirarlo
a indovinare.

``--compare`` misura in una sola esecuzione il set originale e il set di
parafrasi, ciascuno con l'espansione spenta e accesa. Il "prima" del set di
parafrasi e' un fallimento atteso e non fa fallire il comando; il comando esce
in errore solo se una configurazione che deve passare non passa.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lligma.tutor.config import TutorConfig  # noqa: E402
from lligma.tutor.console import setup_utf8  # noqa: E402
from lligma.tutor.service import TutorService  # noqa: E402
from lligma.tutor.text import normalize  # noqa: E402

DEFAULT_EVAL = "data/tutor/eval.jsonl"
DEFAULT_PARAPHRASE_EVAL = "data/tutor/eval-paraphrase.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/tutor.yaml")
    parser.add_argument("--eval", default=DEFAULT_EVAL)
    parser.add_argument(
        "--paraphrase-eval",
        default=DEFAULT_PARAPHRASE_EVAL,
        help="set di parafrasi usato da --compare",
    )
    parser.add_argument(
        "--backend", default=None, help="auto | extractive | openai | local"
    )
    parser.add_argument(
        "--expand",
        action="store_true",
        help="attiva il dizionario di sinonimi per misurare il prima/dopo",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="misura set originale e parafrasi, con espansione spenta e accesa",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--scores", action="store_true", help="stampa i punteggi BM25")
    parser.add_argument("--json", dest="json_out", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.compare and args.expand:
        parser.error("--compare misura gia' entrambe le configurazioni: non usare --expand")
    if args.compare and args.scores:
        parser.error("--scores descrive una sola configurazione: non usarlo con --compare")
    return args


def load_eval(path: Path) -> list[dict]:
    items = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def resolved(path: str, root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def build_service(config_path: str, backend: str | None, expanded: bool) -> TutorService:
    config = TutorConfig.from_file(config_path)
    if backend:
        config.generation.backend = backend
    config.retrieval.expansion.enabled = expanded
    return TutorService(config)


def p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def measure(service: TutorService, items: list[dict], verbose: bool = False) -> dict:
    retrieval_hits = retrieval_total = 0
    top1_hits = 0
    coverage_scores: list[float] = []
    abstention_ok = abstention_total = 0
    unexpected_abstentions = 0
    latencies: list[float] = []
    failures: list[str] = []
    in_domain_scores: list[float] = []
    out_domain_scores: list[float] = []

    for item in items:
        question = item["question"]
        hits, _ = service.retrieve(question, item.get("subject"))
        answer = service.ask(question, subject=item.get("subject"), use_cache=False)
        latencies.append(answer.latency_ms)

        top_score = service.index.search(question, top_k=1, subject=item.get("subject") or None)
        if top_score:
            (out_domain_scores if item.get("should_abstain") else in_domain_scores).append(
                top_score[0].score
            )

        if item.get("expect_doc"):
            retrieval_total += 1
            found = {source.document_id for source in answer.sources} | {
                hit.chunk.document_id for hit in hits
            }
            if item["expect_doc"] in found:
                retrieval_hits += 1
            if hits and hits[0].chunk.document_id == item["expect_doc"]:
                top1_hits += 1
            elif verbose:
                failures.append(
                    f"retrieval: {question} -> atteso {item['expect_doc']}, "
                    f"trovato {sorted(found)}"
                )

        if item.get("should_abstain"):
            abstention_total += 1
            if answer.abstained:
                abstention_ok += 1
            else:
                failures.append(f"avrebbe dovuto astenersi: {question}")
        else:
            if answer.abstained:
                unexpected_abstentions += 1
                failures.append(f"astensione inattesa: {question}")
            keywords = item.get("must_include") or []
            if keywords:
                text = normalize(answer.text)
                found = sum(1 for keyword in keywords if normalize(keyword) in text)
                coverage = found / len(keywords)
                coverage_scores.append(coverage)
                if coverage < 0.5:
                    failures.append(
                        f"copertura {coverage:.0%}: {question} (mancano: "
                        f"{[k for k in keywords if normalize(k) not in text]})"
                    )

        if verbose:
            print(f"Q: {question}")
            print(f"   -> {answer.text[:180]}")
            print(f"   fonti: {[s.title for s in answer.sources]} | {answer.latency_ms} ms")
            print()

    return {
        "questions": len(items),
        "recall_hits": retrieval_hits,
        "retrieval_total": retrieval_total,
        "recall_at_3": retrieval_hits / retrieval_total if retrieval_total else None,
        "top1_hits": top1_hits,
        "coverage_mean": statistics.mean(coverage_scores) if coverage_scores else None,
        "coverage_min": min(coverage_scores) if coverage_scores else None,
        "coverage_max": max(coverage_scores) if coverage_scores else None,
        "abstention_ok": abstention_ok,
        "abstention_total": abstention_total,
        "abstention_accuracy": abstention_ok / abstention_total if abstention_total else None,
        "unexpected_abstentions": unexpected_abstentions,
        "latency_mean_ms": statistics.mean(latencies) if latencies else None,
        "latency_p50_ms": statistics.median(latencies) if latencies else None,
        "latency_p95_ms": p95(latencies),
        "failures": failures,
        "in_domain_scores": in_domain_scores,
        "out_domain_scores": out_domain_scores,
    }


def fraction(numerator: int, denominator: int) -> str:
    return f"{numerator}/{denominator}" if denominator else "n/d"


def percent(numerator: int, denominator: int) -> str:
    return f"{100 * numerator / denominator:.1f}%" if denominator else "n/d"


def coverage(value: float | None) -> str:
    return f"{value:.2f}" if value is not None else "n/d"


def latency(value: float | None) -> str:
    return f"{value:.1f} ms" if value is not None else "n/d"


def print_single(service: TutorService, items: list[dict], args: argparse.Namespace, result: dict) -> None:
    print(f"Backend: {service.backend.name} | passaggi: {len(service.index)} | domande: {len(items)}")
    print()
    print("=" * 68)
    print(
        f"recall@3        {result['recall_hits']:3d}/{result['retrieval_total']:<3d}  "
        f"{percent(result['recall_hits'], result['retrieval_total']):>6}"
    )
    print(
        f"copertura media {result['coverage_mean'] if result['coverage_mean'] is not None else 0:.2f}  "
        f"(min {result['coverage_min'] if result['coverage_min'] is not None else 0:.2f}, "
        f"max {result['coverage_max'] if result['coverage_max'] is not None else 0:.2f})"
    )
    print(
        f"astensione      {result['abstention_ok']:3d}/{result['abstention_total']:<3d}  "
        f"{percent(result['abstention_ok'], result['abstention_total']):>6}"
    )
    if result["latency_mean_ms"] is not None:
        print(
            f"latenza         media {result['latency_mean_ms']:7.1f} ms | "
            f"p50 {result['latency_p50_ms']:7.1f} ms | p95 {result['latency_p95_ms']:7.1f} ms"
        )
    print("=" * 68)

    if result["failures"]:
        print(f"\nCriticita' ({len(result['failures'])}):")
        for failure in result["failures"]:
            print(f"  - {failure}")

    if args.scores:
        print("\nPunteggi BM25 del primo risultato (per calibrare min_score):")
        for label, values in (
            ("in dominio", result["in_domain_scores"]),
            ("fuori dominio", result["out_domain_scores"]),
        ):
            if values:
                print(
                    f"  {label:14s} n={len(values):3d} min={min(values):6.2f} "
                    f"mediana={statistics.median(values):6.2f} max={max(values):6.2f}"
                )


def comparison_row(name: str, configuration: str, result: dict) -> str:
    return (
        f"{name:28} {configuration:10} "
        f"{fraction(result['recall_hits'], result['retrieval_total']):>8} "
        f"{fraction(result['top1_hits'], result['retrieval_total']):>8} "
        f"{coverage(result['coverage_mean']):>9} "
        f"{fraction(result['abstention_ok'], result['abstention_total']):>10} "
        f"{latency(result['latency_p95_ms']):>9}"
    )


def print_comparison(
    service: TutorService,
    eval_name: str,
    paraphrase_name: str,
    baseline_eval: dict,
    expanded_eval: dict,
    baseline_paraphrase: dict,
    expanded_paraphrase: dict,
) -> bool:
    print(
        f"Backend: {service.backend.name} | passaggi: {len(service.index)} | "
        "confronto prima/dopo, due set"
    )
    print()
    print(
        f"{'Set':28} {'Espansione':10} {'recall@3':>8} {'top@1':>8} "
        f"{'copertura':>9} {'astensione':>10} {'p95':>9}"
    )
    print("-" * 88)
    for name, configuration, result in (
        (eval_name, "spenta", baseline_eval),
        (eval_name, "accesa", expanded_eval),
        (paraphrase_name, "spenta", baseline_paraphrase),
        (paraphrase_name, "accesa", expanded_paraphrase),
    ):
        print(comparison_row(name, configuration, result))
    print("-" * 88)
    print(
        f"Il 'prima' del set parafrasi e' il fallimento atteso: "
        f"{baseline_paraphrase['recall_hits']}/{baseline_paraphrase['retrieval_total']} "
        f"recall@3, {baseline_paraphrase['unexpected_abstentions']} astensioni inattese. "
        "Non e' una regressione."
    )

    blocking = []
    for label, result in (
        (f"{eval_name} a espansione spenta", baseline_eval),
        (f"{eval_name} a espansione accesa", expanded_eval),
        (f"{paraphrase_name} a espansione accesa", expanded_paraphrase),
    ):
        if result["failures"]:
            blocking.append((label, result["failures"]))

    if blocking:
        print()
        for label, failures in blocking:
            print(f"Criticita' ({label}, {len(failures)}):")
            for failure in failures:
                print(f"  - {failure}")
        print("\nEsito: FAIL")
        return False

    print("\nEsito: PASS")
    return True


def write_json(path: str, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nRisultati salvati in {output}")


def comparison_payload(
    service: TutorService,
    eval_path: Path,
    paraphrase_path: Path,
    baseline_eval: dict,
    expanded_eval: dict,
    baseline_paraphrase: dict,
    expanded_paraphrase: dict,
    ok: bool,
) -> dict:
    return {
        "backend": service.backend.name,
        "passages": len(service.index),
        "comparison_ok": ok,
        "sets": [
            {
                "name": eval_path.name,
                "path": str(eval_path),
                "baseline": baseline_eval,
                "expanded": expanded_eval,
            },
            {
                "name": paraphrase_path.name,
                "path": str(paraphrase_path),
                "baseline": baseline_paraphrase,
                "expanded": expanded_paraphrase,
            },
        ],
    }


def main() -> int:
    setup_utf8()
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    eval_path = resolved(args.eval, root)
    paraphrase_path = resolved(args.paraphrase_eval, root)
    items = load_eval(eval_path)
    if args.limit:
        items = items[: args.limit]

    if args.compare:
        paraphrase_items = load_eval(paraphrase_path)
        if args.limit:
            paraphrase_items = paraphrase_items[: args.limit]
        baseline_service = build_service(args.config, args.backend, expanded=False)
        expanded_service = build_service(args.config, args.backend, expanded=True)
        baseline_eval = measure(baseline_service, items, args.verbose)
        baseline_paraphrase = measure(baseline_service, paraphrase_items, args.verbose)
        expanded_eval = measure(expanded_service, items, args.verbose)
        expanded_paraphrase = measure(expanded_service, paraphrase_items, args.verbose)
        ok = print_comparison(
            expanded_service,
            eval_path.name,
            paraphrase_path.name,
            baseline_eval,
            expanded_eval,
            baseline_paraphrase,
            expanded_paraphrase,
        )
        if args.json_out:
            write_json(
                args.json_out,
                comparison_payload(
                    expanded_service,
                    eval_path,
                    paraphrase_path,
                    baseline_eval,
                    expanded_eval,
                    baseline_paraphrase,
                    expanded_paraphrase,
                    ok,
                ),
            )
        return 0 if ok else 1

    service = build_service(args.config, args.backend, expanded=args.expand)
    result = measure(service, items, args.verbose)
    print_single(service, items, args, result)

    if args.json_out:
        payload = {
            "backend": service.backend.name,
            "recall_at_3": result["recall_at_3"],
            "coverage_mean": result["coverage_mean"],
            "abstention_accuracy": result["abstention_accuracy"],
            "latency_mean_ms": result["latency_mean_ms"],
            "questions": len(items),
            "failures": result["failures"],
        }
        write_json(args.json_out, payload)

    return 0 if not result["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
