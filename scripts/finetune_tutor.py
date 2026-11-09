#!/usr/bin/env python3
"""Fine-tuning QLoRA dello studente sul dataset di distillazione.

    python scripts/finetune_tutor.py \
        --model Qwen/Qwen3-0.6B \
        --data data/tutor/distill.jsonl \
        --output-dir outputs/tutor-ft \
        --epochs 2 --batch-size 4 --gradient-accumulation-steps 4

Perche' QLoRA e non un training completo:

* il modello base resta in 4 bit (NF4), quindi un 0,6B occupa ~0,4 GB e sta in
  una GPU da 8 GB insieme a ottimizzatore e attivazioni;
* l'adapter LoRA allena lo 0,5-2% dei pesi: si addestra in ore, non in giorni,
  e si puo' rigenerare ogni anno quando cambiano le dispense;
* l'adapter e' separato dai pesi, quindi si possono tenere piu' adapter (uno per
  materia) sullo stesso modello base, cambiando pochi MB invece di GB.

Il prompt di sistema e' lo stesso che il tutor usa in inferenza: se i due
divergono, il modello impara un compito che a scuola non gli verra' chiesto.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lligma.tutor.prompt import SYSTEM_PROMPT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B",
                        help="Modello base piccolo. Alternative: HuggingFaceTB/SmolLM2-360M-Instruct, "
                             "ibm-granite/granite-4.0-nano, LiquidAI/LFM2-350M")
    parser.add_argument("--data", default="data/tutor/distill.jsonl")
    parser.add_argument("--eval-data", default=None, help="File JSONL di validazione (default: 8% del training)")
    parser.add_argument("--output-dir", default="outputs/tutor-ft")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-every", type=int, default=200)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--no-4bit", action="store_true", help="Solo per GPU che non supportano bitsandbytes")
    parser.add_argument("--merge", action="store_true", help="Salva anche il modello fuso in 16 bit")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            question = str(payload.get("question", "")).strip()
            answer = str(payload.get("answer", "")).strip()
            if question and answer:
                records.append({"question": question, "answer": answer})
            else:
                print(f"  riga {number} scartata: campi vuoti")
    return records


def apply_template(tokenizer, messages: list[dict], add_generation_prompt: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )
    except TypeError:
        # Template senza il flag thinking (modelli piu' vecchi o non-Qwen).
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt
        )


def encode_example(tokenizer, record: dict, max_length: int) -> dict:
    """Maschera i token del prompt: la loss si calcola solo sulla risposta.

    Senza maschera il modello impara a riprodurre anche la domanda e il sistema,
    sprecando capacita' su token che non deve generare.
    """
    prompt_text = apply_template(
        tokenizer,
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": record["question"]}],
        add_generation_prompt=True,
    )
    eos = tokenizer.eos_token or ""
    full_text = prompt_text + record["answer"] + eos

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids):]

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
        labels = labels[:max_length]

    return {"input_ids": full_ids, "labels": labels}


class ChatDataset:
    def __init__(self, tokenizer, records, max_length):
        self.examples = []
        dropped = 0
        for record in records:
            encoded = encode_example(tokenizer, record, max_length)
            # Meno di 8 token di risposta significa che la troncatura ha mangiato
            # tutto: l'esempio non insegna nulla.
            if len(encoded["labels"]) - encoded["labels"].count(-100) < 8:
                dropped += 1
                continue
            self.examples.append(encoded)
        if dropped:
            print(f"  {dropped} esempi scartati (risposta troncata o troppo corta)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate(batch, pad_token_id: int):
    import torch

    width = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), width), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), width), -100, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.long)
    for row, item in enumerate(batch):
        length = len(item["input_ids"])
        input_ids[row, :length] = torch.tensor(item["input_ids"], dtype=torch.long)
        labels[row, :length] = torch.tensor(item["labels"], dtype=torch.long)
        attention[row, :length] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}


def main() -> int:
    args = parse_args()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as error:
        raise SystemExit(
            f"Dipendenze mancanti: {error}.\n"
            "  pip install torch transformers peft bitsandbytes datasets accelerate"
        ) from error

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data_path = ROOT / args.data
    if not data_path.is_file():
        raise SystemExit(f"dataset non trovato: {data_path} (crealo con scripts/distill_tutor.py)")
    records = load_records(data_path)
    if len(records) < 20:
        raise SystemExit(f"servono almeno 20 esempi, trovati {len(records)}")

    if args.eval_data:
        eval_records = load_records(ROOT / args.eval_data)
    else:
        random.shuffle(records)
        split = max(1, int(0.08 * len(records)))
        eval_records, records = records[:split], records[split:]

    print(f"Training: {len(records)} esempi | validazione: {len(eval_records)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {"device_map": {"": 0} if args.device == "cuda" else "cpu"}
    if args.device == "cuda":
        load_kwargs["torch_dtype"] = torch.bfloat16
        if not args.no_4bit:
            from transformers import BitsAndBytesConfig

            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            load_kwargs.pop("device_map")

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    if args.device == "cuda" and not args.no_4bit:
        model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False

    available = {name.split(".")[-1] for name, _ in model.named_modules()}
    targets = [name for name in
               ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
               if name in available]
    if not targets:
        raise SystemExit(f"nessun modulo LoRA riconosciuto in {args.model}: trovati {sorted(available)[:20]}")

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        ),
    )
    model.print_trainable_parameters()

    train_data = ChatDataset(tokenizer, records, args.max_length)
    eval_data = ChatDataset(tokenizer, eval_records, args.max_length)
    if not len(train_data):
        raise SystemExit("nessun esempio addestrabile: alza --max-length")

    batch_size = min(args.batch_size, len(train_data))
    steps_per_epoch = max(1, len(train_data) // (batch_size * args.gradient_accumulation_steps))
    total_steps = args.max_steps if args.max_steps > 0 else max(1, int(steps_per_epoch * args.epochs))
    warmup = max(1, int(total_steps * args.warmup_ratio))
    print(f"Step previsti: {total_steps} ({steps_per_epoch}/epoca, warmup {warmup})")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.learning_rate
    )
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    autocast_dtype = torch.bfloat16 if device.type == "cuda" else None

    def evaluate() -> float:
        if not len(eval_data):
            return float("nan")
        model.eval()
        total, batches = 0.0, 0
        with torch.no_grad():
            for start in range(0, len(eval_data), batch_size):
                batch = collate([eval_data[i] for i in range(start, min(start + batch_size, len(eval_data)))],
                                tokenizer.pad_token_id)
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                    enabled=autocast_dtype is not None):
                    total += model(**batch).loss.item()
                batches += 1
        model.train()
        return total / max(batches, 1)

    def save(step: int) -> None:
        model.save_pretrained(output_dir / "adapter")
        tokenizer.save_pretrained(output_dir / "adapter")
        (output_dir / "training_info.json").write_text(
            json.dumps(
                {
                    "base_model": args.model,
                    "step": step,
                    "examples": len(train_data),
                    "lora_r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "max_length": args.max_length,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"adapter salvato (step {step}) in {output_dir / 'adapter'}")

    model.train()
    step = 0
    tokens_seen = 0
    started = time.perf_counter()
    order = list(range(len(train_data)))

    while step < total_steps:
        random.shuffle(order)
        for start in range(0, len(order) - batch_size + 1, batch_size):
            batch = collate([train_data[i] for i in order[start:start + batch_size]],
                            tokenizer.pad_token_id)
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.autocast(device_type=device.type, dtype=autocast_dtype,
                                enabled=autocast_dtype is not None):
                loss = model(**batch).loss / args.gradient_accumulation_steps
            loss.backward()
            tokens_seen += int((batch["labels"] != -100).sum())

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                progress = min(1.0, step / warmup) if step < warmup else 0.5 * (
                    1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup))
                )
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate * progress
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            step += 1
            if step % 10 == 0:
                elapsed = time.perf_counter() - started
                print(f"step {step}/{total_steps} loss={loss.item() * args.gradient_accumulation_steps:.4f} "
                      f"lr={optimizer.param_groups[0]['lr']:.2e} "
                      f"tok/s={tokens_seen / max(elapsed, 1e-9):.0f}", flush=True)
            if args.save_every and step % args.save_every == 0:
                save(step)
            if step >= total_steps:
                break

    save(step)
    eval_loss = evaluate()
    if not math.isnan(eval_loss):
        print(f"loss di validazione: {eval_loss:.4f} (perplexity {math.exp(min(eval_loss, 20)):.2f})")

    if args.merge:
        print("Fusione dell'adapter nei pesi...")
        merged = model.merge_and_unload()
        merged.save_pretrained(output_dir / "merged", safe_serialization=True)
        tokenizer.save_pretrained(output_dir / "merged")
        print(f"Modello fuso salvato in {output_dir / 'merged'}")

    print()
    print("Fatto. Per servirlo:")
    print(f"  imposta local.model_path: {output_dir / 'merged' if args.merge else output_dir / 'adapter'}")
    print("  in configs/tutor.yaml e avvia: python -m lligma.tutor.server")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
