"""Fine-tune LLigMa on Italian extractive QA (SQuAD-it).

The task: given a question and a paragraph, point at the answer inside the
paragraph. The output is a span, never generated text, so a correct answer is
guaranteed to be something the source actually says.

The model stays the pretrained LLigMa decoder; only a 2-logit head is new. That
is deliberate: the same weights keep working as a language model, and the
checkpoint you already have is the starting point, not a throwaway.

Layout is `question \\n context`, not the other way round, because the decoder
mask is causal: at an answer position the model must already have read the
question. Same reason the eval windows slide over the context only.
"""

import argparse
import json
import math
import re
import string
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lligma.model.config import LLigMaConfig
from lligma.model.qa import LLigMaForQuestionAnswering


def parse_args():
    parser = argparse.ArgumentParser(description="Extractive QA fine-tuning for LLigMa")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Directory saved by train.py (config.json + model.safetensors)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="squad_it")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--doc-stride", type=int, default=128)
    parser.add_argument("--max-answer-tokens", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-examples", type=int, default=400)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


# --- data -------------------------------------------------------------------

def load_base_checkpoint(path: Path):
    """Rebuild the QA model on the trained backbone."""
    values = json.loads((path / "config.json").read_text(encoding="utf-8"))
    for key in ("architectures", "dtype", "transformers_version"):
        values.pop(key, None)
    config = LLigMaConfig(**values)
    config.use_cache = False
    model = LLigMaForQuestionAnswering(config)
    state = load_file(str(path / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    backbone = [key for key in state if key.startswith("model.")]
    print(f"loaded {len(backbone)} backbone tensors; new: {sorted(missing)}; "
          f"ignored: {sorted(set(unexpected) - {'lm_head.weight'})}")
    return config, model


def build_training_arrays(tokenizer, dataset, args):
    """One window per example, placed so the answer is inside it.

    A window centred on the answer is legitimate here: at training time the span
    is known, and the alternative is throwing away every long paragraph. At eval
    the windows are answer-agnostic (see `context_windows`).
    """
    sep = tokenizer("\n", add_special_tokens=False)["input_ids"]
    ids_out, starts, ends = [], [], []
    dropped = {"answer_missing": 0, "question_too_long": 0, "answer_too_long": 0}
    raw = dataset
    if args.limit_train:
        raw = raw.select(range(min(args.limit_train, len(raw))))
    for example in raw:
        question_ids = tokenizer(example["question"], add_special_tokens=False)["input_ids"]
        encoded = tokenizer(example["context"], add_special_tokens=False,
                            return_offsets_mapping=True)
        context_ids, offsets = encoded["input_ids"], encoded["offset_mapping"]
        budget = args.max_length - len(question_ids) - len(sep)
        if budget < 64:
            dropped["question_too_long"] += 1
            continue
        answer = example["answers"]
        if not answer["text"]:
            dropped["answer_missing"] += 1
            continue
        char_start = answer["answer_start"][0]
        char_end = char_start + len(answer["text"][0])
        tok_start = next((i for i, (s, e) in enumerate(offsets) if e > char_start), None)
        tok_end = max((i for i, (s, e) in enumerate(offsets) if s < char_end), default=None)
        if tok_start is None or tok_end is None or tok_end < tok_start:
            dropped["answer_missing"] += 1
            continue
        if tok_end - tok_start + 1 > args.max_answer_tokens:
            dropped["answer_too_long"] += 1
            continue
        if len(context_ids) > budget:
            shift = min(max(0, tok_start - budget // 3), len(context_ids) - budget)
            context_ids = context_ids[shift:shift + budget]
            tok_start -= shift
            tok_end -= shift
            if tok_start < 0 or tok_end >= len(context_ids):
                dropped["answer_missing"] += 1
                continue
        ids = question_ids + sep + context_ids
        base = len(question_ids) + len(sep)
        ids_out.append(np.asarray(ids, dtype=np.int32))
        starts.append(base + tok_start)
        ends.append(base + tok_end)
    print(f"training windows: {len(ids_out):,} | dropped {dropped}")
    return ids_out, starts, ends


class SpanDataset(Dataset):
    def __init__(self, ids, starts, ends):
        self.ids, self.starts, self.ends = ids, starts, ends

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        return self.ids[index], self.starts[index], self.ends[index]


def collate(batch, pad_id):
    width = max(len(item[0]) for item in batch)
    input_ids = np.full((len(batch), width), pad_id, dtype=np.int64)
    attention = np.zeros((len(batch), width), dtype=np.int64)
    starts = np.zeros(len(batch), dtype=np.int64)
    ends = np.zeros(len(batch), dtype=np.int64)
    for row, (ids, start, end) in enumerate(batch):
        input_ids[row, :len(ids)] = ids
        attention[row, :len(ids)] = 1
        starts[row] = start
        ends[row] = end
    return (torch.from_numpy(input_ids), torch.from_numpy(attention),
            torch.from_numpy(starts), torch.from_numpy(ends))


# --- evaluation -------------------------------------------------------------

def normalise(text):
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the|il|lo|la|i|gli|le|un|uno|una|l|d|dell|della)\b", " ", text)
    return " ".join(text.split())


def em_f1(prediction, gold):
    predicted, expected = normalise(prediction), normalise(gold)
    if predicted == expected:
        return 1.0, 1.0
    predicted_tokens, expected_tokens = predicted.split(), expected.split()
    common = {}
    for token in predicted_tokens:
        common[token] = common.get(token, 0) + 1
    overlap = 0
    for token in expected_tokens:
        if common.get(token, 0) > 0:
            common[token] -= 1
            overlap += 1
    if not overlap:
        return 0.0, 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(expected_tokens)
    return 0.0, 2 * precision * recall / (precision + recall)


def context_windows(tokenizer, question, context, max_length, stride):
    """Every answer-agnostic window over the context, with its start offset."""
    sep = tokenizer("\n", add_special_tokens=False)["input_ids"]
    question_ids = tokenizer(question, add_special_tokens=False)["input_ids"]
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    budget = max_length - len(question_ids) - len(sep)
    if budget < 64:
        return []
    windows = []
    position = 0
    prefix = len(question_ids) + len(sep)
    while True:
        chunk = context_ids[position:position + budget]
        # prefix = tokens before the context, needed to turn a span index inside
        # the window into a global context token index.
        windows.append((question_ids + sep + chunk, position, len(chunk), prefix))
        if position + budget >= len(context_ids):
            break
        position += stride
    return windows


@torch.no_grad()
def evaluate(model, tokenizer, dataset, args, device, limit):
    model.eval()
    exact = f1_total = 0.0
    counted = 0
    for example in dataset.select(range(min(limit, len(dataset)))):
        best = (-float("inf"), None)
        for ids, offset, length, prefix in context_windows(
                tokenizer, example["question"], example["context"],
                args.max_length, args.doc_stride):
            input_ids = torch.tensor([ids], device=device)
            attention = torch.ones_like(input_ids)
            start, end, score = model.best_span(input_ids, attention,
                                                args.max_answer_tokens)
            if score > best[0]:
                if start < prefix or end >= prefix + length:
                    continue
                best = (score, (offset + start - prefix, offset + end - prefix))
        if best[1] is None:
            continue
        # Decode the chosen span by re-tokenising the context: the window may
        # start mid-paragraph, so a global token index is the only stable handle.
        context_ids = tokenizer(example["context"], add_special_tokens=False)["input_ids"]
        start, end = best[1]
        prediction = tokenizer.decode(context_ids[start:end + 1], skip_special_tokens=True)
        gold = example["answers"]["text"][0]
        exact_score, f1_score = em_f1(prediction, gold)
        exact += exact_score
        f1_total += f1_score
        counted += 1
    model.train()
    if not counted:
        return 0.0, 0.0
    return exact / counted, f1_total / counted


# --- training ---------------------------------------------------------------

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    config, model = load_base_checkpoint(args.checkpoint)
    model = model.to(device)
    model.train()

    from transformers import AutoTokenizer
    from datasets import load_dataset
    tokenizer = AutoTokenizer.from_pretrained(str(args.checkpoint), use_fast=True)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    splits = load_dataset(args.dataset)
    train_split = splits["train"]
    valid_split = splits["validation"] if "validation" in splits else splits["test"]
    ids, starts, ends = build_training_arrays(tokenizer, train_split, args)
    loader = DataLoader(SpanDataset(ids, starts, ends), batch_size=args.batch_size,
                        shuffle=True, drop_last=True, collate_fn=lambda b: collate(b, pad_id),
                        num_workers=0)
    print(f"batches per epoch: {len(loader):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None

    def save(step):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"step": step, "model": model.state_dict(),
                    "optimizer": optimizer.state_dict()},
                   args.output_dir / f"qa-checkpoint-{step:06d}.pt")
        if step and step % 1000 == 0:
            model.model.save_pretrained(args.output_dir / f"step-{step:06d}")
        print(f"saved qa-checkpoint-{step:06d}.pt")

    step = 0
    tokens_seen = 0
    started = time.monotonic()
    model.train()
    while step < args.max_steps:
        for input_ids, attention, start_positions, end_positions in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            attention = attention.to(device, non_blocking=True)
            start_positions = start_positions.to(device)
            end_positions = end_positions.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                out = model(input_ids=input_ids, attention_mask=attention,
                            start_positions=start_positions,
                            end_positions=end_positions)
            loss = out.loss / args.gradient_accumulation_steps
            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                print(f"step {step + 1}: non-finite loss, batch skipped")
                continue
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Warmup then cosine down, so the last steps still move.
                progress = min(1.0, step / max(1, args.warmup_steps))
                scale = progress if step < args.warmup_steps else 0.5 * (
                    1 + math.cos(math.pi * (step - args.warmup_steps)
                                 / max(1, args.max_steps - args.warmup_steps)))
                for group in optimizer.param_groups:
                    group["lr"] = args.learning_rate * scale
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            step += 1
            tokens_seen += int(attention.sum())
            if step % 20 == 0:
                elapsed = time.monotonic() - started
                print(f"step {step}/{args.max_steps} loss={loss.item():.4f} "
                      f"lr={optimizer.param_groups[0]['lr']:.2e} "
                      f"tok_per_s={tokens_seen / max(elapsed, 1e-9):.0f}", flush=True)
            if args.eval_every and step % args.eval_every == 0:
                exact, f1 = evaluate(model, tokenizer, valid_split, args, device,
                                     args.eval_examples)
                print(f"validation step={step} exact_match={exact:.3f} f1={f1:.3f}",
                      flush=True)
            if args.save_every and step % args.save_every == 0:
                save(step)
            if step >= args.max_steps:
                break
    save(step)
    exact, f1 = evaluate(model, tokenizer, valid_split, args, device, args.eval_examples)
    print(f"final validation exact_match={exact:.3f} f1={f1:.3f}")
    print(f"Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
