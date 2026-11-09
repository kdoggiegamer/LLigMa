"""Build a legally clean Italian corpus and its token stream.

Source: Italian Wikipedia (CC BY-SA 4.0), the `wikimedia/wikipedia` dataset at
revision 20231101. Attribution is preserved in the Parquet files (`id`, `url`,
`title` next to `text`), which is what CC BY-SA asks for.

Outputs, committed with the kernel so later sessions can mount them:
    corpus/train.parquet, corpus/valid.parquet   documents + attribution
    tokenizer/                                  32k ByteLevel BPE
    tokens/train.bin, tokens/valid.bin           uint16 token stream, EOS between docs

The token stream exists because a real corpus cannot be re-tokenized inside the
training kernel every session: that would burn GPU-quota time on CPU work.

The work is deliberately bounded (MAX_DOCS) and single-pass: every phase prints
how far it got, so a slow stage is visible instead of looking like a hang.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Kaggle forces /kaggle/working. On a rented GPU the working directory is somewhere
# else, so allow an override instead of editing the constant per host.
WORK = Path(os.environ.get("LLIGMA_WORK", "/kaggle/working"))
CORPUS = WORK / "corpus"
TOKENS = WORK / "tokens"
TOKENIZER = WORK / "tokenizer"
DATASET = "wikimedia/wikipedia"
DATASET_CONFIG = "20231101.it"
VOCAB_SIZE = 32768
MIN_CHARS = 400
VALID_PERCENT = 1
MAX_DOCS = 350_000
TOKENIZER_SAMPLE_DOCS = 60_000


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def find_repo():
    direct = Path("/kaggle/input/lligma-code")
    if (direct / "requirements.txt").is_file():
        return direct
    for candidate in Path("/kaggle/input").rglob("requirements.txt"):
        if (candidate.parent / "lligma").is_dir():
            return candidate.parent
    raise FileNotFoundError("LLigMa code dataset not found under /kaggle/input")


def install_dependencies(repo):
    requirements = (repo / "requirements.txt").read_text(encoding="utf-8").splitlines()
    packages = [line.strip() for line in requirements if line.strip() and not line.strip().lower().startswith("torch")]
    log(f"installing: {packages}")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)
    import datasets
    log(f"datasets {datasets.__version__}")


def digest(value):
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def normalise(text):
    return re.sub(r"\s+", " ", text.strip().lower())


def is_article(text):
    if not isinstance(text, str) or len(text) < MIN_CHARS:
        return False
    letters = sum(character.isalpha() or character.isspace() for character in text)
    return letters >= 0.7 * len(text)


def build_corpus():
    from datasets import load_dataset

    CORPUS.mkdir(parents=True, exist_ok=True)
    log(f"downloading {DATASET} [{DATASET_CONFIG}]")
    raw = load_dataset(DATASET, DATASET_CONFIG, split="train")
    log(f"documents available: {len(raw):,}")
    raw = raw.select(range(min(MAX_DOCS, len(raw))))
    log(f"working subset: {len(raw):,}")

    # One pass: quality filter and exact duplicate removal on normalised text.
    kept, seen, rejected = [], set(), 0
    started = time.time()
    for start in range(0, len(raw), 10_000):
        batch = raw[start:start + 10_000]
        for offset, (article_id, text) in enumerate(zip(batch["id"], batch["text"])):
            if not is_article(text):
                rejected += 1
                continue
            key = digest(normalise(text))
            if key in seen:
                rejected += 1
                continue
            seen.add(key)
            kept.append(start + offset)
        if start % 50_000 == 0:
            log(f" scan {start:,}/{len(raw):,} kept={len(kept):,} rejected={rejected:,}")
    log(f"scan done in {time.time() - started:.0f}s: kept {len(kept):,}, rejected {rejected:,}")

    filtered = raw.select(kept)
    def split_of(article_id):
        return int(digest(str(article_id))[:8], 16) % 100 < VALID_PERCENT

    validation_indices, train_indices = [], []
    for index, article_id in enumerate(filtered["id"]):
        (validation_indices if split_of(article_id) else train_indices).append(index)
    train = filtered.select(train_indices)
    validation = filtered.select(validation_indices)
    columns = [name for name in ("id", "url", "title", "text") if name in train.column_names]

    log(f"writing parquet: train {len(train):,} | validation {len(validation):,}")
    train.select_columns(columns).to_parquet(str(CORPUS / "train.parquet"))
    validation.select_columns(columns).to_parquet(str(CORPUS / "valid.parquet"))
    log(f"parquet bytes: {(CORPUS / 'train.parquet').stat().st_size:,}")
    return train, validation


def train_tokenizer():
    TOKENIZER.mkdir(parents=True, exist_ok=True)
    # A bounded sample is enough to learn a 32k vocabulary and keeps this phase
    # short; the full corpus is still what gets tokenized below.
    from datasets import load_dataset

    sample_path = WORK / "tokenizer-sample.txt"
    dataset = load_dataset("parquet", data_files=str(CORPUS / "train.parquet"), split="train")
    limit = min(TOKENIZER_SAMPLE_DOCS, len(dataset))
    log(f"writing tokenizer sample ({limit:,} documents)")
    with sample_path.open("w", encoding="utf-8") as handle:
        kept, total = 0, 0
        for start in range(0, limit, 2_000):
            for text in dataset[start:min(start + 2_000, limit)]["text"]:
                if text.strip():
                    handle.write(text.replace("\n", " ") + "\n")
                    kept += 1
                    total += len(text)
            if start % 10_000 == 0:
                log(f"  sample {start:,} docs, {total:,} chars")
    log(f"sample ready: {kept:,} documents, {total:,} characters")

    command = [
        sys.executable, str(find_repo() / "scripts" / "train_tokenizer.py"),
        "--input", str(sample_path),
        "--vocab-size", str(VOCAB_SIZE),
        "--output-dir", str(TOKENIZER),
    ]
    log("training tokenizer")
    subprocess.run(command, check=True)
    log("tokenizer done")


def write_token_stream(split, path, batch_size=2000):
    import numpy as np
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, use_fast=True)
    eos = tokenizer.eos_token_id
    total_documents = len(split)
    total_tokens = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with path.open("wb") as handle:
        for start in range(0, total_documents, batch_size):
            encoded = tokenizer(split[start:start + batch_size]["text"], add_special_tokens=False)["input_ids"]
            flat = []
            for sequence in encoded:
                flat.extend(sequence)
                flat.append(eos)
            np.array(flat, dtype=np.uint16).tofile(handle)
            total_tokens += len(flat)
            if start % (batch_size * 5) == 0:
                rate = total_tokens / max(time.time() - started, 1e-9)
                log(f"  {path.name}: {start:,}/{total_documents:,} docs, {total_tokens:,} tokens, {rate:.0f} tok/s")
    log(f"{path.name}: {total_documents:,} documents, {total_tokens:,} tokens")
    return total_documents, total_tokens


def main():
    repo = find_repo()
    log(f"code from {repo}")
    install_dependencies(repo)
    train, validation = build_corpus()
    train_tokenizer()
    TOKENS.mkdir(parents=True, exist_ok=True)
    train_docs, train_tokens = write_token_stream(train, TOKENS / "train.bin")
    valid_docs, valid_tokens = write_token_stream(validation, TOKENS / "valid.bin")

    stats = {
        "source": f"{DATASET} [{DATASET_CONFIG}]",
        "license": "CC BY-SA 4.0",
        "vocab_size": VOCAB_SIZE,
        "max_docs": MAX_DOCS,
        "train_documents": train_docs,
        "train_tokens": train_tokens,
        "valid_documents": valid_docs,
        "valid_tokens": valid_tokens,
        "train_parquet_bytes": (CORPUS / "train.parquet").stat().st_size,
        "valid_parquet_bytes": (CORPUS / "valid.parquet").stat().st_size,
        "train_bin_bytes": (TOKENS / "train.bin").stat().st_size,
        "valid_bin_bytes": (TOKENS / "valid.bin").stat().st_size,
    }
    (WORK / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log("STATS " + json.dumps(stats))
    log("PREP DONE")


if __name__ == "__main__":
    main()
