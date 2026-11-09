"""Train a ByteLevel BPE tokenizer for LLigMa Small on a local text corpus.

At ~0.39B parameters the vocabulary is not a detail: with a tied embedding, every
extra token costs `hidden_size` parameters, and a tokenizer that fragments Italian
words burns both parameters and context. A 32K BPE trained on the actual corpus
beats reusing an English-oriented 50K/128K tokenizer.

Typical use:

    python scripts/train_tokenizer.py \
        --input data/italian/train.parquet data/italian/extra.txt \
        --vocab-size 32768 \
        --output-dir tokenizers/lligma-small-32k
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SUPPORTED_SUFFIXES = {".txt", ".jsonl", ".json", ".parquet"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True,
                        help="Text/JSONL/Parquet files or directories containing them")
    parser.add_argument("--vocab-size", type=int, default=32768)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, default=Path("tokenizers/lligma-small-32k"))
    parser.add_argument("--text-column", default="text", help="Field to read from JSONL/Parquet")
    parser.add_argument("--sample-docs", type=int, default=2000,
                        help="Documents used to report the compression ratio")
    return parser.parse_args()


def collect_files(paths):
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES))
        elif path.suffix.lower() in SUPPORTED_SUFFIXES:
            files.append(path)
        else:
            raise ValueError(f"Unsupported input file type: {path}")
    if not files:
        raise ValueError("No supported input files found")
    return files


def iter_documents(files, text_column):
    """Yield non-empty text documents without loading the corpus into memory."""
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            from datasets import load_dataset

            dataset = load_dataset("parquet", data_files=str(path), split="train")
            column = text_column if text_column in dataset.column_names else dataset.column_names[0]
            for value in dataset[column]:
                if isinstance(value, str) and value.strip():
                    yield value
        elif suffix in {".jsonl", ".json"}:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        payload = payload.get(text_column)
                    if isinstance(payload, str) and payload.strip():
                        yield payload
        else:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        yield line


def main():
    args = parse_args()
    if args.vocab_size < 1000:
        raise ValueError("--vocab-size is unrealistically small")

    from tokenizers import ByteLevelBPETokenizer
    from tokenizers.processors import ByteLevel as ByteLevelProcessor
    from transformers import PreTrainedTokenizerFast

    special_tokens = ["<pad>", "<s>", "</s>", "<unk>"]
    files = collect_files(args.input)
    print(f"Corpus files: {len(files)}")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        iter_documents(files, args.text_column),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
    )
    # ByteLevel BPE assumes a leading space on the first token of a piece of text.
    tokenizer.post_processor = ByteLevelProcessor(trim_offsets=False)

    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(args.output_dir)

    # Report how well the tokenizer compresses the corpus it was trained on.
    sample = []
    for index, document in enumerate(iter_documents(files, args.text_column)):
        if index >= args.sample_docs:
            break
        sample.append(document)
    if sample:
        characters = sum(len(document) for document in sample)
        tokens = sum(len(fast.encode(document)) for document in sample)
        if tokens:
            print(f"Compression on {len(sample)} documents: {characters / tokens:.2f} characters per token")

    print(f"Vocabulary size: {len(fast)}")
    print(f"Saved tokenizer to {args.output_dir}")


if __name__ == "__main__":
    sys.exit(main())
