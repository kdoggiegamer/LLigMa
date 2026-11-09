#!/usr/bin/env python3
"""Train LLigMa from a Hugging Face text dataset or local Parquet files."""
import argparse
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer

from lligma.model import LLigMaConfig, LLigMaForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/lligma_3_0_2b.yaml"))
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--train-file", type=Path, help="Local Parquet/JSONL file (uses --validation-file too)")
    parser.add_argument("--tokens-file", type=Path,
                        help="Pre-tokenized uint16 file: packed blocks, memory mapped, no tokenizer pass")
    parser.add_argument("--validation-tokens-file", type=Path)
    parser.add_argument("--keep-checkpoints", type=int, default=0,
                        help="Keep only the newest N checkpoints (0 keeps all)")
    parser.add_argument("--max-hours", type=float, default=0,
                        help="Stop cleanly after this many hours and save a resumable checkpoint")
    parser.add_argument("--validation-file", type=Path)
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--tokenizer", default="gpt2", help="Hugging Face tokenizer ID or local path")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/lligma-wikitext2"))
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--eval-steps", type=int, default=25)
    parser.add_argument("--max-train-examples", type=int, default=0, help="Optional cap; 0 uses all rows")
    parser.add_argument("--max-eval-examples", type=int, default=0, help="Optional cap; 0 uses all rows")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true",
                        help="Mixed precision with fp16 autocast and loss scaling (the only"
                             " tensor-core path on Turing/sm75, which has no bf16 cores)")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers; keeps block reads off the compute process")
    parser.add_argument("--resume-from", type=Path,
                        help="Checkpoint written by --save-every; resumes step count and optimizer state")
    parser.add_argument("--save-every", type=int, default=0,
                        help="Write a resumable checkpoint every N optimizer steps (0 disables)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_splits(args):
    if args.train_file:
        if not args.validation_file:
            raise ValueError("--validation-file is required with --train-file")
        files = {"train": str(args.train_file), "validation": str(args.validation_file)}
        if args.test_file:
            files["test"] = str(args.test_file)
        kind = "parquet" if args.train_file.suffix == ".parquet" else "json"
        return load_dataset(kind, data_files=files)
    return load_dataset(args.dataset, args.dataset_config)


def make_blocks(split, tokenizer, column, sequence_length, limit):
    rows = split.select(range(min(limit, len(split)))) if limit else split
    token_ids = []
    texts = [text for text in rows[column] if isinstance(text, str) and text.strip()]
    for start in range(0, len(texts), 256):
        encoded = tokenizer(texts[start:start + 256], add_special_tokens=False)
        for sequence in encoded["input_ids"]:
            token_ids.extend(sequence)
    eos = tokenizer.eos_token_id
    if eos is not None:
        token_ids.append(eos)
    block_count = len(token_ids) // sequence_length
    if not block_count:
        raise ValueError(f"Split has fewer than {sequence_length} tokens after tokenization")
    packed = torch.tensor(token_ids[:block_count * sequence_length], dtype=torch.long)
    return TensorDataset(packed.view(block_count, sequence_length))


class TokenBlockDataset(torch.utils.data.Dataset):
    """Fixed-length blocks read from a pre-tokenized uint16 file.

    A real corpus cannot first build a Python list of every token: a few hundred
    million of them do not fit in memory. The file is memory mapped and sliced
    per batch instead, so the corpus size is bounded by disk, not by RAM.
    """

    def __init__(self, path, sequence_length, limit=0):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.sequence_length = sequence_length
        self.count = len(self.tokens) // sequence_length
        if limit:
            self.count = min(self.count, limit)
        if not self.count:
            raise ValueError(f"{path} holds fewer than {sequence_length} tokens")

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        start = index * self.sequence_length
        block = np.asarray(self.tokens[start:start + self.sequence_length], dtype=np.int64)
        # A tuple, like TensorDataset, so the loader keeps the (batch,) shape.
        return (torch.from_numpy(block),)


@torch.no_grad()
def evaluate(model, loader, device, max_batches=0, amp_dtype=None):
    model.eval()
    total_loss = 0.0
    batches = 0
    for (input_ids,) in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype,
                            enabled=amp_dtype is not None and device.type == "cuda"):
            loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
        total_loss += loss.item()
        batches += 1
        if max_batches and batches >= max_batches:
            break
    model.train()
    mean_loss = total_loss / max(batches, 1)
    return mean_loss, math.exp(min(mean_loss, 20.0))


def estimate_parameters(config):
    """Mirror LLigMaForCausalLM exactly so the guard below cannot be fooled."""
    hidden = config.hidden_size
    heads = config.num_attention_heads
    qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

    if config.q_lora_rank > 0:
        q_params = hidden * config.q_lora_rank + config.q_lora_rank * heads * qk_head_dim
        q_norm = config.q_lora_rank
    else:
        q_params = hidden * heads * qk_head_dim
        q_norm = 0

    attention = (
        q_params
        + hidden * (config.kv_lora_rank + config.qk_rope_head_dim)
        + config.kv_lora_rank * heads * (config.qk_nope_head_dim + config.v_head_dim)
        + heads * config.v_head_dim * hidden
        + q_norm
        + config.kv_lora_rank
    )

    dense_mlp = 3 * hidden * config.intermediate_size
    moe = config.num_experts * 3 * hidden * config.moe_intermediate_size
    moe += 3 * hidden * config.moe_intermediate_size * config.num_shared_experts
    moe += hidden * config.num_experts

    layers = 0
    for index in range(config.num_hidden_layers):
        # Same condition as LLigMaDecoderLayer: use_moe gates the whole stack.
        uses_moe = config.use_moe and index >= config.first_k_dense_replace
        layers += attention + 2 * hidden + (moe if uses_moe else dense_mlp)

    embedding = config.vocab_size * hidden
    output = 0 if config.tie_word_embeddings else config.vocab_size * hidden
    return layers + embedding + output + hidden


def main():
    args = parse_args()
    if args.max_steps < 1 or args.sequence_length < 2:
        raise ValueError("--max-steps must be positive and --sequence-length at least 2")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    if args.bf16 and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("--bf16 requires a CUDA device with bfloat16 support")
    if args.fp16 and device.type != "cuda":
        raise RuntimeError("--fp16 requires a CUDA device")
    # Mixed precision keeps the master weights in float32 and runs the matmuls
    # in the low-precision dtype through autocast. fp16 needs a loss scale: on
    # Turing the tensor cores are fp16-only, so this is the fast path there.
    amp_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else None)
    scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    if tokenizer.eos_token_id is None:
        raise ValueError("The tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.tokens_file:
        if not args.validation_tokens_file:
            raise ValueError("--validation-tokens-file is required with --tokens-file")
        splits = None
    else:
        splits = load_splits(args)
        if "validation" not in splits:
            raise ValueError("Dataset must provide a validation split")

    with args.config.open(encoding="utf-8") as handle:
        config_values = yaml.safe_load(handle) or {}
    config_values.pop("model_type", None)
    config_values["vocab_size"] = len(tokenizer)
    config_values["bos_token_id"] = tokenizer.bos_token_id or tokenizer.eos_token_id
    config_values["eos_token_id"] = tokenizer.eos_token_id
    config_values["pad_token_id"] = tokenizer.pad_token_id
    config_values["max_position_embeddings"] = max(args.sequence_length, config_values.get("max_position_embeddings", args.sequence_length))
    config_values["use_cache"] = False
    config = LLigMaConfig(**config_values)
    estimate = estimate_parameters(config)
    print(f"Device: {device}; estimated model parameters: {estimate:,} ({estimate / 1e9:.2f}B)")
    if estimate > 2_000_000_000:
        raise RuntimeError("Configured model exceeds 2B parameters; refusing to allocate it. Use a smaller config for this machine.")
    model = LLigMaForCausalLM(config).to(device)
    actual = sum(parameter.numel() for parameter in model.parameters())
    print(f"Instantiated parameters: {actual:,}")
    if amp_dtype is not None:
        print(f"mixed precision: autocast {str(amp_dtype).replace('torch.', '')}"
              f"{', loss scaling' if args.fp16 else ''}")

    if args.tokens_file:
        train_data = TokenBlockDataset(args.tokens_file, args.sequence_length, args.max_train_examples)
        val_data = TokenBlockDataset(args.validation_tokens_file, args.sequence_length, args.max_eval_examples)
        print(f"pre-tokenized blocks: train {len(train_data):,} | validation {len(val_data):,}"
              f" ({args.sequence_length} tokens each)")
    else:
        train_data = make_blocks(splits["train"], tokenizer, args.text_column, args.sequence_length, args.max_train_examples)
        val_data = make_blocks(splits["validation"], tokenizer, args.text_column, args.sequence_length, args.max_eval_examples)
    # Pinned staging memory plus worker processes: the GPU should never wait on
    # a memmap read or a page fault between steps.
    loader_options = {}
    if args.num_workers:
        loader_options = {
            "num_workers": args.num_workers,
            "prefetch_factor": 4,
            "persistent_workers": True,
        }
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, drop_last=True,
        pin_memory=device.type == "cuda", **loader_options,
    )
    val_loader = DataLoader(val_data, batch_size=args.batch_size,
                            pin_memory=device.type == "cuda")
    if not len(train_loader):
        raise ValueError("No complete training batch; lower batch size or sequence length")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model.train()

    def save_checkpoint(step):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / f"checkpoint-{step:06d}.pt"
        torch.save({"step": step, "model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
        print(f"saved checkpoint {path}")
        if args.keep_checkpoints > 0:
            for old in sorted(args.output_dir.glob("checkpoint-*.pt"))[:-args.keep_checkpoints]:
                old.unlink()
                print(f"pruned checkpoint {old.name}")

    optimizer_steps = 0
    if args.resume_from:
        if not args.resume_from.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {args.resume_from}")
        state = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        optimizer_steps = int(state["step"])
        print(f"Resumed from {args.resume_from} at step {optimizer_steps}")
        if optimizer_steps >= args.max_steps:
            raise ValueError("--max-steps must exceed the resumed step count")

    micro_step = 0
    # GradScaler health. It halves the scale exactly when it found an inf or NaN
    # in the gradients, and that halving happens instead of an optimizer step:
    # watching the scale across step() is how skipped steps are counted exactly.
    skipped_steps = 0
    lowest_scale = None
    started = time.monotonic()
    deadline = started + args.max_hours * 3600.0 if args.max_hours > 0 else None
    tokens_per_micro_step = args.batch_size * args.sequence_length
    stopped_at_deadline = False
    while optimizer_steps < args.max_steps and not stopped_at_deadline:
        for (input_ids,) in train_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype,
                                enabled=amp_dtype is not None and device.type == "cuda"):
                loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
            scaler.scale(loss / args.gradient_accumulation_steps).backward()
            micro_step += 1
            if micro_step % args.gradient_accumulation_steps == 0:
                # Unscale before clipping so the norm is measured on real grads.
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scale_before = scaler.get_scale() if scaler.is_enabled() else None
                scaler.step(optimizer)
                scaler.update()
                if scale_before is not None:
                    scale_after = scaler.get_scale()
                    lowest_scale = scale_after if lowest_scale is None else min(lowest_scale, scale_after)
                    if scale_after < scale_before:
                        skipped_steps += 1
                        if skipped_steps <= 5:
                            print(f"skipped step {optimizer_steps + 1}: non-finite gradients, "
                                  f"scale {scale_before:g} -> {scale_after:g}")
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if optimizer_steps == 1 or optimizer_steps % 10 == 0:
                    elapsed = time.monotonic() - started
                    seen = micro_step * tokens_per_micro_step
                    print(f"step {optimizer_steps}/{args.max_steps} train_loss={loss.item():.4f} "
                          f"tokens={seen:,} elapsed={elapsed:.0f}s tok_per_s={seen / max(elapsed, 1e-9):.0f}")
                if args.save_every and optimizer_steps % args.save_every == 0:
                    save_checkpoint(optimizer_steps)
                if args.eval_steps and optimizer_steps % args.eval_steps == 0:
                    val_loss, ppl = evaluate(model, val_loader, device, amp_dtype=amp_dtype)
                    print(f"validation step={optimizer_steps} loss={val_loss:.4f} perplexity={ppl:.2f}")
            if optimizer_steps >= args.max_steps:
                break
            if deadline is not None and time.monotonic() >= deadline:
                stopped_at_deadline = True
                print(f"reached --max-hours {args.max_hours}: stopping after step {optimizer_steps}")
                if optimizer_steps:
                    save_checkpoint(optimizer_steps)
                break

    if scaler.is_enabled():
        applied = optimizer_steps - skipped_steps
        print(f"scaler summary: attempted_steps={optimizer_steps} skipped_steps={skipped_steps} "
              f"applied_steps={applied} min_scale={lowest_scale if lowest_scale is None else f'{lowest_scale:g}'} "
              f"final_scale={scaler.get_scale():g}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    if splits is not None and "test" in splits:
        test_data = make_blocks(splits["test"], tokenizer, args.text_column, args.sequence_length, args.max_eval_examples)
        test_loss, test_ppl = evaluate(model, DataLoader(test_data, batch_size=args.batch_size), device,
                                       amp_dtype=amp_dtype)
        print(f"test loss={test_loss:.4f} perplexity={test_ppl:.2f}")
    print(f"Saved model and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    main()
