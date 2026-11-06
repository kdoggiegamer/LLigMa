"""Train LLigMa Small in mixed precision, then judge it against equal wall clock.

Why this kernel exists: the fp32 run moved the loss but left the GPU idle. On a
T4 (Turing, sm75) there are no bf16 tensor cores and the fp32 path tops out
around 8 TFLOPS, so 0.39B parameters at 837 tokens/s used ~1.9 TFLOPS. The fix
is fp16 autocast with a GradScaler plus a fused attention kernel, not a smaller
model.

The comparison this kernel must produce is *equal wall clock*, not equal steps:
the previous fp32 run's last checkpoint is mounted as a dataset source, so it is
evaluated here, on this GPU, with the same code and the same held-out blocks as
the new run. Same 1.5 hours each. Then the only difference left is speed. If the
checkpoint is not mounted, the comparison is not measured: the verdict names it
in `missing` instead of reporting a number it does not have.

The GradScaler is instrumented too, because fp16 can silently throw away steps:
skipped steps, scale range and tokens actually learned from are in the verdict.

Everything else is unchanged on purpose: same corpus, same sequence length, same
batch size and accumulation, same learning rate, same seed, same architecture.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

WORK = Path("/kaggle/working")
RUN = WORK / "run"
SEQUENCE_LENGTH = 512
BATCH_SIZE = 4
GRAD_ACCUM = 4
MAX_HOURS = 1.5
SAVE_EVERY = 400
EVAL_STEPS = 200
LEARNING_RATE = 3e-4
KEEP_CHECKPOINTS = 2

# What the fp32 session actually achieved before it hit the wall clock, read
# from its own log (step 551, 4,505,600 tokens, 5,384 s).
OLD_RUN_STEP = 551
OLD_RUN_TOKENS = 4_505_600
OLD_RUN_SECONDS = 5384


def find_input(slug):
    direct = Path(f"/kaggle/input/{slug}")
    if direct.is_dir():
        return direct
    for candidate in Path("/kaggle/input").glob(f"*/{slug}"):
        if candidate.is_dir():
            return candidate
    for candidate in Path("/kaggle/input").rglob(slug):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"{slug} not mounted under /kaggle/input")


def find_repo():
    for candidate in Path("/kaggle/input").rglob("requirements.txt"):
        if (candidate.parent / "lligma").is_dir():
            return candidate.parent
    raise FileNotFoundError("LLigMa code dataset not found")


def install_dependencies(repo):
    requirements = (repo / "requirements.txt").read_text(encoding="utf-8").splitlines()
    packages = [line.strip() for line in requirements if line.strip() and not line.strip().lower().startswith("torch")]
    print("installing:", packages, flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)
    import transformers
    print("torch", torch.__version__, "| transformers", transformers.__version__, flush=True)


def latest_checkpoint(*directories):
    """Newest checkpoint under a mount, wherever the mount puts it.

    A kernel output keeps it in run/, a dataset created with --dir-mode zip
    flattens it to the mount root. Searching recursively handles both.
    """
    best, best_step = None, -1
    for directory in directories:
        if not directory or not Path(directory).is_dir():
            continue
        for path in sorted(Path(directory).rglob("checkpoint-*.pt")):
            try:
                step = int(path.stem.split("-")[1])
            except (IndexError, ValueError):
                continue
            if step > best_step:
                best, best_step = path, step
    return best, best_step


def build_model(repo, config_path, vocab_size):
    sys.path.insert(0, str(repo))
    import yaml
    from lligma.model import LLigMaConfig, LLigMaForCausalLM

    values = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    values.pop("model_type", None)
    values["vocab_size"] = vocab_size
    values["max_position_embeddings"] = max(values.get("max_position_embeddings", 0), SEQUENCE_LENGTH)
    config = LLigMaConfig(**values)
    return config, LLigMaForCausalLM(config)


def _local_copy(checkpoint):
    """The /kaggle/input mount cannot seek, which torch.load requires."""
    source = Path(checkpoint)
    if not str(source).startswith("/kaggle/input"):
        return source
    target = WORK / "mounted" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        print(f"copying {source.name} off the read-only mount", flush=True)
        shutil.copyfile(source, target)
    return target


def load_weights(model, checkpoint):
    """Load a full state dict from a trainer checkpoint.

    save_pretrained drops the tied lm_head from the public weights file, which is
    exactly what made the previous verdict phase fail. The checkpoint written by
    the trainer has every key, so it is the authoritative source; the safetensors
    path stays as a fallback and repairs the one tied key explicitly instead of
    failing on it.
    """
    if checkpoint is not None:
        payload = torch.load(_local_copy(checkpoint), map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model"])
        return f"checkpoint {Path(checkpoint).name}"
    from safetensors.torch import load_file

    state = load_file(str(RUN / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected weights: {unexpected}")
    if missing != ["lm_head.weight"]:
        raise RuntimeError(f"missing weights: {missing}")
    model.lm_head.weight = model.model.embed_tokens.weight
    return "safetensors with tied lm_head repaired"


@torch.no_grad()
def evaluate_perplexity(model, tokens_path, batches=40):
    """Held-out perplexity in the same precision the model trains in."""
    tokens = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    blocks = len(tokens) // SEQUENCE_LENGTH
    if not blocks:
        raise ValueError("validation token file is too short")
    generator = torch.Generator().manual_seed(0)
    picks = torch.randint(0, blocks, (min(batches, blocks),), generator=generator)
    model.eval()
    total, counted = 0.0, 0
    for index in picks.tolist():
        start = index * SEQUENCE_LENGTH
        block = np.asarray(tokens[start:start + SEQUENCE_LENGTH], dtype=np.int64)
        input_ids = torch.from_numpy(block).unsqueeze(0).to("cuda")
        with torch.autocast("cuda", dtype=torch.float16, enabled=model.dtype != torch.float16):
            loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
        total += loss.item()
        counted += 1
    mean = total / max(counted, 1)
    return mean, float(np.exp(min(mean, 20.0)))


def generate_sample(model, prompt_ids, new_tokens=48):
    model.eval()
    generated = prompt_ids
    past = None
    with torch.no_grad():
        for _ in range(new_tokens):
            with torch.autocast("cuda", dtype=torch.float16, enabled=model.dtype != torch.float16):
                if past is None:
                    outputs = model(input_ids=generated, use_cache=True)
                else:
                    outputs = model(
                        input_ids=generated[:, -1:], past_key_values=past,
                        attention_mask=torch.ones(1, generated.shape[1], dtype=torch.long, device="cuda"),
                        use_cache=True,
                    )
            past = outputs.past_key_values
            next_token = outputs.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
    return generated


def assess(repo, config_path, vocab_size, valid_tokens, label, checkpoint):
    """Build the model, load weights, measure held-out loss and perplexity."""
    _, model = build_model(repo, config_path, vocab_size)
    source = load_weights(model, checkpoint)
    step = int(Path(checkpoint).stem.split("-")[1]) if checkpoint else None
    model = model.to("cuda")
    loss, perplexity = evaluate_perplexity(model, valid_tokens)
    print(f"{label.upper()} step={step} from {source}: held-out loss={loss:.4f} perplexity={perplexity:.1f}",
          flush=True)
    return model, loss, perplexity, step, source


def run_training(repo, command, log_path):
    print("RUN " + " ".join(command), flush=True)
    lines = []
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            handle.write(line)
            lines.append(line.rstrip())
        process.wait()
    if process.returncode:
        raise RuntimeError(f"training failed with exit code {process.returncode}; see {log_path}")
    stats = {"steps": None, "tokens": None, "tok_per_s": None,
             "first_loss": None, "last_loss": None, "scaler": None}
    for line in lines:
        if line.startswith("step "):
            fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
            # The step itself is the second token ("step 2680/1000000"), not a
            # key=value field. Reading it as one is what wrote 0 into the
            # previous verdict.
            stats["steps"] = int(line.split()[1].split("/")[0])
            stats["tokens"] = int(fields.get("tokens", "0").replace(",", ""))
            stats["tok_per_s"] = float(fields.get("tok_per_s", "0"))
            stats["last_loss"] = float(fields.get("train_loss", "nan"))
            if stats["first_loss"] is None:
                stats["first_loss"] = stats["last_loss"]
        elif line.startswith("scaler summary:"):
            fields = dict(part.split("=", 1) for part in line.split() if "=" in part)
            stats["scaler"] = {
                "attempted_steps": int(fields["attempted_steps"]),
                "skipped_steps": int(fields["skipped_steps"]),
                "applied_steps": int(fields["applied_steps"]),
                "min_scale": fields["min_scale"],
                "final_scale": fields["final_scale"],
            }
    # Never ship a number that was not measured: no step line or no scaler line
    # means the run did not report what the verdict claims, so stop instead of
    # writing a plausible zero.
    if stats["steps"] is None:
        raise RuntimeError("train.log has no 'step ' lines: refusing to write a verdict without measurements")
    if stats["scaler"] is None:
        raise RuntimeError("train.log has no 'scaler summary:' line: fp16 health was not measured")
    return stats


def find_old_checkpoint():
    """The fp32 run's last checkpoint, mounted as a dataset source.

    A kernel cannot cite a kernel whose last version failed, and the fp32 run
    ended in ERROR after saving its weights. That is why the checkpoint lives in
    a dataset instead of kernel_sources.
    """
    try:
        mount = find_input("lligma-old-checkpoint")
    except FileNotFoundError:
        print("no fp32 checkpoint mounted (dataset lligma-old-checkpoint)", flush=True)
        return None, None
    checkpoint, step = latest_checkpoint(mount)
    print(f"fp32 checkpoint: {checkpoint} (step {step})", flush=True)
    return checkpoint, step


def main():
    repo = find_repo()
    print(f"code from {repo}", flush=True)
    install_dependencies(repo)
    print(f"cuda: {torch.cuda.is_available()} | "
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}", flush=True)

    prep = find_input("lligma-prep")
    corpus_tokens = prep / "tokens" / "train.bin"
    valid_tokens = prep / "tokens" / "valid.bin"
    tokenizer_dir = prep / "tokenizer"
    stats = json.loads((prep / "stats.json").read_text(encoding="utf-8")) if (prep / "stats.json").is_file() else {}
    print(f"prep from {prep} | stats: {json.dumps(stats)}", flush=True)

    config_path = repo / "configs" / "lligma_3_0_small.yaml"
    from transformers import AutoTokenizer

    vocab_size = len(AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True))
    print(f"vocabulary: {vocab_size}", flush=True)
    torch.backends.cudnn.benchmark = True

    # Reference point: the same architecture at random initialisation.
    torch.manual_seed(42)
    _, baseline = build_model(repo, config_path, vocab_size)
    baseline = baseline.to("cuda")
    base_loss, base_ppl = evaluate_perplexity(baseline, valid_tokens)
    print(f"BASELINE random-init held-out loss={base_loss:.4f} perplexity={base_ppl:.1f}", flush=True)
    del baseline
    torch.cuda.empty_cache()

    # The control: what 1.5 hours of fp32 bought last time, measured here with
    # this exact evaluation so precision cannot explain a difference.
    missing = []
    old_loss = old_ppl = None
    old_checkpoint, old_step = find_old_checkpoint()
    if old_checkpoint is not None:
        old_model, old_loss, old_ppl, old_step, _ = assess(
            repo, config_path, vocab_size, valid_tokens, "old", old_checkpoint)
        del old_model
        torch.cuda.empty_cache()
    else:
        missing.append("fp32_previous_run: checkpoint not mounted, so the equal-wall-clock "
                       "comparison was not measured")

    command = [
        # -u so the log streams while the session is alive instead of arriving in
        # 4 KB blocks, which matters when a session is cut at 12 hours.
        sys.executable, "-u", str(repo / "scripts" / "train.py"),
        "--config", str(config_path),
        "--tokens-file", str(corpus_tokens),
        "--validation-tokens-file", str(valid_tokens),
        "--tokenizer", str(tokenizer_dir),
        "--output-dir", str(RUN),
        "--sequence-length", str(SEQUENCE_LENGTH),
        "--batch-size", str(BATCH_SIZE),
        "--gradient-accumulation-steps", str(GRAD_ACCUM),
        "--max-steps", "1000000",
        "--learning-rate", str(LEARNING_RATE),
        "--weight-decay", "0.1",
        "--eval-steps", str(EVAL_STEPS),
        "--max-eval-examples", "64",
        "--save-every", str(SAVE_EVERY),
        "--keep-checkpoints", str(KEEP_CHECKPOINTS),
        "--max-hours", str(MAX_HOURS),
        "--fp16",
        "--num-workers", "2",
        "--device", "cuda",
    ]
    trained = run_training(repo, command, WORK / "train.log")

    new_checkpoint, _ = latest_checkpoint(RUN)
    if new_checkpoint is None:
        missing.append("fp16_this_run.step: no trainer checkpoint in the output")
    new_model, new_loss, new_ppl, new_step, source = assess(
        repo, config_path, vocab_size, valid_tokens, "new", new_checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)
    samples = []
    for prompt in ["L'Italia è", "La storia di Roma", "Nel 1945 l'Europa"]:
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
        generated = generate_sample(new_model, ids)
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        samples.append({"prompt": prompt, "text": text})
        print("SAMPLE " + repr(prompt) + " -> " + repr(text), flush=True)

    scaler = trained["scaler"]
    # Both counts come from the same authoritative place (the scaler summary at
    # the end), never from the periodic step line: that line is printed every 10
    # steps, so pairing it with the final step count made "applied" exceed
    # "seen" in the previous verdict.
    tokens_per_step = BATCH_SIZE * GRAD_ACCUM * SEQUENCE_LENGTH
    tokens_seen = scaler["attempted_steps"] * tokens_per_step
    tokens_applied = scaler["applied_steps"] * tokens_per_step
    print("", flush=True)
    print(f"GRAD SCALER: {scaler['skipped_steps']} of {scaler['attempted_steps']} optimizer steps skipped "
          f"(non-finite gradients), {scaler['applied_steps']} applied, "
          f"scale min {scaler['min_scale']} final {scaler['final_scale']};",
          f"{tokens_applied:,} of {tokens_seen:,} tokens actually learned from", flush=True)
    if scaler["skipped_steps"]:
        print(f"WARNING: {scaler['skipped_steps']} steps were discarded, so the throughput above "
              f"overstates useful work by {100 * scaler['skipped_steps'] / max(scaler['attempted_steps'], 1):.2f}%",
              flush=True)

    print("", flush=True)
    print(f"EQUAL WALL CLOCK ({MAX_HOURS} h each, same corpus, same config, same eval):", flush=True)
    if old_ppl is not None:
        print(f"  fp32 {OLD_RUN_SECONDS}s  step {old_step:5d}  {OLD_RUN_TOKENS:>10,} tokens  "
              f"{OLD_RUN_TOKENS / OLD_RUN_SECONDS:6.0f} tok/s  ppl {old_ppl:8.1f}", flush=True)
    else:
        print("  fp32: NOT MEASURED - checkpoint not mounted (see verdict.json missing[])", flush=True)
    print(f"  fp16 {int(MAX_HOURS * 3600)}s  step {new_step:5d}  {tokens_seen:>10,} tokens  "
          f"{trained['tok_per_s']:6.0f} tok/s  ppl {new_ppl:8.1f}", flush=True)
    if old_ppl is not None:
        print(f"  speedup {trained['tok_per_s'] / (OLD_RUN_TOKENS / OLD_RUN_SECONDS):.2f}x | "
              f"perplexity change {old_ppl / new_ppl:.2f}x lower", flush=True)

    # Only measured numbers go in. Anything absent is named in `missing` instead
    # of appearing as 0 or null without explanation.
    summary = {
        "measurement": {
            "held_out": f"40 blocks of {SEQUENCE_LENGTH} tokens from tokens/valid.bin, fp16 autocast, same GPU",
            "equal_wall_clock_hours": MAX_HOURS,
        },
        "baseline_random_init": {"loss": base_loss, "perplexity": base_ppl},
        "fp32_previous_run": None if old_ppl is None else {
            "step": old_step, "loss": old_loss, "perplexity": old_ppl,
            "tokens": OLD_RUN_TOKENS, "seconds": OLD_RUN_SECONDS,
            "tok_per_s": OLD_RUN_TOKENS / OLD_RUN_SECONDS,
        },
        "fp16_this_run": {
            "step": new_step, "loss": new_loss, "perplexity": new_ppl,
            "tokens_seen": tokens_seen, "tokens_applied": tokens_applied,
            "tok_per_s": trained["tok_per_s"], "seconds": int(MAX_HOURS * 3600),
            "steps_this_session": trained["steps"], "weights_from": source,
        },
        "speedup_vs_fp32": None if old_ppl is None else trained["tok_per_s"] / (OLD_RUN_TOKENS / OLD_RUN_SECONDS),
        "perplexity_improvement": None if old_ppl is None else old_ppl / new_ppl,
        "grad_scaler": scaler,
        "missing": missing,
        "train_loss": {"first": trained["first_loss"], "last": trained["last_loss"]},
        "config": {
            "sequence_length": SEQUENCE_LENGTH, "batch_size": BATCH_SIZE,
            "grad_accum": GRAD_ACCUM, "learning_rate": LEARNING_RATE,
            "precision": "fp16 autocast + GradScaler",
        },
        "samples": samples,
    }
    (WORK / "verdict.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("VERDICT " + json.dumps(summary), flush=True)
    print("TRAIN DONE", flush=True)


if __name__ == "__main__":
    main()
