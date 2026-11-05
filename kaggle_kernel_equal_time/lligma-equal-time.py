"""The same 1.5 hours, two precisions: which one actually learned more?

The fp32 run stopped at step 551 after 5,384 s and 4,505,600 tokens. The fp16
run got the same wall clock. Comparing their perplexities only means something
if both are measured the same way, so this kernel loads both checkpoints into
the same architecture and scores them on the same held-out blocks, in the same
precision, on the same GPU. Nothing is taken from a log.

It also generates from both models with the same prompts and the same greedy
decoding, because a perplexity ratio can hide a model that learned nothing.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

WORK = Path("/kaggle/working")
SEQUENCE_LENGTH = 512
BATCHES = 40
VOCAB_FALLBACK = 32768

OLD_RUN = {"label": "fp32 (bf16 weights)", "step": 551, "tokens": 4_505_600, "seconds": 5384}


def find_input(slug):
    direct = Path(f"/kaggle/input/{slug}")
    if direct.is_dir():
        return direct
    for candidate in Path("/kaggle/input").rglob(slug):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"{slug} not mounted")


def find_repo():
    for candidate in Path("/kaggle/input").rglob("requirements.txt"):
        if (candidate.parent / "lligma").is_dir():
            return candidate.parent
    raise FileNotFoundError("LLigMa code dataset not found")


def install_dependencies(repo):
    requirements = (repo / "requirements.txt").read_text(encoding="utf-8").splitlines()
    packages = [line.strip() for line in requirements if line.strip() and not line.strip().lower().startswith("torch")]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)
    import transformers
    print("torch", torch.__version__, "| transformers", transformers.__version__, flush=True)


def latest_checkpoint(directory):
    """Newest checkpoint under a mount, wherever the mount puts it.

    A kernel output keeps it in run/, while a dataset created with --dir-mode
    zip lands the file at the mount root. Searching recursively handles both
    instead of hardcoding a layout.
    """
    best, best_step = None, -1
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
    return LLigMaForCausalLM(config)


def local_copy(checkpoint):
    """Copy a checkpoint out of /kaggle/input before loading it.

    The dataset mount does not support seek(), and torch.load needs to seek: it
    fails with "you can only torch.load from a file that is seekable". Copying
    to /kaggle/working first is the difference between a working verdict and a
    dead kernel at the last step.
    """
    source = Path(checkpoint)
    target = WORK / "checkpoints" / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists() or target.stat().st_size != source.stat().st_size:
        print(f"copying {source.name} to local disk ({source.stat().st_size / 1e9:.2f} GB)", flush=True)
        shutil.copyfile(source, target)
    return target


def load_into(model, checkpoint):
    payload = torch.load(local_copy(checkpoint), map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    return int(payload.get("step", -1))


@torch.no_grad()
def evaluate(model, tokens_path, batches=BATCHES):
    tokens = np.memmap(tokens_path, dtype=np.uint16, mode="r")
    blocks = len(tokens) // SEQUENCE_LENGTH
    generator = torch.Generator().manual_seed(0)
    picks = torch.randint(0, blocks, (min(batches, blocks),), generator=generator)
    model.eval()
    total, counted = 0.0, 0
    for index in picks.tolist():
        start = index * SEQUENCE_LENGTH
        block = np.asarray(tokens[start:start + SEQUENCE_LENGTH], dtype=np.int64)
        input_ids = torch.from_numpy(block).unsqueeze(0).to("cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_ids=input_ids, labels=input_ids, use_cache=False).loss
        total += loss.item()
        counted += 1
    mean = total / max(counted, 1)
    return mean, float(np.exp(min(mean, 20.0)))


@torch.no_grad()
def generate(model, prompt_ids, new_tokens=32):
    model.eval()
    generated, past = prompt_ids, None
    for _ in range(new_tokens):
        with torch.autocast("cuda", dtype=torch.float16):
            if past is None:
                outputs = model(input_ids=generated, use_cache=True)
            else:
                outputs = model(
                    input_ids=generated[:, -1:], past_key_values=past,
                    attention_mask=torch.ones(1, generated.shape[1], dtype=torch.long, device="cuda"),
                    use_cache=True,
                )
        past = outputs.past_key_values
        generated = torch.cat([generated, outputs.logits[:, -1].argmax(dim=-1, keepdim=True)], dim=1)
    return generated


def check_cached_equivalence(repo):
    """The cache constraint, measured on this GPU: atol 1e-5 / rtol 1e-4.

    The local test runs in float32 on CPU. Speed work must not break it on the
    hardware that matters, so the same comparison runs here, in float32, on the
    card that just trained the model.
    """
    import yaml

    sys.path.insert(0, str(repo))
    from lligma.model import LLigMaConfig, LLigMaForCausalLM

    values = yaml.safe_load((repo / "configs" / "lligma_3_0_small.yaml").read_text(encoding="utf-8")) or {}
    values.pop("model_type", None)
    values["vocab_size"] = 256
    values["max_position_embeddings"] = 64
    torch.manual_seed(0)
    model = LLigMaForCausalLM(LLigMaConfig(**values)).to("cuda").eval()

    tokens = torch.arange(1, 9).unsqueeze(0).to("cuda")
    with torch.no_grad():
        reference = model(input_ids=tokens, attention_mask=torch.ones_like(tokens), use_cache=False).logits
        collected, past = [], None
        for step in range(tokens.shape[1]):
            outputs = model(
                input_ids=tokens[:, step:step + 1],
                attention_mask=torch.ones(1, step + 1, dtype=torch.long, device="cuda"),
                past_key_values=past, use_cache=True,
            )
            past = outputs.past_key_values
            collected.append(outputs.logits[:, -1])
    incremental = torch.stack(collected, dim=1)
    difference = (reference - incremental).abs().max().item()
    torch.testing.assert_close(incremental, reference, atol=1e-5, rtol=1e-4)
    print(f"CACHE CHECK on {torch.cuda.get_device_name(0)}: max|full - cached| = {difference:.3e} "
          f"(atol 1e-5) PASS", flush=True)
    del model
    torch.cuda.empty_cache()
    return difference


def main():
    repo = find_repo()
    print(f"code from {repo}", flush=True)
    install_dependencies(repo)
    print(f"gpu: {torch.cuda.get_device_name(0)} | bf16 report: {torch.cuda.is_bf16_supported()}", flush=True)

    prep = find_input("lligma-prep")
    valid_tokens = prep / "tokens" / "valid.bin"
    tokenizer_dir = prep / "tokenizer"

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True)
    vocab_size = len(tokenizer)
    config_path = repo / "configs" / "lligma_3_0_small.yaml"

    cache_difference = check_cached_equivalence(repo)

    print("mounted under /kaggle/input:", flush=True)
    for mount in sorted(Path("/kaggle/input").iterdir()):
        print(f"  {mount}", flush=True)
        for child in sorted(mount.iterdir())[:8]:
            print(f"    {child}", flush=True)

    old_dir = find_input("lligma-old-checkpoint")
    new_dir = find_input("lligma-train-fp16")
    print(f"looking for checkpoints under {old_dir} and {new_dir}", flush=True)
    print(f"  old glob: {sorted(Path(old_dir).rglob('checkpoint-*.pt'))}", flush=True)
    print(f"  new glob: {sorted(Path(new_dir).rglob('checkpoint-*.pt'))}", flush=True)
    old_checkpoint, old_step = latest_checkpoint(old_dir)
    new_checkpoint, new_step = latest_checkpoint(new_dir)
    if old_checkpoint is None or new_checkpoint is None:
        raise FileNotFoundError(f"missing checkpoints: old={old_checkpoint} new={new_checkpoint}")
    print(f"old checkpoint: {old_checkpoint} (step {old_step})", flush=True)
    print(f"new checkpoint: {new_checkpoint} (step {new_step})", flush=True)

    results = {}
    models = {}
    for label, checkpoint in (("old", old_checkpoint), ("new", new_checkpoint)):
        model = build_model(repo, config_path, vocab_size)
        step = load_into(model, checkpoint)
        model = model.to("cuda")
        loss, perplexity = evaluate(model, valid_tokens)
        results[label] = {"step": step, "loss": loss, "perplexity": perplexity}
        models[label] = model
        print(f"{label.upper()} step={step}: held-out loss={loss:.4f} perplexity={perplexity:.1f}", flush=True)
        torch.cuda.empty_cache()

    prompts = ["L'Italia è", "La storia di Roma"]
    samples = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to("cuda")
        pair = {}
        for label in ("old", "new"):
            text = tokenizer.decode(generate(models[label], ids)[0], skip_special_tokens=True)
            pair[label] = text
            print(f"SAMPLE[{label}] {prompt!r} -> {text!r}", flush=True)
        samples.append({"prompt": prompt, **pair})

    old_rate = OLD_RUN["tokens"] / OLD_RUN["seconds"]
    new_rate = results["new"]["step"] * 8192 / (1.5 * 3600)
    print("", flush=True)
    print("EQUAL WALL CLOCK (1.5 h each, same corpus, same config, same evaluation):", flush=True)
    print(f"  {OLD_RUN['label']:18s} step {results['old']['step']:5d}  {OLD_RUN['tokens']:>10,} tokens  "
          f"{old_rate:6.0f} tok/s  held-out ppl {results['old']['perplexity']:8.1f}", flush=True)
    print(f"  {'fp16 mixed':18s} step {results['new']['step']:5d}  {results['new']['step'] * 8192:>10,} tokens  "
          f"{new_rate:6.0f} tok/s  held-out ppl {results['new']['perplexity']:8.1f}", flush=True)
    print(f"  speed {new_rate / old_rate:.2f}x | perplexity {results['old']['perplexity'] / results['new']['perplexity']:.2f}x lower",
          flush=True)

    summary = {
        "old": {**results["old"], **OLD_RUN},
        "new": {**results["new"], "tokens": results["new"]["step"] * 8192, "seconds": 1.5 * 3600},
        "speedup": new_rate / old_rate,
        "perplexity_ratio": results["old"]["perplexity"] / results["new"]["perplexity"],
        "samples": samples,
        "evaluation": f"{BATCHES} held-out blocks of {SEQUENCE_LENGTH} tokens, fp16 autocast, same GPU",
        "cache_equivalence_max_diff": cache_difference,
    }
    (WORK / "equal_time.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("EQUAL_TIME " + json.dumps(summary), flush=True)
    print("COMPARE DONE", flush=True)


if __name__ == "__main__":
    main()
