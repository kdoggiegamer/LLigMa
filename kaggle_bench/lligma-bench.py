"""Measure where the time actually goes on this GPU, one variant at a time.

The previous training run logged 837 tokens/s on a T4: the model is 386.7M
parameters, so one token costs roughly 2.3 GFLOP of forward+backward, i.e. about
1.9 TFLOPS. A T4 has no bf16 tensor cores (Turing, sm75) and the fp32 path is
capped around 8 TFLOPS, so the fast path has to be fp16 with loss scaling.

Four variants, same weights shape, same data, same device, back to back so the
numbers are comparable:

  A  fp32         + legacy attention (manual matmul -> softmax -> matmul)
  B  fp16 autocast + legacy attention
  C  fp16 autocast + scaled_dot_product_attention
  D  C plus worker-loaded pinned batches

"Legacy" replaces the fused kernel with a faithful copy of the old maths, so the
comparison isolates the kernel change instead of comparing against a number from
a run on a different machine days ago. Then a profiler pass says which kernels
dominate when everything is as fast as this script can make it.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

WORK = Path("/kaggle/working")
SEQUENCE_LENGTH = 512
BATCH_SIZE = 4
WARMUP = 12
MEASURED = 24
VOCAB = 32768

ORIGINAL_SDPA = F.scaled_dot_product_attention


def legacy_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, **kwargs):
    """The pre-change attention maths, cost included: it materialises scores."""
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if attn_mask is not None:
        scores = scores + attn_mask
    elif is_causal:
        rows, columns = q.shape[-2], k.shape[-2]
        allowed = torch.ones(rows, columns, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~allowed, float("-inf"))
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    if dropout_p > 0:
        weights = torch.dropout(weights, dropout_p, True)
    return torch.matmul(weights, v)


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


def build_model(repo, seed=42):
    sys.path.insert(0, str(repo))
    import yaml
    from lligma.model import LLigMaConfig, LLigMaForCausalLM

    values = yaml.safe_load((repo / "configs" / "lligma_3_0_small.yaml").read_text(encoding="utf-8")) or {}
    values.pop("model_type", None)
    values["vocab_size"] = VOCAB
    values["max_position_embeddings"] = max(values.get("max_position_embeddings", 0), SEQUENCE_LENGTH)
    values["use_cache"] = False
    torch.manual_seed(seed)
    return LLigMaForCausalLM(LLigMaConfig(**values)).to("cuda")


class Blocks(torch.utils.data.Dataset):
    def __init__(self, path):
        self.tokens = np.memmap(path, dtype=np.uint16, mode="r")
        self.count = len(self.tokens) // SEQUENCE_LENGTH

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        start = index * SEQUENCE_LENGTH
        block = np.asarray(self.tokens[start:start + SEQUENCE_LENGTH], dtype=np.int64)
        return (torch.from_numpy(block),)


def train_step(model, optimizer, scaler, batch, use_fp16):
    with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
        loss = model(input_ids=batch, labels=batch, use_cache=False).loss
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    return loss.item()


def run_variant(label, repo, tokens_path, use_fp16, legacy, workers, profiler_steps=0):
    if legacy:
        F.scaled_dot_product_attention = legacy_sdpa
        torch.nn.functional.scaled_dot_product_attention = legacy_sdpa
    else:
        F.scaled_dot_product_attention = ORIGINAL_SDPA
        torch.nn.functional.scaled_dot_product_attention = ORIGINAL_SDPA

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_model(repo)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    dataset = Blocks(tokens_path)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        pin_memory=workers > 0,
        **({"num_workers": workers, "prefetch_factor": 4, "persistent_workers": True} if workers else {}),
    )
    stream = iter(loader)

    def next_batch():
        nonlocal stream
        try:
            (batch,) = next(stream)
        except StopIteration:
            stream = iter(loader)
            (batch,) = next(stream)
        return batch.to("cuda", non_blocking=True)

    losses = []
    for _ in range(WARMUP):
        losses.append(train_step(model, optimizer, scaler, next_batch(), use_fp16))
    torch.cuda.synchronize()

    if profiler_steps:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            for _ in range(profiler_steps):
                losses.append(train_step(model, optimizer, scaler, next_batch(), use_fp16))
            torch.cuda.synchronize()
        print(profile.key_averages().table(sort_by="self_cuda_time_total", row_limit=14), flush=True)

    started = time.perf_counter()
    for _ in range(MEASURED):
        losses.append(train_step(model, optimizer, scaler, next_batch(), use_fp16))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    tokens = MEASURED * BATCH_SIZE * SEQUENCE_LENGTH
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    result = {
        "label": label,
        "tok_per_s": tokens / elapsed,
        "step_seconds": elapsed / MEASURED,
        "peak_gb": peak,
        "first_loss": losses[0],
        "last_loss": losses[-1],
    }
    print(f"RESULT {label}: {result['tok_per_s']:.0f} tok/s | "
          f"{result['step_seconds']:.2f} s/step | peak {peak:.2f} GB | "
          f"loss {losses[0]:.4f} -> {losses[-1]:.4f}", flush=True)
    del model, optimizer, loader, dataset
    return result


def main():
    repo = find_repo()
    print(f"code from {repo}", flush=True)
    install_dependencies(repo)
    print(f"gpu: {torch.cuda.get_device_name(0)} capability {torch.cuda.get_device_capability(0)}", flush=True)
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}", flush=True)

    prep = find_input("lligma-prep")
    tokens_path = prep / "tokens" / "train.bin"
    print(f"tokens from {tokens_path}", flush=True)

    torch.backends.cudnn.benchmark = True
    results = []
    results.append(run_variant("A fp32 legacy", repo, tokens_path, False, True, 0))
    results.append(run_variant("B fp16 legacy", repo, tokens_path, True, True, 0))
    results.append(run_variant("C fp16 sdpa", repo, tokens_path, True, False, 0, profiler_steps=6))
    results.append(run_variant("D fp16 sdpa workers", repo, tokens_path, True, False, 2))

    baseline = results[0]["tok_per_s"]
    print("", flush=True)
    for result in results:
        print(f"{result['label']:24s} {result['tok_per_s']:8.0f} tok/s   x{result['tok_per_s'] / baseline:.2f}", flush=True)

    summary = {
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "results": results,
        "speedup_vs_fp32": results[-1]["tok_per_s"] / baseline,
    }
    (WORK / "bench.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("BENCH " + json.dumps(summary), flush=True)
    print("BENCH DONE", flush=True)


if __name__ == "__main__":
    main()
