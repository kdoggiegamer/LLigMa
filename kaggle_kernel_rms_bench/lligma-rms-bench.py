"""Measure what the non-fused RMSNorm path costs in fp16, then decide.

Three checkpoints, one short GPU session:

1. Dispatch. The fp16 run warned "Mismatch dtype between input and weight ...
   Cannot dispatch to fused implementation" for `torch.rms_norm`. We reproduce
   the warning with the current (fp32 weight, fp16 input) call and show the
   aligned call (fp16 weight cast per call) produces none - dispatch to the
   fused kernel is the whole point of the change.
2. Microbenchmark. Real shapes of one micro-step (batch 4, sequence 512):
   hidden [4,512,1024] twice per layer, q-lora [4,512,256] and kv-lora
   [4,512,128] once, plus the final [1024] norm: 4 norms x 24 layers + 1.
   Variants: the manual fp32 path the model uses today, the mismatched
   `nn.RMSNorm` with fp32 weight, and the aligned fused call. Forward, and
   forward+backward, which is what training actually pays.
3. A/B end to end. Real training steps in autocast fp16 with GradScaler, AdamW,
   clipping, grad-accum 4 - identical seed, identical batches. Arm base runs
   the mounted code untouched; arm fused replaces each RMSNorm instance's
   forward with the aligned call. Weights stay fp32 master weights owned by the
   optimizer either way: nothing changes in the checkpoint format or the state
   dict, only a [n] vector is cast at call time next to a [B,S,n] tensor.

No patch is shipped here. The verdict is a measurement, nothing more.
"""

import gc
import json
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

WORK = Path("/kaggle/working")
SEQUENCE_LENGTH = 512
BATCH_SIZE = 4
GRAD_ACCUM = 4
TOKENS_PER_STEP = BATCH_SIZE * SEQUENCE_LENGTH * GRAD_ACCUM
STEPS = 110          # optimizer steps per arm
WARMUP = 10          # discarded before timing
NUM_LAYERS = 24
EPS = 1e-6
# 4 norms per decoder layer + the final model norm.
NORMS_PER_STEP = 4 * NUM_LAYERS + 1


def find_input(slug):
    direct = Path(f"/kaggle/input/{slug}")
    if direct.is_dir():
        return direct
    for candidate in Path("/kaggle/input").glob(f"*/{slug}"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"{slug} not mounted under /kaggle/input")


def find_repo():
    for candidate in Path("/kaggle/input").rglob("requirements.txt"):
        if (candidate.parent / "lligma").is_dir():
            return candidate.parent
    raise FileNotFoundError("LLigMa code dataset not found")


def install_dependencies(repo):
    import subprocess
    requirements = (repo / "requirements.txt").read_text(encoding="utf-8").splitlines()
    packages = [line.strip() for line in requirements
                if line.strip() and not line.strip().lower().startswith("torch")]
    print("installing:", packages, flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)
    import transformers
    print("torch", torch.__version__, "| transformers", transformers.__version__, flush=True)


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


def fused_forward(self, x):
    """Alignment at call time: fp32 master weight down-cast to the activation dtype.

    The fused kernel needs input and weight of the same dtype. Casting the
    [n] weight vector per call is negligible next to the [B,S,n] activation,
    and the kernel itself reduces in fp32, so numerics are preserved within
    fp16 rounding.
    """
    return torch.rms_norm(x, x.shape[-1:], self.weight.to(x.dtype),
                          getattr(self, "variance_epsilon", self.eps))


# --- 1. dispatch -----------------------------------------------------------

def count_mismatch_warnings(make_call):
    """How many 'Mismatch dtype' warnings the call emits (0 = fused dispatch)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make_call()
    return sum("Mismatch dtype" in str(item.message) for item in caught)


def check_dispatch(x16):
    """x16: [B,S,H] fp16 tensor with requires_grad for the backward pass."""
    weight32 = torch.ones(x16.shape[-1], device="cuda", requires_grad=True)
    weight16 = torch.ones(x16.shape[-1], device="cuda", dtype=torch.float16, requires_grad=True)

    mismatched = count_mismatch_warnings(
        lambda: F.rms_norm(x16.detach(), x16.shape[-1:], weight32, EPS))
    aligned = count_mismatch_warnings(
        lambda: torch.rms_norm(x16.detach(), x16.shape[-1:], weight16, EPS))
    return {"mismatched_weight_warnings": mismatched,
            "aligned_weight_warnings": aligned,
            "fused_expected_current": mismatched > 0,
            "fused_ok_after_alignment": aligned == 0}


# --- 2. microbenchmark -----------------------------------------------------

def manual_rms_norm(x, weight, eps):
    """Exactly the RMSNorm.forward the model uses today (fp32 detour)."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return weight * x.to(input_dtype)


def time_variant(fn, iters=100, warmup=20, backward=True):
    for _ in range(warmup):
        out = fn()
        if backward:
            out.sum().backward()
    torch.cuda.synchronize()
    start = time.monotonic()
    for _ in range(iters):
        out = fn()
        if backward:
            out.sum().backward()
    torch.cuda.synchronize()
    return (time.monotonic() - start) / iters * 1000.0  # ms per call


def check_numerics():
    x32 = torch.randn(4, 512, 1024, device="cuda", dtype=torch.float32)
    w32 = torch.randn(1024, device="cuda", dtype=torch.float32)
    fused32 = torch.rms_norm(x32, x32.shape[-1:], w32, EPS)
    manual32 = manual_rms_norm(x32, w32, EPS)
    diff32 = (fused32 - manual32).abs().max().item()

    x16 = torch.randn(4, 512, 1024, device="cuda", dtype=torch.float16)
    w16 = torch.randn(1024, device="cuda", dtype=torch.float16)
    fused16 = torch.rms_norm(x16, x16.shape[-1:], w16, EPS)
    manual16_vs_w16 = manual_rms_norm(x16, w16.float(), EPS)
    diff16 = (fused16 - manual16_vs_w16).abs().max().item()
    return {"fp32_fused_vs_manual_max_abs": diff32,
            "fp16_fused_vs_manual_max_abs": diff16}


def microbench():
    shapes = {"hidden_1024": 1024, "q_lora_256": 256, "kv_lora_128": 128}
    table = {}
    for name, size in shapes.items():
        x16 = torch.randn(4, 512, size, device="cuda", dtype=torch.float16)
        w32 = torch.ones(size, device="cuda", requires_grad=True)
        w16 = torch.ones(size, device="cuda", dtype=torch.float16, requires_grad=True)

        def manual_call():
            out = manual_rms_norm(x16, w32, EPS)
            return out

        def mismatched_call():
            return F.rms_norm(x16, x16.shape[-1:], w32, EPS)

        def fused_call():
            return torch.rms_norm(x16, x16.shape[-1:], w16, EPS)

        table[name] = {}
        for label, fn in (("manual", manual_call), ("mismatched", mismatched_call),
                          ("fused", fused_call)):
            table[name][label] = {
                "forward_ms": time_variant(fn, backward=False),
                "fwd_bwd_ms": time_variant(fn),
            }
        del x16, w32, w16
    return table


def summarise_microbench(table):
    """Steady-state cost of one optimizer step's norms, per variant.

    2x hidden + 1x q_lora + 1x kv_lora per layer, plus the final norm.
    """
    index = {"hidden_1024": 2 * NUM_LAYERS + 1, "q_lora_256": NUM_LAYERS,
             "kv_lora_128": NUM_LAYERS}
    totals = {}
    for label in ("manual", "mismatched", "fused"):
        totals[label] = sum(table[name][label]["fwd_bwd_ms"] * index[name]
                            for name in index) * GRAD_ACCUM
    savings = {
        "manual_minus_fused_ms_per_step": totals["manual"] - totals["fused"],
        "mismatched_minus_fused_ms_per_step":
            totals["mismatched"] - totals["fused"],
        "manual_share_of_step": totals["manual"] / 2325.0,  # 2.33 s/step measured
    }
    return {"per_call_ms": table, "ms_per_optimizer_step": totals,
            "savings_vs_2.33s_step": savings}


# --- 3. end-to-end A/B -----------------------------------------------------

def align_norms(model):
    from lligma.model.lligma import RMSNorm as ManualRMSNorm
    patched = 0
    for module in model.modules():
        if isinstance(module, (ManualRMSNorm, torch.nn.RMSNorm)):
            module.forward = types.MethodType(fused_forward, module)
            patched += 1
    return patched


def train_arm(repo, config_path, vocab, vocab_size, aligned, label):
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    _, model = build_model(repo, config_path, vocab_size)
    model = model.to("cuda")
    model.train()
    patched = align_norms(model) if aligned else 0

    generator = torch.Generator().manual_seed(7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    scaler = torch.amp.GradScaler("cuda")

    losses, times = [], []
    started = time.monotonic()
    for step in range(1, STEPS + 1):
        optimizer.zero_grad(set_to_none=True)
        step_start = time.monotonic()
        loss_value = 0.0
        for _ in range(GRAD_ACCUM):
            batch = torch.randint(0, vocab, (BATCH_SIZE, SEQUENCE_LENGTH), generator=generator)
            batch = batch.to("cuda")
            with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                loss = model(input_ids=batch, labels=batch, use_cache=False).loss
            scaler.scale(loss / GRAD_ACCUM).backward()
            loss_value += loss.item() / GRAD_ACCUM
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss_value)
        times.append(time.monotonic() - step_start)

    elapsed = time.monotonic() - started
    timed = times[WARMUP:]
    steady_s = sum(timed) / len(timed)
    skip = (STEPS - WARMUP) // 2
    second_half_s = sum(timed[skip:]) / (STEPS - WARMUP - skip)
    peak_logit = losses[0], losses[-1]
    print(f"{label}: patched_norms={patched} warmup_first={losses[0]:.4f} "
          f"last={losses[-1]:.4f} steady={steady_s * 1000:.1f} ms/optimizer-step "
          f"({TOKENS_PER_STEP / steady_s:.0f} tok/s) second-half={second_half_s * 1000:.1f} ms", flush=True)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"patched_norms": patched, "first_loss": peak_logit[0], "last_loss": peak_logit[1],
            "steady_ms_per_step": steady_s * 1000.0, "second_half_ms_per_step": second_half_s * 1000.0,
            "steady_tok_per_s": TOKENS_PER_STEP / steady_s,
            "losses": losses, "total_seconds": elapsed}


def main():
    if not torch.cuda.is_available():
        raise SystemExit("this benchmark needs a CUDA GPU")
    torch.cuda.init()
    print(f"device: {torch.cuda.get_device_name(0)}", flush=True)

    repo = find_repo()
    install_dependencies(repo)
    config_path = repo / "configs" / "lligma_3_0_small.yaml"

    # Only the vocabulary size matters here: real tokens change nothing in a
    # speed or numerics measurement, so no tokenizer file is mounted at all.
    # (lligma-prep is a kernel source and skipped a mount in v1, which cost a
    # whole session to find out.)
    vocab_size = 32768
    print(f"vocabulary: {vocab_size}", flush=True)

    verdict = {"device": torch.cuda.get_device_name(0),
               "torch": torch.__version__,
               "vocabulary": vocab_size,
               "tokens_per_step": TOKENS_PER_STEP}

    x_probe = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, 1024, device="cuda",
                          dtype=torch.float16, requires_grad=True)
    verdict["dispatch"] = check_dispatch(x_probe)
    del x_probe
    verdict["numerics"] = check_numerics()
    print("dispatch:", verdict["dispatch"], flush=True)
    print("numerics:", verdict["numerics"], flush=True)

    verdict["microbench"] = summarise_microbench(microbench())
    savings = verdict["microbench"]["savings_vs_2.33s_step"]
    print("microbench per optimizer step (fwd+bwd):",
          {k: round(v, 2) for k, v in verdict["microbench"]["ms_per_optimizer_step"].items()}, flush=True)
    print("savings vs measured 2.33 s/step:", {k: round(v, 4) for k, v in savings.items()}, flush=True)

    base = train_arm(repo, config_path, vocab_size, vocab_size, aligned=False, label="BASE  ")
    fused = train_arm(repo, config_path, vocab_size, vocab_size, aligned=True, label="FUSED ")

    drift = [abs(a - b) for a, b in zip(base["losses"], fused["losses"])]
    verdict["ab"] = {
        "base": {k: v for k, v in base.items() if k != "losses"},
        "fused": {k: v for k, v in fused.items() if k != "losses"},
        "max_abs_loss_drift": max(drift),
        "mean_abs_loss_drift": float(np.mean(drift)),
        "speedup_steady": base["steady_tok_per_s"] / fused["steady_tok_per_s"],
        "speedup_steady_second_half":
            base["second_half_ms_per_step"] / fused["second_half_ms_per_step"],
    }
    print(f"A/B: base {verdict['ab']['base']['steady_tok_per_s']:.0f} tok/s vs "
          f"fused {verdict['ab']['fused']['steady_tok_per_s']:.0f} tok/s  "
          f"({verdict['ab']['speedup_steady']:.3f}x) | max |dloss| "
          f"{verdict['ab']['max_abs_loss_drift']:.4f}", flush=True)

    verdict["norms_per_optimizer_step"] = NORMS_PER_STEP
    (WORK / "verdict.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print("VERDICT " + json.dumps(verdict), flush=True)
    print("BENCH DONE", flush=True)


if __name__ == "__main__":
    main()
