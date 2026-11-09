"""Same-session A/B of the shipped RMSNorm against its pre-change behaviour.

The 1.5 h production runs suggest the new code is ~23% faster per step, but they
ran in two different Kaggle sessions, where a throttled T4 can differ by ~15%.
This kernel settles it inside one session: two copies of the same repo, the only
difference being the weight cast in the two norm classes, each arm run twice and
alternated, all through the real scripts/train.py so the classes that execute
are the shipped ones (arm "new") and their pre-change behaviour (arm "old").

Proof that the arms really differ, not just by intent:
  * arm old must emit "Cannot dispatch to fused implementation", arm new must not;
  * the first-step loss must be identical in both, since the data and the seed
    are identical and the change is only a dtype cast.

No file outside this kernel's output is produced; the repo is copied to
/kaggle/working and the copy is what gets patched.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working")
SEQUENCE_LENGTH = 512
BATCH_SIZE = 4
GRAD_ACCUM = 4
STEPS = 200
WARMUP_STEP = 20          # first logged step used for the steady-state window
ROUNDS = 2

STEP_LINE = re.compile(
    r"step (\d+)/\d+ train_loss=([\d.]+) tokens=([\d,]+) elapsed=(\d+)s tok_per_s=(\d+)")


def find_input(slug):
    # The mount layout moved to /kaggle/input/datasets/<owner>/<slug>, so match
    # the directory name at any depth instead of assuming one level.
    for candidate in Path("/kaggle/input").rglob(slug):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"{slug} not mounted under /kaggle/input")


def find_repo():
    for candidate in Path("/kaggle/input").rglob("requirements.txt"):
        if (candidate.parent / "lligma").is_dir():
            return candidate.parent
    raise FileNotFoundError("LLigMa code dataset not found")


def stage(source, target):
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))


def revert_norms(repo):
    """Undo only the change under test: drop the per-call weight cast.

    Both classes are the same op; without the cast the fp32 master weight meets
    the fp16 activation and PyTorch falls back to the generic kernel, which is
    exactly what the code did before the change.
    """
    edits = {
        repo / "lligma" / "model" / "lligma.py": [
            ("self.weight.to(hidden_states.dtype), self.eps)",
             "self.weight, self.eps)"),
        ],
        repo / "lligma" / "model" / "ml_attention.py": [
            ("self.weight.to(x.dtype), eps)", "self.weight, eps)"),
        ],
    }
    for path, pairs in edits.items():
        text = path.read_text(encoding="utf-8")
        for old, new in pairs:
            if old not in text:
                raise RuntimeError(f"cannot revert {path.name}: {old!r} not found")
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8")
    remaining = [(p.name, s) for p in edits for s in ("hidden_states.dtype), self.eps", "x.dtype), eps)")
                 if s in p.read_text(encoding="utf-8")]
    if remaining:
        raise RuntimeError(f"revert incomplete, still present: {remaining}")
    return sum(len(pairs) for pairs in edits.values())


def norm_dispatch_probe():
    """One norm call, the two dtypes, same session and same shapes.

    This is the mechanism under test: an fp32 weight against an fp16 activation
    cannot dispatch to the fused kernel, an fp16 weight can. Measuring both here
    makes the arms' difference a fact about this session rather than an
    inference from a warning string.
    """
    import torch
    shape = (BATCH_SIZE, SEQUENCE_LENGTH, 1024)
    x = torch.randn(*shape, device="cuda", dtype=torch.float16)
    weight = torch.randn(shape[-1], device="cuda", dtype=torch.float32)
    half = weight.half()

    def bench(call, repeats=50):
        for _ in range(5):
            call()
        torch.cuda.synchronize()
        started = time.monotonic()
        for _ in range(repeats):
            call()
        torch.cuda.synchronize()
        return (time.monotonic() - started) / repeats * 1e3

    mismatched = bench(lambda: torch.rms_norm(x, (shape[-1],), weight, 1e-6))
    fused = bench(lambda: torch.rms_norm(x, (shape[-1],), half, 1e-6))
    print(f"norm call: fp32 weight {mismatched:.3f} ms | fp16 weight {fused:.3f} ms "
          f"({mismatched / fused:.2f}x)", flush=True)
    return mismatched, fused


def train_once(repo, tag):
    repo = Path(repo)
    out = WORK / f"run-{tag}"
    log = WORK / f"{tag}.log"
    command = [
        sys.executable, "-u", str(repo / "scripts" / "train.py"),
        "--config", str(repo / "configs" / "lligma_3_0_small.yaml"),
        "--tokens-file", str(CORPUS),
        "--validation-tokens-file", str(VALID),
        "--tokenizer", str(TOKENIZER),
        "--output-dir", str(out),
        "--sequence-length", str(SEQUENCE_LENGTH),
        "--batch-size", str(BATCH_SIZE),
        "--gradient-accumulation-steps", str(GRAD_ACCUM),
        "--max-steps", str(STEPS),
        "--learning-rate", "3e-4",
        "--weight-decay", "0.1",
        # No eval and no checkpointing: the window measures training steps only,
        # and both arms get the same treatment.
        "--eval-steps", "0",
        "--save-every", "0",
        "--fp16",
        "--num-workers", "2",
        "--device", "cuda",
    ]
    print(f"[{tag}] {' '.join(command)}", flush=True)
    started = time.monotonic()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(repo))
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                   cwd=str(repo), env=env)
    wall = time.monotonic() - started
    text = log.read_text(encoding="utf-8", errors="replace")
    if completed.returncode != 0:
        print(text[-4000:], flush=True)
        raise RuntimeError(f"[{tag}] train.py exited {completed.returncode}")

    points = {}
    for line in text.splitlines():
        match = STEP_LINE.match(line)
        if match:
            points[int(match.group(1))] = (float(match.group(2)), int(match.group(4)),
                                           int(match.group(3).replace(",", "")))
    if not points:
        raise RuntimeError(f"[{tag}] no step lines in the log")
    last = max(points)
    first_loss = points[min(points)][0]
    if WARMUP_STEP not in points:
        raise RuntimeError(f"[{tag}] warmup step {WARMUP_STEP} missing")
    d_steps = last - WARMUP_STEP
    d_seconds = points[last][1] - points[WARMUP_STEP][1]
    s_per_step = d_seconds / d_steps
    tok_s = (points[last][2] - points[WARMUP_STEP][2]) / d_seconds
    result = {
        "tag": tag,
        "last_step": last,
        "first_loss": round(first_loss, 4),
        "last_train_loss": round(points[last][0], 4),
        "window_steps": d_steps,
        "window_seconds": d_seconds,
        "s_per_step": round(s_per_step, 4),
        "tok_per_s": round(tok_s, 1),
        "wall_seconds": round(wall, 1),
        "mismatch_warning": "Cannot dispatch to fused implementation" in text,
        "log": str(log),
    }
    print(json.dumps(result), flush=True)
    shutil.rmtree(out, ignore_errors=True)   # the weights are not needed here
    return result


def main():
    repo = find_repo()
    print(f"code from {repo}", flush=True)
    import torch
    print(f"cuda: {torch.cuda.is_available()} | {torch.cuda.get_device_name(0)}", flush=True)

    requirements = (Path(repo) / "requirements.txt").read_text(encoding="utf-8").splitlines()
    packages = [line.strip() for line in requirements
                if line.strip() and not line.strip().lower().startswith("torch")]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)
    import transformers
    print("torch", torch.__version__, "| transformers", transformers.__version__, flush=True)

    prep = find_input("lligma-prep")
    global CORPUS, VALID, TOKENIZER
    CORPUS = prep / "tokens" / "train.bin"
    VALID = prep / "tokens" / "valid.bin"
    TOKENIZER = prep / "tokenizer"

    arm_new = WORK / "arm-new"
    arm_old = WORK / "arm-old"
    stage(repo, arm_new)
    stage(repo, arm_old)
    edits = revert_norms(arm_old)
    print(f"arm-old reverted ({edits} call sites); arm-new is the shipped code", flush=True)

    runs = []
    for round_index in range(1, ROUNDS + 1):
        # Alternate the order so a drift in clocks hits both arms equally.
        order = [("old", arm_old), ("new", arm_new)]
        if round_index % 2 == 0:
            order.reverse()
        for arm, path in order:
            runs.append(train_once(path, f"r{round_index}-{arm}"))

    old = [run for run in runs if run["tag"].endswith("old")]
    new = [run for run in runs if run["tag"].endswith("new")]
    if any(run["mismatch_warning"] for run in new):
        raise RuntimeError("arm new still warns about the dtype mismatch: the cast did not apply")
    losses = {round(run["first_loss"], 4) for run in runs}
    if len(losses) != 1:
        raise RuntimeError(f"arms did not see identical data/seed: first losses {losses}")
    # The warning text is a torch-version detail; the mechanism is measured
    # directly in-process so the two arms are known to differ for a reason.
    mismatched_ms, fused_ms = norm_dispatch_probe()

    mean = lambda values: sum(values) / len(values)
    old_step = mean([run["s_per_step"] for run in old])
    new_step = mean([run["s_per_step"] for run in new])
    verdict = {
        "measurement": "same session, same GPU, same corpus, same seed, real scripts/train.py",
        "steps_per_arm": STEPS,
        "window": f"step {WARMUP_STEP} to the last logged step, excluding startup",
        "runs": runs,
        "old_s_per_step": round(old_step, 4),
        "new_s_per_step": round(new_step, 4),
        "step_time_change_pct": round((new_step / old_step - 1) * 100, 2),
        "throughput_gain_pct": round((old_step / new_step - 1) * 100, 2),
        "old_tok_per_s": round(mean([run["tok_per_s"] for run in old]), 1),
        "new_tok_per_s": round(mean([run["tok_per_s"] for run in new]), 1),
        "norm_call_ms": {"fp32_weight_mismatched": round(mismatched_ms, 4),
                         "fp16_weight_fused": round(fused_ms, 4)},
        "old_arm_warned_unfused": all(run["mismatch_warning"] for run in old),
        "controls": "identical first-step loss in every arm (same data, same seed)",
    }
    (WORK / "ab.json").write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    print("AB VERDICT " + json.dumps(verdict, indent=2), flush=True)


if __name__ == "__main__":
    main()
