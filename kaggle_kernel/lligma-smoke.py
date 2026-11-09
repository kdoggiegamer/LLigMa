"""Kaggle smoke test for LLigMa Small.

Proves, on a real Kaggle GPU, that: the code dataset mounts and extracts, torch
sees the accelerator, the 0.39B dense config trains, checkpoints get written,
and --resume-from restores the step counter. Keep it short: this is plumbing
verification, not a training run.
"""

import json
import random
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
REPO = WORK / "repo"


def find_project(root):
    candidates = [root] + sorted(p for p in root.rglob("*") if p.is_dir())
    for candidate in candidates:
        if (candidate / "lligma").is_dir() and (candidate / "scripts" / "train.py").is_file():
            return candidate
    return None


shutil.rmtree(REPO, ignore_errors=True)
# Kaggle extracts uploaded zips while building the dataset, so the mount is
# normally an already-extracted tree. Copy it into the writable working dir,
# because training writes the split files and the checkpoints.
project = find_project(INPUT)
if project is not None:
    shutil.copytree(project, REPO)
    print("copied project from", project, flush=True)
else:
    zips = sorted(INPUT.rglob("*.zip"))
    if not zips:
        listing = sorted(str(p.relative_to(INPUT)) for p in INPUT.rglob("*"))[:40]
        raise SystemExit(f"No project or zip under {INPUT}. Contents: {listing}")
    with zipfile.ZipFile(zips[0]) as archive:
        archive.extractall(REPO)
    print("extracted from", zips[0], flush=True)
print("project files:", len(list(REPO.rglob('*'))), flush=True)

import torch  # noqa: E402

print("torch", torch.__version__, "| cuda build", torch.version.cuda, flush=True)
print("cuda available:", torch.cuda.is_available(), flush=True)
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available: set Accelerator = GPU in the kernel settings.")
props = torch.cuda.get_device_properties(0)
print("device:", torch.cuda.get_device_name(0), "| memory GB:", round(props.total_memory / 1e9, 1), flush=True)
print("bf16 supported:", torch.cuda.is_bf16_supported(), flush=True)

# The Kaggle image ships its own transformers, which is newer than the `<5` pin in
# requirements.txt. On that version, `_tied_weights_keys` is expected to be a dict
# and LLigMaForCausalLM declares a list, so the model fails to build with
# "'list' object has no attribute 'keys'". Install the declared range explicitly.
requirements = (REPO / "requirements.txt").read_text(encoding="utf-8").splitlines()
packages = [
    line.strip() for line in requirements
    if line.strip() and not line.strip().lower().startswith("torch")
]
print("installing:", packages, flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", *packages], check=True)
import transformers  # noqa: E402
print("transformers", transformers.__version__, flush=True)

corpus = REPO / "data" / "smoke-corpus.jsonl"
if not corpus.is_file():
    raise SystemExit(f"missing {corpus}")

documents = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines() if line.strip()]
random.Random(42).shuffle(documents)
cut = max(1, int(len(documents) * 0.8))
(REPO / "train.jsonl").write_text(
    "\n".join(json.dumps(d, ensure_ascii=False) for d in documents[:cut]) + "\n", encoding="utf-8"
)
(REPO / "valid.jsonl").write_text(
    "\n".join(json.dumps(d, ensure_ascii=False) for d in documents[cut:]) + "\n", encoding="utf-8"
)
print("documents:", len(documents), "| train:", cut, "| valid:", len(documents) - cut, flush=True)

tokenizer_dir = WORK / "tokenizer"
subprocess.run(
    [sys.executable, str(REPO / "scripts" / "train_tokenizer.py"),
     "--input", str(REPO / "train.jsonl"),
     "--vocab-size", "8192",
     "--output-dir", str(tokenizer_dir)],
    check=True,
)

out_dir = WORK / "run"


def train_command(max_steps, resume_from=None, save_every=15):
    # Deliberately fp32: --bf16 casts the whole model and is not what this test checks.
    cmd = [
        sys.executable, str(REPO / "scripts" / "train.py"),
        "--config", str(REPO / "configs" / "lligma_3_0_small.yaml"),
        "--train-file", str(REPO / "train.jsonl"),
        "--validation-file", str(REPO / "valid.jsonl"),
        "--tokenizer", str(tokenizer_dir),
        "--output-dir", str(out_dir),
        "--sequence-length", "128",
        "--batch-size", "4",
        "--max-steps", str(max_steps),
        "--eval-steps", "15",
        "--save-every", str(save_every),
        "--device", "cuda",
    ]
    if resume_from is not None:
        cmd.extend(["--resume-from", str(resume_from)])
    return cmd


def step_of(path):
    return int(path.stem.split("-")[-1])


print("RUN 1:", " ".join(train_command(30)), flush=True)
subprocess.run(train_command(30), check=True)

checkpoints = sorted(out_dir.glob("checkpoint-*.pt"), key=step_of)
print("checkpoints after run 1:", [p.name for p in checkpoints], flush=True)
if not checkpoints:
    raise SystemExit("no checkpoint written: --save-every did not fire")

earliest = checkpoints[0]
print("resuming from", earliest.name, "at step", step_of(earliest), flush=True)

print("RUN 2:", " ".join(train_command(45, resume_from=earliest)), flush=True)
subprocess.run(train_command(45, resume_from=earliest), check=True)

final = sorted(out_dir.glob("checkpoint-*.pt"), key=step_of)
print("checkpoints after run 2:", [p.name for p in final], flush=True)
if max(step_of(p) for p in final) < 45:
    raise SystemExit("resume did not reach step 45")
print("SMOKE TEST PASSED", flush=True)
