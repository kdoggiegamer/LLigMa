# Renting a GPU for LLigMa

One training session on a rented box, from a cold machine to checkpoints that survive
the instance. Written for the spot instance: assume it can disappear without warning.

## Budget

Reference price: **$1.99/h** (GH200, 96 GB). Monthly allowance **$255**.

| limit | value | why |
| --- | --- | --- |
| hours per month | 128 | 255 / 1.99 |
| hours per session | 6 | ~$12 per session |
| hours per week | 30 | ~$60, leaves $16 of slack per month |

`--max-hours` enforces the session cap in the script itself: it stops at a step
boundary, writes a resumable checkpoint, and exits. Do not rely on remembering to stop.

## 1. Copy the repository

There is no remote, so copy the directory. Exclude the virtualenvs, previous outputs and
scratch folders or you will upload gigabytes of nothing:

```bash
rsync -av --progress \
  --exclude '.venv*' --exclude 'outputs' --exclude '.tmp-*' \
  --exclude '__pycache__' --exclude '.kaggle-staging' \
  ./ user@host:~/lligma/
```

## 2. First ten minutes: prove the GPU works

Do this **before** downloading a dataset. A box whose driver is broken should cost you
five minutes, not an hour.

```bash
nvidia-smi                      # driver + card + CUDA version
python -m venv .venv && . .venv/bin/activate
pip install torch               # on aarch64 (GH200) this is the PyPI wheel
```

If `torch.cuda.is_available()` is `False`, the wheel is a CPU build: reinstall from the
CUDA index that matches the version printed by `nvidia-smi`, e.g.
`pip install torch --index-url https://download.pytorch.org/whl/cu129`.

```bash
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("avail", torch.cuda.is_available(), "device", torch.cuda.get_device_name(0))
a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
b = a @ a
torch.cuda.synchronize()
print("matmul ok", round(b.float().abs().mean().item(), 4))
PY
```

Then the project dependencies and the model itself:

```bash
pip install -r requirements.txt
python -c "
import sys, yaml, torch
sys.path.insert(0, '.')
from lligma.model import LLigMaConfig, LLigMaForCausalLM
values = yaml.safe_load(open('configs/lligma_3_0_small.yaml'))
values.pop('model_type', None)
model = LLigMaForCausalLM(LLigMaConfig(**values)).to('cuda')
print('params', sum(p.numel() for p in model.parameters()))
print('forward ok', model(input_ids=torch.randint(0, 32768, (1, 8), device='cuda')).logits.shape)
"
```

## 3. Build the corpus and the token stream

**Do not run this on the rented box unless you have to.** Tokenising is CPU work and you
would be paying GPU prices for it. It is built once by the `lligma-prep` kernel on Kaggle
(free CPU, `kaggle_kernel_prep/`) and pulled from there:

```bash
pip install kaggle          # needs ~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY
kaggle kernels output angelodellamorte/lligma-prep -p "$HOME/work"
```

That leaves the paid hours for training only. The script below is the fallback: it
downloads Italian Wikipedia, filters it, trains a 32k BPE tokenizer and writes the
`uint16` token stream that training reads. On Kaggle it is forced to `/kaggle/working`;
`LLIGMA_WORK` overrides that here.

```bash
LLIGMA_WORK=$HOME/work python kaggle_kernel_prep/lligma-prep.py
```

Expected outputs, all under `$HOME/work`:

```
corpus/train.parquet  corpus/valid.parquet   documents + CC BY-SA attribution
tokenizer/                                   tokenizer.json
tokens/train.bin  tokens/valid.bin           uint16 stream, EOS between documents
stats.json                                   documents and token counts
```

This is CPU work: it uses the box's cores, not the GPU, but you are still billed for it.
Run it once per session and keep the outputs, rather than rebuilding them every time.

## 4. Train

```bash
python scripts/train.py \
  --config configs/lligma_3_0_small.yaml \
  --tokens-file "$HOME/work/tokens/train.bin" \
  --validation-tokens-file "$HOME/work/tokens/valid.bin" \
  --tokenizer "$HOME/work/tokenizer" \
  --output-dir "$HOME/work/outputs/run1" \
  --sequence-length 1024 \
  --batch-size 16 \
  --gradient-accumulation-steps 2 \
  --learning-rate 3e-4 \
  --bf16 \
  --max-steps 100000 \
  --max-hours 6 \
  --save-every 500 \
  --keep-checkpoints 3 \
  --num-workers 4
```

`--max-steps` is deliberately larger than what six hours can reach: the real stop is
`--max-hours`. The learning rate is the one validated on the Kaggle runs; the effective
token batch here (16 x 1024 x 2 = 32k tokens) is four times larger, so `4e-4` is a
reasonable next experiment, not the first one.

## 5. Resume after a preemption

Spot instances die without warning. Any checkpoint is a valid restart point:

```bash
python scripts/train.py ... (same command) \
  --resume-from "$HOME/work/outputs/run1/checkpoint-000500.pt" \
  --max-steps 100000
```

`--max-steps` must be greater than the step count stored in the checkpoint, or the script
stops immediately. That is a guard, not a bug.

## 6. Before you stop the instance

The instance disk disappears with the instance. In this order:

1. Copy `outputs/run1/*.pt` off the box (`rsync` or object storage).
2. Copy `tokenizer/` and `stats.json`: they are what later runs and the GGUF export need.
3. Recompute `tokens/train.bin` only if you still have the corpus; otherwise copy it too.
4. Check the sizes match what the log reported, then stop the instance.

## 7. Failure modes worth recognising

| symptom | cause | fix |
| --- | --- | --- |
| `cuda avail False` | CPU wheel (common on aarch64) | reinstall from the CUDA index |
| CUDA OOM | batch too large for 96 GB | halve `--batch-size`, double the accumulation steps |
| first steps very slow | kernel selection and allocator warm-up | ignore the first ~20 steps when reading tok/s |
| loss NaN | learning rate too high for the batch | back to `3e-4` before changing anything else |
| step count already past `--max-steps` | resumed checkpoint from a longer run | pass a larger `--max-steps` |
