# LLigMa 3.0

LLigMa 3.0 is an experimental decoder-only language model implementation with latent attention, rotary embeddings, and a routed mixture-of-experts feed-forward stack. The repository contains architecture code and a small training pipeline; it does not contain pretrained weights or evidence of reasoning performance.

## LLigMa Tutor (school assistant)

[`lligma/tutor/`](lligma/tutor/) is a self-contained Italian question-answering
service for an agricultural secondary school: agronomy, biology and Italian
general knowledge. It is deliberately *not* the from-scratch model above — a
0.4B model cannot hold a curriculum — but retrieval over a curated scheda set,
plus a distilled sub-1B student on top. See [docs/TUTOR.md](docs/TUTOR.md) (in
Italian) for the design rationale, the training recipe and the measured limits.

```bash
python scripts/build_tutor_index.py       # 34 schede -> 78 passages, 108 KB index
python -m lligma.tutor.server             # http://127.0.0.1:8080
python -m lligma.tutor.server --backend extractive   # no model at all
```

Measured on the shipped corpus with the extractive reference backend: recall@3
100% (30/30), keyword coverage 1.00 (30/30), abstention 4/4 on out-of-scope
questions, about 0.5 ms mean latency on CPU. Both the no-key `/api/ask` and
`/api/stream` paths are covered by `tests/test_tutor_server_e2e.py`. The
paraphrase set `data/tutor/eval-paraphrase.jsonl` measures the synonym expansion
(0/39 recall without it, 39/39 with it); the shipped configuration enables it,
because with it off a reworded in-scope question is refused instead of answered.
`python scripts/evaluate_tutor.py --compare` prints the before/after report for
both evaluation sets in one run.
The web client's history, interrupted-stream and error paths are covered by
`tests/test_tutor_web_client.py`. The service depends only on `PyYAML`; the
optional student model needs `torch`, `transformers`, `peft` and `bitsandbytes`.

## Alessandria (local)

Alessandria is the local inference playground for the from-scratch checkpoint.
It lives in `web/model/` and uses a standard-library HTTP/SSE server:

```bash
.venv/Scripts/python.exe -m lligma.model.server \
  --model outputs/lligma-small-smoke --port 8081
```

Open `http://127.0.0.1:8081/`. The page has two modes:

- **Tutor + LLigMa** asks the Tutor to retrieve passages, shows the extractive
  answer as the reliable baseline, and sends the numbered context to the
  checkpoint as a rewriter. It calls `POST /api/tutor/stream` with
  `{"question": "..."}`. The SSE metadata includes `sources`, `baseline`,
  `retrieval_ms`, `baseline_ms`, `retrieved`, `context_chars` and `abstained`.
  When the Tutor abstains, the checkpoint is not called at all. The UI shows the
  baseline, the cited sources, and both the Tutor time and the generation
  metrics; TTFT and total time include retrieval and baseline construction.
- **Solo completamento** keeps the raw `POST /api/stream` path.

The page shows the streamed completion, total response time, time to first
token, generated tokens and tokens/s; the measurements come from the server.
The loader detects the checkpoint's missing `lm_head.weight` and enables tied
embeddings, so the output head is not initialized randomly. The included
checkpoint is a smoke model, not an instruction-tuned chat model, so its
completions are expected to be incoherent; the extractive baseline is the
verified answer.

## Model size

The default configuration is **not a 2B-parameter model**. With the current dimensions, its routed experts alone contribute roughly 15B parameters across the MoE layers; total size is roughly 16B parameters, while the active parameters per token are also above 2B. Actual counts can vary with weight tying and configuration. It is not suitable for training from scratch on ordinary consumer hardware. Use the deliberately small debug preset for smoke training and code experiments; it is not representative of the advertised scale.

## Small dense preset (0.39B)

[`configs/lligma_3_0_small.yaml`](configs/lligma_3_0_small.yaml) is a dense preset sized to train on a single consumer GPU or a free Kaggle accelerator. Verified parameter count: **386,721,792** (0.39B) with a 32K vocabulary; the training script's estimate matches the instantiated model exactly.

| Field | Value | Follows |
| --- | --- | --- |
| `hidden_size` / `num_hidden_layers` | 1024 / 24 | Llama-3.2-1B aspect ratio |
| `intermediate_size` | 4096 | Qwen2.5-0.5B range (4x hidden) |
| Attention | MLA, `q_lora_rank` 256, `kv_lora_rank` 128 | DeepSeek-V2/V3 |
| `use_moe` | `false` | the removed experts are what made the default preset ~16B |
| `tie_word_embeddings` | `true` | GPT-2 / Gemma; saves `vocab_size * hidden_size` |

With tied embeddings and a 32K vocabulary the embedding is 33.5M parameters, so the vocabulary is not a rounding error at this scale. Train a tokenizer on your own corpus instead of reusing an English-oriented 50K/128K one:

```bash
python scripts/train_tokenizer.py \
  --input data/italian/train.parquet \
  --vocab-size 32768 \
  --output-dir tokenizers/lligma-small-32k
```

Then train (the tokenizer's real length overrides `vocab_size` in the config):

```bash
python scripts/train.py \
  --config configs/lligma_3_0_small.yaml \
  --train-file data/italian/train.parquet \
  --validation-file data/italian/valid.parquet \
  --tokenizer tokenizers/lligma-small-32k \
  --output-dir outputs/lligma-small \
  --sequence-length 1024 \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --device cuda
```

A 0.39B model needs roughly 6.5 GB for weights, gradients and Adam states before activations, so it fits a 16 GB card at short sequence lengths. It does **not** fit in the guide below for the 16B preset.

### GPU note (AMD RDNA4 on Windows)

On a Windows machine with an AMD Radeon RX 9060 XT, PyTorch from PyPI is a CPU-only build and cannot use the card. The ROCm nightly wheels for Windows (`https://rocm.nightlies.amd.com/v2/gfx120X-all/`) install and report the device correctly, but plain `torch.matmul` segfaults inside `libhipblaslt.dll` (`hipblasLtMatmulAlgoGetHeuristic`) on both `rocm7.13.0a20260416` and `rocm7.11.0a20251215`. Treat local ROCm on Windows as unusable for now.

### Two runnable paths

**[`notebooks/kaggle_train_lligma_small.ipynb`](notebooks/kaggle_train_lligma_small.ipynb) — recommended.** A Linux GPU host with a working CUDA stack, so no driver archaeology. Practical limits: 12 hours per GPU session, about 30 hours per week, and only `/kaggle/working` persists, and only once the notebook is committed. Keep the output under 20 GB.

**[`docker/`](docker/README.md) — experimental.** ROCm inside a container on WSL2, passing the `/dev/dxg` bridge because WSL2 has no `/dev/kfd`. This is a community path: Docker Desktop's GPU passthrough is documented for NVIDIA only. Run the smoke test in that README before any training, and stop if it segfaults in `libhipblaslt`.

### Checkpoints and resuming

Long runs do not finish in one session, so both paths rely on the same flags:

- `--save-every N` writes `checkpoint-{step}.pt` holding model, optimizer state and the step counter.
- `--resume-from PATH` restores all three and continues from that step. `--max-steps` must exceed the resumed step.

```bash
python scripts/train.py --config configs/lligma_3_0_small.yaml ... \
  --save-every 500 --resume-from outputs/lligma-small/checkpoint-005000.pt --max-steps 10000
```

One limitation to know before scaling up: `make_blocks` in `scripts/train.py` packs an entire split into memory before training. A corpus of a few hundred million tokens is fine; a multi-billion-token corpus needs a streaming dataset instead.

## Install

Install a PyTorch build appropriate for your platform, then install the Python requirements:

```bash
python -m pip install -r requirements.txt
```

## Train on WikiText-2

Train and evaluate on the dataset already downloaded under `data/wikitext-2-raw-v1/`:

```bash
python scripts/train.py \
  --config configs/lligma_3_0_debug.yaml \
  --train-file data/wikitext-2-raw-v1/train-00000-of-00001.parquet \
  --validation-file data/wikitext-2-raw-v1/validation-00000-of-00001.parquet \
  --test-file data/wikitext-2-raw-v1/test-00000-of-00001.parquet \
  --output-dir outputs/lligma-wikitext2-debug \
  --sequence-length 64 \
  --max-steps 100 \
  --eval-steps 50 \
  --max-eval-examples 64 \
  --device cpu
```

This command reproduces the saved smoke checkpoint in [outputs/lligma-wikitext2-debug/README.md](outputs/lligma-wikitext2-debug/README.md). The script also supports loading a dataset directly from the Hugging Face Hub. `--max-steps` counts optimizer updates. Use `--max-train-examples` and `--max-eval-examples` to cap rows for quick experiments. WikiText-2 is small and is only appropriate for a pipeline smoke run, not for training a capable general-purpose or reasoning model. Dataset source and license information are in [data/wikitext-2-raw-v1/README.md](data/wikitext-2-raw-v1/README.md).

## Project contents

- `lligma/model/`: configuration, transformer layers, latent attention, MoE, and RoPE
- `lligma/inference/`: generation helper
- `scripts/train.py`: Hub/local dataset loading, tokenization, training, validation, and checkpoint saving
- `configs/`: default architecture and small debug training preset
- `data/wikitext-2-raw-v1/`: downloaded train, validation, and test splits

The model is randomly initialized by the training script. No pretrained checkpoint or benchmark result is included.

`scripts/train_tokenizer.py` trains a ByteLevel BPE tokenizer on local `.txt`, `.jsonl`, or `.parquet` files and reports its compression ratio. On a small Italian Wikipedia sample it reached 4.14 characters per token at an 8K vocabulary, against roughly 3.3 for the GPT-2 tokenizer.
