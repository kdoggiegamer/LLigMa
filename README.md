# LLigMa

LLigMa is an experimental personal language-model project. This repository currently distributes version archives and project documentation.

## Available versions

| Version | Status | Download |
| --- | --- | --- |
| 1.0 | No archive is currently published in this repository. | — |
| 2.0nt | Archive published; detailed version notes are not yet available. | [LLigMa-2.0nt.zip](https://github.com/kdoggiegamer/LLigMa-All-Repositories-partially-untrained-/raw/refs/heads/main/LLigMa-2.0nt.zip) |
| 3.0nt | Archive published as a prerelease. | [LLigMa 3.0nt.zip](https://github.com/kdoggiegamer/LLigMa-All-Repositories-partially-untrained-/raw/refs/heads/main/LLigMa%203.0nt.zip) · [Release notes](https://github.com/kdoggiegamer/LLigMa-All-Repositories-partially-untrained-/releases/tag/v3.0nt) |

In the version names, `nt` means “not trained.” Earlier commit notes also define `bt` as “barely trained” and `ft` as “fully trained.” These labels describe training status; they are not performance guarantees.

### LLigMa 3.0nt

The published project overview describes 3.0 as an experimental decoder-only language model with multi-head latent attention, rotary position embeddings, and a routed mixture-of-experts feed-forward stack. It is research code; no capable pretrained checkpoint or benchmark claim is included in the project overview.

The `lligma_3_0_2b` configuration is not suitable for ordinary consumer hardware. Its total parameter count is substantially above 2B because of its routed experts. The project overview mentions a small debug configuration for smoke training and integration testing.

### LLigMa 1.0 and 2.0nt

The repository has no 1.0 archive at present. The 2.0nt archive is available above, but version-specific architecture, setup, and usage notes have not yet been documented here. Notes will be added when those details can be verified from the uploaded source.

## Future releases

When a new version is uploaded, this page will be updated with its archive or release link, what changed, setup and usage instructions, and known limitations based on the files provided.

## Development setup

The root `requirements.txt` lists Python dependencies for the project. Install a PyTorch build appropriate for your system, then run:

```bash
python -m pip install -r requirements.txt
```

This installs dependencies only; it does not extract or launch a version archive. Follow version-specific instructions when they are included with the archive.

## Community files

Community-provided GGUF files may be shared when accompanied by the source model/version and conversion settings, quantization type, context length, tokenizer compatibility details, checksum, basic load test, and license/redistribution terms. A community file is not an official or verified LLigMa release unless explicitly marked as verified.

## License

No license is currently published. Until license terms are added, reuse and redistribution permissions are not specified. A license should be added before inviting redistribution of source code or model weights.
