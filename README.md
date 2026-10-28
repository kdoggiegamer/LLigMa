# LLigMa

LLigMa is an experimental collection of language-model architectures, training utilities, and model-version repositories.

## Current status

LLigMa 3.0 is an early decoder-only language model implementation with multi-head latent attention, rotary position embeddings, and a routed mixture-of-experts feed-forward stack. It is research code: no capable pretrained checkpoint or benchmark claim is included here.

The original `lligma_3_0_2b` configuration is not suitable for ordinary consumer hardware. Its total parameter count is substantially above 2B because of its routed experts. A small debug configuration is provided for local smoke training and integration testing.

## Repository plan

Each LLigMa version will keep its architecture, configuration, training entry points, conversion tools, and documentation in its own directory or repository. Model weights are kept separate from source code.

Community-provided GGUF files are welcome when they include:

- the source model/version and conversion settings
- quantization type and context length
- tokenizer compatibility details
- a checksum and basic load test
- license and redistribution terms

A community GGUF does not imply an official or verified LLigMa release unless it is explicitly marked as verified.

## Development setup

Install a PyTorch build appropriate for your system, then install the project packages:

```bash
python -m pip install -r requirements.txt
```

## License

License terms will be added before a public model or weight release.
