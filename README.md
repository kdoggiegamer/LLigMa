# LLigMa

LLigMa is an experimental collection of language-model architectures, training utilities, and model-version repositories.

## Version history

This repository brings together the LLigMa versions and their development history. Version notes are based on the material currently published here; earlier release details can be added as their source files and notes are uploaded.

### LLigMa 1.0

The original LLigMa version. A detailed description and archived source for 1.0 have not yet been added to this repository.

### LLigMa 2.0

The second LLigMa version. A detailed description and archived source for 2.0 have not yet been added to this repository.

### LLigMa 3.0

An experimental decoder-only language model implementation with multi-head latent attention, rotary position embeddings, and a routed mixture-of-experts feed-forward stack. The repository includes the 3.0nt archive. This is research code; no capable pretrained checkpoint or benchmark claim is included here.

The original `lligma_3_0_2b` configuration is not suitable for ordinary consumer hardware. Its total parameter count is substantially above 2B because of its routed experts. A small debug configuration is provided for local smoke training and integration testing.

## Future releases

As new LLigMa versions are uploaded, their version notes, source links, and release information will be added here. Future entries will describe what changed, how to run the version, and any known limitations based on the uploaded materials.

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
