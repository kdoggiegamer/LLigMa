# LLigMa training image for AMD Radeon (ROCm on WSL2).
#
# The base image already ships PyTorch built against ROCm, so torch is NEVER
# installed here. Reinstalling it from PyPI would replace the ROCm build with a
# CPU/CUDA one and silently lose the GPU.
#
# Pin the tag to a real entry from https://hub.docker.com/r/rocm/pytorch/tags
# that lists gfx1200 (RDNA4) support. The tag below is a placeholder: verify it
# exists and matches the ROCm version installed in your WSL distro.
ARG ROCM_PYTORCH_TAG=latest-rocm7.2.1-py3.12-pytorch2.9.0
FROM rocm/pytorch:${ROCM_PYTORCH_TAG}

# Non-torch project dependencies only. accelerate is included because the
# training script accepts --device; transformers/datasets/tokenizers do the
# real work.
# Build context is the repository root, so the path keeps the docker/ prefix.
COPY docker/requirements-docker.txt /tmp/requirements-docker.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements-docker.txt

WORKDIR /workspace

# Fail loudly at build time if the base image ever loses its ROCm torch build.
RUN python -c "import torch; assert torch.version.hip is not None, 'base image torch is not a ROCm build'"

CMD ["/bin/bash"]
