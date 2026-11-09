# Running LLigMa on an AMD GPU (ROCm in WSL2)

This directory holds a Docker setup for training **LLigMa 3.0 Small** (`configs/lligma_3_0_small.yaml`, 0.39B dense) on an AMD Radeon GPU, using ROCm inside WSL2.

Read the next section before running anything. This path is fragile.

## What this is, and what it is not

**It is not a supported configuration.** Docker Desktop's GPU passthrough on Windows is documented for NVIDIA only. There is no AMD equivalent. What works, according to community guides, is a **Docker Engine installed inside the WSL2 distro**, which passes the WSL DXCore bridge (`/dev/dxg`) into the container together with the WSL host libraries.

**There is no `/dev/kfd` in WSL2.** That device is what ROCm containers expect on a native Linux host. Its absence is normal here, not a bug to fix.

On native Windows (no WSL, no Docker), the ROCm nightly wheels install and report the GPU but **segfault inside `libhipblaslt.dll` on a plain `torch.matmul`**. That was reproduced with two different builds. If this container fails the same way, the stack is broken for this GPU and you should stop rather than try to fix training on top of it.

## Prerequisites

- Windows 11 with an AMD driver that the ROCm for Radeon matrix lists as compatible with your GPU.
- **WSL2** with a ROCm-capable Linux distro. The kernel and distro must be able to see the GPU.
- **Docker Engine installed inside the WSL distro** (for example the official `docker.io` package, or Docker's apt repository). Do **not** use Docker Desktop: it does not expose the nodes this setup needs.
- Your GPU must appear in the ROCm support matrix for the version inside the image. The **RX 9060 XT is `gfx1200`**.
- The ROCm version installed in your distro and the one in the image should match. A version mismatch is a common cause of silent failures.

## Step 1 - verify the host before building

Run these **inside the WSL distro**:

```bash
# The DXCore bridge must exist. If it does not, stop here.
ls -l /dev/dxg

# Render device, if present. If missing, remove the /dev/dri line from docker-compose.yml.
ls -l /dev/dri

# WSL host libraries.
ls -l /usr/lib/wsl/lib

# ROCDXG shim. If this path exists, uncomment the matching volume in
# docker-compose.yml. If it does not exist, leave the volume commented out.
ls -l /opt/rocm/lib/librocdxg.so.1

# Docker must be the one inside the distro, not Docker Desktop.
docker context ls
```

Then check that your distro's ROCm version matches the image tag you intend to use:

```bash
cat /opt/rocm/.info/version 2>/dev/null || echo "no local ROCm install"
```

Pick the `rocm/pytorch` tag from <https://hub.docker.com/r/rocm/pytorch/tags> that matches that version **and** lists `gfx1200`. The compose file defaults to `latest-rocm7.2.1-py3.12-pytorch2.9.0`; override it if your stack differs:

```bash
export ROCM_PYTORCH_TAG=<real-tag-from-docker-hub>
```

## Step 2 - build

From this directory, inside WSL:

```bash
docker compose build
```

The build fails on purpose if the base image's torch is not a ROCm build. That guard exists because a CPU torch silently replaces the ROCm one and the GPU disappears.

## Step 3 - smoke test before any training

Do not skip this. Run the container:

```bash
docker compose run --rm lligma
```

Then inside the container:

```bash
python -c "
import torch
print('torch', torch.__version__)
print('hip', torch.version.hip)
print('available', torch.cuda.is_available())
print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
a = torch.randn(512, 512, device='cuda')
b = a @ a
torch.cuda.synchronize()
print('matmul ok', round(b.sum().item(), 2))
"
```

**Expected:** `hip` is not `None`, `available` is `True`, the device name mentions Radeon, and `matmul ok` prints a number.

**If it segfaults** (exit code 139, or an `Exception Code: 0xC0000005` trace pointing at `libhipblaslt`): stop. That is the same failure seen on native Windows, and it means the ROCm stack does not work for this GPU in this configuration. Training will not work either. Report the failure and use Kaggle instead.

**If the device is `none` but there is no crash:** the passthrough is incomplete. Check that `/dev/dxg` is in the container (`ls -l /dev/dxg`), that `HSA_ENABLE_DXG_DETECTION=1` is set (`echo $HSA_ENABLE_DXG_DETECTION`), and that the ROCDXG volume is uncommented if the host path exists.

## Step 4 - train

Once the smoke test passes, from the repository root inside the container:

```bash
python scripts/train.py \
  --config configs/lligma_3_0_small.yaml \
  --tokenizer <path-to-tokenizer> \
  --train-file <path-to-train.parquet> \
  --validation-file <path-to-valid.parquet> \
  --output-dir /workspace/outputs/lligma-small \
  --sequence-length 1024 \
  --batch-size 4 \
  --gradient-accumulation-steps 8 \
  --save-every 500 \
  --device cuda
```

`--save-every 500` writes `checkpoint-000500.pt` and so on, so a crash does not lose the run. Resume with:

```bash
python scripts/train.py ... --resume-from /workspace/outputs/lligma-small/checkpoint-000500.pt --max-steps 2000
```

`--max-steps` must be greater than the step stored in the checkpoint.

Memory expectation for the 0.39B model: roughly 6.5 GB for weights, gradients and Adam states, plus activations. On a 16 GB card, sequence length 1024 with batch 4 needs gradient accumulation to keep the micro-batch small if you hit out-of-memory: lower `--batch-size` and raise `--gradient-accumulation-steps` to keep the effective batch size.

## Files here

- `Dockerfile` - builds on a ROCm PyTorch image and adds only the project's non-torch dependencies.
- `requirements-docker.txt` - those dependencies. **torch is deliberately absent**; installing it would replace the ROCm build.
- `docker-compose.yml` - device passthrough, WSL library mounts, and the repository mounted at `/workspace`.

## Alternative

If this does not pass the smoke test, use Kaggle. It is a Linux host with a working CUDA stack, and `notebooks/` contains a notebook for it.
