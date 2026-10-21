# T4 Deployment Notes (Ubuntu, headless)

## Driver & CUDA

- T4 (Turing, `sm_75`) works with NVIDIA driver ≥ 470; use **≥ 530** for the CUDA 12.1 wheels pinned here.
- The repo installs `torch==2.4.1+cu121` by default. Keep `torch < 2.15` — newer major versions are where upstream drops pre-Turing support. CUDA 11.8 wheels (`https://download.pytorch.org/whl/cu118`) also work if the driver is older.

```bash
nvidia-smi            # driver + CUDA version sanity check
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Docker

```bash
# one-time host setup (Ubuntu)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

docker compose up --build
```

The compose file reserves the GPU and sets:

```
NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video
```

`graphics`+`video` are only needed for the **optional** EGL renderer / NVENC path; the default client-side rendering pipeline needs `compute,utility` alone.

## Headless pitfalls

- **No display needed** for the default architecture: tracking runs on CPU (MediaPipe Python) or in the browser (WASM); rendering happens in the client. Do not install a display server.
- **MediaPipe GPU delegate** requires a working EGL stack and is flaky headless — the server tracker intentionally uses CPU. This is well within budget for WebSocket offload (a few ms per frame).
- **EGL offscreen (Option 2):** if `pyrender` falls back to software, check `libegl1`, glvnd ICD files (`/usr/share/glvnd/egl_vendor.d/10_nvidia.json`), and the capabilities env var above. `nvdiffrast` is an alternative that needs no GLX.
- **NVENC:** T4 has one NVENC session (H.264 + HEVC, B-frames, no session limit on datacenter parts). `server/render/webrtc.py` probes `h264_nvenc` via PyAV; aiortc's NVENC path is unmerged upstream (PR #1199), so the patcher is opt-in. CPU `libx264` ultrafast fallback is included.

## Remote access

```bash
# from your workstation
ssh -N -L 8000:localhost:8000 user@t4-server
# open http://localhost:8000
```

For HTTPS/WebRTC deployments put nginx/caddy in front with TLS (browsers require secure contexts for `getUserMedia` unless on `localhost`).

## SLURM clusters (GPU access via sbatch)

On clusters where GPUs are gated by SLURM (interactive shells get `Failed to initialize NVML`), submit work with `sbatch`. Use this header (validated on the GPU-Workstation cluster, 4× T4):

```bash
#!/usr/bin/env bash
#SBATCH --job-name=physir-splat
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=logs/out/%x-%j.out
#SBATCH --error=logs/err/%x-%j.err

set -eo pipefail
```

- Check free resources first: `slurm-avail`
- A ready-made validation job exists: `mkdir -p logs/out logs/err && sbatch scripts/t4_smoke.sh` (GL setup, CUDA/FP16 check, pytest, creation E2E, driving benchmarks, server smoke test). Results land in `logs/out/physir-splat-<jobid>.out`.
- Inside the allocation, `nvidia-smi` and CUDA work normally.
- **Headless MediaPipe:** needs `libGLESv2.so.2`. Without sudo: `bash scripts/setup_headless_gl.sh` extracts `libgles2/libegl1/libgl1` into `~/.local/gl-libs`; `run_demo.sh` and `t4_smoke.sh` pick it up automatically via `LD_LIBRARY_PATH`.

## Performance expectations on T4

| Task | Expected |
|---|---|
| Creation, template backend, 10 s video | ≈ 15–60 s (CPU-bound; scales with keyframe count) |
| Creation, FLAME backend, 24 keyframes | ≈ 40–65 s |
| Server tracking offload (`/ws/track`) | ≈ 5–15 ms inference + RTT; 30 FPS cap by config |
| Driving end-to-end (client WASM) | 30–60 FPS; T4 idle |
| Peak VRAM | ≈ 2–6 GB (well under 16 GB; TRELLIS-class models excluded by design) |

Run `python benchmark.py --mode all` on the server after deployment and attach `outputs/benchmark/benchmark.json` to report numbers.
