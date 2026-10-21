# Setup Guide

## 1. Prerequisites

- **Server (target):** Ubuntu 22.04/24.04, NVIDIA driver ≥ 530 (for CUDA 12.1 wheels), CUDA-capable GPU (T4 validated), FFmpeg (`apt install ffmpeg`)
- **Client:** any modern browser (Chrome/Edge/Firefox) with camera access; reached over SSH tunnel or LAN
- Python 3.10–3.12

## 2. Environment

### Option A — venv (recommended)

```bash
./scripts/setup_env.sh
source .venv/bin/activate
```

The script creates `.venv`, installs PyTorch cu121 wheels (T4/`sm_75` supported) and pinned requirements, then downloads redistributable models.

### Option B — Conda

```bash
conda env create -f environment.yml
conda activate avatar
```

### Option C — Docker (NVIDIA Container Toolkit required)

```bash
docker compose up --build
```

## 3. Model assets

```bash
python scripts/download_models.py
```

Downloads (redistributable):
- `assets/models/face_landmarker.task` — MediaPipe Face Landmarker (server tracker + client WASM)
- `assets/models/mat_final.npy` — ARKit-51 → FLAME-2020 expression mapping

**Optional FLAME (quality up):** register at <https://flame.is.tue.mpg.de>, download FLAME 2020, place `generic_model.pkl` under `assets/models/flame/FLAME2020/`. Also drop `landmark_embedding.npz` there if available (HeadStudio / EMOCA assets) for landmark-supervised fitting. Without these, the pipeline automatically uses the parametric template backend — the system remains fully functional.

Optional extras (FLAME fitting / server rendering / WebRTC):

```bash
pip install -r requirements-optional.txt
```

## 4. Run the demo

```bash
./run_demo.sh
```

From your **client machine**, open an SSH tunnel and browse to `http://localhost:8000`:

```bash
ssh -N -L 8000:localhost:8000 user@t4-server
```

1. **Create:** upload a 5–15 s face video (or use `assets/samples/sample_face.mp4`), press *Upload & build*. The job polls until done and the avatar auto-loads.
2. **Drive:** press *Start webcam*. Choose *Local (MediaPipe WASM)* — recommended — or *Server (WebSocket → T4)* and watch the stats panel for FPS/latency.

## 5. CLI creation (no browser)

```bash
python -c "
from core.creation.pipeline import run_creation_pipeline
import json
print(json.dumps(run_creation_pipeline('path/to/video.mp4', 'outputs/my_avatar'), indent=2))
"
```

Or point the REST API at a server-side path:

```bash
curl -X POST 'http://localhost:8000/api/v1/avatar/create?source_path=/abs/path/video.mp4'
```

## 6. Benchmarks

```bash
python benchmark.py --mode all
# creation only, custom video:
python benchmark.py --mode creation --video assets/samples/sample_face.mp4 --repeats 3
# driving (synthetic frames, no camera needed):
python benchmark.py --mode driving --tracker synthetic --frames 500
```

Results are printed and saved to `outputs/benchmark/benchmark.json` (includes VRAM via NVML when an NVIDIA driver is present).

## 7. Tests

```bash
pip install -r requirements-dev.txt
pytest tests -q
```

## 8. Sample assets

`python scripts/make_sample_video.py` regenerates a synthetic 10 s talking-head clip (no personal data) for smoke tests and benchmarking. Real-face videos produce identity-fitted avatars via MediaPipe tracking.
