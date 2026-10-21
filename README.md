# 3D Avatar Generation & Real-Time Driving Pipeline (NVIDIA T4)

End-to-end system for building a rigged, textured 3D avatar from a short video (or frame sequence) and driving it live from a webcam — designed for a **headless Ubuntu server with a Tesla T4** and a remote browser client.

- **Creation:** video/frames → keyframes → face tracking → identity mesh (FLAME if licensed assets present, parametric template fallback otherwise) → albedo texture → **51 ARKit blendshape morph targets** → `model.glb` (glTF 2.0 binary, VRM-compatible naming)
- **Driving:** browser webcam → MediaPipe Face Landmarker (WASM, ~10–30 ms) **or** server-side Python tracking over WebSocket → smoothed ARKit-52 weights → Three.js / `@pixiv/three-vrm` at 30–60 FPS
- **Photorealism intentionally out of scope** — stylized/parametric rigged avatars keep latency low and fit the T4 budget (see `INVESTIGATION.md` for the SOTA survey and selection rationale)

## Quick start (T4 server)

```bash
./scripts/setup_env.sh          # venv + torch cu121 + pinned deps + model assets
./run_demo.sh                   # FastAPI server on :8000
```

From your workstation:

```bash
ssh -N -L 8000:localhost:8000 user@t4-server
# open http://localhost:8000 → upload a video → Start webcam
```

Docker alternative:

```bash
docker compose up --build
```

Full instructions: [`docs/SETUP.md`](docs/SETUP.md) · T4/headless notes: [`docs/DEPLOYMENT_T4.md`](docs/DEPLOYMENT_T4.md) · Design: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) · Research: [`INVESTIGATION.md`](INVESTIGATION.md)

## Repository layout

```
core/creation/     video ingest, keyframes, tracking, fitting, texture, blendshapes, GLB export
core/animation/    ARKit-52 mapping, trackers (MediaPipe/synthetic), One-Euro smoothing, retargeting
server/            FastAPI app, job manager, /ws/track offload, optional EGL/NVENC extras
client/            Three.js + three-vrm viewer, MediaPipe WASM tracker, WebSocket client
scripts/           model downloader, sample-video generator, env setup
tests/             unit + integration tests (pytest, no GPU required)
benchmark.py       creation/driving/export benchmarks (+ VRAM via NVML)
docs/              architecture, setup, T4 deployment
```

## API

| Endpoint | Description |
|---|---|
| `POST /api/v1/avatar/create` | multipart `file` (video) or `?source_path=` on server → `{job_id}` |
| `GET /api/v1/avatar/jobs/{id}` | job status + progress + timings |
| `GET /api/v1/avatar/jobs/{id}/model.glb` | download rigged avatar (51 morph targets) |
| `GET /api/v1/health` | tracker/CUDA/asset status |
| `WS /ws/track` | JPEG frames in → smoothed ARKit-52 blendshape JSON out |

## CLI

```bash
python -c "from core.creation.pipeline import run_creation_pipeline; run_creation_pipeline('video.mp4', 'outputs/avatar1')"
python benchmark.py --mode all
python scripts/make_sample_video.py          # synthetic test clip
pytest tests -q
```

## FLAME (optional quality upgrade)

Register at <https://flame.is.tue.mpg.de>, drop `generic_model.pkl` into `assets/models/flame/FLAME2020/`, and put `mat_final.npy` (auto-downloaded) alongside. The pipeline then fits real FLAME identity/expression and bakes ARKit morphs from the FLAME expression basis. Without those assets everything still works via the parametric template backend.

## License notes

- Code in this repository: permissive use within this project.
- FLAME model: separate academic/non-commercial license from MPI — not redistributed here.
- MediaPipe models: Apache-2.0. `mat_final.npy` (BEAT2_Tools): check upstream terms for your use case.
