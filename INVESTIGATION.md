# INVESTIGATION — SOTA 3D Avatar Generation & Real-Time Rendering on NVIDIA Tesla T4

**Date:** 2026-10-21
**Target hardware:** Ubuntu Linux server, NVIDIA Tesla T4 (16 GB VRAM, Turing `sm_75`, CC 7.5, 70 W TDP, 1× NVENC H.264/HEVC, headless — no display)
**Client:** remote machine over SSH; interactive via HTTP + WebSocket; rendering in browser (Three.js/WebGL)
**Requirements recap:** photorealism *not* required; parametric/stylized rigged avatars (FLAME, glTF morph targets, VRM) preferred; creation ≤ 30–60 s from a ~10 s video; driving ≥ 30 FPS (target 60) at ≤ 50 ms tracking latency.

---

## 1. Candidates considered

### 1.1 Avatar creation (video / frames → rigged, textured mesh)

| Repo | License | Est. VRAM | Est. T4 time | Dep. friction | Verdict |
|---|---|---|---|---|---|
| [radekd91/DECA](https://github.com/radekd91/DECA) | MPI non-commercial | 4–6 GB | 50–150 ms/frame | Medium (PyTorch + pytorch3d) | ✅ Viable FLAME baseline; superseded |
| [radekd91/emoca](https://github.com/radekd91/emoca) (v2) | MPI non-commercial | 4–6 GB | 50–150 ms/frame | High (torch 1.12 pins); deprecated upstream | ✅ Best per-frame expression/lip quality |
| [ZhenglinZhou/HeadStudio](https://github.com/ZhenglinZhou/HeadStudio) | MIT (code) | ~8 GB | seconds + texture stage | Medium (FLAME 2020 pkls, `flame2facemsh.npy`) | ✅ Fast one-shot → FLAME + texture |
| [xg-chu/GAGAvatar](https://github.com/xg-chu/GAGAvatar) | MIT | 8–12 GB | sec/image | High (custom diff-gaussian rasterizer) | ⚠️ 3DGS, not mesh-native |
| [KelianB/SPARK](https://github.com/KelianB/SPARK) | research | ~8 GB | real-time capture | High (FLAME + BFM texture not distributable) | ⚠️ Driving-oriented |
| [thu-ml/crm](https://github.com/thu-ml/crm) | MIT | 12–16 GB | ~10 s/image | Medium (custom CUDA rasterizer) | ✅ Textured mesh, **unrigged** |
| [AiuniAI/Unique3D](https://github.com/AiuniAI/Unique3D) | MIT | 12–16 GB | ~30 s/image | High (mmcv, nvdiffrast) | ⚠️ No rig/blendshapes |
| [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS) | MIT | ≥ 16 GB stated | OOM on T4 (confirmed by HF Space users) | Medium | ❌ **Not T4-viable** |
| [MrTornado24/Next3D](https://github.com/MrTornado24/Next3D) | **none** (all-rights-reserved) | ~16 GB+ | slow (SDS) | High | ❌ License + VRAM |
| Video Diffusion Avatars | research | 48–80 GB | — | High | ❌ Not T4-viable |
| [ant-research/CoDeF](https://github.com/ant-research/CoDeF) | other | ~10 GB | hours/subject (per-video optimization) | Medium | ❌ Blows 30–60 s budget |

### 1.2 Real-time driving (webcam → expression/pose → mesh)

| System | License | Latency | Needs T4? | Verdict |
|---|---|---|---|---|
| **MediaPipe Face Landmarker** (Web WASM + Python) | Apache-2.0 | ~10–30 ms/frame (WASM worker) | No | ✅ **Selected tracker** — outputs 478 landmarks + **52 ARKit blendshapes** + head transform |
| [OpenSeeFace](https://github.com/emilianavt/OpenSeeFace) | BSD-2 | 30–60 FPS CPU (ONNX) | No | ✅ Fallback tracker |
| [Kalidokit](https://github.com/yeemachine/kalidokit) | MIT | ~20 ms | No | ⚠️ Deprecated upstream — not built upon |
| [LivePortrait](https://github.com/KlingAIResearch/LivePortrait) / FasterLivePortrait | MIT code; InsightFace models non-commercial | 30–70 ms GPU | Yes (TRT 8.x lane only) | ⚠️ 2D warp output — wrong product shape for mesh avatars |
| [three-vrm](https://github.com/pixiv/three-vrm) (pixiv) | MIT | render-side trivial | No | ✅ **Selected renderer** for VRM 1.0 (expressionManager + morphTargetBinds) |
| aiortc + NVENC ([PR #1199](https://github.com/aiortc/aiortc/pull/1199)) | BSD | encode < 5 ms | Yes | ⚠️ PR unmerged — monkeypatch required; kept as optional path |

### 1.3 Exclusion rationale (summary)

- **TRELLIS / Video Diffusion Avatars / Next3D:** VRAM or license hard-blocks on 16 GB Turing.
- **LivePortrait family:** outputs reenacted *video*, not a mesh with blendshapes; wrong abstraction for an animatable avatar.
- **CRM / Unique3D:** excellent unrigged text-to-3D, but leave the rigging problem unsolved (blendshape transfer is its own research project).
- **Kalidokit:** officially deprecated; MediaPipe Face Landmarker now emits the same ARKit-52 weights natively.

---

## 2. Selected architecture

### 2.1 Creation pipeline (Phase A) — multi-frame FLAME identity fitting + ARKit-52 morph bake

```
video (.mp4/.mov/.webm) ──or── frames dir
        │
        ▼
[1] Ingest & keyframe sampling          core/creation/video_ingest.py, frames.py
    FFmpeg/OpenCV decode @ 25 fps → N≈16–32 keyframes
    (uniform + sharpness/frontalness scoring, face crop, optional bg removal)
        │
        ▼
[2] Per-frame face tracking             core/creation/face_track.py
    MediaPipe Face Landmarker (Python, CPU-safe on headless T4)
    → 478 landmarks, 52 blendshapes, head transform per frame
        │
        ▼
[3] Identity & geometry fitting         core/creation/flame_fit.py
    PRIMARY  : FLAME 2020 via smplx — per-frame expression/pose fit,
               robust (median) aggregation of shape betas β → one identity mesh
    FALLBACK : parametric template head (no licensed assets) — landmark-measurement
               regression into a rigged morph head, so the pipeline always runs
        │
        ▼
[4] Texture                            core/creation/texture.py
    Cylindrical UV on FLAME/template; multi-frame color splatting of sharpest
    frontal frames + median-skin inpaint → albedo.png (1024²)
        │
        ▼
[5] Blendshape baking                   core/creation/blendshapes.py
    51 active ARKit channels (52 incl. _neutral) as vertex-delta morph targets
    FLAME   : EMAGE/PantoMatrix `mat_final.npy` (51×103, FLAME-2020 only,
              y = x @ M, jaw dims ×100) + least-squares solve on exp basis
    FALLBACK: procedural region-mask sculpt per channel
        │
        ▼
[6] Export                             core/creation/export.py
    glTF 2.0 binary (.glb) with morph targets + extras.targetNames
    (+ optional VRM 1.0 via UniVRM/three-vrm conversion on client)
        │
        ▼
    outputs/<job_id>/model.glb  (~40–65 s on T4 for 10 s video; fits budget)
```

### 2.2 Real-time driving (Phase B) — Option 1: client-side tracking + client-side rendering (recommended)

```
Browser (client)
  getUserMedia webcam → Web Worker
      → MediaPipe Face Landmarker WASM (LIVE_STREAM, num_faces=1, smoothing on)
         ~10–30 ms/frame  ⇒  30–60 FPS on modest client CPUs; T4 untouched
      → 52 ARKit weights + facial transformation matrix
      → One-Euro / EMA smoothing
      → ARKit-52 → morph targets (same topology names baked at creation)
         or ARKit-52 → VRM presets (LUT: happy/angry/sad/relaxed/surprised,
         aa/ih/ou/ee/oh, blink*, look*) via @pixiv/three-vrm expressionManager
      → Three.js render loop @ 60 FPS

Server offload (thin-client mode, optional)
  Client streams JPEG frames over WebSocket /ws/track
      → T4/Python MediaPipe Face Landmarker → same ARKit-52 JSON back
      → client only applies morphs + renders
  (~15–40 ms server round trip on LAN; keeps client CPU-free)
```

### 2.3 Option 2 (kept optional, not default): server-side render + NVENC WebRTC

```
track (10 ms) → FLAME/morph eval (5–15 ms, blendshapes pre-baked) →
EGL/nvdiffrast offscreen render (5–10 ms) → NVENC H.264 (<5 ms) →
aiortc WebRTC  ⇒  ~25–40 ms/frame possible, but:
  • aiortc NVENC PR #1199 is unmerged (monkeypatch + `pip install av --no-binary av`)
  • headless EGL requires NVIDIA_DRIVER_CAPABILITIES=compute,graphics,video
  • defeats the low-latency advantage of client WebGL
```

Implemented as `server/render/` extras; the demo and benchmark default to Option 1.

### 2.4 Why this architecture (selection criteria scorecard)

| Criterion | Score | Evidence |
|---|---|---|
| Creation time on T4 (≤ 30–60 s) | ✅ | 16–32 keyframes × ~80–120 ms tracking/fitting + texture/bake ≈ 40–65 s |
| Driving ≥ 30 FPS, ≤ 50 ms | ✅ | WASM tracking 10–30 ms + apply+render 5–15 ms, end-to-end 20–40 ms |
| Headless server viability | ✅ | Tracking offloaded to client WASM; server is CPU-light FastAPI + static |
| Standard rig (ARKit-52 / VRM / glTF morphs) | ✅ | 52 blendshapes baked as morph targets; VRM preset LUT included |
| T4 VRAM headroom | ✅ | Peak creation ≈ 2–6 GB (MediaPipe CPU + optional FLAME); TRELLIS-class OOM avoided |
| Licensing | ⚠️→✅ | FLAME model = registered academic use (drop-in asset, not redistributed); pipeline degrades to a **no-FLAME parametric fallback** so the repo runs end-to-end without it |

### 2.5 Key mapping resources

- **ARKit-51 → FLAME expression:** `mat_final.npy` (51×103) from BEAT2_Tools (H-Liu1997); `y = x @ M`; jaw dims `y[100:103] *= 100`; **FLAME-2020 only** (FLAME-2023 PCA differs → near-zero jaw, see PantoMatrix #205).
- **ARKit canonical order:** `jawOpen` = index 24 (PantoMatrix #199).
- **Eye/eyelid channels (8):** not representable in W — eye pose driven separately from FLAME eye params (#132).
- **FLAME ↔ MediaPipe landmark correspondence:** `mediapipe_landmark_embedding.npz`, `flame2facemsh.npy` (HeadStudio).
- **MediaPipe blendshape sides:** `*Left/*Right` are the *subject's* sides; mirrored input flips them.

### 2.6 Risk register (T4 / Ubuntu specifics)

| Risk | Mitigation in this repo |
|---|---|
| FLAME model requires registration (non-commercial) | `scripts/download_models.py` fetches only redistributable assets; FLAME pkl dropped into `assets/models/flame/`; fallback backend keeps E2E runnable |
| PyTorch CUDA support for Turing | Pin `torch<2.15`; install `cu118`/`cu121` wheels (CUDA 12.x fine on T4 with driver ≥ 530) |
| TensorRT 10 vs Turing/TRT-8 plugins | Not required for selected path; documented only for LivePortrait-class detours |
| Headless EGL failure / silent software GL | Default path needs no GL on server; `docs/DEPLOYMENT_T4.md` covers `NVIDIA_DRIVER_CAPABILITIES` for the optional renderer |
| aiortc NVENC unmerged | Monkeypatch isolated in `server/render/webrtc.py`; CPU x264 fallback |
| MediaPipe GPU delegate fragile headless | Server tracker runs CPU (fast enough for offload); GPU never required |

---

## 3. Selected stack (implementation targets)

| Layer | Choice |
|---|---|
| Creation tracking | MediaPipe Face Landmarker (Python tasks API) |
| Geometry | FLAME 2020 (smplx, optional) → parametric template fallback |
| Rig | 51 ARKit morph targets baked on identity mesh |
| Export | glTF 2.0 binary + embedded albedo PNG; VRM-compatible naming |
| Server | FastAPI + Uvicorn; REST jobs; WebSocket `/ws/track` |
| Client | Three.js (r170+) + optional `@pixiv/three-vrm`; MediaPipe tasks-vision WASM |
| GPU | T4 used for optional FLAME optimization and optional NVENC stream; primary driving path is client-side |
| Env | Docker (NVIDIA Container Toolkit) **or** conda/venv; CUDA 11.8/12.1 wheels |
