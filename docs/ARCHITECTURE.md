# Architecture

See `INVESTIGATION.md` for the full SOTA survey and selection rationale. This document describes the implemented system.

## High-level dataflow

```mermaid
flowchart LR
  subgraph Client [Browser client]
    CAM[Webcam] --> WASM[MediaPipe WASM<br/>local mode]
    CAM -->|JPEG frames| WS
    UI[Three.js viewport] --> RIG[AvatarRig<br/>morph targets / VRM]
    WASM -->|ARKit-52 weights| RIG
    UI -->|upload video| REST
  end

  subgraph Server [Ubuntu T4 server - headless]
    REST[FastAPI REST] --> JOBS[JobManager]
    JOBS --> PIPE[core.creation.pipeline]
    PIPE --> GLB[model.glb + meta.json]
    WS[/ws/track WebSocket] --> PYMP[MediaPipe Python<br/>server offload mode]
    PYMP --> RT[Retargeter + One-Euro]
    RT -->|blendshape JSON| WS
  end

  GLB -->|download| UI
  REST -->|poll job| UI
```

## Module map

| Path | Responsibility |
|---|---|
| `core/creation/video_ingest.py` | video/frames → decoded frame buffer (FFmpeg preferred, OpenCV fallback) |
| `core/creation/frames.py` | sharpness/yaw scoring, keyframe selection, face cropping |
| `core/creation/face_track.py` | batch landmark/blendshape tracking over keyframes |
| `core/creation/flame_fit.py` | optional FLAME 2020 identity fitting (smplx + landmark embedding) |
| `core/creation/template_head.py` | parametric morph-head fallback: identity regression + 51 ARKit sculpt deltas |
| `core/creation/texture.py` | multi-frame cylindrical UV albedo splat + inpaint |
| `core/creation/blendshapes.py` | FLAME expression basis → ARKit-51 deltas via `mat_final.npy` pseudo-inverse |
| `core/creation/export.py` | dependency-free GLB writer (morph targets, embedded PNG, `extras.targetNames`) |
| `core/creation/pipeline.py` | orchestrator with per-stage timings + `meta.json` |
| `core/animation/mapping.py` | ARKit-51/52 canonical order, VRM preset LUT, W-matrix helpers |
| `core/animation/tracker.py` | MediaPipe Face Landmarker wrapper + synthetic tracker + quaternion utils |
| `core/animation/smoothing.py` | One-Euro filter (scalars/vectors), quaternion slerp smoother |
| `core/animation/retarget.py` | TrackResult → smoothed blendshapes + VRM presets + gaze + head pose |
| `server/app.py` | FastAPI: upload→job→GLB download, `/ws/track`, health, static client |
| `server/jobs.py` | thread-pool job manager with progress callbacks |
| `server/render/` | optional EGL offscreen renderer + NVENC/WebRTC helpers (Option 2) |
| `client/` | Three.js + optional three-vrm viewer, MediaPipe WASM tracker, WS client |

## Creation backends

- **`flame`** — requires FLAME 2020 `generic_model.pkl` (registered download) + `smplx`, optional `landmark_embedding.npz` and `mat_final.npy`. Per-frame expression/pose/identity fit, median β aggregation, ARKit morph bake. Best quality; matches DECA/EMOCA-class output.
- **`template`** (default fallback, zero licensed assets) — parametric UV-sphere head shaped from MediaPipe landmark measurements; 51 procedural ARKit morph deltas; cylindrical albedo splat. Fully animatable, stylized look — explicitly acceptable per project requirements.
- **`auto`** — uses `flame` when assets + smplx are present, else `template`.

## Animation contract

The exported GLB carries morph targets named exactly like the MediaPipe/ARKit-51 blendshapes (`jawOpen`, `mouthSmileLeft`, …). Any ARKit-51 producer (browser WASM, server Python, OpenSeeFace with a remap) can drive it directly. VRM 1.0 models are driven through `@pixiv/three-vrm` expression presets via the same LUT mirrored in `core/animation/mapping.py` and `client/js/mapping.js`.

Head pose: MediaPipe facial transformation matrix → quaternion → smoothed yaw/pitch applied to the model root (glTF) or the VRM `head` normalized bone.

## WebSocket protocol (`/ws/track`)

Client → server:
- **binary**: JPEG frame (server throttles to `tracker_max_fps`)
- **text JSON**: `{"type":"ping"}` / `{"type":"reset"}`

Server → client:
- `{"type":"blendshapes","weights":{...},"vrm":{...},"gaze":[x,y],"rotation":[w,x,y,z],"translation":[...],"inference_ms":ms,"t":unix}`
- `{"type":"pong"}` / `{"type":"reset_ack"}` / `{"type":"error","error":...}`

## Latency budget (Option 1 — client-side)

| Stage | Cost |
|---|---|
| WASM Face Landmarker (client CPU) | 10–30 ms |
| One-Euro + morph apply | < 1 ms |
| Three.js render | 5–15 ms |
| **end-to-end** | **≈ 20–45 ms → 30–60 FPS** |

Server offload mode adds a JPEG encode + LAN round trip (typically 5–20 ms on LAN; measured 5 ms loopback).
