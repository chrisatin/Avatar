import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { AvatarRig } from "./avatar.js";
import { LocalTracker } from "./tracker.js";
import { ServerTracker, apiCreate, apiHealth, apiJob, modelUrl } from "./network.js";
import { quatToYawPitch } from "./mapping.js";

const $ = (id) => document.getElementById(id);

const canvas = $("canvas");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0e1014);
const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 50);
camera.position.set(0, 0.1, 2.4);
const controls = new OrbitControls(camera, canvas);
controls.target.set(0, 0.05, 0);
scene.add(new THREE.AmbientLight(0xffffff, 0.7));
const key = new THREE.DirectionalLight(0xffffff, 1.6);
key.position.set(0.6, 1.0, 1.2);
scene.add(key);
const rim = new THREE.DirectionalLight(0x88aaff, 0.6);
rim.position.set(-0.8, 0.4, -1.0);
scene.add(rim);

const rig = new AvatarRig(scene);
let currentJobId = null;
let trackingMode = "local";
let localTracker = null;
let serverTracker = null;
let webcamStream = null;
let lastWeights = {};
let lastQuat = null;
let lastFrameTs = performance.now();
let fpsAccum = 0;
let fpsCount = 0;
let lastTrackMs = 0;
let localHeadMatrix = null;
let canvas2d = null;

function resize() {
  const w = canvas.clientWidth || canvas.parentElement.clientWidth;
  const h = canvas.clientHeight || canvas.parentElement.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas.parentElement);
resize();

function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt = Math.min((now - lastFrameTs) / 1000, 0.1);
  lastFrameTs = now;
  fpsAccum += dt;
  fpsCount += 1;
  if (fpsAccum >= 0.5) {
    $("fps").textContent = Math.round(fpsCount / fpsAccum).toFixed(0);
    fpsAccum = 0;
    fpsCount = 0;
  }
  rig.applyWeights(lastWeights, dt);
  if (lastQuat) rig.applyHeadPose(lastQuat, dt);
  else if (localHeadMatrix) {
    const q = matrixToQuat(localHeadMatrix);
    rig.applyHeadPose(q, dt);
  }
  controls.update();
  renderer.render(scene, camera);
  if (trackingMode === "server" && webcamStream && canvas2d) {
    serverTracker.sendFrame(canvas2d);
  }
}
animate();

function matrixToQuat(m) {
  const e = m;
  const m00 = e[0], m01 = e[4], m02 = e[8];
  const m10 = e[1], m11 = e[5], m12 = e[9];
  const m20 = e[2], m21 = e[6], m22 = e[10];
  const tr = m00 + m11 + m22;
  let w, x, y, z;
  if (tr > 0) {
    const s = Math.sqrt(tr + 1) * 2;
    w = 0.25 * s;
    x = (m21 - m12) / s;
    y = (m02 - m20) / s;
    z = (m10 - m01) / s;
  } else if (m00 > m11 && m00 > m22) {
    const s = Math.sqrt(1 + m00 - m11 - m22) * 2;
    w = (m21 - m12) / s;
    x = 0.25 * s;
    y = (m01 + m10) / s;
    z = (m02 + m20) / s;
  } else if (m11 > m22) {
    const s = Math.sqrt(1 + m11 - m00 - m22) * 2;
    w = (m02 - m20) / s;
    x = (m01 + m10) / s;
    y = 0.25 * s;
    z = (m12 + m21) / s;
  } else {
    const s = Math.sqrt(1 + m22 - m00 - m11) * 2;
    w = (m10 - m01) / s;
    x = (m02 + m20) / s;
    y = (m12 + m21) / s;
    z = 0.25 * s;
  }
  return [w, x, y, z];
}

$("healthBtn").addEventListener("click", async () => {
  try {
    const h = await apiHealth();
    $("healthOut").textContent = JSON.stringify(h, null, 2);
    $("backendLabel").textContent = h.tracker;
  } catch (e) {
    $("healthOut").textContent = String(e);
  }
});

$("videoInput").addEventListener("change", () => {
  $("uploadBtn").disabled = !$("videoInput").files.length;
});

$("uploadBtn").addEventListener("click", async () => {
  const file = $("videoInput").files[0];
  if (!file) return;
  setStatus("uploading…");
  try {
    const res = await apiCreate(file);
    currentJobId = res.job_id;
    $("loadModelBtn").disabled = true;
    await pollJob(currentJobId);
  } catch (e) {
    setStatus(String(e), "err");
  }
});

async function pollJob(jobId) {
  const t0 = performance.now();
  while (true) {
    const job = await apiJob(jobId);
    $("jobProgress").value = job.progress;
    setStatus(`${job.status} · ${job.stage} · ${(job.progress * 100).toFixed(0)}%`);
    if (job.status === "done") {
      setStatus(`done in ${((performance.now() - t0) / 1000).toFixed(1)}s`, "ok");
      $("loadModelBtn").disabled = false;
      const link = $("downloadLink");
      link.href = modelUrl(jobId);
      link.textContent = "Download .glb";
      link.classList.remove("hidden");
      $("viewportHint").style.display = "none";
      await loadModel(jobId);
      return job;
    }
    if (job.status === "error") {
      setStatus(job.error || "failed", "err");
      return job;
    }
    await new Promise((r) => setTimeout(r, 800));
  }
}

$("loadModelBtn").addEventListener("click", async () => {
  if (currentJobId) await loadModel(currentJobId);
});

async function loadModel(jobId) {
  setStatus("loading model…");
  try {
    const info = await rig.load(modelUrl(jobId) + `?t=${Date.now()}`);
    $("viewportHint").style.display = "none";
    setStatus(`model loaded (${info.isVRM ? "VRM" : "glTF"}, ${info.morphs} morphs)`, "ok");
  } catch (e) {
    setStatus(`model load failed: ${e}`, "err");
  }
}

$("webcamBtn").addEventListener("click", async () => {
  if (webcamStream) {
    stopWebcam();
    return;
  }
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: "user" },
      audio: false,
    });
  } catch (e) {
    setStatus(`webcam denied: ${e}`, "err");
    return;
  }
  const video = $("webcam");
  video.srcObject = webcamStream;
  video.classList.add("show");
  $("webcamBtn").textContent = "Stop webcam";
  canvas2d = document.createElement("canvas");
  canvas2d.width = 480;
  canvas2d.height = 360;
  await startTracking();
});

function stopWebcam() {
  if (localTracker) localTracker.stop();
  if (serverTracker) serverTracker.close();
  localTracker = null;
  serverTracker = null;
  if (webcamStream) webcamStream.getTracks().forEach((t) => t.stop());
  webcamStream = null;
  $("webcam").classList.remove("show");
  $("webcamBtn").textContent = "Start webcam";
  $("modeLabel").textContent = "–";
}

async function startTracking() {
  trackingMode = $("trackMode").value;
  const video = $("webcam");
  const mirror = () => $("mirror").checked;
  if (trackingMode === "local") {
    localTracker = new LocalTracker();
    try {
      await localTracker.init(video);
    } catch (e) {
      setStatus(`local tracker failed: ${e}; try server mode`, "err");
      return;
    }
    localTracker.onResult = ({ weights, matrix, ms }) => {
      lastWeights = $("mirror").checked ? swapSides(weights) : weights;
      localHeadMatrix = matrix;
      lastQuat = null;
      lastTrackMs = ms;
      $("trackMs").textContent = ms.toFixed(1);
    };
    localTracker.start();
    $("modeLabel").textContent = "local WASM";
  } else {
    serverTracker = new ServerTracker(({ weights, quat, ms, roundtrip }) => {
      // The frame sent to the server is mirrored, so restore the subject's
      // left/right channels before driving the avatar (same as local mode).
      lastWeights = $("mirror").checked ? swapSides(weights) : weights;
      lastQuat = quat;
      localHeadMatrix = null;
      lastTrackMs = roundtrip ?? ms;
      $("trackMs").textContent = (roundtrip ?? ms).toFixed(1);
    });
    try {
      await serverTracker.connect();
      $("modeLabel").textContent = "server ws";
    } catch (e) {
      setStatus(`ws connect failed: ${e}`, "err");
      return;
    }
    const ctx = canvas2d.getContext("2d");
    setInterval(() => {
      if (!webcamStream || !canvas2d) return;
      const v = $("webcam");
      const w = canvas2d.width;
      const h = canvas2d.height;
      const vw = v.videoWidth;
      const vh = v.videoHeight;
      if (!vw || !vh) return;
      const scale = Math.max(w / vw, h / vh);
      const dw = vw * scale;
      const dh = vh * scale;
      if ($("mirror").checked) {
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(v, -(dw - w) / 2 - w, (h - dh) / 2, dw, dh);
        ctx.restore();
      } else {
        ctx.drawImage(v, (w - dw) / 2, (h - dh) / 2, dw, dh);
      }
    }, 33);
  }
  $("viewportHint").style.display = "none";
}

function mirrorName(name) {
  if (name.endsWith("Left")) return name.slice(0, -4) + "Right";
  if (name.endsWith("Right")) return name.slice(0, -5) + "Left";
  return name;
}

function swapSides(weights) {
  const out = {};
  for (const [k, v] of Object.entries(weights)) out[mirrorName(k)] = v;
  return out;
}

function setStatus(text, cls = "") {
  const el = $("jobStatus");
  el.textContent = text;
  el.className = `status ${cls}`;
}

(async () => {
  try {
    const h = await apiHealth();
    $("backendLabel").textContent = h.tracker;
  } catch (_) {}
})();
