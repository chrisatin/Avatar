export async function apiHealth() {
  const r = await fetch("/api/v1/health");
  return r.json();
}

export async function apiCreate(file) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch("/api/v1/avatar/create", { method: "POST", body: fd });
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export async function apiJob(jobId) {
  const r = await fetch(`/api/v1/avatar/jobs/${jobId}`);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

export function modelUrl(jobId) {
  return `/api/v1/avatar/jobs/${jobId}/model.glb`;
}

export class ServerTracker {
  constructor(onResult) {
    this.ws = null;
    this.onResult = onResult;
    this.queue = [];
    this.sending = false;
    this.latency = null;
  }

  connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    this.ws = new WebSocket(`${proto}://${location.host}/ws/track`);
    this.ws.binaryType = "arraybuffer";
    this.ws.onmessage = (ev) => {
      if (typeof ev.data === "string") {
        const msg = JSON.parse(ev.data);
        if (msg.type === "blendshapes" && this.onResult) {
          const t1 = performance.now();
          this.latency = msg.inference_ms ?? null;
          this.onResult({
            weights: msg.weights || {},
            quat: msg.rotation || null,
            ms: msg.inference_ms ?? 0,
            roundtrip: this._sentAt ? t1 - this._sentAt : null,
          });
        }
      }
    };
    return new Promise((resolve, reject) => {
      this.ws.onopen = () => resolve();
      this.ws.onerror = (e) => reject(e);
    });
  }

  sendFrame(canvas, jpegQuality = 0.8) {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;
    if (this.sending) return;
    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        blob.arrayBuffer().then((buf) => {
          if (this.ws.readyState === WebSocket.OPEN) {
            this.sending = true;
            this._sentAt = performance.now();
            this.ws.send(buf);
            setTimeout(() => (this.sending = false), 8);
          }
        });
      },
      "image/jpeg",
      jpegQuality
    );
  }

  close() {
    if (this.ws) this.ws.close();
  }
}
