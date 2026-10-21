const TASK_MODEL_URLS = [
  "/models/face_landmarker.task",
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
];

export class LocalTracker {
  constructor() {
    this.landmarker = null;
    this.video = null;
    this.lastVideoTime = -1;
    this.running = false;
    this.onResult = null;
  }

  async init(videoElement) {
    const { FaceLandmarker, FilesetResolver } = await import(
      "@mediapipe/tasks-vision"
    );
    const fileset = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    let lastErr = null;
    for (const url of TASK_MODEL_URLS) {
      try {
        this.landmarker = await FaceLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: url, delegate: "GPU" },
          runningMode: "VIDEO",
          numFaces: 1,
          outputFaceBlendshapes: true,
          outputFacialTransformationMatrixes: true,
        });
        break;
      } catch (err) {
        lastErr = err;
        try {
          this.landmarker = await FaceLandmarker.createFromOptions(fileset, {
            baseOptions: { modelAssetPath: url, delegate: "CPU" },
            runningMode: "VIDEO",
            numFaces: 1,
            outputFaceBlendshapes: true,
            outputFacialTransformationMatrixes: true,
          });
          break;
        } catch (err2) {
          lastErr = err2;
        }
      }
    }
    if (!this.landmarker) throw lastErr || new Error("FaceLandmarker init failed");
    this.video = videoElement;
    return true;
  }

  start() {
    if (!this.landmarker || !this.video) throw new Error("init() first");
    this.running = true;
    const loop = () => {
      if (!this.running) return;
      if (this.video.currentTime !== this.lastVideoTime && this.video.readyState >= 2) {
        this.lastVideoTime = this.video.currentTime;
        const t0 = performance.now();
        const result = this.landmarker.detectForVideo(this.video, t0);
        const ms = performance.now() - t0;
        if (result.faceBlendshapes?.length && this.onResult) {
          const weights = {};
          for (const cat of result.faceBlendshapes[0]) {
            if (cat.categoryName !== "_neutral") weights[cat.categoryName] = cat.score;
          }
          let matrix = null;
          const m = result.facialTransformationMatrixes?.[0]?.data;
          if (m) matrix = Array.from(m);
          this.onResult({ weights, matrix, ms });
        }
      }
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }

  stop() {
    this.running = false;
  }
}
