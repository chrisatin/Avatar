import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { VRMLoaderPlugin } from "@pixiv/three-vrm";
import { ARKIT_51, arkitToVrm, quatToYawPitch } from "./mapping.js";

export class AvatarRig {
  constructor(scene) {
    this.scene = scene;
    this.root = new THREE.Group();
    scene.add(this.root);
    this.model = null;
    this.vrm = null;
    this.morphMap = new Map();
    this.smoothed = {};
    this.pose = { yaw: 0, pitch: 0 };
    this.isVRM = false;
  }

  clear() {
    if (this.model) {
      this.root.remove(this.model);
      this.model.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
      });
      this.model = null;
      this.vrm = null;
      this.morphMap.clear();
      this.isVRM = false;
    }
  }

  async load(url) {
    this.clear();
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));
    const gltf = await loader.loadAsync(url);
    if (gltf.userData?.vrm) {
      this.vrm = gltf.userData.vrm;
      this.model = this.vrm.scene;
      this.isVRM = true;
    } else {
      this.model = gltf.scene;
      this.isVRM = false;
    }
    this.model.traverse((obj) => {
      if (obj.isMesh && obj.morphTargetDictionary) {
        for (const [name, idx] of Object.entries(obj.morphTargetDictionary)) {
          if (!this.morphMap.has(name)) this.morphMap.set(name, []);
          this.morphMap.get(name).push({ mesh: obj, idx });
        }
      }
    });
    const box = new THREE.Box3().setFromObject(this.model);
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const scale = 1.6 / Math.max(size.y, 1e-4);
    this.model.scale.setScalar(scale);
    this.model.position.sub(center.multiplyScalar(scale));
    this.model.position.y -= box.min.y * scale * 0.0;
    this.root.add(this.model);
    return { isVRM: this.isVRM, morphs: this.morphMap.size };
  }

  get loaded() {
    return this.model !== null;
  }

  applyWeights(rawWeights, dt = 1 / 60) {
    if (!this.loaded) return;
    const alpha = 1 - Math.exp(-dt * 18.0);
    const weights = {};
    for (const name of ARKIT_51) {
      const target = Math.max(0, rawWeights[name] || 0);
      const prev = this.smoothed[name] || 0;
      const v = prev + (target - prev) * alpha;
      this.smoothed[name] = v;
      weights[name] = v;
    }
    if (this.isVRM && this.vrm?.expressionManager) {
      const vrmW = arkitToVrm(weights);
      for (const [preset, value] of Object.entries(vrmW)) {
        try {
          this.vrm.expressionManager.setValue(preset, value);
        } catch (_) {}
      }
      this.vrm.expressionManager.setValue("aa", Math.min(1, weights.jawOpen));
      return;
    }
    for (const [name, bindings] of this.morphMap) {
      const w = weights[name] || 0;
      for (const b of bindings) {
        b.mesh.morphTargetInfluences[b.idx] = w;
      }
    }
  }

  applyHeadPose(quatOrNull, dt = 1 / 60) {
    if (!this.loaded || !quatOrNull) return;
    const { yaw, pitch } = quatToYawPitch(quatOrNull);
    const alpha = 1 - Math.exp(-dt * 12.0);
    this.pose.yaw += (yaw - this.pose.yaw) * alpha;
    this.pose.pitch += (pitch - this.pose.pitch) * alpha;
    if (this.isVRM && this.vrm?.humanoid) {
      const head = this.vrm.humanoid.getNormalizedBoneNode("head");
      if (head) {
        head.rotation.y = this.pose.yaw;
        head.rotation.x = this.pose.pitch;
      }
      this.vrm.update?.(dt);
    } else {
      this.root.rotation.y = this.pose.yaw * 0.85;
      this.root.rotation.x = -this.pose.pitch * 0.85;
    }
  }
}
