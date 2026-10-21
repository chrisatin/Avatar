export const ARKIT_51 = [
  "browDownLeft","browDownRight","browInnerUp","browOuterUpLeft","browOuterUpRight",
  "cheekPuff","cheekSquintLeft","cheekSquintRight",
  "eyeBlinkLeft","eyeBlinkRight","eyeLookDownLeft","eyeLookDownRight","eyeLookInLeft","eyeLookInRight",
  "eyeLookOutLeft","eyeLookOutRight","eyeLookUpLeft","eyeLookUpRight","eyeSquintLeft","eyeSquintRight",
  "eyeWideLeft","eyeWideRight","jawForward","jawLeft","jawOpen","jawRight","mouthClose",
  "mouthDimpleLeft","mouthDimpleRight","mouthFrownLeft","mouthFrownRight","mouthFunnel","mouthLeft",
  "mouthLowerDownLeft","mouthLowerDownRight","mouthPressLeft","mouthPressRight","mouthPucker","mouthRight",
  "mouthRollLower","mouthRollUpper","mouthShrugLower","mouthShrugUpper","mouthSmileLeft","mouthSmileRight",
  "mouthStretchLeft","mouthStretchRight","mouthUpperUpLeft","mouthUpperUpRight","noseSneerLeft","noseSneerRight",
];

export const VRM_LUT = {
  happy: { mouthSmileLeft: 0.8, mouthSmileRight: 0.8, cheekSquintLeft: 0.5, cheekSquintRight: 0.5 },
  angry: { browDownLeft: 0.8, browDownRight: 0.8, noseSneerLeft: 0.5, noseSneerRight: 0.5 },
  sad: { browInnerUp: 0.9, mouthFrownLeft: 0.7, mouthFrownRight: 0.7 },
  relaxed: { eyeSquintLeft: 0.4, eyeSquintRight: 0.4, mouthSmileLeft: 0.25, mouthSmileRight: 0.25 },
  surprised: { eyeWideLeft: 0.9, eyeWideRight: 0.9, browOuterUpLeft: 0.8, browOuterUpRight: 0.8, jawOpen: 0.35 },
  aa: { jawOpen: 1.0 },
  ih: { mouthStretchLeft: 0.6, mouthStretchRight: 0.6, jawOpen: 0.25 },
  ou: { mouthPucker: 0.9, jawOpen: 0.15 },
  ee: { mouthStretchLeft: 0.9, mouthStretchRight: 0.9, jawOpen: 0.12 },
  oh: { mouthFunnel: 0.8, jawOpen: 0.3 },
  blinkL: { eyeBlinkLeft: 1.0 },
  blinkR: { eyeBlinkRight: 1.0 },
  lookLeft: { eyeLookInRight: 0.8, eyeLookOutLeft: 0.8 },
  lookRight: { eyeLookInLeft: 0.8, eyeLookOutRight: 0.8 },
  lookUp: { eyeLookUpLeft: 0.8, eyeLookUpRight: 0.8 },
  lookDown: { eyeLookDownLeft: 0.8, eyeLookDownRight: 0.8 },
};

export function arkitToVrm(weights) {
  const out = {};
  for (const [preset, contributors] of Object.entries(VRM_LUT)) {
    let acc = 0;
    for (const [ch, coeff] of Object.entries(contributors)) {
      acc += Math.max(weights[ch] || 0, 0) * coeff;
    }
    out[preset] = Math.min(1, Math.max(0, acc));
  }
  return out;
}

export function quatToYawPitch(q) {
  const [w, x, y, z] = q;
  const yaw = Math.atan2(2 * (w * y + x * z), 1 - 2 * (y * y + x * x));
  const pitch = Math.asin(Math.max(-1, Math.min(1, 2 * (w * x - y * z))));
  return { yaw, pitch };
}
