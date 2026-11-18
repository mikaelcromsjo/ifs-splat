// Does the cross-fade band remove LOD popping?
//
// Aggregate measures cannot answer this -- thirty thousand splats average each
// other out.  Popping is a property of ONE splat: it is there at full opacity
// in this frame and gone in the next.  So match splats between consecutive
// frames by node identity and ask the only question that matters:
//
//     what is the largest opacity that appears or vanishes in one frame?
//
// With a hard cut that is the blob's full opacity, every time a node crosses
// the threshold.  With a band it should be about (how much t moved) times
// that, which at any sane camera speed is small.

import { Attractor, FlyCamera, AdaptiveStream } from '../js/adaptive.js';

const W = 1600, H = 900, FOV = 55;
const focal = (H / 2) / Math.tan((FOV * Math.PI) / 360);
const tanY = Math.tan((FOV * Math.PI) / 360), tanX = tanY * (W / H);
const view = { focal, tanX, tanY, znear: 1e-4, W, H };

function measure(fade, speed, frames, settle) {
  const att = new Attractor({});
  const cam = new FlyCamera(att);
  cam.f = [att.C[0] - cam.pos[0], att.C[1] - cam.pos[1], att.C[2] - cam.pos[2]];
  cam.u = [0, 0, 1];
  cam.ortho();
  const st = new AdaptiveStream(att, 400000);
  st.budgetBy = 'splats';
  st.target = 40000;
  if (!fade) { st.lo = 1.8; st.hi = 1.8001; }
  for (let i = 0; i < settle; i++) { st.expand(cam, view); cam.dive(0.05, 0.15); cam.reroot(); }
  st.budgetBy = 'frozen';

  let prev = new Map();
  let worst = 0, sumTop = 0, sumAlpha = 0, n = 0;
  for (let fr = 0; fr < frames; fr++) {
    cam.dive(speed, 0.15);
    cam.reroot();
    st.expand(cam, view);
    // only splats well inside the frustum: a node entering at the screen edge
    // appears at full opacity no matter what the LOD does, and that is a
    // property of the camera turning, not of the level of detail
    const cur = new Map();
    const base = st.offset;
    for (let j = 0; j < st.count; j++) {
      const i = base + j;
      const dx = st.pos[3 * i] - cam.pos[0], dy = st.pos[3 * i + 1] - cam.pos[1],
            dz = st.pos[3 * i + 2] - cam.pos[2];
      const z = dx * cam.f[0] + dy * cam.f[1] + dz * cam.f[2];
      if (z <= view.znear) continue;
      const sx = focal * (dx * cam.r[0] + dy * cam.r[1] + dz * cam.r[2]) / z;
      const sy = focal * (dx * cam.u[0] + dy * cam.u[1] + dz * cam.u[2]) / z;
      if (Math.abs(sx) > W * 0.35 || Math.abs(sy) > H * 0.35) continue;
      cur.set(st.id[i], st.rgba[4 * i + 3] / 255);
    }
    if (prev.size) {
      let popped = 0, hard = 0;
      for (const [k, a] of cur) {
        const d = Math.abs(a - (prev.get(k) ?? 0));
        popped += d;
        if (d > 0.5) hard++;
      }
      for (const [k, a] of prev) if (!cur.has(k)) { popped += a; if (a > 0.5) hard++; }
      worst = Math.max(worst, hard);
      sumTop += hard; n++;
      sumAlpha += popped / Math.max(cur.size, 1);
    }
    prev = cur;
  }
  return { worst, mean: sumTop / Math.max(n, 1),
           alpha: sumAlpha / Math.max(n, 1), depth: cam.depth };
}

console.log('splats in the middle of the screen whose opacity jumps by more than');
console.log('0.5 in a single frame -- a pop you would see\n');
console.log('   speed   cross-fade band           hard cut                  reduction');
for (const sp of [0.05, 0.02, 0.008]) {
  const on = measure(true, sp, 160, 90);
  const off = measure(false, sp, 160, 90);
  console.log(`  ${String(sp).padStart(6)}   `
    + `${on.mean.toFixed(1)}/frame (worst ${on.worst})`.padEnd(26)
    + `${off.mean.toFixed(1)}/frame (worst ${off.worst})`.padEnd(26)
    + `${(off.mean / Math.max(on.mean, 1e-9)).toFixed(0)}x fewer`);
}
