// Stage 2's success criterion, as a measurement: a continuous zoom over many
// orders of magnitude, with no popping and no precision breakdown.
//
//   depth / scale   how far in the flight actually got
//   f32 err/radius  THE PRECISION CLAIM.  A splat's position has to be right
//                   to much less than its own size or it jitters.  Measured
//                   both in the re-rooted frame (what we render) and in world
//                   coordinates (what a renderer without re-rooting is stuck
//                   with), so the difference is visible rather than asserted.
//   ink change      THE POPPING CLAIM.  "Ink" is sum(alpha * projected area):
//                   a continuous LOD changes it smoothly frame to frame, a
//                   popping one does not.  The LOD threshold is FROZEN during
//                   the measurement, so what is left is the cross-fade's doing
//                   and not the budget controller's.

import { Attractor, FlyCamera, AdaptiveStream, PALETTES } from '../js/adaptive.js';

const FRAMES = Number(process.argv[2] ?? 600);
const SPEED = Number(process.argv[3] ?? 0.05);
const TARGET = Number(process.argv[4] ?? 40000);
const W = 1600, H = 900, FOV = 55;
const focal = (H / 2) / Math.tan((FOV * Math.PI) / 360);
const tanY = Math.tan((FOV * Math.PI) / 360), tanX = tanY * (W / H);
const view = { focal, tanX, tanY, znear: 1e-4, W, H };

function setup(fade) {
  const att = new Attractor({ jitter: 0.16, seed: 3, palette: PALETTES.tree });
  const cam = new FlyCamera(att);
  const st = new AdaptiveStream(att, 400000);
  st.budgetBy = 'splats';
  st.target = TARGET;
  if (!fade) { st.lo = 1.8; st.hi = 1.8001; }      // a hard cut, no band
  cam.f = [att.C[0] - cam.pos[0], att.C[1] - cam.pos[1], att.C[2] - cam.pos[2]];
  cam.u = [0, 0, 1];
  cam.ortho();
  // settle the budget controller before anything is measured
  for (let i = 0; i < 90; i++) { st.expand(cam, view); cam.dive(SPEED, 0.08); cam.reroot(); }
  st.budgetBy = 'frozen';                          // nothing adapts from here on
  return { att, cam, st };
}

function run(fade, label) {
  const { cam, st } = setup(fade);
  const rec = [];
  let worstF32 = 0, worstF32World = 0, totalMs = 0, maxMs = 0, budget = 0;

  for (let fr = 0; fr < FRAMES; fr++) {
    cam.dive(SPEED, 0.08);
    cam.reroot();
    const t0 = process.hrtime.bigint();
    const n = st.expand(cam, view);
    const ms = Number(process.hrtime.bigint() - t0) / 1e6;
    totalMs += ms; if (ms > maxMs) maxMs = ms;
    if (st.budgetHit) budget++;

    let ink = 0;
    const ws = cam.scale;
    const base = st.offset;
    for (let j = 0; j < n; j++) {
      const i = base + j, o = 6 * i;
      const r = Math.sqrt(Math.max((st.cov[o] + st.cov[o + 3] + st.cov[o + 5]) / 3, 1e-300));
      const dx = st.pos[3 * i] - cam.pos[0], dy = st.pos[3 * i + 1] - cam.pos[1],
            dz = st.pos[3 * i + 2] - cam.pos[2];
      const z = dx * cam.f[0] + dy * cam.f[1] + dz * cam.f[2];
      if (z > 1e-7) {
        // a splat cannot cover more than the screen, however close or large
        // it is -- without this clamp an ancestor rod that the camera is
        // level with contributes 1e30 and swamps the measurement
        const rp = focal * r / z;
        ink += (st.rgba[4 * i + 3] / 255) * Math.min(Math.PI * rp * rp, W * H);
      }
      if ((j & 63) === 0) {
        for (let k = 0; k < 3; k++) {
          const p = st.pos[3 * i + k];
          let e = Math.abs(Math.fround(p) - p) / r;
          if (e > worstF32) worstF32 = e;
          const pw = p * ws;                        // the same splat, in world
          e = Math.abs(Math.fround(pw) - pw) / (r * ws);
          if (e > worstF32World) worstF32World = e;
        }
      }
    }
    rec.push({ n, ink, depth: cam.depth, scale: ws, nodes: st.nodes });
  }

  const inks = rec.map((r) => r.ink);
  const med = inks.slice().sort((a, b) => a - b)[inks.length >> 1];
  let jump = 0, jumpAt = 0, jsum = 0;
  for (let i = 1; i < rec.length; i++) {
    const j = Math.abs(rec[i].ink - rec[i - 1].ink) / Math.max(med, 1e-12);
    jsum += j;
    if (j > jump) { jump = j; jumpAt = i; }
  }
  const last = rec[rec.length - 1];
  const ns = rec.map((r) => r.n);
  console.log(`\n  ${label}`);
  console.log(`    depth ${last.depth}, local scale ${last.scale.toExponential(2)} `
            + `(${(-Math.log10(last.scale)).toFixed(1)} orders of magnitude), `
            + `${cam.reroots} re-roots`);
  console.log(`    splats mean ${Math.round(ns.reduce((a, b) => a + b, 0) / ns.length)} `
            + `max ${Math.max(...ns)}   expand mean ${(totalMs / FRAMES).toFixed(1)} ms `
            + `max ${maxMs.toFixed(1)} ms` + (budget ? `   BUDGET HIT x${budget}` : ''));
  console.log(`    ink change per frame   mean ${(jsum / (rec.length - 1) * 100).toFixed(3)}%   `
            + `worst ${(jump * 100).toFixed(2)}% (frame ${jumpAt})`);
  console.log(`    float32 pos error / splat radius   re-rooted `
            + `${worstF32.toExponential(2)}   world ${worstF32World.toExponential(2)}`
            + (worstF32 === 0 ? '   (re-rooted positions are exact in float32)' : ''));
  return { jump, jmean: jsum / (rec.length - 1), worstF32, worstF32World, last };
}

console.log(`flight: ${FRAMES} frames, ${SPEED}/frame, ${TARGET} splat target, `
          + `${W}x${H} @ ${FOV} deg`);
const on = run(true, 'cross-fade ON  (band 0.9 - 3.2 px of blob sigma)');
const off = run(false, 'cross-fade OFF (hard cut at 1.8 px)');
console.log(`\n  cross-fade vs hard cut: worst pop `
          + `${(off.jump / Math.max(on.jump, 1e-12)).toFixed(1)}x smaller, `
          + `mean change ${(off.jmean / Math.max(on.jmean, 1e-12)).toFixed(1)}x smaller`);
console.log(`  re-rooting: float32 positions carry `
          + `${(on.worstF32World * 100).toFixed(1)}% of a splat radius of error in world `
          + `coordinates at this depth, and ${(on.worstF32 * 100).toFixed(3)}% re-rooted`);
