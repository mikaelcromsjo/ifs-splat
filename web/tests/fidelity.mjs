// Is the adaptive level of detail faithful?
//
// The honest test is not "does it look plausible" but "does it look like the
// thing it is standing in for".  Ground truth here is the same view expanded
// until every splat is sub-pixel and no moment-matched blob is large enough to
// see; the LOD image is the same view on a normal frame budget.  PSNR between
// them is what the approximation actually costs.
//
//   node tests/fidelity.mjs [budget] [truth] [W] [H]

import { writeFileSync, mkdirSync } from 'node:fs';
import { SETS, SET_NAMES, build } from '../js/maps.js';
import { Attractor, FlyCamera, AdaptiveStream, PALETTES } from '../js/adaptive.js';
import { render, png, tile } from '../js/cpu_raster.js';

const BUDGET = Number(process.argv[2] ?? 60000);
const TRUTH = Number(process.argv[3] ?? 900000);
const W = Number(process.argv[4] ?? 360), H = Number(process.argv[5] ?? 290);
const focal = (H / 2) / Math.tan((55 * Math.PI) / 360);
const tanY = Math.tan((55 * Math.PI) / 360);
const view = { focal, tanX: tanY * (W / H), tanY, znear: 1e-5, W, H };

function frame(name, target, dive) {
  const b = build(SETS[name]);
  const att = new Attractor({
    maps: b.maps, trunkA: b.trunkA, sig0: b.sig0, mu0: b.mu0,
    omega: SETS[name].omega ?? 0.28,
    palette: PALETTES[SETS[name].palette] || PALETTES.spectral,
  });
  const cam = new FlyCamera(att);
  cam.f = [att.C[0] - cam.pos[0], att.C[1] - cam.pos[1], att.C[2] - cam.pos[2]];
  cam.u = [0, 0, 1];
  cam.ortho();
  const st = new AdaptiveStream(att, Math.round(target * 1.4));
  st.budgetBy = 'splats'; st.target = target;
  for (let i = 0; i < (dive || 25); i++) {
    st.expand(cam, view);
    if (dive) { cam.dive(0.05, 0.2); cam.reroot(); }
  }
  for (let i = 0; i < 12; i++) st.expand(cam, view);
  const n = st.expand(cam, view);
  return { im: render({ pos: st.pos, cov: st.cov, rgba: st.rgba, off: st.offset,
    count: n, eye: cam.pos, B: { r: cam.r, u: cam.u, f: cam.f }, W, H }), n, cam, att };
}
const psnr = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.img.length; i++) { const d = a.img[i] - b.img[i]; s += d * d; }
  return 10 * Math.log10((255 * 255) / (s / a.img.length));
};

mkdirSync('out', { recursive: true });
console.log(`adaptive LOD at ${BUDGET} splats vs ground truth at ${TRUTH}, ${W}x${H}\n`);
console.log('  set        truth     LOD    ratio   PSNR     verdict');
const ims = [];
let worst = 99;
for (const name of SET_NAMES) {
  const gt = frame(name, TRUTH, 0);
  const lo = frame(name, BUDGET, 0);
  const p = psnr(lo.im, gt.im);
  worst = Math.min(worst, p);
  const ratio = gt.n / lo.n;
  console.log(`  ${name.padEnd(9)} ${String(gt.n).padStart(7)} ${String(lo.n).padStart(7)}`
    + `  ${ratio.toFixed(1).padStart(5)}x  ${p.toFixed(1).padStart(5)} dB  `
    + (p > 30 ? 'faithful' : p > 25 ? 'visible' : 'POOR'));
  ims.push(lo.im, gt.im);
}
writeFileSync('out/fidelity.png', png(tile(ims, 4)));
console.log(`\n  worst ${worst.toFixed(1)} dB   -> out/fidelity.png (LOD, truth, LOD, truth, ...)`);
