// Does ordered expansion still hold up in a CROWD?
//
// Stage 0 measured the approximation inside one attractor: the stream is not
// sorted, it is a lexicographic walk under a per-node digit permutation, and
// the question was whether the resulting order is close enough to a true
// back-to-front sort that nobody can see the difference.  It is -- 14.7% mean
// and 32.3% worst opacity-weighted inversion rate, which looks fine.
//
// A field asks the question again one level up, and the answer is NOT implied
// by the first one.  Within a subtree the order is exact.  BETWEEN two
// separate attractors nothing orders them except the order they are streamed,
// and that order is a sort by nearest point of the bounding sphere -- a
// heuristic, not a truth.  Two objects side by side at similar depth can be
// streamed in the wrong order wholesale, and that is a different kind of error
// from a few misplaced splats inside one object: it is every splat of one
// object against every splat of the other.
//
// So this measures the whole frame against an exact per-splat sort, and also
// renders both, because the inversion rate is a proxy and the image is not.
//
//   node tests/order_field.mjs [dirs] [W] [H]

import { writeFileSync, mkdirSync } from 'node:fs';
import { AdaptiveStream } from '../js/adaptive.js';
import { World, FieldWorld, VoidCamera, ringLayout } from '../js/world.js';
import { InversionCounter } from '../js/inversions.js';
import { render, png, tile } from '../js/cpu_raster.js';

const DIRS = Number(process.argv[2] || 24);
const W = Number(process.argv[3] || 300);
const H = Number(process.argv[4] || 230);

const FOV = 55;
const tanY = Math.tan((FOV * Math.PI) / 360);
const view = { focal: (H / 2) / tanY, tanX: tanY * (W / H), tanY, znear: 1e-5, W, H };

/** The exact back-to-front order of the live stream, farthest first. */
function exactOrder(st, n, cam) {
  const off = st.offset;
  const z = new Float64Array(n);
  const idx = new Int32Array(n);
  for (let k = 0; k < n; k++) {
    const p = 3 * (off + k);
    z[k] = (st.pos[p] - cam.pos[0]) * cam.f[0]
         + (st.pos[p + 1] - cam.pos[1]) * cam.f[1]
         + (st.pos[p + 2] - cam.pos[2]) * cam.f[2];
    idx[k] = k;
  }
  // farthest first, which is the order the stream claims to already be in
  const order = Array.from(idx).sort((a, b) => z[b] - z[a]);
  return Int32Array.from(order);
}

/** Re-pack the stream into a given order, so it can be rasterised. */
function permute(st, n, order) {
  const off = st.offset;
  const pos = new Float32Array(3 * n), cov = new Float32Array(6 * n);
  const rgba = new Uint8Array(4 * n);
  for (let k = 0; k < n; k++) {
    const src = off + order[k];
    for (let c = 0; c < 3; c++) pos[3 * k + c] = st.pos[3 * src + c];
    for (let c = 0; c < 6; c++) cov[6 * k + c] = st.cov[6 * src + c];
    for (let c = 0; c < 4; c++) rgba[4 * k + c] = st.rgba[4 * src + c];
  }
  return { pos, cov, rgba };
}

function shade(cam, world, st, counter, wantImages) {
  const n = world.expand(st, cam, view);
  if (n < 200) return null;
  const off = st.offset;
  const alpha = new Float64Array(n);
  for (let k = 0; k < n; k++) alpha[k] = st.rgba[4 * (off + k) + 3] / 255;
  const stream = new Int32Array(n);
  for (let k = 0; k < n; k++) stream[k] = k;
  const truth = exactOrder(st, n, cam);
  const r = counter.measure(stream, truth, alpha, n);

  let imgs = null, pix = null;
  if (wantImages) {
    const common = { eye: cam.pos, B: { r: cam.r, u: cam.u, f: cam.f }, W, H,
                     fogNear: 1e9, fogFar: 2e9 };
    const a = render({ pos: st.pos, cov: st.cov, rgba: st.rgba, off, count: n, ...common });
    const p = permute(st, n, truth);
    const b = render({ pos: p.pos, cov: p.cov, rgba: p.rgba, off: 0, count: n, ...common });
    let sum = 0, max = 0;
    for (let i = 0; i < a.img.length; i++) {
      const d = Math.abs(a.img[i] - b.img[i]);
      sum += d; if (d > max) max = d;
    }
    pix = { mean: sum / a.img.length, max };
    imgs = [b, a];
  }
  return { rate: r.rate, n, visible: world.visible, pix, imgs };
}

function sweep(world, cam, label, { images = false } = {}) {
  const st = new AdaptiveStream(world.get(world.first()).att, 400000);
  st.budgetBy = 'splats';
  st.target = 120000;
  const counter = new InversionCounter(1 << 16);

  // let the LOD controller settle before measuring anything
  for (let i = 0; i < 8; i++) world.expand(st, cam, view);

  const rates = [], vis = [], pixes = [];
  let shots = null, worst = -1, worstAt = 0;
  for (let d = 0; d < DIRS; d++) {
    cam.orient((2 * Math.PI) / DIRS, d % 6 === 5 ? 0.22 : 0);
    const want = images && d % Math.ceil(DIRS / 3) === 0;
    const r = shade(cam, world, st, counter, want);
    if (!r) continue;
    rates.push(r.rate); vis.push(r.visible);
    if (r.pix) {
      pixes.push(r.pix);
      if (!shots) shots = [];
      shots.push(...r.imgs);
    }
    if (r.rate > worst) { worst = r.rate; worstAt = d; }
  }
  if (!rates.length) throw new Error(`${label}: every direction was empty`);
  const mean = rates.reduce((a, b) => a + b, 0) / rates.length;
  const sorted = [...rates].sort((a, b) => a - b);
  const med = sorted[sorted.length >> 1];
  const vmean = vis.reduce((a, b) => a + b, 0) / vis.length;

  console.log(`  ${label.padEnd(22)} inversions  mean ${(mean * 100).toFixed(2)}%`
            + `  median ${(med * 100).toFixed(2)}%  worst ${(worst * 100).toFixed(2)}%`
            + ` (dir ${worstAt})`);
  console.log(`  ${''.padEnd(22)} objects in view  mean ${vmean.toFixed(1)}`
            + `  max ${Math.max(...vis)}`);
  if (pixes.length) {
    const pm = pixes.reduce((a, b) => a + b.mean, 0) / pixes.length;
    const px = Math.max(...pixes.map((p) => p.max));
    console.log(`  ${''.padEnd(22)} vs exact sort    mean ${pm.toFixed(3)}/255`
              + `  worst pixel ${px.toFixed(1)}/255`);
  }
  return { mean, med, worst, vmean, shots };
}

console.log(`ordered expansion in a crowd: ${DIRS} directions, ${W}x${H}\n`);

// ---- baseline: one attractor, the number stage 0 established
const ring = new World(ringLayout(), { morphing: false });
const rcam = new VoidCamera(ring, ring.first());
{
  const h = ring.get(ring.first());
  rcam.pos = [h.att.C[0], h.att.C[1] - h.att.R * 2.4, h.att.C[2]];
  rcam.f = [0, 1, 0]; rcam.u = [0, 0, 1]; rcam.ortho();
}
const base = sweep(ring, rcam, 'ring (7 objects)');

// ---- the field
const field = new FieldWorld({ morphing: false });
const fcam = new VoidCamera(field, field.first());
{
  const h = field.get(field.first());
  const c = h.worldCentre(), r = h.worldRadius();
  fcam.pos = [(c[0] - r * 3 - h.t[0]) / h.s, (c[1] - h.t[1]) / h.s, (c[2] - h.t[2]) / h.s];
  fcam.f = [1, 0, 0]; fcam.u = [0, 0, 1]; fcam.ortho();
}
console.log();
const fld = sweep(field, fcam, 'field', { images: true });

if (fld.shots && fld.shots.length) {
  mkdirSync('out', { recursive: true });
  writeFileSync('out/order_field.png', png(tile(fld.shots, 2, 6)));
  console.log('\n  out/order_field.png   exact sort | ordered expansion, per row');
}

// ---- verdict
console.log();
const LIMIT = 0.40;
let bad = 0;
if (fld.worst > LIMIT) {
  console.log(`  FAIL  worst-case inversion rate ${(fld.worst * 100).toFixed(2)}% is over the`
            + ` ${(LIMIT * 100).toFixed(0)}% limit.`);
  console.log('        Off-ramp: lower rho so objects sit further apart, or split each');
  console.log("        object's stream into depth slabs and merge across objects.");
  bad = 1;
} else if (fld.worst > base.worst * 1.6) {
  console.log(`  WARN  the crowd is ${(fld.worst / base.worst).toFixed(2)}x worse than one`
            + ' attractor at its worst. Under the limit, but worth watching.');
} else {
  console.log(`  ok    a crowd costs ${(fld.mean / base.mean).toFixed(2)}x the single-object`
            + ' inversion rate, and stays under the limit.');
}
console.log(`        single object  mean ${(base.mean * 100).toFixed(2)}%`
          + `  worst ${(base.worst * 100).toFixed(2)}%`);
console.log(`        the field      mean ${(fld.mean * 100).toFixed(2)}%`
          + `  worst ${(fld.worst * 100).toFixed(2)}%`);
process.exit(bad);
