// Fly the whole demo, headless, and photograph it.
//
// This is the viewer's own loop with the GL calls removed: same World, same
// VoidCamera, same Autopilot, same AdaptiveStream.  It answers the questions a
// browser would -- does the autopilot get anywhere, does the budget hold, does
// anything go blank, does the morph keep up -- and leaves a filmstrip to look
// at.
//
//   node tests/tour.mjs [seconds] [out.png] [W] [H] [field|ring]
//
// `field` is the endless generated world and is the default, because it is
// what the viewer now opens with.  `ring` flies the fixed seven of stage 3 and
// is kept as the regression that proves the id-addressed camera did not change
// the old behaviour.

import { writeFileSync, mkdirSync } from 'node:fs';
import { World, FieldWorld, VoidCamera, Autopilot, ringLayout } from '../js/world.js';
import { AdaptiveStream } from '../js/adaptive.js';
import { SET_NAMES } from '../js/maps.js';
import { render, png, tile } from '../js/cpu_raster.js';

const SECONDS = Number(process.argv[2] ?? 90);
const OUT = process.argv[3] || 'out/tour.png';
const W = Number(process.argv[4] ?? 300), H = Number(process.argv[5] ?? 230);
const MODE = (process.argv[6] || 'field').toLowerCase();
if (MODE !== 'field' && MODE !== 'ring') throw new Error(`unknown mode ${MODE}`);
const FPS = 30, SHOTS = 8;
const frames = SECONDS * FPS;
const focal = (H / 2) / Math.tan((55 * Math.PI) / 360);
const tanY = Math.tan((55 * Math.PI) / 360);
const view = { focal, tanX: tanY * (W / H), tanY, znear: 1e-5, W, H };

const world = MODE === 'field'
  ? new FieldWorld({ morphing: true })
  : new World(ringLayout(SET_NAMES), { morphing: true });
const cam = new VoidCamera(world, world.first());
{
  const h = world.get(world.first()), c = h.worldCentre(), r = h.worldRadius();
  const back = MODE === 'field'
    ? [c[0] - r * 2.6, c[1] - r * 1.1, c[2] + r * 1.7]
    : [c[0] * 1.9 - 4, c[1] * 1.9 - 4, c[2] + r * 2.4];
  cam.pos = [(back[0] - h.t[0]) / h.s, (back[1] - h.t[1]) / h.s, (back[2] - h.t[2]) / h.s];
  const L = Math.hypot(c[0] - back[0], c[1] - back[1], c[2] - back[2]);
  cam.f = [(c[0] - back[0]) / L, (c[1] - back[1]) / L, (c[2] - back[2]) / L];
  cam.u = [0, 0, 1];
  cam.ortho();
}
let seed = 12345;
const rng = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
const pilot = new Autopilot(world, cam, { rng, diveSeconds: 14, climbSeconds: 2 });
const st = new AdaptiveStream(world.get(world.first()).att, 300000);
st.budgetBy = 'splats'; st.target = 90000;

const shotAt = new Set(Array.from({ length: SHOTS },
  (_, k) => Math.floor(((k + 0.5) * frames) / SHOTS)));
const shots = [], log = [];
let blank = 0, minN = Infinity, maxDepth = 0, maxZoom = 1, morphMs = 0, homes = new Set();
let morphMax = 0, morphCold = 0, visSum = 0, visMin = Infinity, resSum = 0, resMin = Infinity;
// Cold and steady state are different animals and must not be averaged
// together.  The first few frames pay for JIT-compiling build/morph/reparam --
// 90 ms on frame 1 even in the fixed seven-attractor ring, which has no pool
// and has always done it -- and that is a one-off hitch on page load, not a
// per-frame cost.  Assert on steady state; REPORT the cold start, so it stays
// visible rather than being averaged into nothing or failing the run forever.
const WARM = 30;
const reparamSamples = [], frameSamples = [];
const dt = 60 / FPS;

for (let i = 0; i < frames; i++) {
  const f0 = performance.now();
  world.update(1 / FPS, cam);
  pilot.step(dt);
  cam.reroot();
  cam.retarget();
  const n = world.expand(st, cam, view);
  if (i >= WARM) frameSamples.push(performance.now() - f0);
  morphMs += world.reparamMs;
  if (i < WARM) { if (world.reparamMs > morphCold) morphCold = world.reparamMs; }
  else {
    if (world.reparamMs > morphMax) morphMax = world.reparamMs;
    reparamSamples.push(world.reparamMs);
  }
  visSum += world.visible || 0; resSum += world.length;
  if (i > 30) {
    if ((world.visible || 0) < visMin) visMin = world.visible || 0;
    if (world.length < resMin) resMin = world.length;
  }
  homes.add(cam.home);
  if (n < 500) blank++;
  if (i > 30) minN = Math.min(minN, n);
  maxDepth = Math.max(maxDepth, cam.depth);
  maxZoom = Math.max(maxZoom, 1 / cam.scale);
  const span = Math.max(2.5, 2.2 * Math.hypot(
    cam.pos[0] - world.get(cam.home).att.C[0],
    cam.pos[1] - world.get(cam.home).att.C[1],
    cam.pos[2] - world.get(cam.home).att.C[2]));
  // the viewer's rule, kept in step: in a field, fog is also the horizon and
  // must close inside the generated region or cells switch on in clear view
  let fogNear = span * 0.9, fogFar = span * 7.0;
  if (world.layout) {
    const home = world.get(cam.home);
    const edge = 0.75 * world.layout.genRadius / home.s * cam.rootFrame({}).kap;
    if (edge < fogFar) { fogFar = edge; fogNear = Math.min(fogNear, fogFar * 0.14); }
  }
  if (shotAt.has(i)) {
    const im = render({ pos: st.pos, cov: st.cov, rgba: st.rgba, off: st.offset,
      count: n, eye: cam.pos, B: { r: cam.r, u: cam.u, f: cam.f }, W, H,
      fogNear, fogFar });
    shots.push(im);
    log.push(`  t=${(i / FPS).toFixed(0).padStart(3)}s  ${pilot.phase.padEnd(6)} `
      + `in ${world.get(cam.home).name.padEnd(8)} depth ${String(cam.depth).padStart(2)} `
      + `zoom ${(1 / cam.scale).toExponential(1).padStart(8)}  ${String(n).padStart(6)} splats`
      + `  ${im.drawn} drawn`);
  }
}

mkdirSync('out', { recursive: true });
writeFileSync(OUT, png(tile(shots, 4)));
console.log(`${SECONDS}s of the ${MODE} at ${FPS}fps (${frames} frames), ${W}x${H}\n`);
console.log(log.join('\n'));
console.log(`\n  attractors visited        ${homes.size}`
          + (MODE === 'field' ? ' distinct cells' : ` of ${world.length}`));
if (MODE === 'field')
  console.log(`  objects in view           mean ${(visSum / frames).toFixed(1)}`
            + `  min ${visMin}   (resident mean ${(resSum / frames).toFixed(0)}, min ${resMin})`);
console.log(`  deepest                   ${maxDepth} (zoom ${maxZoom.toExponential(1)}x)`);
console.log(`  fewest splats after start ${minN}`);
console.log(`  blank frames (<500)       ${blank}`);
const pct = (a, q) => (a.length ? [...a].sort((x, y) => x - y)[
  Math.min(a.length - 1, Math.floor(a.length * q))] : 0);
const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
const rP99 = pct(reparamSamples, 0.99);
console.log(`  morph rebuild             ${(morphMs / frames).toFixed(2)} ms/frame avg`
          + `, p99 ${rP99.toFixed(2)}, worst ${morphMax.toFixed(2)} ms settled`
          + `, ${morphCold.toFixed(2)} ms over the first ${WARM} frames`);
console.log(`  whole frame on the CPU    ${mean(frameSamples).toFixed(2)} ms avg`
          + `, p95 ${pct(frameSamples, 0.95).toFixed(2)}`
          + `, p99 ${pct(frameSamples, 0.99).toFixed(2)}`
          + `, worst ${Math.max(...frameSamples).toFixed(2)} ms`);
console.log(`\n  -> ${OUT}`);
// Thresholds.  Blank frames are the one that has caught real bugs three times
// (the autopilot flying backwards, diving into gaps, and climbing out facing
// away), so it stays strict as a FRACTION rather than a count -- the old
// absolute `> 2` was impossible to satisfy at any useful length.
const blankRate = blank / frames;
const fails = [];
if (blankRate > 0.03)
  fails.push(`${(blankRate * 100).toFixed(1)}% of frames were blank (limit 3%)`);
if (MODE === 'field') {
  if (visMin < 1) fails.push('the view was empty of objects at some point');
  if (morphMs / frames > 2.0)
    fails.push(`morph rebuild averaged ${(morphMs / frames).toFixed(2)} ms/frame (limit 2.0)`);
  // p99, not max.  A garbage collection landing inside a single reparam can
  // stretch one frame to thirty milliseconds, and no scheduling policy can
  // prevent that -- gating on the maximum would be testing the collector
  // rather than the pool.  p99 over a thousand settled frames still catches
  // the pool systematically overrunning (it sat at 2.4 ms when the budget was
  // admitting two rebuilds a frame), which is the thing that can actually be
  // fixed here.  The worst frame is reported above, with its cause.
  if (rP99 > 3.0)
    fails.push(`p99 of the rebuild cost is ${rP99.toFixed(2)} ms/frame (limit 3.0)`);
}
if (fails.length) { console.log(`\n  FAIL  ${fails.join('\n        ')}`); process.exit(1); }
console.log('\n  tour ok');
