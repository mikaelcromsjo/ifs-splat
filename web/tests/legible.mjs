// Is there anything to SEE?
//
// tour.mjs already fails a flight that goes blank -- fewer than 500 splats on
// screen.  Blank is not the only way to show nothing.  The opposite failure is
// just as bad and far more common: the camera dives until a handful of splats
// each cover a third of the screen, and the frame becomes a flat wash of
// colour with no structure in it at all.  Measured over a stage-4 flight, 28%
// of frames were like that -- more than a quarter of the time the autopilot
// was showing a fog bank.
//
// "Washed out" is made precise here as: the frame is bright enough to be
// showing something, and has almost no spatial detail in it.  Both halves are
// needed.  Low detail alone catches empty space, which is a different fault
// and already measured; brightness alone catches a legitimate close-up of a
// dense object.
//
//   node tests/legible.mjs [seconds] [W] [H] [seed]
//
// The seed is not decoration. The autopilot used to take Math.random, so every
// run was a different flight -- and the dive is chaotic (tests/pilot.mjs: one
// part in a million of speed puts the heading 60 degrees away in three
// seconds), so two runs of the SAME code differ as much as two variants do.
// Every A/B measured with this file before it was seeded was one sample
// against one sample. Seeded, a comparison is paired: the same flight, flown
// by two builds, and a difference that survives several seeds is a
// difference.

import { writeFileSync, mkdirSync } from 'node:fs';
import { AdaptiveStream } from '../js/adaptive.js';
import { FieldWorld, VoidCamera, Autopilot } from '../js/world.js';
import { render, png, tile } from '../js/cpu_raster.js';

const SECONDS = Number(process.argv[2] || 150);
const W = Number(process.argv[3] || 280);
const H = Number(process.argv[4] || 210);
const SEED = Number(process.argv[5] || 1);
const DT = 1 / 30;            // seconds per frame
const TICKS = DT * 60;       // the same step, in the 60fps ticks Autopilot counts
const FOV = 55;
const tanY = Math.tan((FOV * Math.PI) / 360);
const view = { focal: (H / 2) / tanY, tanX: tanY * (W / H), tanY, znear: 1e-5, W, H };

/**
 * Mean luminance, and mean absolute neighbour difference -- a cheap proxy for
 * how much structure the frame has.  Both on the rendered RGB buffer, which is
 * stride 3, not 4.  (It is stride 3.  A probe that assumed 4 once read the
 * green of the next pixel as the alpha of this one and reported NaNs that were
 * not there.)
 *
 * `floor` is the luminance the EMPTY frame already has, and it is the whole
 * difference between this probe meaning something and not.
 *
 * `lit` was a count of pixels over an absolute 8/255. The flat clear colour
 * this rasteriser fills is [0.030, 0.035, 0.047], which comes out at 10-16
 * after gamma -- so every pixel of an empty frame counted as lit, every
 * frame of the crossing scored lit = 1.00 with nothing drawn in it at all,
 * and 99.6% of cruise frames were reported as "washed out" when what they
 * actually were was empty. The two faults want opposite fixes (dive less far
 * in / aim at something on the way over), so a measure that cannot tell them
 * apart is worse than no measure. Counting only pixels that are brighter
 * than the background can tell them apart.
 */
export function detail(im, W, H, floor = 0) {
  const img = im.img;
  let sum = 0, grad = 0, n = 0, lit = 0;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = 3 * (y * W + x);
      const l = (img[i] + img[i + 1] + img[i + 2]) / 3;
      sum += l;
      if (l > floor + 8) lit++;
      if (x + 1 < W && y + 1 < H) {
        const r = 3 * (y * W + x + 1), d = 3 * ((y + 1) * W + x);
        const lr = (img[r] + img[r + 1] + img[r + 2]) / 3;
        const ld = (img[d] + img[d + 1] + img[d + 2]) / 3;
        grad += Math.abs(l - lr) + Math.abs(l - ld);
        n++;
      }
    }
  }
  return { lum: sum / (W * H), grad: grad / Math.max(n, 1), lit: lit / (W * H) };
}

/** Covering most of the screen, and almost no structure in it. */
export const washedOut = (d) => d.lit > 0.45 && d.grad < 3.0;
/** The opposite fault: next to nothing drawn anywhere. */
export const empty = (d) => d.lit < 0.08;

/** What an untouched frame scores, so `lit` can be measured against it. */
export function bgFloor(bg = [0.030, 0.035, 0.047], gamma = 1.1) {
  return (255 / 3) * (Math.pow(bg[0], 1 / gamma) + Math.pow(bg[1], 1 / gamma)
                    + Math.pow(bg[2], 1 / gamma));
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const world = new FieldWorld({});
  const cam = new VoidCamera(world, world.first());
  // The same stream every run, so two builds fly the same flight.
  let sd = (SEED * 2654435761) >>> 0;
  const rng = () => ((sd = (Math.imul(sd, 1103515245) + 12345) >>> 0) / 4294967296);
  const auto = new Autopilot(world, cam, { rng });
  const st = new AdaptiveStream(world.get(world.first()).att, 500000);
  st.budgetBy = 'splats'; st.target = 140000;

  const frames = Math.round(SECONDS / DT);
  const FLOOR = bgFloor();
  let wash = 0, blank = 0, hollow = 0, n = 0;
  const byPhase = new Map();
  const signals = [];
  const shots = [];
  for (let f = 0; f < frames; f++) {
    // Autopilot counts in 60fps ticks, not seconds: TICKS of them per frame.
    // Exactly the viewer's order, and the tour's. Dropping cam.reroot() from
    // this loop leaves depth pinned at 0 while the camera dives: the
    // coordinate chain never follows it, the stream has nothing to draw, and
    // the probe reports 41% blank and 77% washed out for a flight the tour
    // measures at 0 blank. A probe that does not run the real loop measures
    // itself.
    world.update(DT, cam);
    const phase = auto.step(TICKS);
    cam.reroot();
    cam.retarget();
    n = world.expand(st, cam, view);
    // The viewer's fog rule, kept in step: in a field, fog is the horizon and
    // must close inside the generated region.
    const home = world.get(cam.home);
    const span = Math.max(home.att.R * 2, Math.hypot(
      cam.pos[0] - home.att.C[0], cam.pos[1] - home.att.C[1], cam.pos[2] - home.att.C[2]));
    let fogNear = span * 0.9, fogFar = span * 7.0;
    const edge = 0.75 * world.layout.genRadius / home.s * cam.rootFrame({}).kap;
    if (edge < fogFar) { fogFar = edge; fogNear = Math.min(fogNear, fogFar * 0.14); }
    const im = render({ pos: st.pos, cov: st.cov, rgba: st.rgba, off: st.offset, count: n,
                        eye: cam.pos, B: { r: cam.r, u: cam.u, f: cam.f }, W, H,
                        fogNear, fogFar });
    const d = detail(im, W, H, FLOOR);
    const w = washedOut(d);
    const e = empty(d);
    if (w) wash++;
    if (e) hollow++;
    if (n < 500) blank++;
    if (!byPhase.has(phase)) byPhase.set(phase, { n: 0, w: 0, e: 0 });
    const p = byPhase.get(phase); p.n++; if (w) p.w++; if (e) p.e++;
    // Which cheap in-engine signal predicts this? Recorded so the autopilot's
    // turn condition can be chosen from data rather than guessed at.
    signals.push({ w, n, depth: cam.depth, grad: d.grad, lit: d.lit });
    if (f % 150 === 0)
      process.stdout.write(`    ${f}/${frames}  ${phase} d${cam.depth}`
        + `  wash ${(wash / (f + 1) * 100).toFixed(0)}%`
        + `  empty ${(hollow / (f + 1) * 100).toFixed(0)}%\n`);
    if (f % Math.floor(frames / 8) === 0 && shots.length < 8) shots.push(im);
  }

  console.log(`${frames} frames, ${SECONDS}s of autopilot, seed ${SEED}\n`);
  for (const [k, v] of byPhase)
    console.log(`  ${k.padEnd(8)} ${String(v.n).padStart(5)} frames`
              + `  washed out ${(v.w / v.n * 100).toFixed(1)}%`
              + `  nearly empty ${(v.e / v.n * 100).toFixed(1)}%`);
  const rate = wash / frames;
  console.log(`\n  washed-out frames   ${wash}/${frames} = ${(rate * 100).toFixed(1)}%`);
  console.log(`  nearly-empty frames ${hollow}/${frames} = ${(hollow / frames * 100).toFixed(1)}%`);
  console.log(`  blank frames        ${blank}/${frames} = ${(blank / frames * 100).toFixed(1)}%`);
  mkdirSync('out', { recursive: true });
  writeFileSync('out/legible.png', png(tile(shots, 4)));
  console.log('\n  out/legible.png   eight frames across the flight');

  // how well does each candidate signal separate washed-out frames?
  const wash1 = signals.filter((s) => s.w), ok1 = signals.filter((s) => !s.w);
  const avg = (a, k) => (a.length ? a.reduce((x, y) => x + y[k], 0) / a.length : NaN);
  if (wash1.length && ok1.length) {
    console.log('\n  signal           washed-out      legible');
    for (const k of ['n', 'depth', 'grad', 'lit'])
      console.log(`  ${k.padEnd(16)} ${avg(wash1, k).toFixed(2).padStart(10)}`
                + ` ${avg(ok1, k).toFixed(2).padStart(12)}`);
  }

  const LIMIT = 0.10;
  const ok = rate <= LIMIT && blank / frames <= 0.03;
  console.log(ok ? '\n  legible' : `\n  FAIL  over the ${LIMIT * 100}% washed-out limit`);
  process.exit(ok ? 0 : 1);
}
