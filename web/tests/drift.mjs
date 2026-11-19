// Where a SPLAT is, drawn against true, frame by frame.
//
// Everything else that measures the morph measures a parameter or a table
// entry. A viewer sees neither: they see a splat at a position, and that
// position is the premultiplied table composed down an address, times the
// interpolated moment centre, plus the interpolated translations. Each of
// those is blended separately and a product of blends is not the blend of a
// product, so the only honest place to ask "is the interpolation right" is
// at the position.
//
// So: walk fixed addresses the way visit() walks them, through the drawn
// tables at the drawn fraction, and through an exact build at the same morph
// parameter, and difference the two points. Fixed addresses and one hash
// chain for both walks -- the root hash decides which jitter entry every node
// reads, so walking the reference from its own root compares two different
// realisations of the same set and calls the difference an error. That
// mistake read 1.5e-2 of the radius whichever table was read, which is how it
// was caught.
//
// Three questions, in the order a viewer would notice them:
//
//   error       how far the drawn splat is from the true one
//   step        how far the drawn splat moves in a frame, against how far the
//               true one moves -- a JUMP is a step far bigger than the truth
//   direction   the dot product of the two steps; negative is moving the
//               wrong way along the morph
//
// The second is what this was written for, and it was red: 17 slot-frames of
// 10,552 stepped more than 8x the true motion, up to 36x, every one of them
// on a rebuild frame, plus 7 that moved against the morph and a worst
// position error of 9.6e-3 of the radius. Three separate things caused it,
// and each had to go:
//
//   1. The build was made at the parameter the morph had ALREADY reached, so
//      the fraction reset to exactly 1 on every rebuild and the blend could
//      only ever extrapolate. The hand-over from one secant to the next then
//      happened at the tip of the old one, where two lines disagree most.
//      Building one schedule quantum AHEAD puts the pair either side of the
//      present: the fraction runs 0 -> 1 and the hand-over lands on the
//      sample both lines share, where they agree exactly.
//   2. A table resize threw the pair away, so a slot drew one shape for a
//      whole rebuild interval and then resumed: 9.4e-3 at p99 inside 40
//      frames of a resize against 6.0e-5 settled, and the pool resizes 150
//      times a minute.
//   3. The resize also FORCED a rebuild immediately, a quarter of a quantum
//      after the last one, which asked for a fraction of -3.06 -- clamped,
//      and a clamp is a freeze whose release is a step.
//
// With all three fixed this read 0 and 0 -- and the picture was still jumpy
// at times, which is a FOURTH thing and the reason for section 2 below.
//
// The step column was the clue: 4.28e-2 of the radius at its worst against
// 2.61e-3 at p99, 86x one frame's own motion. Every one of the fourteen
// largest steps in the flight was a frame on which a slot's jitter TABLE
// changed size. A node picks its rotation with `hash & tableMask`, so a
// resize re-draws the branch jitter of every node whose index moves -- a
// different realisation of the same set, in one frame, on whichever object
// had just become the most important one on screen. The pool does it 150
// times a minute.
//
// Nothing here could see it, and that is worth stating plainly: this test
// compares the drawn table against an exact build AT THE SAME MASK, so the
// reference re-draws its jitter in the same frame and the same way. The
// error stayed at 1e-5 while the picture snapped. Only the step saw it, and
// the step was only ever checked as a RATIO against the true motion -- which
// was equally large, so the ratio was 1.00.
//
// The fix is Attractor.migrateRebuilds: a resize is ramped over several
// rebuilds, starting from exact copies of the entries the nodes were already
// reading, so the mask moves without the picture moving and the new draw
// arrives at one fourth of a rebuild's own step at a time. Section 2 is the
// assertion that was missing: an absolute bound on the worst step in the
// whole flight, not a ratio.
//
// While a ramp is running the drawn shape is deliberately BETWEEN two
// realisations, so an exact build at the new size is not what it should
// equal and section 1's tracking checks do not apply to those frames. They
// are reported separately and bounded by their step instead, which is the
// honest statement: a blend of two valid realisations of the same set at the
// same morph parameter is a valid shape, and what must be true of it is that
// it arrives smoothly.
import { ShapePool } from '../js/field.js';
import { SETS, SET_NAMES, morphPath, build } from '../js/maps.js';
import { Attractor } from '../js/adaptive.js';
import { mix } from '../js/ifs_core.js';

const SPECS = SET_NAMES.map((n) => SETS[n]);
const FPS = 60, DT = 1 / FPS, FRAMES = 1500, SETTLE = 240;
const DEPTH = 4, NADDR = 24;

// Fixed addresses: the same splats every frame, so a difference is the
// interpolation and not a different splat.
const addrs = [];
for (let a = 0; a < NADDR; a++) {
  const d = []; let x = (a * 2654435761) >>> 0;
  for (let k = 0; k < DEPTH; k++) { x = (x * 1664525 + 1013904223) >>> 0; d.push(x % 4); }
  addrs.push(d);
}

// The node centre each address lands on: CA/CB/CC in Attractor.visit(), by
// hand, at whatever fraction the slot is drawing.
const where = (att, blend, out, root) => {
  const CM = att._cmB, CMA = att._cmA, MB = att.mapB, m = att.m, tm = att.tableMask;
  const bt = blend ? att._bt : 1, C = att.C;
  for (let a = 0; a < addrs.length; a++) {
    let A = [1, 0, 0, 0, 1, 0, 0, 0, 1], b = [0, 0, 0], h = root;
    for (const i of addrs[a]) {
      const e = 9 * ((mix(h, i) & tm) * m + i), E = [];
      for (let k = 0; k < 9; k++) {
        const v = CM[e + k], p = CMA[e + k];
        E.push(bt === 1 ? v : p + (v - p) * bt);
      }
      const nb = [0, 0, 0], nA = new Array(9).fill(0);
      for (let r = 0; r < 3; r++) {
        nb[r] = b[r];
        for (let c = 0; c < 3; c++) {
          nb[r] += A[3 * r + c] * MB[3 * i + c];
          for (let k = 0; k < 3; k++) nA[3 * r + c] += A[3 * r + k] * E[3 * k + c];
        }
      }
      A = nA; b = nb; h = mix(h, i);
    }
    for (let r = 0; r < 3; r++)
      out[3 * a + r] = A[3 * r] * C[0] + A[3 * r + 1] * C[1] + A[3 * r + 2] * C[2] + b[r];
  }
};
// Mean distance over the addresses, in units of the object's own radius, so
// eight objects of different sizes can be averaged without the big ones
// drowning the small.
const dist = (x, y, R) => {
  let s = 0;
  for (let i = 0; i < x.length; i += 3)
    s += Math.hypot(x[i] - y[i], x[i + 1] - y[i + 1], x[i + 2] - y[i + 2]);
  return s / (x.length / 3) / R;
};

// The demo's own settings. The priority MOVES, because a flight promotes and
// demotes objects continuously and that is what drives the table resizes --
// with every slot equally important nothing is ever resized and the part of
// this that covers resizes is vacuous.
const pool = new ShapePool({ slots: 8, seed: 7, rebuildHz: 7, interpolate: true });
const prio = new Float64Array(pool.slots);
const ref = new Attractor({ jitterTableBits: 10, migrateRebuilds: 1 });

const nd = 3 * NADDR, mk = () => pool.shapes.map(() => new Float64Array(nd));
const dB = mk(), dN = mk(), dT = mk(), pB = mk(), pT = mk();
const err = [], errNo = [], steps = [], tsteps = [], settled = [], afterResize = [];
const jumps = [], revs = [];
// Steps split by whether a resize ramp is in flight (or has just finished,
// so that the older half of the blend's pair still holds a ramped build).
const stepRamp = [], stepCalm = [];
const since = new Int32Array(pool.slots).fill(9999);
const sinceMig = new Int32Array(pool.slots).fill(9999);
const bits = pool.shapes.map((s) => s.att.tableBits);
let sf = 0, resizes = 0, pinned = 0, ramping = 0;

for (let f = 0; f < FRAMES; f++) {
  for (let k = 0; k < pool.slots; k++)
    prio[k] = 0.5 + 0.5 * Math.sin(0.02 * f + k * 1.7);
  pool.update(DT, prio);
  for (let k = 0; k < pool.slots; k++) {
    const sh = pool.shapes[k], att = sh.att;
    const u = pool._u + (k * SET_NAMES.length) / pool.slots;
    const spec = morphPath(SPECS, u), b = build(spec);
    ref.setTableBits(att.tableBits);
    ref.reparam({ maps: b.maps, trunkA: b.trunkA, sig0: b.sig0, mu0: b.mu0,
                  omega: spec.omega ?? 0.28, jitter: spec.jitter ?? 0.16 });
    where(att, true, dB[k], att.rootHash);     // what is drawn
    where(att, false, dN[k], att.rootHash);    // what would be drawn uninterpolated
    where(ref, false, dT[k], att.rootHash);    // the truth at this parameter
    if (att.tableBits !== bits[k]) { bits[k] = att.tableBits; since[k] = 0; resizes++; }
    else since[k]++;
    // A ramp, or the tail of one: `_migW` reaches 1 on the rebuild that
    // lands the last step, and the pair still carries the one before it for
    // a rebuild interval after that.
    const mig = att._migW < 1 || att._migDrop;
    if (mig) sinceMig[k] = 0; else sinceMig[k]++;
    const calm = sinceMig[k] > 40;
    const R = att.R || 1;
    if (f > SETTLE) {
      const s = dist(dB[k], pB[k], R), ts = dist(dT[k], pT[k], R);
      steps.push(s);
      (calm ? stepCalm : stepRamp).push(s);
      if (!calm) ramping++;
      if (att._bt >= att._btSafe - 1e-12 || att._bt <= att._btSafeLo + 1e-12) pinned++;
      // Section 1 -- how well the drawn shape tracks an exact build at the
      // same parameter -- is only defined where no realisation hand-over is
      // in flight; see the header.
      if (calm) {
        const e = dist(dB[k], dT[k], R);
        err.push(e); errNo.push(dist(dN[k], dT[k], R));
        (since[k] > 40 ? settled : afterResize).push(e);
        tsteps.push(ts);
        let dot = 0;
        for (let i = 0; i < nd; i++) dot += (dB[k][i] - pB[k][i]) * (dT[k][i] - pT[k][i]);
        if (dot < 0) revs.push(s);
        if (ts > 1e-12 && s > 8 * ts) jumps.push({ f, k, x: s / ts, bt: att._bt });
      }
      sf++;
    }
    pB[k].set(dB[k]); pT[k].set(dT[k]);
  }
}

const q = (a, p) => { const s = a.slice().sort((x, y) => x - y);
  return s.length ? s[Math.min(s.length - 1, Math.floor(p * s.length))] : NaN; };
const e3 = (v) => (Number.isNaN(v) ? '          -' : v.toExponential(2).padStart(11));
const row = (label, a) => console.log('  ' + label.padEnd(20) +
  e3(q(a, .5)) + e3(q(a, .9)) + e3(q(a, .99)) + e3(q(a, 1)));

console.log(`  ${sf} slot-frames, ${NADDR} addresses at depth ${DEPTH}, ${resizes} table`
          + ` resizes, ${(100 * ramping / sf).toFixed(0)}% of slot-frames mid-resize`
          + `\n  positions in units of each object's own radius\n`);
console.log('                            p50        p90        p99        max');
row('error, drawn', err);
row('error, no interp', errNo);
row('step, drawn', steps);
row('step, true', tsteps);
row('error, settled pair', settled);
row('error, after resize', afterResize);
console.log('');
row('step, no resize', stepCalm);
row('step, mid-resize', stepRamp);

let fail = 0;
const check = (ok, msg) => { console.log((ok ? '  ok    ' : '  FAIL  ') + msg); if (!ok) fail++; };
console.log('');
// The headline: no frame may move the drawn splat much further than the true
// splat moved. This is the one that was red, at 17 frames and up to 36x.
check(jumps.length === 0, `no frame jumps: every drawn step is within 8x the true`
  + ` motion (${jumps.length} over 8x`
  + (jumps.length ? `, worst ${q(jumps.map((j) => j.x), 1).toFixed(1)}x` : '') + ')');
check(revs.length === 0,
  `nothing ever moves against the morph (${revs.length} slot-frames with a negative step)`);
// Tracking, not just smoothness: a shape can move smoothly and be the wrong
// shape. Against the exact build, and against the alternative of not
// interpolating at all.
check(q(err, 1) < 1e-3, `the worst drawn splat is within 1e-3 of the radius of the`
  + ` true one (${e3(q(err, 1)).trim()})`);
check(q(err, .5) < 0.1 * q(errNo, .5), `and interpolating is what does it:`
  + ` ${(q(errNo, .5) / q(err, .5)).toFixed(0)}x closer than the newest build alone`);
// The resize used to throw the blend's pair away, which stopped a slot moving
// for a whole rebuild interval: 9.4e-3 at p99 against 6.0e-5 settled.
check(q(afterResize, .99) < 4 * q(settled, .99),
  `a table resize is not a special case any more (p99 ${e3(q(afterResize, .99)).trim()}`
  + ` after one against ${e3(q(settled, .99)).trim()} settled)`);
check(pinned === 0, `no slot-frame is pinned at a blend bound (${pinned})`);
// Speed, as a distribution: the drawn motion should BE the true motion, not
// merely be of the same order.
for (const p of [0.5, 0.9, 0.99]) {
  const r = q(steps, p) / q(tsteps, p);
  check(r > 0.9 && r < 1.1,
    `the drawn motion is the true motion at p${(100 * p).toFixed(0)} (${r.toFixed(3)}x)`);
}

// ---- 2. the step, absolutely: a resize may not be a movement of its own.
//
// Everything above is a ratio against the true motion, and a resize moved
// the reference exactly as far as it moved the picture, so a ratio could not
// see it. These are absolute, against the distribution of ordinary frames.
console.log('');
const calm99 = q(stepCalm, .99), calm50 = q(stepCalm, .5);
check(q(steps, 1) < 1.5 * calm99,
  `no slot-frame in the flight steps further than 1.5x the p99 of an ordinary`
  + ` frame (worst ${e3(q(steps, 1)).trim()} against ${e3(calm99).trim()};`
  + ` ${(q(steps, 1) / calm50).toFixed(0)}x the median, where switching a`
  + ` resize instead of ramping it is 86x)`);
// Vacuous with the ramp switched off (migrateRebuilds 1), which is the A/B:
// there the headline above is what goes red, at 86x.
if (stepRamp.length) {
check(q(stepRamp, 1) < calm99 * 1.5,
  `a resize is not a step any more: worst mid-resize frame`
  + ` ${e3(q(stepRamp, 1)).trim()} against an ordinary p99 of`
  + ` ${e3(calm99).trim()}`);
check(q(stepRamp, .5) < 2 * calm50,
  `and it is not a lurch either: median mid-resize step`
  + ` ${e3(q(stepRamp, .5)).trim()} against ${e3(calm50).trim()} ordinary`);
}
check(ramping > 0.02 * sf, `the resizes are actually being exercised`
  + ` (${(100 * ramping / sf).toFixed(0)}% of slot-frames mid-resize)`);

console.log(fail ? `\n  ${fail} FAILED` : '\n  drift ok');
process.exit(fail ? 1 : 0);
