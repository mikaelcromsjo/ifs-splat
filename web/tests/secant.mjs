// Is a straight line good enough to read the morph with?
//
// There are two interpolations in the morph and only one of them is a spline.
// The PATH between map sets is Catmull-Rom -- in log space for quantities that
// must stay positive, with unwrapped angles -- and it is sampled by whatever
// rebuilds the budget affords, 3 to 7 times a second per slot. What reaches
// the screen is those samples read at display rate by the secant through the
// two most recent of them, evaluated at the parameter the morph is actually
// at. That read is linear, and it extrapolates: the fraction is a parameter
// position divided by the distance between the two builds, and the next build
// arrives when the budget allows, so it sits past 1 essentially always.
//
// The pair brackets the present: the build is made about one schedule quantum
// AHEAD of the morph, so the fraction runs 0 -> 1 between rebuilds and the
// hand-over from one secant to the next lands on the sample they share. It
// was not always so -- building at the parameter the morph had already
// reached made every frame an extrapolation and put the hand-over at the tip
// of the old line, where the two disagree, which is what section 6 of
// tests/drift.mjs measures as a jump.
//
// The question a spline would answer is how much the secant misses the curve
// by. This measures it directly: build the exact spec at the same parameter,
// through the same reparam the pool uses, and difference the premultiplied
// child table entry by entry -- the table being where the drawn geometry
// lives. Three errors, all in the same units, so they can be compared:
//
//   secant    the two-table read, extrapolated, as the traversal reads it
//   newest    the newest build alone -- what it drew before interpolating
//   step      how far the drawn table moved since the frame before
//
// `step` is the yardstick. An error well under one frame's own movement is
// below the noise floor of the animation it is drawing, and buying it back
// would cost a third resident table (288 KB a slot) and a third set of reads
// in the hot loop -- which is the budget the blend was moved into the read to
// avoid in the first place.
//
// A spline is even less called for now than it was when the blend only ever
// extrapolated: reading a secant BETWEEN its samples is where a straight line
// is at its best, and the error fell by a factor of 40 at the tail when the
// pair was moved to bracket the present, with no change to the arithmetic.
import { ShapePool } from '../js/field.js';
import { SETS, SET_NAMES, morphPath, build } from '../js/maps.js';
import { Attractor } from '../js/adaptive.js';

const SPECS = SET_NAMES.map((n) => SETS[n]);
const FPS = 60, DT = 1 / FPS, FRAMES = 1800, SETTLE = 180;
// The demo's own settings, not `auto`: under `auto` a headless machine with
// nothing else to do rebuilds 20+ times a second and the secant is reading
// samples four times closer together than a browser ever gives it.
const pool = new ShapePool({ slots: 8, rebuildHz: 7, interpolate: true });
const prio = new Float64Array(pool.slots).fill(1);
// The jitter axis table is derived from the entry index alone, so a reference
// attractor at the same table size holds the same rotations and the two
// tables are comparable entry for entry.
const ref = new Attractor({ jitterTableBits: 10, migrateRebuilds: 1 });

const secant = [], newest = [], step = [], rel = [];
// Bucketed by how far past the newest build the read is extrapolating, which
// is the one thing that should make a straight line worse.
const bins = [[-Infinity, 0], [0, 0.5], [0.5, 1], [1, 1.5], [1.5, Infinity]]
  .map(([a, b]) => ({ a, b, e: [] }));
let below1 = 0, pinned = 0, skipped = 0;
const prev = pool.shapes.map((sh) => new Float64Array(sh.att._cmB.length));
// Frames since this slot finished re-cutting its jitter table.
//
// A resize is ramped over a few rebuilds now (Attractor.migrateRebuilds), and
// while that runs the drawn table is deliberately part way between the
// realisation of the old size and the realisation of the new one -- so an
// exact build at either size is not what it should equal, and the difference
// is the hand-over, not the secant. Those slot-frames are counted and left
// out; tests/drift.mjs is where the hand-over itself is measured, as the
// step it puts on the screen. The pair carries a ramped build for one more
// rebuild interval after the ramp ends, hence the tail.
const sinceMig = new Int32Array(pool.slots).fill(9999);
const seeded = pool.shapes.map(() => false);

for (let f = 0; f < FRAMES; f++) {
  pool.update(DT, prio);
  if (f < SETTLE) continue;                 // the warm-up and the first builds
  for (let k = 0; k < pool.slots; k++) {
    const att = pool.shapes[k].att;
    if (att._migW < 1 || att._migDrop) sinceMig[k] = 0; else sinceMig[k]++;
    if (sinceMig[k] <= 40) { skipped++; seeded[k] = false; continue; }
    const u = pool._u + (k * SET_NAMES.length) / pool.slots;
    const spec = morphPath(SPECS, u);
    const b = build(spec);
    ref.setTableBits(att.tableBits);
    ref.reparam({ maps: b.maps, trunkA: b.trunkA, sig0: b.sig0, mu0: b.mu0,
                  omega: spec.omega ?? 0.28, jitter: spec.jitter ?? 0.16 });
    const T = ref._cmB, A = att._cmA, B = att._cmB, bt = att._bt, P = prev[k];
    const n = 9 * att.m * (att.tableMask + 1);
    let dE = 0, nE = 0, sE = 0, scale = 0;
    for (let i = 0; i < n; i++) {
      const drawn = bt === 1 ? B[i] : A[i] + (B[i] - A[i]) * bt;
      const t = T[i], ad = Math.abs(drawn - t);
      if (ad > dE) dE = ad;
      const an = Math.abs(B[i] - t); if (an > nE) nE = an;
      const as = Math.abs(drawn - P[i]); if (as > sE) sE = as;
      const at = Math.abs(t); if (at > scale) scale = at;
      P[i] = drawn;
    }
    // The first measured frame of a slot has nothing to difference the step
    // against -- `prev` is zeros -- so it is used to seed it and not counted.
    // It read 0.74 on one entry, which is a hundred times the real p99.
    if (!seeded[k]) { seeded[k] = true; continue; }
    secant.push(dE); newest.push(nE); step.push(sE); rel.push(dE / scale);
    if (bt <= 1) below1++;
    if (bt >= att._btSafe - 1e-12 || bt <= att._btSafeLo + 1e-12) pinned++;
    for (const g of bins) if (bt >= g.a && bt < g.b) g.e.push(dE);
  }
}

const q = (a, p) => a[Math.min(a.length - 1, Math.floor(p * a.length))];
const med = (a) => q(a, 0.5);
const e = (v) => v.toExponential(2).padStart(11);
for (const a of [secant, newest, step, rel]) a.sort((x, y) => x - y);

console.log(`  ${secant.length} slot-frames, ${pool.slots} slots, rebuild `
          + `${pool.rebuildHz} Hz, ${FPS} fps`
          + (skipped ? `, ${skipped} left out mid-resize` : '') + `\n`);
console.log('  max abs table error            p50        p90        p99        max');
const row = (label, a) => console.log('  ' + label.padEnd(28) +
  e(q(a, 0.5)) + e(q(a, 0.9)) + e(q(a, 0.99)) + e(a[a.length - 1]));
row('secant vs exact build', secant);
row('newest build vs exact', newest);
row('frame-to-frame step', step);
row('secant, relative to entry', rel);

console.log('\n  by blend fraction      slot-frames        p50        p99        max');
for (const g of bins) {
  g.e.sort((x, y) => x - y);
  const name = g.a === -Infinity ? '< 0'
             : g.b === Infinity ? `>= ${g.a}` : `${g.a} - ${g.b}`;
  console.log('  ' + name.padEnd(22) + String(g.e.length).padStart(10) +
    (g.e.length ? e(q(g.e, 0.5)) + e(q(g.e, 0.99)) + e(g.e[g.e.length - 1]) : ''));
}

let fail = 0;
const check = (ok, msg) => { console.log((ok ? '  ok    ' : '  FAIL  ') + msg); if (!ok) fail++; };
console.log('');
check(med(secant) < 0.5 * med(step),
  `the secant's error is a fraction of one frame's own movement `
  + `(${(med(secant) / med(step)).toFixed(3)} at p50)`);
check(med(secant) < 0.25 * med(newest),
  `and well under what drawing the newest build alone costs `
  + `(${(med(secant) / med(newest)).toFixed(3)} at p50)`);
check(q(rel, 0.99) < 0.01,
  `the drawn table is within 1% of the exact build at p99 (${(100 * q(rel, 0.99)).toFixed(3)}%)`);
// The regime, stated as a check so it cannot quietly go back: the pair
// brackets the present, so this is an INTERPOLATION on all but the frames the
// schedule ran late for. It was the other way round -- every frame an
// extrapolation -- and that is what jumped.
check(below1 > 0.95 * secant.length,
  `the fraction is inside its pair: ${(100 * below1 / secant.length).toFixed(1)}%`
  + ` of slot-frames interpolate rather than extrapolate`);
// A bound that binds is a frozen shape, and releasing it is a step. Both
// bounds exist for a real fault (negative variances; see Attractor._btLimit)
// but in the ordinary case neither should ever be reached.
check(pinned === 0, `no slot-frame is pinned at either bound (${pinned})`);
// And with no extrapolation there is no tail: the thing a spline would have
// been asked to fix.
check(q(rel, 1) < 1e-3,
  `the worst single slot-frame is within 0.1% of the exact build `
  + `(${(100 * q(rel, 1)).toFixed(4)}%)`);

console.log(fail ? `\n  ${fail} FAILED` : '\n  secant ok');
process.exit(fail ? 1 : 0);
