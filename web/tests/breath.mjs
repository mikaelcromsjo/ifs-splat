// Does the world actually breathe, and does it never stop or jump?
//
// Stage 4 morphed in bursts: hold a set for 10 seconds, cross-fade to the next
// over 14, hold again.  That has two faults.  The holds are corners -- the
// shape is motionless, then abruptly moving, then motionless -- and between
// them the rate is constant, so a transition reads as a mechanical slide
// rather than as something alive.
//
// Stage 5 replaces both: a closed Catmull-Rom spline through the map sets, so
// the path has no corners, driven by a breathing RATE rather than a breathing
// position.  The distinction matters.  The holds were the only thing making a
// shape legible -- long enough in one configuration to read as a particular
// fractal.  A constant rate along a smooth path would be perpetual mush.  A
// rate that falls almost to rest gives the legibility back without putting the
// corner back.
//
// So, four claims:
//
//   1. the rate is a breath: periodic, smooth, bounded, strictly positive
//   2. the path is C1 -- in particular ACROSS a waypoint, which is exactly
//      where the old chain of straight morphs had its corner
//   3. the morph never stalls and never jumps
//   4. it fits the budget, and the cheap frames are the legible ones
//   5. and the shape DRAWN moves at the display's rate, not the rebuild rate
//
// Claim 5 is a different claim from 3. Three is about the path through shape
// space: smooth, never still. Five is about what reaches the screen, which is
// a sampling of that path -- and a smooth path sampled 20 times a second and
// held between samples is judder. It was judder: before read-time
// interpolation, 61% of slot-frames drew exactly what the frame before drew.
//
//   node tests/breath.mjs

import { SETS, SET_NAMES, build, morph, morphPath } from '../js/maps.js';
import { ShapePool } from '../js/field.js';

const SPECS = SET_NAMES.map((n) => SETS[n]);
let bad = 0;
const ok = (cond, name, detail = '') => {
  if (!cond) bad = 1;
  console.log(`  ${name.padEnd(52)} ${detail}  ${cond ? 'ok' : 'FAILED'}`);
};

// A single number summarising a shape, so a path through shape space can be
// differentiated.  Every scalar the spline touches, concatenated.
// Angles go in as their cosine and sine, never as degrees.  The spline
// unwraps each segment's angle controls around that segment's own start, so
// two segments can describe the same orientation 360 apart -- a difference
// that means nothing to the shape but would show up here as an enormous jump
// and make every smoothness measurement below a measurement of bookkeeping.
const ang = (d) => { const r = (d * Math.PI) / 180; return [Math.cos(r), Math.sin(r)]; };
const vec = (spec) => {
  const v = [];
  for (const m of spec.maps) {
    v.push(m.s, ...ang(m.tilt), ...ang(m.roll), ...ang(m.twist), ...m.aniso, ...m.b);
  }
  v.push(...spec.sig0, ...spec.mu0, spec.omega, spec.jitter);
  return v;
};
const dist = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s);
};

console.log('the breathing morph\n');

// ------------------------------------------------------------- 1. the breath
console.log('1. the rate is a breath');
{
  const pool = new ShapePool({ slots: 4, seed: 11 });
  const dt = 1 / 60;
  const T = pool.breathSeconds;
  const rates = [], us = [];
  for (let i = 0; i < Math.round((T * 3) / dt); i++) {
    pool.update(dt, null);
    rates.push(pool.breath); us.push(pool._u);
  }
  const lo = Math.min(...rates), hi = Math.max(...rates);
  ok(lo >= pool.breathMin - 1e-9 && hi <= pool.breathMax + 1e-9,
     'the rate stays between breathMin and breathMax',
     `${lo.toFixed(3)} .. ${hi.toFixed(3)} vs [${pool.breathMin}, ${pool.breathMax}]`);
  ok(lo > 0, 'and never reaches zero -- it slows, it does not stop',
     `slowest ${lo.toFixed(4)} (${(lo / hi * 100).toFixed(1)}% of fastest)`);

  // periodic: the rate one period later is the same rate
  const n = Math.round(T / dt);
  let worstP = 0;
  for (let i = 0; i + n < rates.length; i++)
    worstP = Math.max(worstP, Math.abs(rates[i + n] - rates[i]));
  ok(worstP < 1e-3, 'the period is the breathSeconds it claims',
     `${T}s; worst mismatch one period apart ${worstP.toExponential(1)}`);

  // smooth: the rate's own derivative has no jump
  let worstJerk = 0;
  for (let i = 2; i < rates.length; i++) {
    const d1 = rates[i] - rates[i - 1], d0 = rates[i - 1] - rates[i - 2];
    worstJerk = Math.max(worstJerk, Math.abs(d1 - d0));
  }
  const span = (hi - lo);
  ok(worstJerk < span * 1e-3, 'the rate itself is smooth -- no corner in the breath',
     `worst second difference ${(worstJerk / span).toExponential(1)} of the rate's range`);

  // monotone: u only ever goes forward
  let back = 0;
  for (let i = 1; i < us.length; i++) if (us[i] <= us[i - 1]) back++;
  ok(back === 0, 'the morph only ever moves forward', `${back} steps that stalled or reversed`);
}

// --------------------------------------------------------- 2. C1 at waypoints
console.log('\n2. the path has no corner at a waypoint');
{
  // One-sided derivatives either side of an integer u -- a waypoint, where the
  // old chain of straight morphs changed which pair it was interpolating and
  // the velocity jumped.
  const h = 1e-5;
  let worstSpline = 0, worstLinear = 0;
  for (let i = 0; i < SET_NAMES.length; i++) {
    const before = dist(vec(morphPath(SPECS, i - h)), vec(morphPath(SPECS, i - 2 * h))) / h;
    const after = dist(vec(morphPath(SPECS, i + 2 * h)), vec(morphPath(SPECS, i + h))) / h;
    const rel = Math.abs(after - before) / Math.max(before, after, 1e-9);
    worstSpline = Math.max(worstSpline, rel);

    // the same measurement on the old scheme, for contrast
    const A = SETS[SET_NAMES[(i - 1 + SET_NAMES.length) % SET_NAMES.length]];
    const B = SETS[SET_NAMES[i]], C = SETS[SET_NAMES[(i + 1) % SET_NAMES.length]];
    const lb = dist(vec(morph(A, B, 1 - h)), vec(morph(A, B, 1 - 2 * h))) / h;
    const la = dist(vec(morph(B, C, 2 * h)), vec(morph(B, C, h))) / h;
    worstLinear = Math.max(worstLinear, Math.abs(la - lb) / Math.max(la, lb, 1e-9));
  }
  ok(worstSpline < 0.02, 'velocity is continuous across every waypoint',
     `worst change in speed ${(worstSpline * 100).toFixed(3)}%`);
  console.log(`  the old chain of straight morphs, measured the same way:`
            + ` ${(worstLinear * 100).toFixed(1)}% -- that is the corner this removes`);
  ok(worstLinear > worstSpline * 10, 'and the spline is the thing that fixed it',
     `${(worstLinear / Math.max(worstSpline, 1e-9)).toFixed(0)}x smoother at the joins`);
}

// ------------------------------------------------- 3. never stalls, never jumps
console.log('\n3. the shape never stalls and never jumps');
{
  const STEPS = 2000, du = SET_NAMES.length / STEPS;
  let min = Infinity, max = 0, prev = vec(morphPath(SPECS, 0));
  for (let i = 1; i <= STEPS; i++) {
    const v = vec(morphPath(SPECS, i * du));
    const d = dist(v, prev) / du;
    if (d < min) min = d;
    if (d > max) max = d;
    prev = v;
  }
  ok(min > 0, 'the shape is always changing', `slowest ${min.toFixed(3)} per unit u`);
  ok(max / min < 60, 'and never lurches', `fastest/slowest ${(max / min).toFixed(1)}x`);
}

// ---------------------------------------------------------------- 4. the cost
console.log('\n4. the cost of morphing continuously');
{
  const pool = new ShapePool({ slots: 8, seed: 7, rebuildHz: 7 });
  const dt = 1 / 60, STEPS = Math.round(pool.breathSeconds * 3 / dt);
  let rb = 0, ms = 0, worst = 0, slowRb = 0, slowN = 0, fastRb = 0, fastN = 0;
  for (let i = 0; i < STEPS; i++) {
    pool.update(dt, null);
    rb += pool.rebuildsThisFrame; ms += pool.reparamMs;
    if (pool.reparamMs > worst) worst = pool.reparamMs;
    // is the breath's slow phase also the cheap phase?
    const slow = pool.breath < (pool.breathMin + pool.breathMax) / 2;
    if (slow) { slowRb += pool.rebuildsThisFrame; slowN++; }
    else { fastRb += pool.rebuildsThisFrame; fastN++; }
  }
  const perFrame = rb / STEPS;
  console.log(`  ${STEPS} frames, ${rb} rebuilds`);
  ok(ms / STEPS <= pool.rebuildBudgetMs, 'mean frame stays inside the rebuild budget',
     `${(ms / STEPS).toFixed(2)} ms vs ${pool.rebuildBudgetMs} ms, worst ${worst.toFixed(2)}`);
  ok(pool.worstAdmit <= pool.rebuildBudgetMs,
     'never starts a rebuild the budget cannot afford',
     `worst admission claimed ${pool.worstAdmit.toFixed(2)} ms`);
  ok(perFrame > 0.2, 'the world is actually morphing, not quietly stalled',
     `${perFrame.toFixed(2)} rebuilds/frame`);
  const slowRate = slowRb / Math.max(slowN, 1), fastRate = fastRb / Math.max(fastN, 1);
  ok(slowRate < fastRate, 'the legible moments are the cheap ones',
     `${slowRate.toFixed(2)} rebuilds/frame while slow vs ${fastRate.toFixed(2)} while quick`);
  console.log(`  deferred ${pool.deferred} slots on the last frame`);
}

// ------------------------------------------------------- 5. at display rate
//
// Measured on the table as the TRAVERSAL reads it -- the two builds blended
// by the slot's current fraction -- because that is the shape on screen. The
// drawn table is the one thing here that is not a parameter: the pool can be
// morphing perfectly and still be showing a frozen shape, which is exactly
// the fault this section exists to catch.
console.log('\n5. the shape on screen moves every frame, not every rebuild');
{
  const SAMPLES = 64;
  const probe = (interpolate) => {
    const pool = new ShapePool({ slots: 8, seed: 7, interpolate });
    const prio = new Float64Array(pool.slots).fill(1);
    const prev = pool.shapes.map(() => null);
    const steps = [];
    let still = 0, total = 0, changes = 0;
    const FRAMES = 1200, WARM = 120;
    for (let i = 0; i < FRAMES; i++) {
      pool.update(1 / 60, prio);
      for (let k = 0; k < pool.slots; k++) {
        const a = pool.shapes[k].att, t = a._bt;
        const n = 9 * a.m * (a.tableMask + 1);
        const cur = new Float64Array(SAMPLES);
        for (let j = 0; j < SAMPLES; j++) {
          // spread over the whole table: a stride coprime with it, so the
          // sample is not all one child or all one rotation
          const e = (j * 277) % n;
          cur[j] = a._cmA[e] + (a._cmB[e] - a._cmA[e]) * t;
        }
        if (i >= WARM && prev[k]) {
          let d = 0, m = 0;
          for (let j = 0; j < SAMPLES; j++) {
            d += (cur[j] - prev[k][j]) ** 2; m += cur[j] ** 2;
          }
          const rel = Math.sqrt(d / Math.max(m, 1e-30));
          total++;
          if (rel < 1e-12) still++; else { steps.push(rel); changes++; }
        }
        prev[k] = cur;
      }
    }
    steps.sort((x, y) => x - y);
    const q = (f) => steps[Math.min(steps.length - 1, Math.floor(steps.length * f))];
    return { frozen: still / total, hz: (changes / total) * 60,
             med: q(0.5), p90: q(0.9) };
  };
  const on = probe(true), off = probe(false);
  console.log(`  held still: ${(off.frozen * 100).toFixed(1)}% of slot-frames without`
    + ` interpolation, ${(on.frozen * 100).toFixed(1)}% with`);
  ok(on.frozen < 0.05, 'the drawn shape changes nearly every frame',
     `${on.hz.toFixed(1)} Hz against a 60 Hz display`);
  ok(on.frozen * 4 < off.frozen, 'and that is interpolation doing it, not the budget',
     `${(off.hz).toFixed(1)} Hz with it switched off`);
  // Smoothness is not just "it moved": many small steps rather than a few
  // big ones. The same path crossed in three times as many frames means
  // steps a third the size, and a flat tail means no frame carries a lurch.
  ok(on.med < off.med * 0.6, 'and in smaller steps than a rebuild makes',
     `median step ${on.med.toExponential(2)} vs ${off.med.toExponential(2)}`);
  ok(on.p90 < on.med * 6, 'with no frame carrying a jump',
     `p90 ${on.p90.toExponential(2)} is ${(on.p90 / on.med).toFixed(1)}x the median`);

  // And an extrapolated shape must still be a possible shape. Interpolation
  // between two valid builds cannot go wrong -- every quantity is a convex
  // combination -- but this runs PAST the newest build, and a variance that
  // is shrinking goes negative if it is carried far enough. It did: -0.0204
  // on sigma0 in the first two seconds, from the one pair of builds that are
  // not a rebuild apart (the warm-up's pure set, then the first spline
  // build). Both the bound in Attractor._btLimit and the reset that stops
  // that first pair being extrapolated at all are checked here, because
  // either one alone leaves a negative variance on the screen.
  const pool = new ShapePool({ slots: 8, seed: 7 });
  const prio = new Float64Array(pool.slots).fill(1);
  let minSig = Infinity, minS = Infinity, minQ = Infinity, minR = Infinity, nf = 0;
  for (let i = 0; i < 3600; i++) {
    pool.update(1 / 60, prio);
    for (let k = 0; k < pool.slots; k++) {
      const a = pool.shapes[k].att;
      for (let j = 0; j < 3; j++) {
        if (a.sig0[j] < minSig) minSig = a.sig0[j];
        if (a.rodSig[j] < minSig) minSig = a.rodSig[j];
      }
      for (let j = 0; j < a.m; j++) if (a.sArr[j] < minS) minS = a.sArr[j];
      if (a.qMax < minQ) minQ = a.qMax;
      if (a.R < minR) minR = a.R;
      if (!Number.isFinite(a.R + a.qMax + a.sig0[0] + a.sArr[0] + a._bt)) nf++;
    }
  }
  ok(minSig > 0, 'every variance the blend produces is positive',
     `smallest sigma over 60s x 8 slots ${minSig.toExponential(3)}`);
  ok(minS > 0 && minQ > 0 && minR > 0, 'and so are the scales and the extents',
     `child scale ${minS.toFixed(3)}, qMax ${minQ.toFixed(3)}, R ${minR.toFixed(3)}`);
  ok(nf === 0, 'and nothing is non-finite', `${nf} non-finite slot-frames`);
}

// ------------------------------------------------- 6. resizing the jitter table
//
// The pool sizes each slot's jitter table by how important the slot is on
// screen, which is the dial that makes a 60 Hz morph affordable. A node picks
// its rotation with `hash & tableMask`, so the mask and the table it indexes
// have to move together: grow the mask and the traversal reads entries above
// the ones the last build wrote, which hold whatever that slot's table held
// the last time it was that big -- another build, seconds of morph ago.
//
// Only a rebuild can write the table, and rebuilds are budgeted, so the
// request and the fill are not the same frame whenever the budget is spent.
// Hence: a slot's drawn mask may change only on a frame it was rebuilt. The
// priority has to MOVE for any of this to happen at all -- with every slot
// equally important nothing is ever resized and the check is vacuous, so the
// count of resizes is asserted too.
console.log('\n6. a table resize reaches the screen with the build that fills it');
{
  // A budget of 0.1 ms is one rebuild a frame -- the first of a frame always
  // goes ahead -- which is 7.5 a second per slot across eight, so the morph
  // is barely slowed and the resize is made to wait its turn almost every
  // time. At the full budget the fault shows in 11 to 47 of 150 resizes
  // depending on how busy the machine is, since the budget is spent in real
  // milliseconds; this makes the test's sensitivity its own rather than the
  // machine's.
  const pool = new ShapePool({ slots: 8, seed: 7, rebuildHz: 7, rebuildBudgetMs: 0.1 });
  const prio = new Float64Array(pool.slots);
  const bits = pool.shapes.map((sh) => sh.att.tableBits);
  let resizes = 0, early = 0, grew = 0;
  for (let i = 0; i < 3600; i++) {
    // objects coming towards the camera and going away again
    for (let k = 0; k < pool.slots; k++)
      prio[k] = 0.5 + 0.5 * Math.sin(0.02 * i + k * 1.7);
    pool.update(1 / 60, prio);
    for (let k = 0; k < pool.slots; k++) {
      const sh = pool.shapes[k], b = sh.att.tableBits;
      if (b === bits[k]) continue;
      resizes++;
      if (b > bits[k]) grew++;
      // `lastBuilt` is -1 from the moment the resize is asked for until the
      // rebuild that serves it, so this is "drawn at the new mask with the
      // old table".
      if (sh.lastBuilt === -1) early++;
      bits[k] = b;
    }
  }
  console.log(`  ${resizes} resizes over 60 s of promotions and demotions,`
            + ` ${grew} of them growths`);
  ok(resizes > 20, 'the dial is actually being turned', `${resizes} resizes`);
  ok(grew > 0, 'and in the direction that can read unwritten entries', `${grew} growths`);
  ok(early === 0, 'no frame is drawn at a mask its table was not built for',
     `${early} slot-frames`);
}

console.log(bad ? '\nthe breath is NOT sound' : '\nthe breath is sound');
process.exit(bad);
