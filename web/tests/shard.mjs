// The walk, split across real threads, against the walk that is not.
//
// The engine's central claim is that it never sorts: the expansion emits
// back-to-front by construction.  Splitting that walk across threads is the
// one optimisation most likely to break it silently -- the output would be
// the right splats in the wrong order, which looks like a rendering bug and
// not like a threading bug.
//
// So this is not a benchmark.  It walks a scene serially, walks the same scene
// sharded over k worker_threads, and compares the two SPLAT FOR SPLAT, in
// order, bit for bit: position, covariance, colour and effect mask.  The
// speedup is printed afterwards, as an aside.
//
// Why the whole mechanism is testable here when none of the GPU work is:
// worker_threads and SharedArrayBuffer are both in node, so this is the real
// thing on real threads, not a transcription.  What it does NOT cover is the
// browser's own wiring -- Worker with a module URL, and the COOP/COEP headers
// SharedArrayBuffer needs there, which is a root.md task.
//
//   node tests/shard.mjs [k] [steps]
import { Worker } from 'node:worker_threads';
import { AdaptiveStream } from '../js/adaptive.js';
import { split, offsets, capacities, combine } from '../js/shard.js';
import { setup, frameList, walkRun, costOf, view } from './shard_setup.mjs';

const K = Number(process.argv[2] ?? 4);
const STEPS = Number(process.argv[3] ?? 40);
const MAX = 400000;
// A threshold that gives a frame's worth of work rather than a toy's.  At the
// shipped 4.0 this ring at this distance is a couple of thousand splats and
// the measurement is all hand-off; the real demo draws 100-300k, which is the
// regime the question is about.
const LOD = { lodScale: Number(process.argv[4] ?? 0.25), lo: 0.9, hi: 3.2, rodFade: true };

let fail = 0;
const check = (ok, msg) => { console.log((ok ? '  ok    ' : '  FAIL  ') + msg); if (!ok) fail++; };

// ------------------------------------------------------------- the units
// js/shard.js on its own, including the cases a live scene never produces.
{
  let bad = 0, why = '';
  const cases = [[[1570, 100, 100, 100, 80, 70, 60, 40, 30, 14], 4],
                 [[100, 100, 100, 100, 100, 100, 100, 100], 4], [[5, 3], 4], [[7], 4],
                 [[1, 1, 1], 2], [[9, 1, 1, 1, 1], 3], [[1], 1], [[3, 3, 3, 3], 2],
                 // all zero is the FIRST FRAME, before any object has been
                 // walked and measured: it must not collapse to one shard
                 [[0, 0, 0, 0], 3], [[0, 0, 0, 0, 0, 0, 0], 4], [[2, 0, 0], 3],
                 [[1e6, 1, 1], 2]];
  for (const [cs, k] of cases) {
    const r = split(cs, k), want = Math.min(k, cs.length);
    const contig = r.every((x, i) => x.i0 === (i ? r[i - 1].i1 : 0))
                && r[r.length - 1].i1 === cs.length && r.every((x) => x.i1 > x.i0);
    if (r.length !== want || !contig) { bad++; why = `${JSON.stringify(cs)} k=${k}`; }
  }
  check(split([], 3).length === 0, 'an empty view splits into no shards at all');
  check(bad === 0, `split gives min(k,n) non-empty contiguous runs on ${cases.length} `
    + `shapes of cost, all-zero included${bad ? ` -- broke on ${why}` : ''}`);
  const o = offsets([10, 20, 30]);
  check(o.off[0] === 50 && o.off[1] === 30 && o.off[2] === 0 && o.total === 60,
    `offsets put the NEAREST shard last: ${JSON.stringify(o.off)} for counts 10,20,30`);
  check(capacities(3, 400000).every((x) => x % 4 === 0),
    `capacities are multiples of four, so a shard's base needs no rounding`);
  const cb = combine([{ ms: 5, count: 10, hit: false }, { ms: 9, count: 20, hit: false },
                      { ms: 4, count: 5, hit: true }]);
  check(cb.walkMs === 9 && cb.count === 35 && cb.hit === true && cb.serialMs === 18,
    'combine reports the slowest shard, the summed count, and any hit');
  console.log('');
}

// ----------------------------------------------------------------- serial
const { world, cam } = setup(STEPS);
const frames = frameList(world, cam);
const serial = new AdaptiveStream(frames[0].att, MAX);
serial.lodScale = LOD.lodScale; serial.lo = LOD.lo; serial.hi = LOD.hi;
serial.rodFade = LOD.rodFade; serial.budgetBy = 'frozen';
// Warm up before timing anything.  The first walk of a cold process is mostly
// V8 compiling the traversal -- 26 ms against 2 ms warm on this scene -- and
// a speedup measured against that is a speedup over the compiler.
const perObject = new Array(frames.length).fill(0);
for (let w = 0; w < 12; w++) {
  serial.begin(cam, view);
  walkRun(serial, frames, 0, frames.length, perObject);
  serial.end();
}
let serialMs = Infinity, nSerial = 0;
for (let r = 0; r < 5; r++) {
  const t0 = performance.now();
  serial.begin(cam, view);
  walkRun(serial, frames, 0, frames.length, perObject);
  nSerial = serial.end();
  serialMs = Math.min(serialMs, performance.now() - t0);
}

console.log(`${frames.length} objects in view, ${nSerial} splats, `
  + `${serialMs.toFixed(2)} ms serial (best of 5, warm), sharded ${K} ways\n`);

// The granularity limit, before anything is split.  Sharding BY OBJECT cannot
// divide the largest single object: its run is indivisible, so the slowest
// shard holds at least that share of the splats however the rest are
// arranged.  It is the number that decides whether this is worth having on a
// given scene, and on a ring of ten objects it is not generous -- a field,
// with 25-38 objects in view, spreads better.
//
// It bounds the SPLAT SHARE and not the time, and the difference is real: a
// thread walking one object has a smaller working set than the serial walk
// going through all of them, so the measured speedup here comes out above it.
const biggest = Math.max(...perObject);
const bound = nSerial / biggest;
console.log(`  the largest single object is ${biggest} of ${nSerial} splats`
  + ` (${((100 * biggest) / nSerial).toFixed(0)}%), so the slowest shard can`
  + ` hold no less than that -- a ${bound.toFixed(2)}x ceiling on splat share\n`);

// ------------------------------------------------------------------ shards
// Two cost proxies, and the difference between them is the finding.
//
// `costOf` is apparent size, Rw/(d+Rw) -- a formula, and the one
// FieldWorld.frames() already computes for its own priorities.  `perObject` is
// how many splats each object actually emitted last frame.  The walk counts
// that anyway, and the camera moves by a fraction of a percent a frame, so it
// is nearly free and nearly exact.
const formula = frames.map(costOf);
const runsF = split(formula, K);
const runs = split(perObject, K);
const worstOf = (rs, cost) => {
  const tot = cost.reduce((a, b) => a + b, 0);
  let mx = 0;
  for (const r of rs) {
    let c = 0;
    for (let i = r.i0; i < r.i1; i++) c += cost[i];
    if (c > mx) mx = c;
  }
  return { mx, imbalance: mx / (tot / rs.length) };
};
const byF = worstOf(runsF, perObject), byM = worstOf(runs, perObject);
console.log(`  split by apparent size: slowest shard holds ${byF.mx} splats`
  + ` (${byF.imbalance.toFixed(2)}x the mean)`);
console.log(`  split by measured count: slowest shard holds ${byM.mx} splats`
  + ` (${byM.imbalance.toFixed(2)}x the mean)\n`);
const caps = capacities(runs.length, MAX);
const STRIDE = 41;
const sab = new SharedArrayBuffer(STRIDE * caps[0] * runs.length);

console.log('  the split, nearest first:');
for (let s = 0; s < runs.length; s++)
  console.log(`    shard ${s}: objects ${runs[s].i0}..${runs[s].i1 - 1}`
    + `  cost ${runs[s].cost.toFixed(4)}  capacity ${caps[s]}`);

const workers = runs.map((r, s) => new Worker(new URL('./shard_worker.mjs', import.meta.url), {
  workerData: { sab, splatOffset: s * caps[0], cap: caps[s], steps: STEPS, ...LOD },
}));
// Both listeners come off again on either outcome: this is awaited once per
// shard per round and node warns at ten accumulated listeners, which on a
// warm-up loop is a leak and not a warning to ignore.
const once = (w) => new Promise((res, rej) => {
  const ok = (m) => { w.off('error', bad); res(m); };
  const bad = (e) => { w.off('message', ok); rej(e); };
  w.once('message', ok); w.once('error', bad);
});
const ready = await Promise.all(workers.map(once));
check(ready.every((r) => r.objects === frames.length),
  `every thread built the same scene (${ready[0].objects} objects each)`);

const round = async () => {
  const t = performance.now();
  const rs = await Promise.all(runs.map((r, s) => {
    const p = once(workers[s]);
    workers[s].postMessage({ cmd: 'walk', i0: r.i0, i1: r.i1 });
    return p;
  }));
  return { rs, wall: performance.now() - t };
};
// The same warm-up the serial side got, for the same reason.
for (let w = 0; w < 12; w++) await round();
let best = null;
for (let r = 0; r < 5; r++) {
  const got = await round();
  if (!best || got.wall < best.wall) best = got;
}
const results = best.rs, wallMs = best.wall;
for (const w of workers) w.postMessage({ cmd: 'stop' });

const counts = results.map((r) => r.count);
const { off, total } = offsets(counts);
const comb = combine(results.map((r) => ({ ms: r.ms, count: r.count, hit: r.hit })));

console.log('');
for (let s = 0; s < runs.length; s++)
  console.log(`    shard ${s}: ${String(counts[s]).padStart(7)} splats`
    + `  ${results[s].ms.toFixed(1)} ms  -> buffer offset ${off[s]}`);
console.log('');

check(total === nSerial,
  `the shards emit exactly as many splats as the serial walk (${total} vs ${nSerial})`);
check(results.every((r) => !r.hit), 'no shard ran out of room');

// --------------------------------------------------- the splat-for-splat test
//
// Reassemble: for each shard, in nearest-first order, its valid run is the
// `count` splats ending at the top of its own region.  offsets() says where
// that run belongs in the final buffer -- shard 0 last, because nearest is
// drawn last.  If that is right, the concatenation is the serial stream.
const sPos = new Float32Array(sab), sU8 = new Uint8Array(sab);
let worstPos = 0, worstCov = 0, badCol = 0, badFx = 0, wrong = 0;
for (let s = 0; s < runs.length; s++) {
  const cap = caps[s], base = STRIDE * s * caps[0];
  const fBase = base / 4;                       // in floats
  const head = cap - counts[s];                 // the shard's own valid start
  for (let k = 0; k < counts[s]; k++) {
    const src = head + k;                       // index inside the shard
    const dst = off[s] + k;                     // index in the serial stream
    const sj = serial.offset + dst;
    for (let c = 0; c < 3; c++) {
      const a = sPos[fBase + 3 * src + c], b = serial.pos[3 * sj + c];
      if (a !== b) { wrong++; worstPos = Math.max(worstPos, Math.abs(a - b)); }
    }
    for (let c = 0; c < 6; c++) {
      const a = sPos[fBase + 3 * cap + 6 * src + c], b = serial.cov[6 * sj + c];
      if (a !== b) { wrong++; worstCov = Math.max(worstCov, Math.abs(a - b)); }
    }
    for (let c = 0; c < 4; c++)
      if (sU8[base + 36 * cap + 4 * src + c] !== serial.rgba[4 * sj + c]) badCol++;
    if (sU8[base + 40 * cap + src] !== serial.fx[sj]) badFx++;
  }
}
check(wrong === 0,
  `every splat's position and covariance is bit-identical to the serial walk`
  + (wrong ? ` (${wrong} differ, worst pos ${worstPos}, cov ${worstCov})` : ''));
check(badCol === 0 && badFx === 0,
  `and so is every colour and effect mask${badCol + badFx ? ` (${badCol}/${badFx} differ)` : ''}`);

// The order is the thing most likely to be wrong, so it is also asserted the
// other way round: reversing the shards must NOT reproduce the serial stream,
// or the test would pass on a scene where order happens not to matter.
if (runs.length > 1) {
  const rev = offsets([...counts].reverse()).off.reverse();
  let same = 0, n = 0;
  for (let s = 0; s < runs.length; s++) {
    const cap = caps[s], fBase = (STRIDE * s * caps[0]) / 4;
    const head = cap - counts[s];
    for (let k = 0; k < counts[s]; k += 7) {
      const sj = serial.offset + rev[s] + k;
      if (sj >= serial.max) continue;
      n++;
      if (sPos[fBase + 3 * (head + k)] === serial.pos[3 * sj]) same++;
    }
  }
  check(same < n * 0.9,
    `and the reversed order does NOT match, so the test can see the difference `
    + `(${same} of ${n} sampled splats coincide)`);
}

// ------------------------------------------------------------- the speedup
console.log('');
console.log(`  slowest shard ${comb.walkMs.toFixed(2)} ms, all shards summed `
  + `${comb.serialMs.toFixed(2)} ms, wall ${wallMs.toFixed(2)} ms`);
console.log(`  imbalance ${comb.imbalance.toFixed(2)}x `
  + `(1.00 = a perfect split; 2.00 = half the threads idle)`);
console.log(`  against ${serialMs.toFixed(2)} ms serial: `
  + `${(serialMs / comb.walkMs).toFixed(2)}x on the slowest shard, `
  + `${(serialMs / wallMs).toFixed(2)}x wall including the hand-off`);
console.log(`  the splat-share ceiling was ${bound.toFixed(2)}x; time can beat it `
  + `because a shard's working set is smaller -- ${(nSerial / serialMs).toFixed(0)} `
  + `splats/ms serial against ${(comb.count / comb.serialMs).toFixed(0)} summed over shards`);
check(comb.count === nSerial, 'the combined count the controller sees is the frame total');
check(comb.walkMs <= comb.serialMs + 1e-9,
  'the time it reports is the slowest shard, not their sum -- they run at once');
check(runs.length === Math.min(K, frames.length),
  `the split used every thread it was given (${runs.length} of `
  + `${Math.min(K, frames.length)} possible)`);
check(byM.imbalance <= byF.imbalance + 1e-9,
  `the measured cost splits at least as well as the formula `
  + `(${byM.imbalance.toFixed(2)}x against ${byF.imbalance.toFixed(2)}x)`);
if (runs.length > 1)
  check(runs.some((r) => r.i1 - r.i0 === 1 && perObject[r.i0] === biggest),
    `the largest object got a shard to itself, which is the most the split can do`);

console.log(fail ? `\n  ${fail} FAILED` : '\n  shard ok');
process.exit(fail ? 1 : 0);
