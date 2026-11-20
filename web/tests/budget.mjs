// The detail controller, driven with synthetic load.
//
// The controller decides how much detail is on screen, and until now its only
// input was the walk's own time -- so a cheaper DRAW produced no more detail,
// it just left the frame half empty (budget.js carries the argument).  The
// frame term that fixes that cannot be tested in a browser from here: there is
// no GL and no vsync on this machine.  What can be tested is the arithmetic,
// which is the part with a quiet way to be wrong.
//
// So: a toy world where splat count is a known function of the threshold, a
// cost model that turns splats into milliseconds, and a loop that asks whether
// the controller lands where it should and stays there.
//
// The model.  The LOD threshold is a length on screen and the splat count goes
// about as its square, which is the reason the detail ladder doubles: so
//   count   = K / lodScale^2
//   walkMs  = count / WALK_RATE
//   gpuMs   = count / GPU_RATE + gpuFixed
// The rates are per-millisecond throughputs, i.e. exactly what items 2-6 of
// this round of work change.  Nothing here claims those are real numbers for
// any real machine; what is being checked is the loop's response to them.
import { makeBudget, budgetStep, FRAME_FILL, MISS_AT } from '../js/budget.js';

const K = 900000;                       // splats at lodScale 1
const count = (ls) => K / (ls * ls);

function run(opts) {
  const { frames = 600, walkRate = 60000, gpuRate = 90000, gpuFixed = 1.5,
          refresh = 16.667, haveQuery = true, targetMs = 6, target = 96000,
          budgetBy = 'both', useFrame = true, lodScale = 4.0, maxSplats = 400000,
          cpuFixed = 0.8 } = opts;
  const st = makeBudget({ targetMs, target, budgetBy, useFrame, lodScale });
  const hist = [];
  for (let i = 0; i < frames; i++) {
    const c0 = count(st.lodScale);
    const hit = c0 > maxSplats;
    const c = Math.min(c0, maxSplats);
    const walkMs = c / walkRate;
    const gpuMs = c / gpuRate + gpuFixed;
    const cpuMs = walkMs + cpuFixed;
    // A vsync-locked display: the frame lands on a refresh boundary, so the
    // interval is the period until the work no longer fits, and then it is
    // however many periods it does take.  This is exactly why the interval
    // alone cannot say how much headroom there is.
    const workMs = cpuMs + gpuMs;
    const interval = refresh * Math.max(1, Math.ceil(workMs / refresh));
    budgetStep(st, { walkMs, count: c, cpuMs,
                     gpuMs: haveQuery ? gpuMs : null, interval, hit });
    hist.push({ ls: st.lodScale, c, walkMs, gpuMs, cpuMs, interval,
                binding: st.binding, err: st.err });
  }
  const tail = hist.slice(-120);
  const mean = (f) => tail.reduce((a, h) => a + f(h), 0) / tail.length;
  const lsMin = Math.min(...tail.map((h) => h.ls));
  const lsMax = Math.max(...tail.map((h) => h.ls));
  return { st, hist, tail,
           ls: mean((h) => h.ls), splats: mean((h) => h.c),
           walkMs: mean((h) => h.walkMs), gpuMs: mean((h) => h.gpuMs),
           frameMs: mean((h) => h.cpuMs + h.gpuMs),
           interval: mean((h) => h.interval),
           // peak-to-peak of the threshold over the settled tail, as a
           // fraction of its own mean: this is the pumping test
           ripple: (lsMax - lsMin) / mean((h) => h.ls),
           binding: tail[tail.length - 1].binding };
}

let fail = 0;
const check = (ok, msg) => { console.log((ok ? '  ok    ' : '  FAIL  ') + msg); if (!ok) fail++; };
const f1 = (x) => x.toFixed(1);

console.log('The detail controller, on a toy world: count = 900000 / lodScale^2\n');

// ---------------------------------------------------------------- 1. the old
// Walk-only, which is what every headless test in this repository measures and
// what the browser did before budget.js existed.  It must still hold the walk
// at its target and must NOT be moved by anything downstream.
const walkOnly = run({ useFrame: false, budgetBy: 'time', target: 0 });
console.log('1. walk-only (the controller as it was)');
console.log(`   lodScale ${f1(walkOnly.ls)}  splats ${Math.round(walkOnly.splats)}`
  + `  walk ${f1(walkOnly.walkMs)} ms  frame ${f1(walkOnly.frameMs)} ms`
  + `  interval ${f1(walkOnly.interval)} ms`);
check(Math.abs(walkOnly.walkMs - 6) < 0.3, `holds the walk at its 6 ms target (${f1(walkOnly.walkMs)})`);
check(walkOnly.ripple < 0.02, `and does not pump (${(walkOnly.ripple * 100).toFixed(2)}% ripple)`);

// The fault being fixed, stated as a measurement: with the walk-only
// controller, a frame that cannot fit in a refresh period is held anyway.
const overloaded = run({ useFrame: false, budgetBy: 'time', target: 0, gpuRate: 12000 });
console.log('\n2. walk-only, with a GPU 7.5x slower');
console.log(`   lodScale ${f1(overloaded.ls)}  splats ${Math.round(overloaded.splats)}`
  + `  walk ${f1(overloaded.walkMs)} ms  frame ${f1(overloaded.frameMs)} ms`
  + `  interval ${f1(overloaded.interval)} ms`);
check(Math.abs(overloaded.ls - walkOnly.ls) < 0.05,
  'the threshold is identical -- the controller cannot see the GPU at all');
check(overloaded.interval > 2.5 * 16.667,
  `so the frame rate collapses unchecked (${f1(1000 / overloaded.interval)} fps)`);

// ------------------------------------------------------------- 3. the brake
// No timer query: the interval is all there is, and a vsync-locked frame holds
// at the period however much headroom is left -- so the brake can only learn
// by missing.  What it must not do is oscillate: grant detail, miss, take it
// away, grant it again.  That is the limit cycle the latch and the remembered
// cliff exist to break, and the ripple figure is the test of them.
const brake = run({ haveQuery: false, budgetBy: 'time', target: 0, gpuRate: 12000 });
console.log('\n3. the same slow GPU, no timer query: the interval brake');
console.log(`   lodScale ${f1(brake.ls)}  splats ${Math.round(brake.splats)}`
  + `  frame ${f1(brake.frameMs)} ms  interval ${f1(brake.interval)} ms`
  + `  ripple ${(brake.ripple * 100).toFixed(1)}%`);
check(brake.ls > overloaded.ls * 1.3,
  `detail is given up (threshold ${f1(brake.ls)} against ${f1(overloaded.ls)})`);
check(brake.interval < 16.667 * MISS_AT,
  `and frames stop being missed (${f1(brake.interval)} ms, ${f1(1000 / brake.interval)} fps)`);
check(brake.ripple < 0.10,
  `without pumping: the remembered cliff holds it steady (${(brake.ripple * 100).toFixed(1)}% ripple)`);

// The brake must be a brake and nothing more: with frames on time it may not
// grant detail the walk budget has not granted.
const idle = run({ haveQuery: false, budgetBy: 'time', target: 0 });
check(Math.abs(idle.ls - walkOnly.ls) < 1e-9,
  'with frames on time the brake is inert -- bit-identical to walk-only');

// ------------------------------------------------------------ 4. the point
// With a timer query the term is two-sided, so a cheaper draw becomes detail.
//
// The rung's own ceilings are set generously here (the `highest` rung's 24 ms
// against a 600k splat cap) so that the GPU is what binds.  That is not a
// contrivance: when the detail dial is up and the walk is cheap, the draw IS
// the constraint, and before this change nothing in the engine could tell.
const RUNG = { targetMs: 24, target: 600000, maxSplats: 800000 };
const slow = run({ ...RUNG, gpuRate: 30000 });
const fast = run({ ...RUNG, gpuRate: 60000 });
console.log('\n4. with a timer query: a cheaper draw becomes detail');
console.log(`   GPU 30k splats/ms:  lodScale ${f1(slow.ls)}  splats ${Math.round(slow.splats)}`
  + `  frame ${f1(slow.frameMs)} ms  binding ${slow.binding}`);
console.log(`   GPU 60k splats/ms:  lodScale ${f1(fast.ls)}  splats ${Math.round(fast.splats)}`
  + `  frame ${f1(fast.frameMs)} ms  binding ${fast.binding}`);
check(fast.splats > slow.splats * 1.3,
  `doubling draw throughput buys ${(fast.splats / slow.splats).toFixed(2)}x the splats`);
check(slow.binding === 'gpu' && fast.binding === 'gpu',
  `and the controller says so: binding term is ${slow.binding}/${fast.binding}`);
check(slow.frameMs < 16.667 * FRAME_FILL * 1.08
   && fast.frameMs < 16.667 * FRAME_FILL * 1.08,
  `both land under ${(FRAME_FILL * 100) | 0}% of the refresh period `
  + `(${f1(slow.frameMs)} and ${f1(fast.frameMs)} ms of 16.7)`);
check(slow.ripple < 0.08 && fast.ripple < 0.08,
  `and neither pumps (${(slow.ripple * 100).toFixed(1)}% and ${(fast.ripple * 100).toFixed(1)}%)`);

// The walk ceiling still binds when it is the tighter of the two: a fast GPU
// must not be allowed to drive the walk past its own budget, nor the splat
// count past the rung's.
const fastGpu = run({ gpuRate: 1e9, gpuFixed: 0.2, targetMs: 6, target: 600000 });
console.log(`\n   GPU effectively free: walk ${f1(fastGpu.walkMs)} ms, binding ${fastGpu.binding}`);
check(fastGpu.walkMs < 6.4 && fastGpu.binding === 'time',
  `the walk's own ceiling still holds (${f1(fastGpu.walkMs)} ms, binding ${fastGpu.binding})`);
const capped = run({ ...RUNG, target: 120000, gpuRate: 1e9, gpuFixed: 0.2 });
check(Math.round(capped.splats) <= 120001 && capped.binding === 'splats',
  `and so does the rung's splat ceiling (${Math.round(capped.splats)} of 120000, binding ${capped.binding})`);

// ------------------------------------------------------- 5. the refresh rate
// The period is observed, not configured.  A 120 Hz display must get half the
// frame budget and converge on its own.
const hz120 = run({ ...RUNG, refresh: 8.333, gpuRate: 30000 });
console.log('\n5. the refresh period is observed, not configured');
console.log(`   120 Hz: refresh seen ${hz120.st.refreshMs.toFixed(2)} ms`
  + `  frame ${f1(hz120.frameMs)} ms  splats ${Math.round(hz120.splats)}`);
check(Math.abs(hz120.st.refreshMs - 8.333) < 0.2,
  `the period is recovered from the intervals (${hz120.st.refreshMs.toFixed(2)} ms)`);
check(hz120.frameMs < 8.333 * FRAME_FILL * 1.12,
  `and the frame fits inside it (${f1(hz120.frameMs)} ms)`);
check(hz120.splats < slow.splats * 0.6,
  `which costs detail against 60 Hz, as it must (${Math.round(hz120.splats)} vs ${Math.round(slow.splats)})`);

// ------------------------------------------------------------- 6. transients
// A step change in cost: the response is bounded per frame, quicker to give
// detail up than to take it back, and it must recover rather than ring.
const st6 = makeBudget({ targetMs: 24, target: 600000, budgetBy: 'both', lodScale: 4.0 });
let worstStep = 0, recover = -1, violated = 0;
const settled = [];
for (let i = 0; i < 400; i++) {
  // the cliff: at frame 200 the GPU gets six times slower
  const gpuRate = i < 200 ? 90000 : 15000;
  const c = Math.min(count(st6.lodScale), 800000);
  const walkMs = c / 60000, gpuMs = c / gpuRate + 1.5, cpuMs = walkMs + 0.8;
  const frameMs = cpuMs + gpuMs;
  const interval = 16.667 * Math.max(1, Math.ceil(frameMs / 16.667));
  const before = st6.lodScale;
  budgetStep(st6, { walkMs, count: c, cpuMs, gpuMs, interval, hit: c >= 800000 });
  const rel = Math.abs(st6.lodScale - before) / before;
  if (i > 5) worstStep = Math.max(worstStep, rel);
  if (i >= 200) {
    // The controller's setpoint IS the budget, so "recovered" means back AT
    // it within a couple of percent, not under it -- converging exactly onto
    // 14.2 ms would otherwise count as 200 frames of violation.
    const over = 16.667 * FRAME_FILL * 1.02;
    if (frameMs > over) violated++;
    if (recover < 0 && violated > 0 && frameMs <= over) recover = i - 200;
    if (i > 340) settled.push(st6.lodScale);
  }
}
const ripple6 = (Math.max(...settled) - Math.min(...settled))
              / (settled.reduce((a, b) => a + b, 0) / settled.length);
console.log('\n6. a step change: the GPU gets 6x slower at frame 200');
console.log(`   worst single-frame threshold move ${(worstStep * 100).toFixed(1)}%`
  + `  over budget for ${violated} frames  recovered in ${recover}`
  + `  settled ripple ${(ripple6 * 100).toFixed(2)}%`);
check(worstStep <= 0.30 + 1e-9,
  `no frame moves the threshold by more than the 30% bound (${(worstStep * 100).toFixed(1)}%)`);
// 45, not the 40 this once asserted: the frame term no longer divides the
// whole frame by the budget, it fits what a splat costs and holds the rest
// aside (see scenario 8), and a fit has to be relearned after a step this
// size.  FIT_A buys the difference back directly -- the figures are in
// budget.js -- and 45 frames is 0.75 s at 60 Hz for a GPU that got six times
// slower between one frame and the next.
check(recover > 0 && recover <= 45,
  `the frame budget is met again within 45 frames (${recover})`);
// 8%, not the 5% this once asserted.  The refresh period is now read from a
// window rather than from the running minimum, so it moves a little as the
// window turns over, and the budget moves with it.  The cross-fade band exists
// to absorb exactly this: threshold motion is not visible motion.
check(ripple6 < 0.08, `and it settles rather than ringing (${(ripple6 * 100).toFixed(2)}%)`);

// ------------------------------------------------------------- 7. disjoint
// EXT_disjoint_timer_query's own failure mode: the GPU was interrupted and
// every number in flight is rubbish, which GpuClock reports as null.  That
// must read as "no timer query" -- fall back to the brake -- and never as "the
// GPU costs nothing", which would hand out detail until the frame rate died.
const st7 = makeBudget({ targetMs: 24, target: 600000, budgetBy: 'both', lodScale: 4.0 });
const seen = [];
for (let i = 0; i < 500; i++) {
  const c = Math.min(count(st7.lodScale), 800000);
  const walkMs = c / 60000, gpuMs = c / 15000 + 1.5, cpuMs = walkMs + 0.8;
  const interval = 16.667 * Math.max(1, Math.ceil((cpuMs + gpuMs) / 16.667));
  // the query works for 200 frames, then goes disjoint for good
  budgetStep(st7, { walkMs, count: c, cpuMs, gpuMs: i < 200 ? gpuMs : null,
                    interval, hit: false });
  seen.push({ ls: st7.lodScale, interval, binding: st7.binding });
}
const withQ = seen[199], after = seen.slice(-120);
const ivAfter = after.reduce((a, h) => a + h.interval, 0) / after.length;
const lsAfter = after.reduce((a, h) => a + h.ls, 0) / after.length;
console.log('\n7. the timer query goes disjoint at frame 200');
console.log(`   with the query: lodScale ${f1(withQ.ls)}, binding ${withQ.binding}`);
console.log(`   after it fails: lodScale ${f1(lsAfter)}, interval ${f1(ivAfter)} ms,`
  + ` binding ${after[after.length - 1].binding}`);
check(withQ.binding === 'gpu', 'while the query works the GPU term is what binds');
check(lsAfter > 1.5,
  `once it fails the brake holds detail back rather than letting it run (${f1(lsAfter)})`);
check(ivAfter < 16.667 * MISS_AT,
  `and frames keep landing on time (${f1(ivAfter)} ms, ${f1(1000 / ivAfter)} fps)`);
check(Math.abs(lsAfter - withQ.ls) / withQ.ls < 0.5,
  `at a threshold close to what the query had found (${f1(lsAfter)} against ${f1(withQ.ls)})`);

// ---------------------------------------------------------------------------
// 8. A FRAME IS NOT ALL SPLATS.
//
// Every scenario above models the frame as a function of the splat count --
// which is the assumption the first version of the frame term was built on,
// and it is false.  The sky, the pool rebuild, the uniforms and the present
// cost the same whether there are 300k splats or one.  That version divided
// the WHOLE frame by the budget, so when the floor alone exceeded the budget
// no number of splats could satisfy it: the controller pushed the threshold
// to its clamp and the remembered cliff pinned it there.  One splat, 900 px
// across, and a detail key that could not win because the loop rewrote the
// threshold every frame.  It shipped, and a live machine showed it within the
// hour -- so this scenario has a known floor AND a known marginal cost, and
// checks the controller recovers both rather than checking it against itself.
function frameSplit(floorMs, usPerSplat, frames = 2400) {
  const st = makeBudget({ lodScale: 4.0 });
  st.targetMs = 6.0; st.target = 2000000; st.budgetBy = 'time'; st.useFrame = true;
  const cpu = 2.5;
  let last = {};
  for (let i = 0; i < frames; i++) {
    const c = Math.min(2000000, Math.round(900000 / (st.lodScale ** 2)));
    const gpuMs = (floorMs - cpu) + (c * usPerSplat) / 1000;
    budgetStep(st, { walkMs: Math.min(6, c / 200000), count: c, cpuMs: cpu,
                     gpuMs, interval: null, hit: c >= 2000000 });
    last = { count: c, frame: cpu + gpuMs };
  }
  return { st, ...last };
}
console.log('\n8. a frame is not all splats: a known floor, and a known cost each');
console.log('   floor   splats     frame    floor fit   us/splat (true 0.030)  binding');
const sp = {};
for (const fl of [5, 10, 15, 20]) {
  const r = sp[fl] = frameSplit(fl, 0.03);
  console.log(`   ${String(fl).padStart(2)} ms ${String(r.count).padStart(9)}`
    + ` ${f1(r.frame).padStart(8)} ms ${f1(r.st.floorMs).padStart(8)}`
    + `   ${(r.st.fitK * 1000).toFixed(3).padStart(14)}      ${r.st.binding}`);
}
const half = frameSplit(5, 0.015);
check(sp[5].count > 100000,
  `a low floor leaves most of the budget to splats (${sp[5].count})`);
check(sp[10].count < sp[5].count,
  `a higher floor costs detail (${sp[10].count} against ${sp[5].count})`);
check(sp[15].count > 20000 && sp[20].count > 20000,
  `a floor OVER the whole budget still leaves a scene (${sp[15].count}, ${sp[20].count})`);
check(sp[15].st.lodScale < 16 && sp[20].st.lodScale < 16,
  `at a threshold that is still a fractal, not one splat `
  + `(${f1(0.9 * sp[15].st.lodScale)} px, ${f1(0.9 * sp[20].st.lodScale)} px)`);
check(half.count > 1.8 * sp[5].count,
  `halving the cost of a splat buys ${(half.count / sp[5].count).toFixed(2)}x the splats`);
// The floor is tested where it is USED.  At 5 and 10 ms it is a control
// input -- it sets how much of the budget the splats may have -- so it has to
// be close.  At 15 and 20 it has already decided the term stands down, the
// count is pinned by the walk clock and no longer identifies the slope, so
// the number is a diagnostic and all that matters is that it still says
// "there is no headroom".
for (const fl of [5, 10])
  check(Math.abs(sp[fl].st.floorMs - fl) < 1.5,
    `the fitted floor finds ${fl} ms, which is a control input (${f1(sp[fl].st.floorMs)})`);
for (const fl of [15, 20])
  check(Math.abs(sp[fl].st.floorMs - fl) < 3.5,
    `and ${fl} ms well enough to know there is no headroom (${f1(sp[fl].st.floorMs)})`);
// When the floor owns the budget the splats get a MINIMUM share of it, not
// none and not all of it.  Standing the term down entirely was the first
// attempt and it fails the other way: with only the walk clock to answer to
// the detail runs free, which on a 120 Hz display measured 1.2 million splats
// in a 42 ms frame.  Both bounds are asserted here, because this controller
// has now been wrong in both directions.
check(sp[15].count > 20000 && sp[15].count < 400000,
  `a floor over budget leaves a bounded scene, neither empty nor runaway `
  + `(${sp[15].count})`);
check(sp[15].st.binding === 'floor' || sp[15].st.binding === 'gpu'
   || sp[15].st.binding === 'time' || sp[15].st.binding === 'splats',
  `and the panel names what is holding it (${sp[15].st.binding})`);

// ---------------------------------------------------------------------------
// 9. requestAnimationFrame RUNS EARLY, not only late.
//
// The refresh period was the running minimum of the intervals, and that is the
// second bug this controller shipped.  After a long frame the browser catches
// up and delivers the next callback almost immediately, so a 60 Hz display
// hands out the occasional 6 ms gap.  A minimum believes it: the budget
// becomes 0.85 x 6 ms and the controller spends the session chasing 200 fps,
// taking detail away every frame to get there.  A scene that starts as a
// fractal and shrinks away -- which is how it was reported.  One such gap cut
// a 255k-splat scene to 88k and left it oscillating twenty seconds later.
//
// A missed frame makes the interval a multiple of the period; coalescing makes
// it a fraction.  Only a statistic robust at BOTH ends gets this right.
function ivRun(ivOf, frames = 1500) {
  const st = makeBudget({ lodScale: 4.0 });
  st.targetMs = 6.0; st.target = 2000000; st.budgetBy = 'time'; st.useFrame = true;
  let c = 0;
  for (let i = 0; i < frames; i++) {
    c = Math.min(2000000, Math.round(900000 / (st.lodScale ** 2)));
    budgetStep(st, { walkMs: Math.min(6, c / 200000), count: c, cpuMs: 2.5,
                     gpuMs: 4.0 + (c * 0.03) / 1000, interval: ivOf(i), hit: false });
  }
  return { st, count: c };
}
console.log('\n9. rAF runs early as well as late: the refresh period must not believe it');
const ivCases = [
  ['a clean 60 Hz', () => 16.667, 16.667],
  ['one 6 ms gap at frame 30', (i) => (i === 30 ? 6.0 : 16.667), 16.667],
  ['a 6 ms gap every 30 frames', (i) => (i % 30 === 0 ? 6.0 : 16.667), 16.667],
  ['a 6 ms gap every 4 frames', (i) => (i % 4 === 0 ? 6.0 : 16.667), 16.667],
  ['every other frame missed', (i) => (i % 2 ? 16.667 : 33.3), 16.667],
  ['a real 120 Hz display', () => 8.333, 8.333],
  ['60 Hz, then 120 at frame 500', (i) => (i < 500 ? 16.667 : 8.333), 8.333],
];
const ivGot = {};
for (const [name, fn, want] of ivCases) {
  const r = ivGot[name] = ivRun(fn);
  console.log(`   ${name.padEnd(30)} reads ${f1(r.st.refreshMs).padStart(5)} ms`
    + ` (${String(Math.round(1000 / r.st.refreshMs)).padStart(3)} fps)`
    + `   splats ${String(r.count).padStart(7)}`);
}
for (const [name, , want] of ivCases)
  check(Math.abs(ivGot[name].st.refreshMs - want) < 0.1,
    `${name} reads as ${Math.round(1000 / want)} Hz`);
const clean = ivGot['a clean 60 Hz'].count;
for (const name of ['one 6 ms gap at frame 30', 'a 6 ms gap every 30 frames',
                    'a 6 ms gap every 4 frames', 'every other frame missed'])
  check(Math.abs(ivGot[name].count - clean) / clean < 0.01,
    `and costs no detail against a clean 60 (${ivGot[name].count} vs ${clean})`);

console.log(fail ? `\n  ${fail} FAILED` : '\n  budget ok');
process.exit(fail ? 1 : 0);
