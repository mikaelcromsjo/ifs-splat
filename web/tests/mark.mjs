// The other half of the link: rebuild the moment, check it, draw it.
//
//   node tests/mark.mjs                 self-test, and the A/B that says the
//                                       detector can see a known fault
//   node tests/mark.mjs '<link>'        a link from the demo (or a file
//   node tests/mark.mjs mark.txt        holding one): diagnose and draw it
//
// js/snapshot.js packs one frame of the simulation into a URL fragment. What
// makes that worth doing is that it is EXACT rather than descriptive: every
// heavy array in the engine is a pure function of a few parameters, so the
// two 288 KB tables a slot is drawing from can be rebuilt here from ten
// numbers -- the two build parameters, the mask, and the migration weight.
//
// This file rebuilds them through the same ShapePool code path the browser
// took, and then does not trust itself: every rebuild record in the window
// carries the state the browser reached (mask, migration weight, which
// entries were still merging, whether the mask drop was pending), and the
// replay is checked against all of them, then against the per-frame splat
// movement the browser measured with its own probe. A replay that cannot
// reproduce the numbers says so instead of drawing a picture of something
// else.
//
// What the diagnosis is for: "it went jumpy" has at least four mechanisms
// behind it and only one of them is the morph. The window carries the frame
// cost and the splat count as well, so a slow frame (twelve frames of motion
// arriving at once) and a change in WHAT is drawn (the budget, the LOD) can
// be ruled in or out here, from the link alone, before anything is blamed on
// the interpolation.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { SETS, SET_NAMES, morphPath, build } from '../js/maps.js';
import { Attractor, AdaptiveStream, FlyCamera, PALETTES } from '../js/adaptive.js';
import { ShapePool } from '../js/field.js';
import { JumpLog, NADDR } from '../js/mark.js';
import { snapshot, encode, decode } from '../js/snapshot.js';
import { render, png, tile } from '../js/cpu_raster.js';

const SPECS = SET_NAMES.map((n) => SETS[n]);
const ND = 3 * NADDR;
// One JumpLog, used for nothing but its probe() and dist() -- the same walk
// down the same fixed addresses the browser measured with, so a step here and
// a step there are the same quantity.
const P = new JumpLog();

const e2 = (v) => (Number.isFinite(v) ? v.toExponential(2) : '-');
const q = (a, p) => {
  const s = a.slice().sort((x, y) => x - y);
  return s.length ? s[Math.min(s.length - 1, Math.floor(p * s.length))] : NaN;
};

// ---------------------------------------------------------------- the replay

/** The rebuild records belonging to one slot, oldest first. */
const recsOf = (snap, k) => snap.tail.reb.filter((x) => x[1] === k);

/**
 * Rebuild one slot across the window.
 *
 * The first record in the window is the Attractor's construction, so the pair
 * only becomes the browser's pair at the second: `from` below is the first
 * frame whose shape this can claim to have reproduced, and everything is
 * measured from there.
 *
 * `want` and not `bits` is what gets asked for, because they are not always
 * the same number -- a shrink keeps its mask for one more rebuild so that
 * both halves of the blend's pair agree about which entries exist, and a
 * replay that asserted the mask instead of requesting it would skip the ramp
 * the whole thing is about.
 */
export function replaySlot(snap, k, { draw = [], W = 260, H = 190,
                                      target = 40000 } = {}) {
  // The panel's own camera parameters, from the demo's field of view.
  const tanY = Math.tan((55 * Math.PI) / 360), tanX = tanY * (W / H);
  const focal = (H / 2) / tanY;
  const recs = recsOf(snap, k);
  const out = { k, nrec: recs.length, from: -1, frames: [], bad: [], shots: [],
                ok: true, why: '' };
  if (recs.length < 2) {
    out.ok = false;
    out.why = `only ${recs.length} rebuild${recs.length === 1 ? '' : 's'} inside`
            + ` the window -- nothing to replay a pair from`;
    return out;
  }
  const seed = snap.pool.seed + k;
  const mk = (u) => {
    const spec = morphPath(SPECS, u);
    return { spec, b: build(spec) };
  };
  const first = mk(recs[0][2]);
  const att = new Attractor({
    maps: first.b.maps, trunkA: first.b.trunkA, sig0: first.b.sig0,
    mu0: first.b.mu0, omega: first.spec.omega ?? 0.28,
    jitter: first.spec.jitter ?? 0.16, seed,
    jitterTableBits: recs[0][4],
    interp: !!snap.pool.interpolate,
    migrateRebuilds: snap.pool.migrateRebuilds,
    palette: PALETTES[first.spec.palette] || PALETTES.spectral,
  });
  out.att = att;

  const pos = new Float64Array(ND), prev = new Float64Array(ND);
  let have = false, ri = 1;
  const bt = snap.tail.bt[k], step = snap.tail.step[k], F = snap.tail.f;
  let st = null, cam = null;

  for (let i = 0; i < F.length; i++) {
    const f = F[i];
    while (ri < recs.length && recs[ri][0] === f) {
      const [, , u, want, bits, w, g, lo, drop] = recs[ri];
      const { spec, b } = mk(u);
      att.setTableBits(want);
      att.reparam({ maps: b.maps, trunkA: b.trunkA, sig0: b.sig0, mu0: b.mu0,
                    omega: spec.omega ?? 0.28, jitter: spec.jitter ?? 0.16,
                    palette: PALETTES[spec.palette] || att.palette });
      // The state the browser reached, against the state this reached. Any
      // of these differing means the replay is not the same table, and
      // nothing below it is worth looking at.
      const got = { bits: att.tableBits, w: att._migW, g: att._migGrow ? 1 : 0,
                    lo: att._migLo, drop: att._migDrop ? 1 : 0 };
      if (got.bits !== bits || Math.abs(got.w - w) > 2e-6 || got.g !== g
          || got.lo !== lo || got.drop !== drop)
        out.bad.push({ f, want, had: { bits, w, g, lo, drop }, got });
      if (out.from < 0) out.from = f;
      ri++;
    }
    if (out.from < 0) continue;                 // before the first real pair
    att.setBlend(bt[i], snap.pool.blendMax, snap.pool.blendMin);
    P.probe(att, pos, 0);
    const s = have ? P.dist(pos, prev, 0, att.R) : NaN;
    prev.set(pos); have = true;
    out.frames.push({ f, bt: bt[i], btGot: att._bt, step: s, had: step[i],
                      mig: att._migW, drop: att._migDrop ? 1 : 0,
                      built: recs.some((x) => x[0] === f) ? 1 : 0 });
    if (draw.includes(f)) {
      if (!st) {
        st = new AdaptiveStream(att, Math.max(400000, 3 * target));
        st.budgetBy = 'splats';
        st.target = target;
        cam = new FlyCamera(att);
        // Looking at the object from outside, from a direction fixed for the
        // whole strip: the panels have to differ by what the shape did, not
        // by where the camera went.
        // 1.8 radii out: at this field of view (tanY 0.52) an object of
        // radius R fills the frame's height at about 1.9, and the panels are
        // there to be compared with each other, not to be flown through.
        cam.pos = [att.C[0] + att.R * 1.05, att.C[1] - att.R * 1.35, att.C[2] + att.R * 0.5];
        cam.f = [att.C[0] - cam.pos[0], att.C[1] - cam.pos[1], att.C[2] - cam.pos[2]];
        cam.u = [0, 0, 1];
        cam.ortho();
        // The level-of-detail controller is a feedback loop on the splat
        // count, and it takes about twenty frames to converge from its
        // starting threshold -- eight left the first panel at 964 splats
        // against a target of 40,000, which would have drawn one panel of
        // the strip at a different detail from the rest. Measured: 25 is
        // converged (40,004 splats) and 40 moves it by 5e-9.
        for (let j = 0; j < 25; j++)
          st.expand(cam, { focal, tanX, tanY, znear: 1e-5, W, H });
      }
      const n = st.expand(cam, { focal, tanX, tanY, znear: 1e-5, W, H });
      out.shots.push({ f, n, im: render({
        pos: st.pos, cov: st.cov, rgba: st.rgba, off: st.offset, count: n,
        eye: cam.pos, B: { r: cam.r, u: cam.u, f: cam.f }, W, H,
        fogNear: 1e9, fogFar: 2e9,
      }) });
    }
  }
  return out;
}

/** Relative error between the replayed steps and the recorded ones. */
function agreement(frames) {
  const rel = [];
  for (const fr of frames) {
    if (!Number.isFinite(fr.step) || !(fr.had > 0)) continue;
    rel.push(Math.abs(fr.step - fr.had) / fr.had);
  }
  return rel;
}

// ------------------------------------------------------------- the diagnosis

export function diagnose(snap) {
  const L = [];
  const say = (s) => L.push(s);
  const S = snap.pool.slots, T = snap.tail;
  say(`  build ${snap.build}, frame ${snap.frame}, ${snap.t}s of flight`);
  say(`  pool: ${S} slots, rebuild ${snap.pool.rebuildHz} Hz (${snap.pool.hz} effective)`
    + `, u ${snap.pool.u}, breath ${snap.pool.breath}`
    + `, morph ${snap.pool.morphing ? 'on' : 'OFF'}`
    + `, interpolate ${snap.pool.interpolate ? 'on' : 'OFF'}`
    + `, ramp ${snap.pool.migrateRebuilds}`);
  if (snap.cam)
    say(`  camera: cell ${snap.cam.home}, depth ${snap.cam.depth}`
      + `, zoom ${snap.cam.scale ? (1 / snap.cam.scale).toExponential(1) : '-'}x`
      + `, ${snap.view.splats} splats at detail ${snap.view.detail}`
      + `, ${snap.view.fps} fps, ${snap.world.visible}/${snap.world.resident} cells`);
  if (!T.f.length) { say('  no window in this snapshot'); return L.join('\n'); }
  say(`  window: ${T.f.length} frames, ${T.f[0]} to ${T.f[T.f.length - 1]}`
    + `, ${T.reb.length} rebuilds`);
  say('');

  // 1. Was it the frame rate? A 200 ms frame delivers twelve frames of
  // motion at once and nothing in the morph is wrong.
  const dt = T.dt.filter((x) => x > 0);
  const mdt = q(dt, 0.5), wdt = q(dt, 1);
  const slow = dt.filter((x) => x > 2 * mdt).length;
  say(`  frame cost: median ${mdt.toFixed(1)} ms, worst ${wdt.toFixed(1)} ms`
    + `, ${slow} frame${slow === 1 ? '' : 's'} over 2x the median`
    + (wdt > 2.5 * mdt && wdt > 40
      ? `  <-- a HITCH. At ${mdt.toFixed(0)} ms a frame this is `
        + `${(wdt / mdt).toFixed(1)} frames of motion in one, which looks `
        + `exactly like a jump and is not the morph.`
      : '  -- steady, so what was seen was not the frame rate'));

  // 2. Was it what is drawn, rather than where it is?
  let popF = -1, popX = 0;
  for (let i = 1; i < T.n.length; i++) {
    if (T.n[i - 1] < 1000) continue;
    const x = Math.abs(T.n[i] - T.n[i - 1]) / T.n[i - 1];
    if (x > popX) { popX = x; popF = T.f[i]; }
  }
  say(`  splat count: ${Math.min(...T.n)} to ${Math.max(...T.n)}`
    + `, worst one-frame change ${(100 * popX).toFixed(1)}% at frame ${popF}`
    + (popX > 0.25 ? '  <-- a POP: the budget or the LOD changed what is drawn'
                   : '  -- no pop'));
  say('');

  // 3. The morph, per slot, as the browser's own probe measured it.
  say('  slot  shape                 bits  rebuilds   worst step   x median   at frame');
  const worst = [];
  for (let k = 0; k < S; k++) {
    const sl = snap.slots[k], st = T.step[k] || [];
    let wi = -1, wv = 0;
    for (let i = 0; i < st.length; i++) if (st[i] > wv) { wv = st[i]; wi = i; }
    const x = sl.med > 0 ? wv / sl.med : NaN;
    worst.push({ k, v: wv, x, f: wi >= 0 ? T.f[wi] : -1, i: wi });
    say(`  s${k}    ${(sl.from + '>' + sl.to).padEnd(20)} ${String(sl.bits).padStart(4)}`
      + `  ${String(recsOf(snap, k).length).padStart(8)}   ${e2(wv).padStart(10)}`
      + `   ${(Number.isFinite(x) ? x.toFixed(1) + 'x' : '-').padStart(8)}`
      + `   ${String(worst[k].f).padStart(8)}`);
  }
  worst.sort((a, b) => b.x - a.x);
  say('');
  const top = worst[0];
  say(`  the worst frame in the window is slot ${top.k} at frame ${top.f}:`
    + ` ${e2(top.v)} of the object's radius, ${top.x.toFixed(1)}x that slot's`
    + ` own median frame`);
  say(`  (the probe in js/mark.js flags ${snap.probe ? snap.probe.ratio : 8}x;`
    + ` the whole flight's worst was ${snap.probe ? snap.probe.worst : '-'}x`
    + ` on slot ${snap.probe ? snap.probe.slot : '-'} at frame`
    + ` ${snap.probe ? snap.probe.frame : '-'})`);
  return { text: L.join('\n'), worst, top };
}

/** The per-frame table for one slot: the thing that shows a freeze and its
 *  release, which is what a clamped blend fraction looks like. */
export function perFrame(snap, k, rep) {
  const L = [], T = snap.tail;
  L.push(`  slot ${k}, frame by frame (B a rebuild landed, M a table`
       + ` migration is running, ! over 8x the median)`);
  L.push('   frame      bt     step  x med   replayed    rel   flags');
  const med = snap.slots[k].med || NaN;
  const byF = new Map(rep ? rep.frames.map((x) => [x.f, x]) : []);
  for (let i = 0; i < T.f.length; i++) {
    const f = T.f[i], s = T.step[k][i], r = byF.get(f);
    const x = med > 0 ? s / med : NaN;
    const rel = r && Number.isFinite(r.step) && s > 0
      ? Math.abs(r.step - s) / s : NaN;
    const reb = T.reb.find((z) => z[0] === f && z[1] === k);
    L.push(`  ${String(f).padStart(6)}  ${(T.bt[k][i] ?? 0).toFixed(3).padStart(6)}`
      + `  ${e2(s).padStart(8)}  ${(Number.isFinite(x) ? x.toFixed(1) : '-').padStart(5)}`
      + `  ${(r ? e2(r.step) : '-').padStart(9)}`
      + `  ${(Number.isFinite(rel) ? (100 * rel).toFixed(1) + '%' : '-').padStart(6)}`
      + `   ${reb ? 'B' : ' '}${r && r.mig < 1 ? 'M' : ' '}`
      + `${Number.isFinite(x) && x > 8 ? ' !' : ''}`
      + `${reb ? `  want ${reb[3]} bits ${reb[4]} w ${reb[5]}` : ''}`);
  }
  return L.join('\n');
}

// ------------------------------------------------------------------ the mode

async function fromLink(arg) {
  let text = arg;
  try { text = readFileSync(arg, 'utf8'); } catch (_) { /* it is the link itself */ }
  const snap = await decode(text);
  if (snap.v !== 1) console.log(`  (snapshot version ${snap.v}, expected 1)`);
  const d = diagnose(snap);
  console.log(d.text);
  const k = d.top.k, T = snap.tail;
  // Four panels around the worst frame, and one from the start of the window
  // to compare them against.
  const i = d.top.i;
  const draw = [T.f[0], T.f[Math.max(0, i - 2)], T.f[Math.max(0, i - 1)],
                T.f[i], T.f[Math.min(T.f.length - 1, i + 1)],
                T.f[Math.min(T.f.length - 1, i + 2)]].filter((x) => x >= 0);
  const rep = replaySlot(snap, k, { draw });
  console.log('');
  if (!rep.ok) {
    console.log(`  slot ${k} cannot be replayed: ${rep.why}`);
  } else {
    const rel = agreement(rep.frames);
    console.log(`  replay of slot ${k}: ${rep.nrec} rebuild records,`
      + ` ${rep.bad.length} disagreed with the state they produced,`
      + ` ${rep.frames.length} frames re-measured from frame ${rep.from}`);
    for (const b of rep.bad)
      console.log(`    FAIL  frame ${b.f}: want ${b.want} -> `
        + `${JSON.stringify(b.got)}, the browser had ${JSON.stringify(b.had)}`);
    console.log(`  splat movement, replayed against recorded:`
      + ` p50 ${(100 * q(rel, 0.5)).toFixed(2)}%, p99 ${(100 * q(rel, 0.99)).toFixed(2)}%,`
      + ` worst ${(100 * q(rel, 1)).toFixed(2)}%`
      + (q(rel, 0.99) < 0.05
        ? '  -- the same frames, so the pictures below are the real ones'
        : '  -- NOT the same frames; treat the pictures as indicative only'));
    console.log('');
    console.log(perFrame(snap, k, rep));
    if (rep.shots.length) {
      mkdirSync('out', { recursive: true });
      const out = process.argv[3] || 'out/mark.png';
      writeFileSync(out, png(tile(rep.shots.map((s) => s.im), 3)));
      console.log('');
      console.log(`  frames ${rep.shots.map((s) => s.f).join(' ')} of slot ${k}`
        + ` -> ${out} (left to right, top row first)`);
    }
  }
}

// ------------------------------------------------------------- the self-test

const DT = 1 / 60;

function flight(migrate, frames, log) {
  const pool = new ShapePool({ slots: 8, seed: 7, rebuildHz: 7,
                               interpolate: true, migrateRebuilds: migrate });
  const prio = new Float64Array(pool.slots);
  for (let f = 0; f < frames; f++) {
    // A moving priority, because that is what drives the table resizes: with
    // every slot equally important nothing is ever re-cut and the part of
    // this that is about resizes would be vacuous. Same drive as
    // tests/drift.mjs.
    for (let k = 0; k < pool.slots; k++)
      prio[k] = 0.5 + 0.5 * Math.sin(0.02 * f + k * 1.7);
    pool.update(DT, prio);
    log.tick(DT, pool, { splats: 120000 + 2000 * Math.sin(f * 0.1), depth: 11,
                         visible: 7, resident: 23 });
  }
  return { pool, prio };
}

function stub(pool, log) {
  return {
    world: { pool, _clock: 12.5, length: 23, visible: 7, _centre: [1, 2, 3] },
    jumpLog: log, fly: null, detail: 2, dissolve: true, dilate: true,
    autopilot: true, opts: { fovy: 55 }, look: { mask: 3 },
    stats: { phase: 'dive', splats: 120000, fps: 60 },
    canvas: null, fogNear: 2.5, fogFar: 18,
  };
}

async function selfTest() {
  let fail = 0;
  const check = (ok, msg) => {
    console.log((ok ? '  ok    ' : '  FAIL  ') + msg);
    if (!ok) fail++;
  };

  // 1. The detector against the fault it was written for. A resize switched
  // rather than ramped is the 86x jump STATUS.md records; if the probe
  // cannot see that, nothing it says about a quiet flight means anything.
  const onLog = new JumpLog({ build: 'selftest-ramp-on' });
  flight(4, 900, onLog);
  const offLog = new JumpLog({ build: 'selftest-ramp-off' });
  const off = flight(1, 900, offLog);
  console.log(`  ramp on  (migrateRebuilds 4): worst ${onLog.worst.toFixed(1)}x`
    + ` the slot's median, ${onLog.jumps} over ${onLog.ratio}x`);
  console.log(`  ramp off (migrateRebuilds 1): worst ${offLog.worst.toFixed(1)}x`
    + ` the slot's median, ${offLog.jumps} over ${offLog.ratio}x`);
  check(offLog.jumps > 0 && offLog.worst > 20,
    `the probe sees a switched resize: ${offLog.worst.toFixed(0)}x on slot`
    + ` ${offLog.worstSlot}, ${offLog.jumps} frames flagged`);
  check(onLog.jumps === 0 && onLog.worst < onLog.ratio,
    `and the ramped one is quiet: worst ${onLog.worst.toFixed(1)}x,`
    + ` nothing flagged`);
  console.log(`  probe cost ${(onLog.probeMs * 1000).toFixed(1)} us/frame`
    + ` for ${onLog.slots} slots`);
  check(onLog.probeMs < 0.3, `and it is cheap enough to leave on:`
    + ` ${onLog.probeMs.toFixed(3)} ms a frame against the pool's own 1.5 ms`
    + ` rebuild budget`);
  console.log('');

  // 2. Reaction time, simulated: carry on past the jump by as long as a hand
  // takes, which is what the window exists to cover.
  const prio = off.prio, pool = off.pool;
  let seen = -1;
  for (let f = 900; f < 2400 && seen < 0; f++) {
    for (let k = 0; k < pool.slots; k++)
      prio[k] = 0.5 + 0.5 * Math.sin(0.02 * f + k * 1.7);
    pool.update(DT, prio);
    offLog.tick(DT, pool, { splats: 120000, depth: 11, visible: 7, resident: 23 });
    const last = offLog.events[offLog.events.length - 1];
    if (last && last.kind === 'MORPH' && last.f === offLog.frame) seen = offLog.frame;
  }
  check(seen > 0, `a jump happened to be caught live (frame ${seen})`);
  for (let i = 0; i < 18; i++) {                 // ~0.3 s of human reaction
    for (let k = 0; k < pool.slots; k++)
      prio[k] = 0.5 + 0.5 * Math.sin(0.02 * (900 + i) + k * 1.7);
    pool.update(DT, prio);
    offLog.tick(DT, pool, { splats: 120000, depth: 11, visible: 7, resident: 23 });
  }
  const v = stub(pool, offLog);
  offLog.mark(pool, { splats: 120000 }, 'self-test');
  const snap = snapshot(v);
  check(!!snap && snap.tail.f.length > 60,
    `the snapshot carries a window of ${snap.tail.f.length} frames and`
    + ` ${snap.tail.reb.length} rebuilds`);
  check(snap.tail.f.includes(seen),
    `and the jump at frame ${seen} is inside it (frames ${snap.tail.f[0]}`
    + `-${snap.tail.f[snap.tail.f.length - 1]}), which is what the window is for`);

  // 3. The link itself.
  const frag = await encode(snap);
  const back = await decode(`https://example.invalid/splat/#${frag}`);
  const raw = JSON.stringify(snap).length;
  console.log(`  snapshot ${(raw / 1024).toFixed(1)} kB of JSON ->`
    + ` ${(frag.length / 1024).toFixed(1)} kB of link`
    + ` (${(100 * frag.length / raw).toFixed(0)}%)`);
  check(JSON.stringify(back) === JSON.stringify(snap),
    `the link round-trips to the same state exactly`);
  check(frag.length < 8192, `and it is short enough to paste: ${frag.length} chars`);
  console.log('');

  // 4. The replay, against what the browser measured.
  const d = diagnose(back);
  const k = d.top.k;
  const rep = replaySlot(back, k, { draw: [d.top.f] });
  check(rep.ok, `slot ${k} replays: ${rep.nrec} rebuild records,`
    + ` ${rep.frames.length} frames from frame ${rep.from}`);
  check(rep.bad.length === 0, `every rebuild in the window reproduces the state`
    + ` the browser reached -- mask, migration weight, merging entries,`
    + ` pending drop (${rep.bad.length} disagreements)`);
  for (const b of rep.bad)
    console.log(`        frame ${b.f}: got ${JSON.stringify(b.got)},`
      + ` had ${JSON.stringify(b.had)}`);
  const rel = agreement(rep.frames);
  console.log(`  replayed splat movement against recorded: p50`
    + ` ${(100 * q(rel, 0.5)).toFixed(3)}%, p99 ${(100 * q(rel, 0.99)).toFixed(3)}%,`
    + ` worst ${(100 * q(rel, 1)).toFixed(3)}%`);
  check(q(rel, 0.5) < 0.02 && q(rel, 1) < 0.2,
    `the reconstruction moves the splats the same way the browser did`);
  check(rep.shots.length > 0 && rep.shots[0].n > 1000,
    `and it draws: ${rep.shots.length} frame(s), ${rep.shots[0]?.n} splats`);

  // 5. The whole report, on the real path, so the thing a viewer will
  // actually run is what was tested.
  console.log('');
  console.log('  ---- the report a link produces ----');
  await fromLink(`https://example.invalid/splat/#${frag}`);

  console.log(fail ? `\n  ${fail} FAILED` : '\n  mark ok');
  process.exit(fail ? 1 : 0);
}

if (process.argv[2]) await fromLink(process.argv[2]);
else await selfTest();
