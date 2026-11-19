// The two morph switches, and the two rates the panel shows for them.
//
// A switch in a panel is a claim: "interpolation off" has to mean the shape
// stops moving between rebuilds, "morph off" has to mean it stops moving at
// all, and the fps figures beside them have to be the rate the shape ACTUALLY
// changes rather than the frame rate wearing a different label. Each of those
// is checked here against the drawn table, not against the flag that was set.
//
// Why the drawn table: every one of these switches has a way to fail that
// leaves the flag correct. Turning interpolation off without reconciling the
// blend leaves the shape frozen part-way between two builds; turning it back
// on without resetting the pair lerps from a table that stopped being updated
// an arbitrary time ago, which walks the shape backwards through a morph it
// already did. Reading `pool.interpolate` would see neither.
import { ShapePool } from '../js/field.js';

// The demo's own pool settings (world.js builds it with rebuildHz 7), not
// `auto`. Under `auto` the quantum follows the frame rate, so on a machine
// with nothing else to do the pool rebuilds 20+ times a second per slot and
// interpolation looks like a much smaller win than it is in a browser that
// is also drawing. Measuring the switch against settings the demo does not
// use would be measuring the wrong thing.
const FPS = 60, DT = 1 / FPS;
const pool = new ShapePool({ slots: 8, rebuildHz: 7, interpolate: true });
const prio = new Float64Array(pool.slots).fill(1);

// What the traversal reads: the first child map of slot k, blended exactly as
// adaptive.js's hot loop blends it. If this does not move, the shape on screen
// did not move.
const probe = (k) => {
  const att = pool.shapes[k].att, bt = att._bt;
  let h = 0;
  for (let i = 0; i < 9; i++) {
    const a = att._cmA[i], b = att._cmB[i];
    h = (h * 31 + (bt === 1 ? b : a + (b - a) * bt)) % 1e9;
  }
  return h;
};

// The rest of what the shape IS: the moments, the bounding radius, the child
// scales and the translations, which the traversal reads alongside the table
// and the camera reads instead of it. These are snapshotted either side of a
// rebuild and interpolated like the table, so a switch has to reconcile them
// like the table -- and if it does not, `pool.interpolate` and the table
// probe above both read perfectly correct while the field draws the newest
// build's geometry at an old build's level of detail, with its children in
// the wrong places.
const dstate = (k) => {
  const a = pool.shapes[k].att;
  let h = 0;
  const put = (v) => { h = (h * 31 + v) % 1e9; };
  put(a.R); put(a.qMax); put(a.rodMax);
  for (const arr of [a.C, a.sig0, a.mu0, a.Q, a.s, a.mapB, a.rodMu, a.rodSig])
    for (let i = 0; i < arr.length; i++) put(arr[i]);
  return h;
};

// `frames` frames, counting slot-frames in which the drawn shape moved.
//
// The meters are averaged over the whole run rather than read at the end: a
// meter is a half-second window, and comparing one window against a ten
// second count is comparing two different quantities and calling the
// difference a bug.
const run = (frames) => {
  const prev = new Float64Array(pool.slots);
  for (let k = 0; k < pool.slots; k++) prev[k] = probe(k);
  let moved = 0, total = 0, bad = 0, sHz = 0, rHz = 0;
  for (let f = 0; f < frames; f++) {
    pool.update(DT, prio);
    sHz += pool.shapeHz; rHz += pool.rebuildHzNow;
    for (let k = 0; k < pool.slots; k++) {
      const h = probe(k);
      if (h !== prev[k]) moved++;
      prev[k] = h;
      total++;
      const att = pool.shapes[k].att;
      if (!(att.R > 0) || !(att.sig0[0] > 0 && att.sig0[1] > 0 && att.sig0[2] > 0)) bad++;
    }
  }
  return { frac: moved / total, bad, shapeHz: sHz / frames, rebuildHz: rHz / frames };
};

// A switch must not change the picture by itself: whatever is on screen when
// it is flipped has to still be on screen the instant after. That is the one
// property both failure modes break -- an unreconciled blend jumps the shape
// to the newest build, a stale table pair jumps it to an old one.
const snap = () => Array.from({ length: pool.slots },
  (_, k) => [probe(k), dstate(k)]);
const jumped = (a, b, j) => a.reduce((n, v, i) => n + (v[j] === b[i][j] ? 0 : 1), 0);
let jumps = 0, dJumps = 0;
const flip = (fn) => {
  const a = snap(); fn(); const b = snap();
  jumps += jumped(a, b, 0); dJumps += jumped(a, b, 1);
};

run(120);                                    // settle: warm-up, first rebuilds
const on = run(600);
flip(() => pool.setInterpolate(false));
const off = run(600);
flip(() => pool.setInterpolate(true));
const back = run(600);
flip(() => pool.setMorphing(false));
const held = run(120);
flip(() => pool.setMorphing(true));
const again = run(300);

const pct = (x) => (100 * x).toFixed(1) + '%';
const hz = (x) => x.toFixed(1).padStart(5);
console.log(`  ${FPS} fps, ${pool.slots} slots, rebuild budget ${pool.rebuildBudgetMs} ms\n`);
console.log('                      slot-frames that moved   shape fps   rebuild fps');
const row = (label, r) => console.log('  ' + label.padEnd(22) +
  pct(r.frac).padStart(12) + hz(r.shapeHz).padStart(14) + hz(r.rebuildHz).padStart(14));
row('interpolation on', on);
row('interpolation off', off);
row('on again', back);
row('morph off', held);
row('morph on again', again);

let fail = 0;
const check = (ok, msg) => { console.log((ok ? '  ok    ' : '  FAIL  ') + msg); if (!ok) fail++; };
console.log('');
check(on.frac > 0.95, `interpolation on: the shape moves nearly every slot-frame (${pct(on.frac)})`);
check(off.frac < 0.25, `interpolation off: it moves only when rebuilt (${pct(off.frac)})`);
check(off.frac > 0.01, 'interpolation off: rebuilds still land -- the morph is not stopped');
check(back.frac > 0.95, `switched back on: moving every frame again (${pct(back.frac)})`);
check(held.frac === 0, `morph off: nothing moves at all (${pct(held.frac)})`);
check(again.frac > 0.95, `morph on again: it carries on (${pct(again.frac)})`);
check(on.bad + off.bad + back.bad + held.bad + again.bad === 0,
  'no slot is left with a non-positive radius or variance through any switch');
check(jumps === 0, `no switch moves the drawn table by itself (${jumps} slots jumped)`);
check(dJumps === 0,
  `no switch moves the moments, radius or translations either (${dJumps} slots jumped)`);

// The meters are what the panel prints, so they have to agree with the
// slot-frames counted here rather than merely being plausible.
console.log('');
check(Math.abs(on.shapeHz - FPS * on.frac) < 2,
  `shape fps matches the count with interpolation (${on.shapeHz.toFixed(1)} vs ${(FPS * on.frac).toFixed(1)})`);
check(Math.abs(off.shapeHz - FPS * off.frac) < 2,
  `shape fps matches the count without it (${off.shapeHz.toFixed(1)} vs ${(FPS * off.frac).toFixed(1)})`);
check(on.shapeHz > 4 * on.rebuildHz,
  `interpolation is the difference: ${on.shapeHz.toFixed(1)} fps drawn from ${on.rebuildHz.toFixed(1)} rebuilds/s`);
check(Math.abs(off.shapeHz - off.rebuildHz) < 1.5,
  'without interpolation the two rates are the same number, as they must be');
check(held.shapeHz === 0 && held.rebuildHz === 0,
  'morph off reads zero rather than holding its last value');

console.log(fail ? `\n  ${fail} FAILED` : '\n  toggles ok');
process.exit(fail ? 1 : 0);
