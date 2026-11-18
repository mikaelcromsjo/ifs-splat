// Does the endless field actually hold together?
//
// Five things, in the order they would break something visible:
//
//   1 determinism   fly away, fly back, get the same objects bit for bit --
//                   otherwise "nothing is stored" is a lie and the field
//                   reshuffles behind the camera
//   2 separation    no two objects ever intersect, over the WHOLE morph cycle,
//                   because draw order between objects is stream order and
//                   stream order is only valid while their spheres are
//                   disjoint.  Also re-measures REACH_MAX, which the entire
//                   guarantee is built on.
//   3 density       the field is as full as it was asked to be, and the sizes,
//                   slots and tints are spread the way they were meant to be
//   4 never empty   a long flight never arrives somewhere with nothing in it,
//                   and the boundary-cross early-out is actually firing
//   5 jitter        the per-node jitter rotation cannot push geometry outside
//                   |C| + R.  REACH_MAX comes from invariantMoments, which
//                   never sees the jitter; if the drawn object were bigger
//                   than its own bound, check 2 would be measuring the wrong
//                   sphere and check 2 would still pass.

const SUBDIV_T = 2;   // mirrors SUBDIV in field.js; the test states it independently
import { FieldWorld } from '../js/world.js';
import { REACH_MAX, cellHash, cellContents, cellObjects, ShapePool, FieldLayout, cellKey }
  from '../js/field.js';
import { SETS, SET_NAMES, build, morph, morphPath } from '../js/maps.js';
const SPECS = SET_NAMES.map((x) => SETS[x]);
import { Attractor } from '../js/adaptive.js';

let bad = 0;
const ok = (c, label, detail) => {
  if (!c) bad++;
  console.log(`  ${label.padEnd(52)} ${detail}  ${c ? 'ok' : 'FAILED'}`);
  return c;
};
const ms = () => Number(process.hrtime.bigint()) / 1e6;

const OPTS = { L: 10, density: 0.35, seed: 1, rho: 0.25, uMin: 0.35, slots: 8 };
const FIELDS = ['scale', 'yaw', 'tilt', 'slot', 'hash'];

function same(a, b) {
  if (!a || !b) return a === b;
  for (const f of FIELDS) if (!Object.is(a[f], b[f])) return false;
  for (let i = 0; i < 3; i++) {
    if (!Object.is(a.t[i], b.t[i])) return false;
    if (!Object.is(a.tint[i], b.tint[i])) return false;
  }
  return a.key === b.key && Object.is(a.reach, b.reach);
}

// ------------------------------------------------------------------ 1
console.log('1. determinism\n');

{
  const field = new FieldLayout(null, { ...OPTS, genRadius: 55 });
  field.update([0, 0, 0]);
  const snap = new Map();
  for (const r of field.resident()) snap.set(r.key, r);
  const n0 = field.size;

  // fly a long way off: every one of them must be evicted
  field.update([40000, -17000, 9000]);
  let survivors = 0;
  for (const k of snap.keys()) if (field.get(k)) survivors++;

  // and back
  field.update([0, 0, 0]);
  let missing = 0, differing = 0, worstT = 0;
  for (const [k, a] of snap) {
    const b = field.get(k);
    if (!b) { missing++; continue; }
    if (!same(a, b)) differing++;
    for (let i = 0; i < 3; i++) worstT = Math.max(worstT, Math.abs(a.t[i] - b.t[i]));
  }
  ok(n0 > 100, 'a home region holds objects', `${n0} cells resident`);
  ok(survivors === 0, 'flying away evicts all of them', `${survivors} survived`);
  ok(missing === 0 && differing === 0 && field.size === n0,
     'flying back rematerialises every field exactly',
     `${n0} cells, ${missing} missing, ${differing} differing, dt ${worstT.toExponential(1)}`);

  // the pure function, twice, and out of order
  const coords = [];
  for (let i = -12; i < 12; i++) coords.push([i, (i * 7) % 11, (i * 13) % 17]);
  let diff2 = 0;
  for (const c of coords) {
    const a = cellContents(c[0], c[1], c[2], OPTS);
    const b = cellContents(c[0], c[1], c[2], OPTS);
    if (!same(a, b)) diff2++;
  }
  ok(diff2 === 0, 'cellContents is a pure function', `${coords.length} cells, ${diff2} differ`);

  // call order must not matter -- a hidden counter or a shared scratch buffer
  // would show up here and nowhere else
  const fwd = coords.map((c) => cellContents(c[0], c[1], c[2], OPTS));
  const rev = [];
  for (let i = coords.length - 1; i >= 0; i--)
    rev[i] = cellContents(coords[i][0], coords[i][1], coords[i][2], OPTS);
  let diff3 = 0;
  for (let i = 0; i < fwd.length; i++) if (!same(fwd[i], rev[i])) diff3++;
  ok(diff3 === 0, 'independent of the order cells are asked for', `${diff3} differ`);

  // negative coordinates must hash, and must not mirror the positive side
  let collide = 0;
  for (let i = 1; i < 300; i++)
    if (cellHash(-i, -i * 2, -i * 3, 1) === cellHash(i, i * 2, i * 3, 1)) collide++;
  ok(collide === 0, 'negative coordinates are their own cells', `${collide}/299 mirrored`);
}

// ------------------------------------------------------------------ 2
console.log('\n2. separation\n');

// REACH_MAX first: it is the number everything below is measured against.
{
  const reachOf = (spec) => {
    const b = build(spec);
    const a = new Attractor({ maps: b.maps, trunkA: b.trunkA, sig0: b.sig0, mu0: b.mu0,
                              omega: spec.omega ?? 0.28, jitter: 0.16, seed: 3 });
    return Math.hypot(a.C[0], a.C[1], a.C[2]) + a.R;
  };
  let worst = 0, where = '', n = 0;
  for (const name of SET_NAMES) {
    const v = reachOf(SETS[name]); n++;
    if (v > worst) { worst = v; where = name; }
  }
  let pairs = 0;
  for (let i = 0; i < SET_NAMES.length; i++)
    for (let j = i + 1; j < SET_NAMES.length; j++) {
      pairs++;
      for (let k = 0; k <= 10; k++) {
        const v = reachOf(morph(SETS[SET_NAMES[i]], SETS[SET_NAMES[j]], k / 10)); n++;
        if (v > worst) { worst = v; where = `${SET_NAMES[i]}->${SET_NAMES[j]} @ ${k / 10}`; }
      }
    }
  const worstLinear = worst;
  // And the spline, which is what the pool actually flies.  This is the case
  // that matters: Catmull-Rom leaves the hull of its control points, so the
  // path reaches shapes that no straight morph between two sets ever visits,
  // and it reached 11% past the old constant.  Sampling it finely, because
  // the maximum sits inside a segment rather than at a control point -- 40
  // steps a segment missed nothing against 200, but coarse sampling here
  // would be measuring the wrong thing cheaply.
  const SPLINE_STEPS = 60;
  let splineWorst = 0, splineAt = '';
  for (let i = 0; i < SET_NAMES.length; i++)
    for (let k = 0; k < SPLINE_STEPS; k++) {
      const u = i + k / SPLINE_STEPS;
      const v = reachOf(morphPath(SPECS, u)); n++;
      if (v > splineWorst) { splineWorst = v; splineAt = `u=${u.toFixed(3)}`; }
    }
  if (splineWorst > worst) { worst = splineWorst; where = `spline ${splineAt}`; }
  console.log(`  ${n} shapes (${SET_NAMES.length} sets, ${pairs} morph pairs x 11 steps,`
            + ` spline ${SET_NAMES.length}x${SPLINE_STEPS} steps)`);
  console.log(`  straight morphs reach ${worstLinear.toFixed(5)}, the spline `
            + `${splineWorst.toFixed(5)} -- the spline is what the pool flies`);
  ok(worst <= REACH_MAX,
     'max(|C| + R) <= REACH_MAX',
     `${worst.toFixed(6)} <= ${REACH_MAX} (worst: ${where}), headroom `
     + `${(REACH_MAX - worst).toExponential(2)}`);
}

// Now the sweep.  The argument is the same one, restated for a hierarchy, and
// it gets SHORTER rather than longer.
//
// A cell that holds an object does not subdivide, so no object's cell contains
// another object's cell.  Cells at any two levels are either nested or
// disjoint, so every pair of object-holding cells is disjoint.  Each object
// lies inside the sphere inscribed in its own cell -- jitter <= L_l*(0.5-rho),
// reach <= rho*L_l, and those sum to L_l/2 at EVERY level, because both are
// written in terms of that level's own side length.  Disjoint boxes, each
// containing its object: the objects cannot intersect, whatever levels they
// are at.
//
// What that leaves to check by brute force is only proximity.  Every object
// sits inside some level-0 cell, and two level-0 cells at Chebyshev distance
// >= 2 have centres >= 2L apart with every object within L/2 of its own
// centre -- a gap of at least L. So only objects in the same root cell, or in
// touching root cells, can be close, and those are what the sweep compares.
const SWEEP = 40;                          // 40^3 = 64,000 root cells
const idx = (x, y, z) => ((z * SWEEP) + y) * SWEEP + x;
let occupied = new Array(SWEEP * SWEEP * SWEEP).fill(null);
let allObjs = [];
let nOcc = 0;
{
  const t0 = ms();
  for (let z = 0; z < SWEEP; z++)
    for (let y = 0; y < SWEEP; y++)
      for (let x = 0; x < SWEEP; x++) {
        const list = cellObjects(x - 20, y - 20, z - 20, OPTS, []);
        if (list.length) {
          occupied[idx(x, y, z)] = list;
          nOcc++;
          for (const r of list) allObjs.push(r);
        }
      }
  const byLevel = new Map();
  for (const r of allObjs) byLevel.set(r.level, (byLevel.get(r.level) || 0) + 1);
  console.log(`  swept ${SWEEP ** 3} root cells in ${(ms() - t0).toFixed(0)} ms, `
    + `${nOcc} non-empty, ${allObjs.length} objects`);
  console.log('  by level: ' + [...byLevel.entries()].sort((a, b) => a[0] - b[0])
    .map(([k, v]) => `L${k} ${v}`).join('   '));
}

{
  // The 13 forward half of the 26-neighbourhood, so each pair is tested once.
  const NB = [];
  for (let dz = -1; dz <= 1; dz++) for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
    if (dz > 0 || (dz === 0 && (dy > 0 || (dy === 0 && dx > 0)))) NB.push([dx, dy, dz]);
  }
  let pairs = 0, worst = Infinity, worstAt = '', violations = 0;
  let cross = 0, worstCross = Infinity, sameCell = 0;
  const test = (a, b) => {
    pairs++;
    const d = Math.hypot(a.t[0] - b.t[0], a.t[1] - b.t[1], a.t[2] - b.t[2]);
    const m = d - a.reach - b.reach;
    if (m < worst) { worst = m; worstAt = `${a.key} (L${a.level}) / ${b.key} (L${b.level})`; }
    if (a.level !== b.level) { cross++; if (m < worstCross) worstCross = m; }
    if (m < 0) violations++;
  };
  for (let z = 0; z < SWEEP; z++)
    for (let y = 0; y < SWEEP; y++)
      for (let x = 0; x < SWEEP; x++) {
        const A = occupied[idx(x, y, z)];
        if (!A) continue;
        // Within one root cell: a cluster's own members, which is where the
        // hierarchy packs objects closest and where it is most likely to be
        // wrong.  The flat field never had this case at all.
        for (let i = 0; i < A.length; i++)
          for (let j = i + 1; j < A.length; j++) { test(A[i], A[j]); sameCell++; }
        for (const [dx, dy, dz] of NB) {
          const nx = x + dx, ny = y + dy, nz = z + dz;
          if (nx < 0 || ny < 0 || nz < 0 || nx >= SWEEP || ny >= SWEEP || nz >= SWEEP) continue;
          const B = occupied[idx(nx, ny, nz)];
          if (!B) continue;
          for (const a of A) for (const b of B) test(a, b);
        }
      }
  ok(NB.length === 13, 'half-neighbourhood covers each pair once', `${NB.length} offsets`);
  ok(violations === 0 && pairs > 0,
     'no two neighbouring objects intersect',
     `${pairs} pairs, ${violations} intersecting`);
  ok(cross > 0, 'and the cross-level pairs were actually exercised',
     `${cross} of them, ${sameCell} inside a single root cell, `
     + `worst cross-level margin ${worstCross.toFixed(4)}`);
  const budget = 2 * OPTS.rho * OPTS.L;
  console.log(`  worst-case margin observed ${worst.toFixed(6)} world units `
    + `(${(100 * worst / budget).toFixed(2)}% of the ${budget} unit budget), at ${worstAt}`);
  console.log('  (this number is the thing to watch: it erodes silently, and a '
    + 'negative\n   one is a flat sorting seam cutting through two objects)');

  // and the structural claim the whole argument rests on -- now per level,
  // against that level's own side length, which is the thing that makes the
  // proof carry across levels at all
  let outside = 0, worstIn = 0;
  for (const a of allObjs) {
    const Ll = OPTS.L / SUBDIV_T ** a.level;
    for (let i = 0; i < 3; i++) {
      const c = ([a.ix, a.iy, a.iz][i] + 0.5) * Ll;
      const over = Math.abs(a.t[i] - c) + a.reach - Ll / 2;
      if (over > worstIn) worstIn = over;
      if (over > 0) outside++;
    }
  }
  ok(outside === 0, 'every object fits inside its own cell',
     `worst overhang ${worstIn.toExponential(2)} units`);
}

// ------------------------------------------------------------------ 3
console.log('\n3. density, size, slots, tint\n');

{
  // Objects per root cell, against the branching process the hierarchy is.
  // A cell holds an object with probability `density`, or subdivides into 8
  // with probability P_SUB[level], so counting from the deepest level up:
  //
  //     E[level MAX] = density                     (cannot subdivide)
  //     E[level l]   = density + 8*P_SUB[l]*E[l+1]
  //
  // This is the check that a change to P_SUB or MAX_LEVEL has to come past.
  // It replaces the flat field's density/L^3, which is no longer what the
  // field does: that formula counts one object per cell and now undercounts
  // by 3x, which is exactly the sort of silent mismatch worth a gate.
  const P_SUB_T = [0.14, 0.10, 0];
  let want = OPTS.density;
  for (let l = 2 - 1; l >= 0; l--) want = OPTS.density + 8 * P_SUB_T[l] * want;
  const got = allObjs.length / (SWEEP ** 3);
  const err = Math.abs(got - want) / want;
  ok(err < 0.04, 'objects per root cell matches the branching process',
     `${got.toFixed(4)} vs ${want.toFixed(4)} expected, off by `
     + `${(100 * err).toFixed(2)}%`);
  ok(nOcc / SWEEP ** 3 < 0.6, 'and most of space is still empty -- there are voids',
     `${(100 * (1 - nOcc / SWEEP ** 3)).toFixed(1)}% of root cells yield nothing`);

  let sMin = Infinity, sMax = -Infinity, sSum = 0, sSq = 0;
  const slotN = new Int32Array(OPTS.slots);
  let tLo = Infinity, tHi = -Infinity, lumWorst = 0;
  for (const r of allObjs) {
    sMin = Math.min(sMin, r.scale); sMax = Math.max(sMax, r.scale);
    sSum += r.scale; sSq += r.scale * r.scale;
    slotN[r.slot]++;
    for (const c of r.tint) { tLo = Math.min(tLo, c); tHi = Math.max(tHi, c); }
    const lum = 0.2126 * r.tint[0] + 0.7152 * r.tint[1] + 0.0722 * r.tint[2];
    lumWorst = Math.max(lumWorst, Math.abs(lum - 1));
  }
  const mean = sSum / allObjs.length;
  const sd = Math.sqrt(sSq / allObjs.length - mean * mean);
  const cap = OPTS.rho * OPTS.L / REACH_MAX;
  console.log(`  scale  mean ${mean.toFixed(4)}  sd ${sd.toFixed(4)}  `
    + `range ${sMin.toFixed(4)}..${sMax.toFixed(4)}  (cap ${cap.toFixed(4)}, `
    + `mean is ${(100 * mean / cap).toFixed(1)}% of it)`);
  ok(sMax <= cap + 1e-12, 'no object exceeds the reach budget',
     `max scale ${sMax.toFixed(6)} <= ${cap.toFixed(6)}`);

  // The size distribution is the point of the hierarchy: sizes should fall
  // geometrically with level while the count of cells able to hold them
  // rises, which is a power law nobody had to choose the exponent of.
  {
    const byLevel = new Map();
    for (const r of allObjs) {
      if (!byLevel.has(r.level)) byLevel.set(r.level, { n: 0, max: 0 });
      const e = byLevel.get(r.level);
      e.n++; if (r.scale > e.max) e.max = r.scale;
    }
    const lv = [...byLevel.entries()].sort((a, b) => a[0] - b[0]);
    console.log('  size by level:');
    for (const [l, e] of lv)
      console.log(`    L${l}  ${String(e.n).padStart(6)} objects   max scale `
        + `${e.max.toFixed(4)}`);
    ok(lv.length >= 3, 'every level is populated', `${lv.length} levels in use`);
    let geometric = true;
    for (let i = 1; i < lv.length; i++) {
      const ratio = lv[i - 1][1].max / lv[i][1].max;
      if (Math.abs(ratio - SUBDIV_T) > 0.08 * SUBDIV_T) geometric = false;
    }
    ok(geometric, 'and each level is SUBDIV times smaller than the one above',
       `max-scale ratios ${lv.slice(1).map((e, i) =>
         (lv[i][1].max / e[1].max).toFixed(3)).join(', ')} vs ${SUBDIV_T}`);
    ok(sMax / sMin > 5, 'so the field spans a real range of sizes',
       `${(sMax / sMin).toFixed(1)}x from smallest to largest`);
  }

  const exp = allObjs.length / OPTS.slots;
  let chi = 0, lo = Infinity, hi = -Infinity;
  for (const n of slotN) { chi += (n - exp) ** 2 / exp; lo = Math.min(lo, n); hi = Math.max(hi, n); }
  console.log(`  slots  ${Array.from(slotN).join(' ')}  (expect ${exp.toFixed(0)} each)`);
  // chi-square, 7 d.o.f.: 24.3 is p = 0.001.  A hash that put a bias in the
  // low 3 bits would be hundreds, not twenty.
  ok(chi < 24.3, 'pool slots are uniform (chi-square, 7 d.o.f.)',
     `chi2 ${chi.toFixed(1)} < 24.3, spread ${lo}..${hi}`);

  ok(tLo >= 0.75 && tHi <= 1.25, 'tint stays in [0.75, 1.25]',
     `${tLo.toFixed(4)}..${tHi.toFixed(4)}`);
  ok(lumWorst < 1e-12, 'tint is luminance-neutral (Rec.709)',
     `worst |lum - 1| ${lumWorst.toExponential(2)}`);
}

// Drop the sweep before timing anything.  43,725 frozen records with two
// frozen arrays each is ~40 MB held live, and a major GC landing inside a
// reparam gets timed as if ShapePool had caused it -- a 25 ms frame that is
// this test's own garbage, not the pool's.
occupied = null;

// ------------------------------------------------------------------ 4
console.log('\n4. a 2000-step flight\n');

{
  const pool = new ShapePool({ slots: 8, rebuildBudgetMs: 1.5 });
  const field = new FieldLayout(pool, { ...OPTS, genRadius: 55, evictFactor: 1.2 });
  console.log(`  pool warm-up ${pool.warmupMs.toFixed(1)} ms in the constructor, `
    + `seeding the per-rebuild estimate at ${pool.reparamEst.toFixed(3)} ms`);

  // A deterministic walk with persistence: a pure random walk barely leaves the
  // first few cells in 2000 steps, which would not test eviction at all.
  let s = 20260101;
  const rnd = () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
  const pos = [0, 0, 0];
  const dir = [1, 0, 0];
  const prio = new Float64Array(pool.slots);

  const STEPS = 2000, SPEED = 2.5, DT = 1 / 60;
  let minR = Infinity, maxR = 0, sumR = 0, emptyAt = -1;
  let poolSum = 0, poolMax = 0, poolMaxAt = -1;
  let reparamSum = 0, reparamMax = 0, rebuilds = 0, deferred = 0, maxDeferred = 0;
  const reparamEach = new Float64Array(STEPS);
  const rebuildHist = new Int32Array(16);
  let enumSum = 0, enumMax = 0, travelled = 0;
  const early = [];

  field.update(pos);
  // The camera's home cell must survive being left behind: the camera holds an
  // ADDRESS inside it, not a world position (the whole point of world.js), so
  // evicting it strands the camera in coordinates that refer to nothing.  Cell
  // (0,0,0) is empty 65% of the time, so pin a cell that actually exists.
  let homeKey = null, homeD = Infinity;
  for (const r of field.resident()) {
    const d = Math.hypot(r.t[0], r.t[1], r.t[2]);
    if (d < homeD) { homeD = d; homeKey = r.key; }
  }
  const homeRec = field.pin(homeKey);

  for (let i = 0; i < STEPS; i++) {
    for (let k = 0; k < 3; k++) dir[k] += (rnd() - 0.5) * 0.35;
    const n = Math.hypot(dir[0], dir[1], dir[2]) || 1;
    for (let k = 0; k < 3; k++) { dir[k] /= n; pos[k] += dir[k] * SPEED; }
    travelled += SPEED;

    // on-screen importance, the way the viewer computes it: the largest
    // apparent size any cell on that slot has from here
    prio.fill(0);
    for (const r of field.resident()) {
      const d = Math.hypot(r.t[0] - pos[0], r.t[1] - pos[1], r.t[2] - pos[2]);
      const w = r.reach / Math.max(d, r.reach);
      if (w > prio[r.slot]) prio[r.slot] = w;
    }

    const p0 = ms();
    pool.update(DT, prio);
    const p1 = ms() - p0;
    poolSum += p1;
    if (p1 > poolMax) { poolMax = p1; poolMaxAt = i; }
    reparamSum += pool.reparamMs;
    reparamMax = Math.max(reparamMax, pool.reparamMs);
    reparamEach[i] = pool.reparamMs;
    rebuildHist[Math.min(15, pool.rebuildsThisFrame)]++;
    rebuilds += pool.rebuildsThisFrame;
    deferred += pool.deferred;
    maxDeferred = Math.max(maxDeferred, pool.deferred);
    if (i < 4) early.push(`f${i} ${pool.reparamMs.toFixed(2)}ms/${pool.rebuildsThisFrame}`);

    field.update(pos);
    enumSum += field.lastEnumeration;
    enumMax = Math.max(enumMax, field.lastEnumeration);

    const r = field.size;
    if (r === 0 && emptyAt < 0) emptyAt = i;
    minR = Math.min(minR, r); maxR = Math.max(maxR, r); sumR += r;
  }

  ok(emptyAt < 0, 'residency never drops to zero',
     `min ${minR}, mean ${(sumR / STEPS).toFixed(1)}, max ${maxR}`);
  console.log(`  flew ${travelled.toFixed(0)} world units (${(travelled / OPTS.L).toFixed(0)} `
    + `cells) from the origin to ${pos.map((v) => v.toFixed(0)).join(', ')}`);
  console.log(`  churn  ${(field.totalMaterialised / STEPS).toFixed(2)} materialised and `
    + `${(field.totalEvicted / STEPS).toFixed(2)} evicted per step`);
  console.log(`  enumerated on ${field.enumerations} of ${field.updates} updates `
    + `(${(100 * field.enumerations / field.updates).toFixed(1)}% -- the rest crossed no `
    + `cell boundary)`);
  console.log(`  enumeration ${(enumSum / Math.max(field.enumerations, 1)).toFixed(3)} ms `
    + `each, ${(enumSum / STEPS).toFixed(3)} ms/frame amortised, worst `
    + `${enumMax.toFixed(2)} ms`);

  console.log(`\n  ShapePool.update  mean ${(poolSum / STEPS).toFixed(3)} ms/frame, worst `
    + `${poolMax.toFixed(2)} ms (frame ${poolMaxAt})`);
  const sorted = Float64Array.from(reparamEach).sort();
  const pct = (q) => sorted[Math.min(STEPS - 1, Math.floor(q * STEPS))];
  console.log(`  reparam           mean ${(reparamSum / STEPS).toFixed(3)} ms/frame, p50 `
    + `${pct(0.5).toFixed(3)}, p99 ${pct(0.99).toFixed(3)}, worst ${reparamMax.toFixed(2)} ms`);
  console.log(`  per-rebuild       estimate converged to ${pool.reparamEst.toFixed(3)} ms `
    + `(warm-up seeded it at the constructor, ${pool.warmupPasses} passes, `
    + `${pool.warmupMs.toFixed(0)} ms)`);
  console.log(`  rebuilds          ${rebuilds} over ${STEPS} frames `
    + `(${(rebuilds / STEPS).toFixed(2)}/frame), ${deferred} slot-frames deferred `
    + `(${(deferred / STEPS).toFixed(2)}/frame, worst ${maxDeferred} at once)`);
  let maxRb = 0;
  for (let n = 0; n < 16; n++) if (rebuildHist[n]) maxRb = n;
  const hz = (rebuilds / (STEPS * DT)) / pool.slots;
  console.log(`  rebuilds/frame    ` + Array.from(rebuildHist.slice(0, maxRb + 1))
    .map((c, n) => `${n}:${c}`).join('  '));
  console.log(`  effective rate    ${hz.toFixed(1)} Hz per slot against the ${pool.rebuildHz} Hz `
    + `asked for -- the budget, not the schedule, is what sets it`);
  console.log(`  first frames      ${early.join('  ')}`);

  // The budget is predictive now, so a frame may overshoot by at most the
  // error in the estimate, not by a whole extra rebuild.  1.6x was the old
  // behaviour; anything near that means the estimate has stopped tracking.
  // Gate on the DECISION, not on the wall clock.  What this class controls is
  // how many rebuilds it starts; what a started rebuild then costs is V8's
  // business -- a GC landing inside one reparam stretched a single rebuild to
  // 11 ms in this run and to 33 ms in another, and no scheduling policy can
  // prevent that.  So: having admitted n rebuilds at an estimated cost each,
  // n*est must fit the budget, with the first always allowed so the pool
  // cannot deadlock on a budget smaller than one rebuild.
  //
  // This is exactly the defect.  The old test-after-the-fact check admitted a
  // second rebuild at 1.19 ms each against a 1.5 ms budget and landed at
  // 2.4 ms, 1.6x over, every frame in steady state; floor(1.5/1.19) = 1, so
  // this assertion fails on that behaviour and passes on this one.
  const allowed = Math.max(1, Math.floor(pool.rebuildBudgetMs / pool.reparamEst));
  ok(pool.worstAdmit <= pool.rebuildBudgetMs,
     'never starts a rebuild the budget cannot afford',
     `worst admission claimed ${pool.worstAdmit.toFixed(2)} ms against a `
     + `${pool.rebuildBudgetMs} ms budget; worst frame started ${maxRb}, `
     + `~${allowed} affordable at `
     + `${pool.reparamEst.toFixed(2)} ms each`);
  ok(reparamSum / STEPS <= pool.rebuildBudgetMs,
     'and the mean frame stays inside the budget',
     `mean ${(reparamSum / STEPS).toFixed(2)} ms vs ${pool.rebuildBudgetMs} ms; p99 `
     + `${pct(0.99).toFixed(2)}, worst ${reparamMax.toFixed(2)} ms (one reparam, stretched `
     + `by a GC -- no frame ever started ${maxRb + 1})`);
  // The cold start must not be a stall.  Bounded ABSOLUTELY, against the
  // budget, not as a ratio to the median: a rebuild now costs ~0.33 ms and
  // most frames do none at all, so the median is 0.00 and every ratio to it
  // is meaningless.  What matters on the first frame a person ever sees is
  // how many milliseconds it takes, and four times the budget is still a
  // quarter of a 60 fps frame.  This caught the real defect at 52 ms and
  // would catch it again.
  const earlyMax = Math.max(...early.map((e) => parseFloat(e.slice(e.indexOf(' ') + 1))));
  ok(earlyMax <= 4 * pool.rebuildBudgetMs,
     'frames 0-3 are not a stall',
     `worst of the first four ${earlyMax.toFixed(2)} ms vs p50 ${pct(0.5).toFixed(2)} ms`);

  const far = Math.hypot(pos[0] - homeRec.t[0], pos[1] - homeRec.t[1], pos[2] - homeRec.t[2]);
  ok(!!field.get(homeKey) && field.size > 0,
     'the pinned home cell survives being left behind',
     `cell ${homeKey}, now ${far.toFixed(0)} units away (evict radius `
     + `${(field.genRadius * field.evictFactor).toFixed(0)})`);
  field.unpin(homeKey);
  field.ci = NaN;                       // force the next update to re-enumerate
  field.update(pos);
  ok(!field.get(homeKey), 'and is evicted once unpinned', 'gone');
}

// ------------------------------------------------------------------ 5
console.log('\n5. motion is safe for the whole flight, not just at t=0\n');

{
  // Stage 4 could check separation once per placement, because a placement
  // never changed. Objects now drift, breathe and turn, so "no two spheres
  // intersect" stopped being a fact about a field and became a claim about
  // every instant of it.
  //
  // The argument says this cannot fail: each object's envelope -- every
  // position and size it can ever take -- is the same sphere the static field
  // already fitted inside its own cell, because the drift amplitude is drawn
  // from what is LEFT after the base offset and the size breathes downward
  // from the cap. This samples it anyway. An argument that has not been
  // checked against the running code is a comment.
  const world = new FieldWorld({ morphing: false, seed: 5 });
  const L = world.layout.L;
  const buckets = new Map();
  const bk = (a, b, c) => `${a},${b},${c}`;
  let worst = Infinity, worstAt = '', violations = 0, samples = 0, pairs = 0;
  let maxDrift = 0;
  const t0 = new Map();
  for (const [k, it] of world.items) t0.set(k, it.t.slice());

  const STEP = 0.5, SPAN = 240;          // four minutes, half-second samples
  for (let time = 0; time <= SPAN; time += STEP) {
    world.update(STEP, null);
    samples++;
    buckets.clear();
    for (const it of world.items.values()) {
      const key = bk(Math.floor(it.t[0] / L), Math.floor(it.t[1] / L), Math.floor(it.t[2] / L));
      let b = buckets.get(key);
      if (!b) { b = []; buckets.set(key, b); }
      b.push(it);
      const s = t0.get(it.cell.key);
      if (s) {
        const d = Math.hypot(it.t[0] - s[0], it.t[1] - s[1], it.t[2] - s[2]);
        if (d > maxDrift) maxDrift = d;
      }
    }
    for (const it of world.items.values()) {
      const ax = Math.floor(it.t[0] / L), ay = Math.floor(it.t[1] / L), az = Math.floor(it.t[2] / L);
      for (let dx = -1; dx <= 1; dx++)
        for (let dy = -1; dy <= 1; dy++)
          for (let dz = -1; dz <= 1; dz++) {
            const nb = buckets.get(bk(ax + dx, ay + dy, az + dz));
            if (!nb) continue;
            for (const q of nb) {
              if (q === it || q.cell.key <= it.cell.key) continue;
              pairs++;
              const d = Math.hypot(q.t[0] - it.t[0], q.t[1] - it.t[1], q.t[2] - it.t[2]);
              const m = d - it.reach - q.reach;
              if (m < worst) { worst = m; worstAt = `${it.cell.key} / ${q.cell.key} at t=${time}s`; }
              if (m < 0) violations++;
            }
          }
    }
  }
  ok(violations === 0, 'no two objects intersect at ANY sampled instant',
     `${samples} instants over ${SPAN}s, ${pairs.toLocaleString()} pair-tests, `
     + `${violations} intersecting`);
  ok(maxDrift > 0.1, 'and the objects really were moving',
     `largest drift from the t=0 placement ${maxDrift.toFixed(3)} world units`);
  console.log(`  worst margin over the whole run ${worst.toFixed(6)} at ${worstAt}`);
}

console.log('\n6. what REACH_MAX actually bounds\n');

{
  // REACH_MAX comes from |C| + R, and C and R come from invariantMoments,
  // which works from the maps alone and never sees the per-node jitter
  // rotation the traversal applies (adaptive.js:130).  So "the object fits in
  // its reach" is an assumption, not a theorem, and check 2 would pass just as
  // happily while measuring the wrong sphere.  Walk the real jittered child
  // maps and measure three nested hulls, because they are three different
  // numbers and only one of them is the guarantee:
  //
  //   support    max over the DEEPEST level of |A C + b| + kap R.  kap falls
  //              off geometrically, so this converges from above on the
  //              attractor's actual support.  THIS is the separation
  //              guarantee: if it is <= reach, no two objects' geometry can
  //              ever overlap, whatever the jitter draws.
  //   cull ball  the same quantity over EVERY level, which is what visit()
  //              culls and sorts with.  Shallow balls are loose -- a depth-1
  //              ball is kap*R with kap near 1 -- so this one can poke outside
  //              reach without any geometry doing so.
  //   splat hull  blob centres + 3 sigma and rod positions + 3 sigma: what is
  //              actually rasterised.  A Gaussian has no edge, and att.R is a
  //              bound on the SUPPORT, not on a 3-sigma tail, so this is the
  //              largest of the three and always will be.
  const DEPTH = 8;
  const tmp = new Float64Array(9), A2 = new Float64Array(9);
  let wSup = 0, wSupAt = '', wBall = 0, wBallAt = '', wHull = 0, wHullAt = '';
  let nodes = 0;
  for (const name of SET_NAMES) {
    for (const seed of [3, 101, 370]) {
      const b = build(SETS[name]);
      const att = new Attractor({ maps: b.maps, trunkA: b.trunkA, sig0: b.sig0, mu0: b.mu0,
                                  omega: SETS[name].omega ?? 0.28, jitter: 0.16, seed });
      const reach = Math.hypot(att.C[0], att.C[1], att.C[2]) + att.R;
      const C = att.C, tA = att.trunkA, mu = att.mu0;
      const tag = `${name} seed ${seed}`;
      const walk = (A, bb, h, d, kap) => {
        nodes++;
        const bx = A[0] * C[0] + A[1] * C[1] + A[2] * C[2] + bb[0];
        const by = A[3] * C[0] + A[4] * C[1] + A[5] * C[2] + bb[1];
        const bz = A[6] * C[0] + A[7] * C[1] + A[8] * C[2] + bb[2];
        const dc = Math.hypot(bx, by, bz);
        const ball = (dc + kap * att.R) / reach;
        if (ball > wBall) { wBall = ball; wBallAt = `${tag} depth ${DEPTH - d}`; }
        if (d === 0 && ball > wSup) { wSup = ball; wSupAt = tag; }
        let hull = (dc + 3 * kap * att.qMax) / reach;
        const rx = A[2] * tA[8] * mu[2] + bb[0];
        const ry = A[5] * tA[8] * mu[2] + bb[1];
        const rz = A[8] * tA[8] * mu[2] + bb[2];
        hull = Math.max(hull, (Math.hypot(rx, ry, rz) + 3 * kap * att.rodMax) / reach);
        if (hull > wHull) { wHull = hull; wHullAt = tag; }
        if (d === 0) return;
        for (let i = 0; i < att.m; i++) {
          att.childLinear(tmp, h, i);
          for (let p = 0; p < 3; p++)
            for (let q = 0; q < 3; q++)
              A2[p * 3 + q] = A[p * 3] * tmp[q] + A[p * 3 + 1] * tmp[3 + q]
                            + A[p * 3 + 2] * tmp[6 + q];
          const mb = att.maps[i].b;
          walk(Float64Array.from(A2),
               [A[0] * mb[0] + A[1] * mb[1] + A[2] * mb[2] + bb[0],
                A[3] * mb[0] + A[4] * mb[1] + A[5] * mb[2] + bb[1],
                A[6] * mb[0] + A[7] * mb[1] + A[8] * mb[2] + bb[2]],
               att.mixHash(h, i), d - 1, kap * att.sArr[i]);
        }
      };
      walk(new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), [0, 0, 0], att.rootHash, DEPTH, 1);
    }
  }
  console.log(`  ${nodes} jittered nodes, ${SET_NAMES.length} sets x 3 seeds, depth ${DEPTH}`);
  ok(wSup <= 1, 'the support is inside reach -- objects cannot overlap',
     `${wSup.toFixed(5)} of reach (${wSupAt}), ${((1 - wSup) * 100).toFixed(2)}% to spare`);
  // The cull ball is loose at shallow depth and the excess does not grow with
  // depth (measured flat from depth 1 to depth 11), so it is the bound being
  // conservative, not geometry escaping.  Tripwire, not guarantee.
  ok(wBall <= 1.02, 'the per-node cull ball is within 2% of reach',
     `${wBall.toFixed(5)} of reach (${wBallAt})`);
  // The 3-sigma blob hull is 3*qMax where R is about 2.7*qMax (adaptive.js:198),
  // so it is ~11% over by construction and this is true of stage 3's ring
  // layout too, not something the field introduced.  At 3 sigma a blob carries
  // exp(-4.5) = 1.1% of its opacity, and visit() already clips those tails at
  // the frustum edge.  The tripwire is here so a future map set whose Gaussian
  // stand-in is wildly bigger than its own support is noticed.
  ok(wHull <= 1.30, 'the 3-sigma splat hull is within 30% of reach',
     `${wHull.toFixed(5)} of reach (${wHullAt})`);
  console.log('  (only the first of those three is the separation guarantee; the other\n'
    + '   two are the splat approximation being fuzzy at the edge, and they are the\n'
    + '   reason the worst-case margin in check 2 is worth keeping comfortable)');
}

console.log(bad ? `\n${bad} checks FAILED` : '\nthe field is sound');
process.exit(bad ? 1 : 0);
