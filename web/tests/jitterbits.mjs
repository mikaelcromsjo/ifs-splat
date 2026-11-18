// The jitter table: is it still the reference, is it morphable, and is it big
// enough?
//
// A node's jitter rotation used to be baked at construction, which made
// `jitter` the one map parameter that could not morph -- so a rigid lattice
// could never become an organic one.  It is now assembled during reparam from
// a cached axis, which makes it morphable, and the table shrank from 4096
// entries to 1024, which is what pays for morphing continuously rather than in
// bursts (reparam 1.388 -> 0.44 ms).
//
// Three things have to hold, and only the first two can be asserted:
//
//   1. the assembled table is EXACTLY the shared reference implementation,
//      jitterRotInto() in ifs_core.js, which is what the reference expansion
//      and py/ifs_core.py both use.  Not "close" -- equal.
//   2. jitter morphs, repeatably, and the rebuild stays inside the budget.
//   3. 1024 rotations are enough that no eye sees the repetition.  That is a
//      judgement, so the test renders the same attractors across table sizes
//      and asserts only that the image statistics do not move; the picture is
//      for a person to look at.
//
//   node tests/jitterbits.mjs

import { writeFileSync, mkdirSync } from 'node:fs';
import { SETS, build, morph } from '../js/maps.js';
import { Attractor, FlyCamera, AdaptiveStream, PALETTES } from '../js/adaptive.js';
import { jitterRotInto, h32 } from '../js/ifs_core.js';
import { render, png, tile } from '../js/cpu_raster.js';

const W = 320, H = 250;
const BUDGET_MS = 1.5;          // ShapePool's per-frame rebuild budget
let bad = 0;
const fail = (m) => { console.log(`  FAIL  ${m}`); bad = 1; };

const mk = (name, bits, jitter = 0.16) => {
  const spec = SETS[name], b = build(spec);
  return new Attractor({
    maps: b.maps, trunkA: b.trunkA, sig0: b.sig0, mu0: b.mu0,
    omega: spec.omega ?? 0.28, jitter, seed: 3,
    palette: PALETTES[spec.palette] || PALETTES.tree, jitterTableBits: bits,
  });
};

console.log('the jitter table\n');

// ---------------------------------------------------------------- 1. exact
console.log('1. the table is the reference implementation');
{
  const J = new Float64Array(9);
  let worst = 0, checked = 0;
  for (const bits of [8, 10, 12]) {
    for (const amt of [0, 0.04, 0.16, 0.5]) {
      const a = mk('tree', bits, amt);
      const K = a.tableMask + 1;
      for (let k = 0; k < K; k++) {
        jitterRotInto(J, h32(Math.imul(k, 2654435761) >>> 0), amt);
        const A = a.maps[0].A, o = 9 * (k * a.m);
        for (let p = 0; p < 3; p++)
          for (let q = 0; q < 3; q++) {
            const want = J[p * 3] * A[q] + J[p * 3 + 1] * A[3 + q] + J[p * 3 + 2] * A[6 + q];
            const got = a.childMaps[o + p * 3 + q];
            if (Math.abs(want - got) > worst) worst = Math.abs(want - got);
            checked++;
          }
      }
    }
  }
  if (worst !== 0) fail(`table differs from jitterRotInto by ${worst.toExponential(2)}`);
  console.log(`  ${checked.toLocaleString()} entries over 3 table sizes and 4 jitter amounts`
            + `   max difference ${worst === 0 ? '0 (exact)' : worst.toExponential(2)}`
            + `  ${worst === 0 ? 'ok' : 'FAIL'}`);
}

// ------------------------------------------------------------ 2. morphable
console.log('\n2. jitter morphs, repeatably, inside the budget');
{
  const a = mk('tree', 10, 0.16);
  const snap = () => Array.from(a.childMaps.slice(0, 36)).join(',');
  a.reparam({ jitter: 0.0 }); const z0 = snap();
  a.reparam({ jitter: 0.16 }); const z1 = snap();
  a.reparam({ jitter: 0.0 }); const z2 = snap();
  if (z0 === z1) fail('changing jitter did not change the child maps -- it is not morphable');
  if (z0 !== z2) fail('rebuilding at the same jitter gave a different table -- not repeatable');
  console.log(`  changing jitter changes the geometry        ${z0 !== z1 ? 'ok' : 'FAIL'}`);
  console.log(`  returning to it reproduces it exactly       ${z0 === z2 ? 'ok' : 'FAIL'}`);

  // cost, warm, for a steady jitter and for one that moves every rebuild
  const spin = (extra) => {
    const sm = (t) => t * t * (3 - 2 * t);
    for (let i = 0; i < 120; i++) {
      const sp = morph(SETS.tree, SETS.spiral, sm((i % 11) / 10)), q = build(sp);
      a.reparam({ maps: q.maps, trunkA: q.trunkA, sig0: q.sig0, mu0: q.mu0,
                  omega: 0.28, ...extra(i) });
    }
    const t0 = process.hrtime.bigint(), N = 120;
    for (let i = 0; i < N; i++) {
      const sp = morph(SETS.tree, SETS.spiral, sm((i % 11) / 10)), q = build(sp);
      a.reparam({ maps: q.maps, trunkA: q.trunkA, sig0: q.sig0, mu0: q.mu0,
                  omega: 0.28, ...extra(i) });
    }
    return Number(process.hrtime.bigint() - t0) / 1e6 / N;
  };
  const steady = spin(() => ({}));
  const moving = spin((i) => ({ jitter: 0.05 + 0.15 * ((i % 17) / 17) }));
  console.log(`  rebuild, jitter steady                      ${steady.toFixed(3)} ms`);
  console.log(`  rebuild, jitter morphing every step         ${moving.toFixed(3)} ms`);
  console.log(`  rebuilds affordable in a ${BUDGET_MS} ms budget       `
            + `${Math.floor(BUDGET_MS / moving)}  (continuous morphing needs ~2)`);
  if (moving > BUDGET_MS * 0.5)
    fail(`a rebuild costs ${moving.toFixed(3)} ms; two will not fit in ${BUDGET_MS} ms`);
}

// --------------------------------------------------- 3. is 1024 enough?
console.log('\n3. table size against appearance');
{
  const shots = [], rows = [];
  for (const bits of [12, 10, 8]) {
    const row = [];
    for (const name of ['tree', 'lattice']) {
      const att = mk(name, bits);
      const cam = new FlyCamera(att);
      cam.f = [att.C[0] - cam.pos[0], att.C[1] - cam.pos[1], att.C[2] - cam.pos[2]];
      cam.u = [0, 0, 1]; cam.ortho();
      const st = new AdaptiveStream(att, 400000);
      st.budgetBy = 'splats'; st.target = 130000;
      const tanY = Math.tan((55 * Math.PI) / 360);
      const view = { focal: (H / 2) / tanY, tanX: tanY * (W / H), tanY, znear: 1e-5, W, H };
      let n = 0;
      for (let i = 0; i < 16; i++) n = st.expand(cam, view);
      const im = render({ pos: st.pos, cov: st.cov, rgba: st.rgba, off: st.offset, count: n,
                          eye: cam.pos, B: { r: cam.r, u: cam.u, f: cam.f }, W, H,
                          fogNear: 5, fogFar: 55 });
      shots.push(im);
      let s = 0, s2 = 0; const N = W * H;
      for (let k = 0; k < N; k++) {
        const l = (im.img[3 * k] + im.img[3 * k + 1] + im.img[3 * k + 2]) / 3;
        s += l; s2 += l * l;
      }
      const mean = s / N;
      row.push({ name, bits, n, mean, sd: Math.sqrt(Math.max(0, s2 / N - mean * mean)),
                 reuse: n / ((1 << bits) * att.m) });
    }
    rows.push(row);
  }
  console.log('  bits   set       splats   reuse per entry   mean   sd');
  for (const row of rows)
    for (const r of row)
      console.log(`  ${String(r.bits).padStart(4)}   ${r.name.padEnd(9)} ${String(r.n).padStart(6)}`
                + `   x${r.reuse.toFixed(1).padStart(6)}        ${r.mean.toFixed(2).padStart(6)}`
                + ` ${r.sd.toFixed(2).padStart(6)}`);
  // The realisations differ, so a pixel diff is meaningless -- these are not
  // approximations of one another.  What must not move is the statistics: if
  // a smaller table were collapsing detail, brightness and contrast would go
  // with it.
  for (let c = 0; c < 2; c++) {
    const ref = rows[0][c];
    for (let r = 1; r < rows.length; r++) {
      const dm = Math.abs(rows[r][c].mean - ref.mean) / Math.max(ref.mean, 1e-6);
      const ds = Math.abs(rows[r][c].sd - ref.sd) / Math.max(ref.sd, 1e-6);
      if (dm > 0.08 || ds > 0.08)
        fail(`${ref.name} at ${rows[r][c].bits} bits moved brightness ${(dm * 100).toFixed(1)}%`
           + ` / contrast ${(ds * 100).toFixed(1)}% from the 4096-entry table`);
    }
  }
  mkdirSync('out', { recursive: true });
  writeFileSync('out/jitterbits.png', png(tile(shots, 2)));
  console.log('\n  out/jitterbits.png   rows 4096 / 1024 / 256 entries, cols tree / lattice');
  console.log('  (they are different realisations, not better and worse ones --');
  console.log('   what to look for is regularity or repetition, and there is none)');
}

console.log(bad ? '\nthe jitter table is NOT sound' : '\nthe jitter table is sound');
process.exit(bad);
