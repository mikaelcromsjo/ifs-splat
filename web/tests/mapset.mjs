// Validate a map set: does it actually define a bounded attractor that the
// LOD machinery can use?
//
//   contraction  every map must contract, or the "attractor" is unbounded
//   bound        the Hutchinson radius must converge, and not be enormous
//   moments      the mean/covariance fixed point must converge
//   shape        how the mass is distributed -- this is what tells the sets
//                apart: a flat one, a dusty one, a lattice
//   cost         nodes and splats to fill a screen, which is the real budget

import { SETS, SET_NAMES, build, morph, morphPath } from '../js/maps.js';
import { Attractor, FlyCamera, AdaptiveStream } from '../js/adaptive.js';
import { opNorm } from '../js/moments.js';

const W = 1600, H = 900, FOV = 55;
const focal = (H / 2) / Math.tan((FOV * Math.PI) / 360);
const tanY = Math.tan((FOV * Math.PI) / 360), tanX = tanY * (W / H);
const view = { focal, tanX, tanY, znear: 1e-4, W, H };

/**
 * `cost` renders a frame to measure what the set costs to look at.  The sweeps
 * below do not need it: whether a spec is VALID is a question about its
 * contraction, its Hutchinson radius and its moment fixed point, all of which
 * are already computed by the time the Attractor is constructed.  Rendering
 * for each of them turned a gate into an eleven-minute job -- 650 specs each
 * expanding 41 frames -- which is a gate nobody runs.
 */
export function check(spec, verbose = true, cost = true) {
  const b = build(spec);
  const s = b.maps.map((m) => opNorm(m.A));
  const worst = Math.max(...s);
  const att = new Attractor({
    maps: b.maps, trunkA: b.trunkA, sig0: b.sig0, mu0: b.mu0,
    omega: spec.omega ?? 0.28, jitter: spec.jitter ?? 0.16, seed: 3,
  });
  // eigenvalues of Q say what shape the mass has
  const Q = att.Q;
  const tr = Q[0] + Q[4] + Q[8];
  const l1 = opNorm(Q);
  const flat = l1 / tr;                 // 1/3 = ball, ->1 = a line
  const ok = worst < 0.999 && isFinite(att.R) && att.R < 1e4 &&
             isFinite(tr) && tr > 1e-12;

  if (!cost) {
    if (verbose) console.log(`  ${spec.name.padEnd(12)} contraction ${worst.toFixed(3)}  `
      + `R ${att.R.toFixed(2).padStart(7)}  ${ok ? 'ok' : 'FAILED'}`);
    return { ok, worst, R: att.R, tr, flat, n: 0, ms: 0 };
  }

  // what does it cost to look at?
  const cam = new FlyCamera(att);
  cam.f = [att.C[0] - cam.pos[0], att.C[1] - cam.pos[1], att.C[2] - cam.pos[2]];
  cam.u = [0, 0, 1];
  cam.ortho();
  const st = new AdaptiveStream(att, 300000);
  st.budgetBy = 'splats'; st.target = 120000;
  for (let i = 0; i < 40; i++) { st.expand(cam, view); cam.dive(0.02, 0.1); cam.reroot(); }
  const t0 = process.hrtime.bigint();
  const n = st.expand(cam, view);
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;

  if (verbose)
    console.log(`  ${spec.name.padEnd(12)} contraction ${worst.toFixed(3)}  `
      + `R ${att.R.toFixed(2).padStart(7)}  spread ${Math.sqrt(tr).toFixed(3)}  `
      + `anisotropy ${flat.toFixed(2)}  ${String(n).padStart(6)} splats `
      + `${ms.toFixed(1)}ms  ${ok ? 'ok' : 'FAILED'}`);
  return { ok, worst, R: att.R, tr, flat, n, ms };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const names = Object.keys(SETS);
  console.log(`checking ${names.length} map sets\n`);
  let bad = 0;
  for (const k of names) if (!check(SETS[k]).ok) bad++;
  // every pair must be morphable, and every point along the way must be valid
  console.log('\nmorphs (11 steps each):');
  for (let i = 0; i < names.length; i++)
    for (let j = i + 1; j < names.length; j++) {
      let worst = 0, fail = 0;
      for (let k = 0; k <= 10; k++) {
        const r = check(morph(SETS[names[i]], SETS[names[j]], k / 10), false, false);
        if (!r.ok) fail++;
        worst = Math.max(worst, r.worst);
      }
      console.log(`  ${names[i]} <-> ${names[j]}: worst contraction `
        + `${worst.toFixed(3)}  ${fail ? `${fail}/11 INVALID` : 'all valid'}`);
      if (fail) bad++;
    }
  // And the closed spline, which is the path the field actually flies.  The
  // pairs above are no longer what runs: Catmull-Rom leaves the hull of its
  // controls, so it reaches configurations no straight morph between two sets
  // ever visits, and those are exactly the ones most likely to be invalid.
  // This is the gate for every set added from here on.
  const SPECS = SET_NAMES.map((n) => SETS[n]);
  const STEPS = 60;
  console.log(`\nthe closed spline (${SET_NAMES.length} segments x ${STEPS} steps):`);
  let sWorst = 0, sFail = 0, sR = 0, sSig = Infinity, sAt = '';
  for (let i = 0; i < SET_NAMES.length; i++) {
    let segWorst = 0, segFail = 0, segR = 0;
    for (let k = 0; k < STEPS; k++) {
      const u = i + k / STEPS;
      const spec = morphPath(SPECS, u);
      const r = check(spec, false, false);
      if (!r.ok) { segFail++; sFail++; }
      segWorst = Math.max(segWorst, r.worst);
      segR = Math.max(segR, r.R);
      sSig = Math.min(sSig, ...spec.sig0);
      if (r.R > sR) { sR = r.R; sAt = `u=${u.toFixed(2)}`; }
    }
    sWorst = Math.max(sWorst, segWorst);
    console.log(`  ${SET_NAMES[i]} -> ${SET_NAMES[(i + 1) % SET_NAMES.length]}:`
      + ` worst contraction ${segWorst.toFixed(3)}  max R ${segR.toFixed(2)}`
      + `  ${segFail ? `${segFail}/${STEPS} INVALID` : 'all valid'}`);
  }
  if (sFail) bad++;
  console.log(`  overall: contraction ${sWorst.toFixed(4)} (must stay under 1),`
    + ` max R ${sR.toFixed(3)} at ${sAt}, min sig0 ${sSig.toExponential(2)}`);
  // sig0 is a variance: splining it linearly drove it to -0.021 on the way
  // down between sets three orders of magnitude apart, which is a degenerate
  // splat rather than a small one.  It is splined in log space now, and this
  // is the assertion that keeps it that way.
  if (!(sSig > 0)) {
    console.log('  FAIL  sig0 went non-positive along the spline');
    bad++;
  }

  console.log(bad ? `\n${bad} problems` : '\nall map sets ok');
  process.exit(bad ? 1 : 0);
}
