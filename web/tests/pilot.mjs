// What the dive aims at, and what banking does to the basis.
//
// Everything here is on a bare FlyCamera over one attractor, so nothing
// depends on the field, the pool or the renderer.
//
// What this test does NOT try to measure: whether the flight looks better.
// The first version of it compared "how deep did 1200 dive steps get" with
// the aim point on and off and reported a 23-against-4 win, which was noise:
// a 1e-6 relative change in speed moves that same number from 23 to 27, and
// over an ensemble of ten map sets and six seeds the two settings are within
// each other's spread. The dive's child choice is a discrete decision inside
// a self-similar structure, so any single-run outcome is chaotic and a
// difference in one is evidence of nothing. The claim that the dive now shows
// more structure is measured where it is visible -- the washed-out rate over
// a real flight, in tests/legible.mjs. What is checked here is the mechanism:
// that the aim point is where the geometry says it should be, that the choice
// does lean outward where the children differ, and that banking leaves the
// basis intact.
//
//   node tests/pilot.mjs [steps]

import { Attractor, FlyCamera, PALETTES } from '../js/adaptive.js';
import { SETS, build } from '../js/maps.js';

const STEPS = Number(process.argv[2] ?? 900);
let fail = 0;
const ok = (good, msg, detail = '') => {
  console.log((good ? '  ok    ' : '  FAIL  ') + msg + (detail ? '   ' + detail : ''));
  if (!good) fail++;
};

function attractor(name = null, seed = 3) {
  if (!name) return new Attractor({ seed, jitter: 0.16 });
  const sp = SETS[name], b = build(sp);
  return new Attractor({ maps: b.maps, trunkA: b.trunkA, sig0: b.sig0, mu0: b.mu0,
    omega: sp.omega ?? 0.28, seed, jitter: 0.16,
    palette: PALETTES[sp.palette] || PALETTES.spectral });
}

function aimed(cam) {
  const C = cam.att.C;
  cam.f = [C[0] - cam.pos[0], C[1] - cam.pos[1], C[2] - cam.pos[2]];
  cam.u = [0, 0, 1];
  cam.ortho();
  return cam;
}

/**
 * Fly, and report where it chose to go -- the outwardness of the child it
 * picked, recomputed here from the maps rather than read out of the camera's
 * own scores, so what is checked is the choice and not the arithmetic that
 * made it.  1.0 is the outermost child of the node, 0 one sitting on the
 * node's own centre of mass.
 */
function fly({ att, edgeBias, edgeAim, steps = STEPS, ticks = 1, speed = 0.05 }) {
  const cam = aimed(new FlyCamera(att));
  cam.edgeBias = edgeBias; cam.edgeAim = edgeAim;
  const AE = new Float64Array(9);
  let outSum = 0, outN = 0, maxDepth = 0, orthoWorst = 0, rollWorst = 0;
  let aimWorst = 0, aimMargin = 0, aimBack = 0, aimN = 0;
  for (let i = 0; i < steps; i++) {
    const h = cam.hashes[cam.hashes.length - 1], a0 = cam.att, C0 = a0.C;
    const p0 = cam.pos.slice();
    cam.dive(speed * ticks, 0.14, ticks);
    const pick = cam._aimv ? cam._aimv[3] : -1;
    if (cam._diveHash === h && pick >= 0) {
      let mx = 0, chosen = 0, ox = 0, oy = 0, oz = 0, cx = 0, cy = 0, cz = 0;
      for (let k = 0; k < a0.m; k++) {
        a0.childLinear(AE, h, k);
        const b = a0.maps[k].b;
        const lx = AE[0] * C0[0] + AE[1] * C0[1] + AE[2] * C0[2] + b[0];
        const ly = AE[3] * C0[0] + AE[4] * C0[1] + AE[5] * C0[2] + b[1];
        const lz = AE[6] * C0[0] + AE[7] * C0[1] + AE[8] * C0[2] + b[2];
        const o = Math.hypot(lx - C0[0], ly - C0[1], lz - C0[2]);
        if (o > mx) mx = o;
        if (k === pick) {
          chosen = o;
          ox = lx - C0[0]; oy = ly - C0[1]; oz = lz - C0[2];
          cx = lx; cy = ly; cz = lz;
        }
      }
      if (mx > 1e-12) { outSum += chosen / mx; outN++; }
      // The aim point, in the same local frame: camera position plus the
      // camera-relative aim the dive recorded.
      const ax = p0[0] + cam._aimv[0], ay = p0[1] + cam._aimv[1],
            az = p0[2] + cam._aimv[2];
      const off = Math.hypot(ax - cx, ay - cy, az - cz);
      const rad = a0.s[pick] * a0.R;
      // how far the aim sits off the child's centre, as a fraction of the
      // child's own radius, and whether it is on the outward side
      if (chosen > 1e-12) {
        const dot = ((ax - cx) * ox + (ay - cy) * oy + (az - cz) * oz) / chosen;
        if (dot < aimBack) aimBack = dot;
        const frac = off / rad;
        if (frac > aimWorst) aimWorst = frac;
        aimMargin += frac; aimN++;
      }
    }
    cam.reroot();
    if (cam.depth > maxDepth) maxDepth = cam.depth;
    if (Math.abs(cam._roll) > rollWorst) rollWorst = Math.abs(cam._roll);
    const dev = [
      cam.r[0] * cam.u[0] + cam.r[1] * cam.u[1] + cam.r[2] * cam.u[2],
      cam.r[0] * cam.f[0] + cam.r[1] * cam.f[1] + cam.r[2] * cam.f[2],
      cam.u[0] * cam.f[0] + cam.u[1] * cam.f[1] + cam.u[2] * cam.f[2],
      Math.hypot(cam.r[0], cam.r[1], cam.r[2]) - 1,
      Math.hypot(cam.u[0], cam.u[1], cam.u[2]) - 1,
      Math.hypot(cam.f[0], cam.f[1], cam.f[2]) - 1,
    ].map(Math.abs);
    const w = Math.max(...dev);
    if (w > orthoWorst) orthoWorst = w;
  }
  return { cam, out: outN ? outSum / outN : NaN, maxDepth, orthoWorst, rollWorst,
           aimWorst, aimMean: aimN ? aimMargin / aimN : NaN, aimBack };
}

// ---------------------------------------------------------------- the aim point
//
// The one thing about the aim point that is not a matter of taste: it must
// stay inside `descendMargin`. The descent test is distance from the child's
// CENTRE under s_i * R * descendMargin, so an aim point further out than that
// is a camera that circles the shell and never gets in -- the infinite zoom,
// quietly switched off.
const tree = attractor();
const edge = fly({ att: tree, edgeBias: 0.55, edgeAim: 0.52 });
const centre = fly({ att: tree, edgeBias: 0, edgeAim: 0 });

console.log(`${STEPS} dive steps, default tree set\n`);
console.log(`  aim offset from the child's centre, in child radii:`
  + ` mean ${edge.aimMean.toFixed(3)}, worst ${edge.aimWorst.toFixed(3)}`);
console.log(`  with the offset switched off:`
  + ` mean ${centre.aimMean.toFixed(3)}, worst ${centre.aimWorst.toFixed(3)}`);
console.log('');
ok(Math.abs(edge.aimMean - 0.52) < 1e-9 && Math.abs(edge.aimWorst - 0.52) < 1e-9,
   'the aim point sits edgeAim of a child radius off its centre',
   `${edge.aimMean.toFixed(6)} radii`);
ok(edge.aimWorst < tree.R && edge.aimWorst < edge.cam.descendMargin,
   'and inside descendMargin, so the dive can still get in',
   `${edge.aimWorst.toFixed(3)} < ${edge.cam.descendMargin}`);
ok(edge.aimBack >= -1e-12,
   'the offset is outward, never back toward the node centre',
   `worst inward component ${edge.aimBack.toExponential(1)}`);
ok(centre.aimWorst < 1e-12,
   'edgeAim = 0 reproduces the old aim-at-the-centre dive exactly');

// ------------------------------------------------------------ the child choice
//
// The outward bias only bites where the children are at different distances
// from the node's centre of mass. In most of the shipped sets they are not --
// a Sierpinski tetrahedron's four children are symmetric and every one is the
// outermost -- so the bias changes nothing and must not pretend to. `lattice`
// is the one that has an inner child competitive with the outer ones, and it
// is the case the bias exists for: an inner child is a dive into the middle
// of the mass.
console.log('');
for (const name of ['lattice', 'sierpinski']) {
  const att = attractor(name);
  const a = fly({ att, edgeBias: 0, edgeAim: 0.52 });
  const b = fly({ att, edgeBias: 0.55, edgeAim: 0.52 });
  console.log(`  ${name.padEnd(11)} outwardness of the child chosen:`
    + ` ${a.out.toFixed(3)} without the bias, ${b.out.toFixed(3)} with it`);
  if (name === 'lattice')
    ok(b.out > a.out + 0.1, 'the bias moves the choice outward where it can',
       `${a.out.toFixed(3)} -> ${b.out.toFixed(3)}`);
  else
    ok(Math.abs(b.out - a.out) < 1e-9 && a.out > 0.99,
       'and changes nothing where every child is already outermost');
}

// ----------------------------------------------------------------- the basis
console.log('');
ok(edge.orthoWorst < 1e-9, 'the basis stays orthonormal while banking',
   `worst deviation ${edge.orthoWorst.toExponential(1)}`);
ok(edge.rollWorst > 0.02, 'the camera does bank',
   `worst lean ${(edge.rollWorst * 57.2958).toFixed(1)} deg`);
ok(edge.rollWorst <= edge.cam.bankMax + 1e-9, 'and never past its cap',
   `cap ${(edge.cam.bankMax * 57.2958).toFixed(1)} deg`);
{
  const cam = new FlyCamera(tree);
  cam._roll = 0.4;                        // as if it had just come out of a turn
  for (let i = 0; i < 300; i++) cam.bank(0, 1 / 60);
  ok(Math.abs(cam._roll) < 0.01, 'the lean unwinds when the turning stops',
     `${(cam._roll * 57.2958).toFixed(2)} deg left after 5 s`);
}

// -------------------------------------------------------------- the frame rate
//
// Two things used to make 30fps a different flight from 60fps: the steering
// follower integrated 1/60 s per call however long the frame actually was,
// and the forward step closed a bare FRACTION of the remaining distance,
// which does not compose (0.95^2 = 0.9025, not 0.90). Both are fixed.
//
// Checked over half a second, and that is not hedging -- it is the horizon
// over which the question is answerable. The chaos probe below is the reason:
// perturb the speed by one part in a million and the heading is 60 degrees
// away three seconds later. Two clients at different frame rates will always
// end up in different parts of the attractor; what has to match is the near
// term, which is what a viewer would call "the same flight on a slower
// machine".
console.log('');
{
  const a = fly({ att: tree, edgeBias: 0.55, edgeAim: 0.52, steps: 30, ticks: 1 });
  const b = fly({ att: tree, edgeBias: 0.55, edgeAim: 0.52, steps: 15, ticks: 2 });
  const dot = a.cam.f[0] * b.cam.f[0] + a.cam.f[1] * b.cam.f[1] + a.cam.f[2] * b.cam.f[2];
  const ang = Math.acos(Math.max(-1, Math.min(1, dot))) * 57.2958;
  const gap = Math.hypot(a.cam.pos[0] - b.cam.pos[0], a.cam.pos[1] - b.cam.pos[1],
                         a.cam.pos[2] - b.cam.pos[2]) / tree.R;
  console.log(`  0.5 s at 60fps against 30fps: headings ${ang.toFixed(2)} deg apart,`
    + ` positions ${(gap * 100).toFixed(2)}% of a radius apart`);
  ok(ang < 2.0, 'half the frame rate flies the same flight', `${ang.toFixed(2)} deg`);
  ok(gap < 0.02, 'and arrives in the same place', `${(gap * 100).toFixed(2)}% of R`);
}
{
  const a = fly({ att: tree, edgeBias: 0.55, edgeAim: 0.52, steps: 180, speed: 0.05 });
  const b = fly({ att: tree, edgeBias: 0.55, edgeAim: 0.52, steps: 180,
                  speed: 0.05 * (1 + 1e-6) });
  const dot = a.cam.f[0] * b.cam.f[0] + a.cam.f[1] * b.cam.f[1] + a.cam.f[2] * b.cam.f[2];
  const ang = Math.acos(Math.max(-1, Math.min(1, dot))) * 57.2958;
  console.log(`  chaos probe: one part in a million of speed, 3 s later the`
    + ` heading is ${ang.toFixed(1)} deg away (depth ${a.maxDepth} against ${b.maxDepth})`);
  console.log('  -- which is why no outcome of a single dive is evidence of anything.');
}

console.log(fail ? `\n  ${fail} FAILED` : '\n  pilot ok');
process.exit(fail ? 1 : 0);
