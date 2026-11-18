// Does re-rooting actually express the same world, just in another frame?
//
// Take a point with a known address, compute where it lands two ways:
//   (a) straight down from the true root, in root coordinates, then map it
//       into the camera's frame with rootFrame();
//   (b) the camera's frame directly.
// They must agree, and the camera's own position must come back to where it
// started after a round trip up and down the path.

import { Attractor, FlyCamera } from '../js/adaptive.js';

const att = new Attractor({});
const cam = new FlyCamera(att);
// Both errors are measured in LOCAL units and then multiplied by the local
// scale, which puts them back into world units.  That is the only way to state
// the claim depth-independently: an absolute threshold in local units is a
// different test at every depth, because a local unit at depth 52 is 3e-12 of
// a world unit.
let worstRound = 0, worstMap = 0, atRound = 1, atMap = 1;

for (let step = 0; step < 400; step++) {
  cam.dive(0.05, 0.15);
  cam.reroot();
  if (cam.depth < 2) continue;

  // (1) the camera's position, taken up to the true root and back down
  let p = cam.pos.slice();
  const AE = new Float64Array(9);
  for (let k = cam.path.length - 1; k >= 0; k--) {
    const i = cam.path[k], b = att.maps[i].b;
    att.childLinear(AE, cam.hashes[k], i);
    p = [AE[0] * p[0] + AE[1] * p[1] + AE[2] * p[2] + b[0],
         AE[3] * p[0] + AE[4] * p[1] + AE[5] * p[2] + b[1],
         AE[6] * p[0] + AE[7] * p[1] + AE[8] * p[2] + b[2]];
  }
  const world = p.slice();
  for (let k = 0; k < cam.path.length; k++) {
    const i = cam.path[k], b = att.maps[i].b, s = att.s[i], inv = 1 / (s * s);
    att.childLinear(AE, cam.hashes[k], i);
    const q = [p[0] - b[0], p[1] - b[1], p[2] - b[2]];
    p = [(AE[0] * q[0] + AE[3] * q[1] + AE[6] * q[2]) * inv,
         (AE[1] * q[0] + AE[4] * q[1] + AE[7] * q[2]) * inv,
         (AE[2] * q[0] + AE[5] * q[1] + AE[8] * q[2]) * inv];
  }
  const round = Math.hypot(p[0] - cam.pos[0], p[1] - cam.pos[1], p[2] - cam.pos[2]);
  if (round > worstRound) { worstRound = round; atRound = cam.scale; }

  // (2) rootFrame() must take that world point back to the camera frame
  const rf = cam.rootFrame({});
  const m = [rf.A[0] * world[0] + rf.A[1] * world[1] + rf.A[2] * world[2] + rf.b[0],
             rf.A[3] * world[0] + rf.A[4] * world[1] + rf.A[5] * world[2] + rf.b[1],
             rf.A[6] * world[0] + rf.A[7] * world[1] + rf.A[8] * world[2] + rf.b[2]];
  const err = Math.hypot(m[0] - cam.pos[0], m[1] - cam.pos[1], m[2] - cam.pos[2]);
  if (err > worstMap) { worstMap = err; atMap = cam.scale; }
}

console.log(`depth reached ${cam.depth}, local scale ${cam.scale.toExponential(2)}`);
const roundW = worstRound * atRound, mapW = worstMap * atMap;
console.log(`  up-and-back-down round trip   ${worstRound.toExponential(2)} local`
  + `  = ${roundW.toExponential(2)} world`);
console.log(`  rootFrame() world -> camera   ${worstMap.toExponential(2)} local`
  + `  = ${mapW.toExponential(2)} world`);
console.log('\n  Both are float64 epsilon in WORLD units, which is the whole argument for');
console.log('  re-rooting: a round trip through world coordinates is exact to one ulp of a');
console.log('  world unit, and at this depth one ulp of a world unit is a ten-thousandth of');
console.log('  a local one. The camera keeps an address instead of a position precisely so');
console.log('  that it never has to make that round trip.');
// And the depth has to be asserted, not merely printed.
//
// Both numbers above are ERRORS, so they are smallest when nothing happens:
// this test passed for some time while reporting "depth reached 0", because
// the dive it uses to get deep had started flying out of the attractor
// instead of into it -- a camera facing away from its target closed a
// fraction of a distance its own motion was increasing, and accelerated into
// the void (fixed in FlyCamera.dive, which now scales the step by alignment
// the way VoidCamera.cruise always has). A precision claim measured at depth
// 0 is not a precision claim, and nothing here noticed.
const deep = cam.depth >= 12;
if (!deep) console.log(`\n  the dive never got deep -- depth ${cam.depth}`);
const ok = roundW < 1e-13 && mapW < 1e-13 && deep;
console.log(ok ? 'REROOT OK' : 'REROOT FAILED');
process.exit(ok ? 0 : 1);
