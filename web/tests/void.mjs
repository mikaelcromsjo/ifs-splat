// Does the void's coordinate chain actually hold?
//
// Drawing attractor j means pushing its root through
//   j's local -> world -> home's local -> the camera's frame,
// and getting any one of those three backwards puts the object somewhere
// plausible-looking but wrong -- or, as re-rooting did once, at infinity.  So
// check it against the composition done the long way round, in world
// coordinates, at depth 0 and after the camera has descended.

import { writeFileSync, mkdirSync } from 'node:fs';
import { World, VoidCamera, ringLayout } from '../js/world.js';
import { AdaptiveStream, PALETTES } from '../js/adaptive.js';
import { SET_NAMES } from '../js/maps.js';
import { render, png } from '../js/cpu_raster.js';

const W = 520, H = 400;
const focal = (H / 2) / Math.tan((55 * Math.PI) / 360);
const tanY = Math.tan((55 * Math.PI) / 360);
const view = { focal, tanX: tanY * (W / H), tanY, znear: 1e-5, W, H };

const layout = ringLayout(SET_NAMES.length >= 3 ? SET_NAMES
  : ['tree', 'tree', 'tree', 'tree'], { radius: 16, rise: 5 });
const world = new World(layout, { morphing: false });
const cam = new VoidCamera(world, 0);

// look from outside the ring, at its middle
cam.pos = world.get(world.first()).worldCentre();
{
  const h = world.get(world.first());
  const mid = [0, 0, 0];
  const back = [-34, -34, 15];
  cam.pos = [(back[0] - h.t[0]) / h.s, (back[1] - h.t[1]) / h.s, (back[2] - h.t[2]) / h.s];
  cam.f = [mid[0] - back[0], mid[1] - back[1], mid[2] - back[2]];
  cam.u = [0, 0, 1];
  cam.ortho();
}

let bad = 0;
function checkFrames(label) {
  const home = world.get(cam.home);
  const rf = cam.rootFrame({});
  const fr = world.frames(cam);
  let worst = 0;
  for (const f of fr) {
    const it = f.item, C = it.att.C;
    // the long way: j's local -> world -> home's local -> camera
    const w = it.toWorld(C);
    const rel = [w[0] - home.t[0], w[1] - home.t[1], w[2] - home.t[2]];
    const hl = [ (home.R[0] * rel[0] + home.R[3] * rel[1] + home.R[6] * rel[2]) / home.s,
                 (home.R[1] * rel[0] + home.R[4] * rel[1] + home.R[7] * rel[2]) / home.s,
                 (home.R[2] * rel[0] + home.R[5] * rel[1] + home.R[8] * rel[2]) / home.s ];
    const want = [rf.A[0] * hl[0] + rf.A[1] * hl[1] + rf.A[2] * hl[2] + rf.b[0],
                  rf.A[3] * hl[0] + rf.A[4] * hl[1] + rf.A[5] * hl[2] + rf.b[1],
                  rf.A[6] * hl[0] + rf.A[7] * hl[1] + rf.A[8] * hl[2] + rf.b[2]];
    // the short way: what frames() produced
    const got = [f.A[0] * C[0] + f.A[1] * C[1] + f.A[2] * C[2] + f.b[0],
                 f.A[3] * C[0] + f.A[4] * C[1] + f.A[5] * C[2] + f.b[1],
                 f.A[6] * C[0] + f.A[7] * C[1] + f.A[8] * C[2] + f.b[2]];
    const scale = Math.max(Math.hypot(...want), 1e-12);
    worst = Math.max(worst, Math.hypot(got[0] - want[0], got[1] - want[1],
                                       got[2] - want[2]) / scale);
    // and the radius it will be culled by
    const wantKap = rf.kap * it.s / home.s;
    worst = Math.max(worst, Math.abs(f.kap - wantKap) / Math.max(wantKap, 1e-12));
  }
  const ok = worst < 1e-12;
  if (!ok) bad++;
  console.log(`  ${label.padEnd(34)} worst relative error ${worst.toExponential(2)}  ${ok ? 'ok' : 'WRONG'}`);
}

console.log(`${world.length} attractors: ${world.items.map((i) => i.name).join(', ')}\n`);
console.log('frame composition, short way vs long way round:');
checkFrames('from outside, depth 0');

const st = new AdaptiveStream(world.get(world.first()).att, 500000);
st.budgetBy = 'splats'; st.target = 110000;
for (let i = 0; i < 14; i++) world.expand(st, cam, view);
const n = world.expand(st, cam, view);
mkdirSync('out', { recursive: true });
writeFileSync('out/void.png', png(render({
  pos: st.pos, cov: st.cov, rgba: st.rgba, off: st.offset, count: n,
  eye: cam.pos, B: { r: cam.r, u: cam.u, f: cam.f }, W, H,
  fogNear: 30, fogFar: 150,
})));
console.log(`\n  the whole void: ${n} splats -> out/void.png`);

// now fly into one and check the chain still holds as the camera descends
cam.pos = [...world.get(world.first()).att.C];
cam.pos[1] -= world.get(world.first()).att.R * 2.2;
cam.f = [0, 1, 0]; cam.u = [0, 0, 1]; cam.ortho();
for (let i = 0; i < 120; i++) { world.expand(st, cam, view); cam.dive(0.05, 0.15); cam.reroot(); }
checkFrames(`after diving to depth ${cam.depth}`);
console.log(`  (local scale ${cam.scale.toExponential(2)}; the other attractors are now `
  + `${(1 / cam.scale).toExponential(1)} local units away, which is why they fog out)`);

const n2 = world.expand(st, cam, view);
writeFileSync('out/void_deep.png', png(render({
  pos: st.pos, cov: st.cov, rgba: st.rgba, off: st.offset, count: n2,
  eye: cam.pos, B: { r: cam.r, u: cam.u, f: cam.f }, W, H,
  fogNear: 5, fogFar: 55,
})));
console.log(`  inside one of them: ${n2} splats -> out/void_deep.png`);

// and that retarget hands the camera over without moving it in the world
cam.path.length = 0; cam.hashes.length = 1;
cam.pos = [0, 0, 0];
const before = cam.worldPos();
const moved = cam.retarget();
const after = cam.worldPos();
const drift = Math.hypot(after[0] - before[0], after[1] - before[1], after[2] - before[2]);
const ok = drift < 1e-9;
if (!ok) bad++;
console.log(`\n  retarget: home ${moved ? 'changed' : 'unchanged'}, camera drifted `
  + `${drift.toExponential(2)} world units  ${ok ? 'ok' : 'WRONG'}`);

console.log(bad ? `\n${bad} checks FAILED` : '\nvoid ok');
process.exit(bad ? 1 : 0);
