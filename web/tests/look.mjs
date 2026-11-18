// The look effects: iridescence, rim light, and the per-object masking.
//
// What this has to establish is not "it looks nice" but that the switches
// mean what they say: that each effect changes the picture, that the UI's
// master mask can turn each one off completely, and -- the one that is easy
// to get wrong -- that an object which did not opt in is left alone even
// while its neighbours are lit.
import { writeFileSync, mkdirSync } from 'node:fs';
import { FieldWorld, VoidCamera } from '../js/world.js';
import { FX_IRID, FX_RIM } from '../js/field.js';
import { AdaptiveStream } from '../js/adaptive.js';
import { render, png } from '../js/cpu_raster.js';

const W = 420, H = 320, FOV = 55;
const tanY = Math.tan((FOV * Math.PI) / 360);
const view = { focal: (H / 2) / tanY, tanX: tanY * (W / H), tanY, znear: 1e-5, W, H };

// Park the camera on an object and look at it, rather than flying and hoping.
//
// A camera parked where VoidCamera starts is not necessarily pointed at
// anything, and a frame of pure background compares equal to itself under
// every effect -- which is a test that passes its masking check and tells you
// nothing about whether the effects work.
//
// Flying a fixed 90 frames and trusting where it ended up was the same
// mistake one step further on. It worked until the autopilot changed, and then
// the 90th frame of the new flight was a distant haze: the effects still
// fired, on so few lit pixels that the whole-image means fell under the
// threshold, and checks failed with nothing wrong in the thing they test. The
// dive is chaotic by construction (tests/pilot.mjs), so no fixed frame number
// is safe, and searching the flight for a bright frame only moves the problem
// -- what it finds is still whatever that flight happened to contain.
//
// A rim light needs lit surface across the frame, which a camera 2.2 radii
// off an object's centre, pointed at it, has: every pixel of the plain frame
// is above the background, drawn by a few hundred very large splats. That is
// two lines and does not depend on the autopilot at all. Measured there, the
// rim moves the mean pixel by 43.7 of 255 and the iridescence by 11.7 -- the
// thresholds below are not being squeezed past.
//
// And it has to be an object that OPTED IN. The effects are per-object by
// design (that is the third thing this file tests), so parking on whichever
// cell the camera starts in means a 16% chance of pointing the test at an
// object that has declined both effects and measuring nothing -- which is how
// it first came back green-adjacent and wrong. The per-object masking checks
// further down read the resident field rather than this frame, so choosing a
// cell here does not weaken them.
const world = new FieldWorld({});
let seat = null;
world.each((id, it) => { if (seat === null && it.fx === (FX_IRID | FX_RIM)) seat = id; });
if (seat === null) throw new Error('no resident cell wants both effects');
const cam = new VoidCamera(world, seat);
const home0 = world.get(seat);
const st = new AdaptiveStream(home0.att, 400000);
st.budgetBy = 'splats'; st.target = 120000;
const C0 = home0.att.C, R0 = home0.att.R;
cam.pos[0] = C0[0];
cam.pos[1] = C0[1] - 2.2 * R0;
cam.pos[2] = C0[2];
cam.f = [0, 1, 0];
cam.u = [0, 0, 1];
cam.ortho();
console.log(`  parked 2.2 radii off cell ${seat}, which wants both effects`);
let n = 0;
// The stream grows its budget over a few frames, and the LOD cut needs the
// camera it is actually drawing for.
for (let i = 0; i < 16; i++) {
  world.update(1 / 30, cam);
  n = world.expand(st, cam, view);
}

// Fog as the viewer sets it, so the splats are actually lit rather than
// faded out: near the home object's own span.
const home = world.get(cam.home);
const span = Math.max(home.att.R * 2, Math.hypot(
  cam.pos[0] - home.att.C[0], cam.pos[1] - home.att.C[1], cam.pos[2] - home.att.C[2]));

const LOOK = { mask: 3, iridAmt: 0.35, iridScale: 1.1, iridPhase: 0,
               rimAmt: 0.55, rimPow: 3, rimColor: [0.55, 0.78, 1.0] };
const shoot = (look) => render({
  pos: st.pos, cov: st.cov, rgba: st.rgba, off: st.offset, count: n,
  fx: st.fx, look,
  eye: cam.pos, B: { r: cam.r, u: cam.u, f: cam.f }, W, H,
  fogNear: span * 0.9, fogFar: span * 7.0 });

const base = shoot(null);
const all  = shoot(LOOK);
const irid = shoot({ ...LOOK, mask: FX_IRID });
const rim  = shoot({ ...LOOK, mask: FX_RIM });
const none = shoot({ ...LOOK, mask: 0 });

const diff = (a, b) => {
  let s = 0, worst = 0;
  for (let i = 0; i < a.img.length; i++) {
    const d = Math.abs(a.img[i] - b.img[i]);
    s += d; if (d > worst) worst = d;
  }
  return { mean: s / a.img.length, worst };
};
// The rasteriser's buffer is RGB, three bytes a pixel, not four.
const lum = (im) => {
  let s = 0; for (let i = 0; i < im.img.length; i += 3)
    s += 0.2126 * im.img[i] + 0.7152 * im.img[i + 1] + 0.0722 * im.img[i + 2];
  return s / (im.img.length / 3);
};

console.log(`${n} splats drawn, ${W}x${H}, ${base.drawn} passed the cull\n`);
const rows = [['both effects', all], ['iridescence only', irid],
              ['rim only', rim], ['mask 0 (all off)', none]];
for (const [label, im] of rows) {
  const d = diff(base, im);
  console.log('  ' + label.padEnd(18) + ' vs plain:  mean |d| ' +
    d.mean.toFixed(3).padStart(7) + '   worst ' + String(d.worst).padStart(3) +
    '   mean luminance ' + lum(im).toFixed(2));
}

let fail = 0;
const check = (ok, msg) => { console.log((ok ? '  ok    ' : '  FAIL  ') + msg); if (!ok) fail++; };
console.log('');
// The master mask must be absolute: mask 0 has to reproduce the plain image
// exactly, or a UI that says "off" is lying.
check(diff(base, none).worst === 0, 'mask 0 is bit-identical to no effects at all');
check(diff(base, irid).mean > 0.05, 'iridescence alone changes the picture');
check(diff(base, rim).mean > 0.05, 'rim light alone changes the picture');
check(lum(rim) > lum(base), 'rim light adds light rather than removing it');
check(diff(irid, rim).mean > 0.05, 'the two effects are distinguishable');

// Per-object opt-in.  Count how the resident field divided itself up, and
// confirm both effects are genuinely in use and genuinely declined.
const tally = new Map();
world.each((id, it) => tally.set(it.fx, (tally.get(it.fx) || 0) + 1));
const tot = [...tally.values()].reduce((a, b) => a + b, 0);
const share = (bit) => [...tally].filter(([k]) => k & bit)
  .reduce((a, [, v]) => a + v, 0) / tot;
console.log('');
console.log('  objects resident ' + tot +
  ':  want iridescence ' + (100 * share(FX_IRID)).toFixed(0) + '%' +
  ',  want rim ' + (100 * share(FX_RIM)).toFixed(0) + '%' +
  ',  want neither ' + (100 * ((tally.get(0) || 0) / tot)).toFixed(0) + '%');
check(share(FX_IRID) > 0.3 && share(FX_IRID) < 0.8, 'iridescence is a mix, not all or nothing');
check(share(FX_RIM) > 0.3 && share(FX_RIM) < 0.9, 'rim is a mix, not all or nothing');
check((tally.get(0) || 0) > 0, 'some objects opt out of everything');

// Determinism: the same cell must ask for the same effects every time it is
// generated, or effects would flicker as cells are evicted and remade.
const w2 = new FieldWorld({});
const c2 = new VoidCamera(w2, w2.first());
for (let i = 0; i < 40; i++) w2.update(1 / 30, c2);
let same = true, cmp = 0;
w2.each((id, it) => { const a = world.get(id); if (a) { cmp++; if (a.fx !== it.fx) same = false; } });
check(same && cmp > 50, `effect choice is deterministic across worlds (${cmp} cells compared)`);

mkdirSync('out', { recursive: true });
const sbs = { W: W * 2, H, img: new Uint8Array(W * 2 * H * 3) };
for (let y = 0; y < H; y++) {
  sbs.img.set(base.img.subarray(y * W * 3, (y + 1) * W * 3), y * W * 6);
  sbs.img.set(all.img.subarray(y * W * 3, (y + 1) * W * 3), y * W * 6 + W * 3);
}
writeFileSync('out/look.png', png(sbs));
console.log('\n  out/look.png   plain on the left, iridescence + rim on the right');
console.log(fail ? `\n  ${fail} FAILED` : '\n  look ok');
process.exit(fail ? 1 : 0);
