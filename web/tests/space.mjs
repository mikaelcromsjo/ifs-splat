// The void's background, and the one thing that can go wrong with stars.
//
// The nebula is three smooth exponential lobes and a band, chosen that way
// because anything with detail finer than a pixel crawls when the camera
// moves -- the comment in shaders.js is explicit that noise was ruled out for
// it. A star field is points, which is the ruled-out thing, so it is here on
// the same terms the splats are: never ask for something smaller than the
// sampling grid. Each star's angular sigma is floored at 1.3 pixels, so no
// star is ever a sub-pixel spike, and its profile is a Gaussian, so it has no
// edge to alias either.
//
// What distinguishes a field that crawls from one that does not is not how
// much it changes when the camera turns -- a star that moves SHOULD change
// the pixels it moves across -- it is whether the change is PROPORTIONAL to
// the motion. Turn by a quarter of a pixel, a half, one, two, and measure:
//
//   floored    sigma >= 1.3 px, as shipped
//   spike      the same field with the floor removed (sigma 0.08 px), which
//              is what a star field written without thinking about it does
//
// A band-limited field answers a 4x bigger turn with a 4x bigger change. A
// sub-pixel spike answers every turn, however small, by flipping one pixel
// from background to white and back: its worst pixel is already saturated at
// a quarter of a pixel of motion and does not grow. That saturation is
// exactly what crawling looks like.
//
// It also writes a picture, because the background is most of what the flight
// looks like and until now every headless frame had a flat grey one.
//
//   node tests/space.mjs [W] [H]

import { writeFileSync, mkdirSync } from 'node:fs';
import { FieldWorld, VoidCamera, Autopilot } from '../js/world.js';
import { AdaptiveStream } from '../js/adaptive.js';
import { render, png, tile, skyColour } from '../js/cpu_raster.js';

const W = Number(process.argv[2] ?? 420), H = Number(process.argv[3] ?? 300);
const FOV = 55;
const tanY = Math.tan((FOV * Math.PI) / 360), tanX = tanY * (W / H);
const view = { focal: (H / 2) / tanY, tanX, tanY, znear: 1e-5, W, H };
const PX = (2 * tanY) / H;              // one pixel, in radians

// The viewer's own look, kept in step with viewer.js by hand (the same deal
// the two rasterisers have had from the start).
const LOOK = {
  mask: 3, iridAmt: 0.38, iridScale: 1.1, iridPhase: 0,
  rimAmt: 0.62, rimPow: 2.8, rimColor: [0.60, 0.74, 1.0],
  sky: [0.040, 0.043, 0.068], deep: [0.006, 0.008, 0.018],
  lobeCol: [[0.090, 0.024, 0.100], [0.016, 0.062, 0.086], [0.082, 0.044, 0.018]],
  lobeDir: [[0.6, 0.5, -0.6], [-0.7, 0.2, 0.68], [0.2, -0.8, 0.55]],
  lobeK: [3.6, 6.0, 9.0],
  bandCol: [0.046, 0.040, 0.066], bandAxis: [0.25, 0.93, 0.27], bandK: 5.5,
  nebGain: 1.0, starAmt: 0.85, starDens: 40.0, starFill: 0.17, starWarm: 0.55,
};

let fail = 0;
const ok = (good, msg, detail = '') => {
  console.log((good ? '  ok    ' : '  FAIL  ') + msg + (detail ? '   ' + detail : ''));
  if (!good) fail++;
};

// ------------------------------------------------------- sub-pixel stability

/** The sky alone, as a plain RGB buffer, for a given basis and star sigma. */
function skyOnly(B, look, px) {
  const img = new Uint8Array(W * H * 3);
  for (let y = 0; y < H; y++) {
    const ny = 1 - (2 * (y + 0.5)) / H;
    for (let x = 0; x < W; x++) {
      const nx = (2 * (x + 0.5)) / W - 1;
      const vx = nx * tanX, vy = ny * tanY;
      let d = [B.r[0] * vx + B.u[0] * vy + B.f[0],
               B.r[1] * vx + B.u[1] * vy + B.f[1],
               B.r[2] * vx + B.u[2] * vy + B.f[2]];
      const l = Math.hypot(d[0], d[1], d[2]);
      d = [d[0] / l, d[1] / l, d[2] / l];
      const c = skyColour(d, look, px);
      const o = 3 * (y * W + x);
      for (let k = 0; k < 3; k++)
        img[o + k] = Math.round(255 * Math.min(Math.max(c[k], 0), 1) ** (1 / 1.1));
    }
  }
  return { img, W, H };
}

/**
 * The sky as the viewer now draws it: rendered at a fraction of the viewport
 * and stretched back bilinearly (viewer.js nebTarget/BLIT_FRAG).
 *
 * The star floor comes from the REDUCED target, not the screen, because
 * uStarPx is handed the height the sky is actually drawn at -- so a half-res
 * star is floored at one half-res pixel and comes out twice as wide in angle.
 * That is the one visible cost of the optimisation, and the question this
 * measures is whether the field is still band-limited afterwards, which is
 * the whole reason the floor exists.
 *
 * GL_LINEAR on a CLAMP_TO_EDGE RGBA8 texture, drawn over a full-screen quad:
 * sample centres land at (x+0.5)/W in texture coordinates, so the source
 * coordinate is (x+0.5)*sw/W - 0.5, clamped at the edges.
 */
function skyScaled(B, look, scale) {
  const sw = Math.max(1, Math.round(W * scale)), sh = Math.max(1, Math.round(H * scale));
  const spx = (2 * tanY) / sh;            // one pixel OF THE REDUCED TARGET
  const src = new Uint8Array(sw * sh * 3);
  for (let y = 0; y < sh; y++) {
    const ny = 1 - (2 * (y + 0.5)) / sh;
    for (let x = 0; x < sw; x++) {
      const nx = (2 * (x + 0.5)) / sw - 1;
      const vx = nx * tanX, vy = ny * tanY;
      let d = [B.r[0] * vx + B.u[0] * vy + B.f[0],
               B.r[1] * vx + B.u[1] * vy + B.f[1],
               B.r[2] * vx + B.u[2] * vy + B.f[2]];
      const l = Math.hypot(d[0], d[1], d[2]);
      d = [d[0] / l, d[1] / l, d[2] / l];
      const c = skyColour(d, look, spx);
      const o = 3 * (y * sw + x);
      for (let k = 0; k < 3; k++)
        src[o + k] = Math.round(255 * Math.min(Math.max(c[k], 0), 1) ** (1 / 1.1));
    }
  }
  const img = new Uint8Array(W * H * 3);
  const cl = (v, n) => Math.min(Math.max(v, 0), n - 1);
  for (let y = 0; y < H; y++) {
    const fy = ((y + 0.5) * sh) / H - 0.5;
    const y0 = Math.floor(fy), wy = fy - y0;
    const ya = cl(y0, sh), yb = cl(y0 + 1, sh);
    for (let x = 0; x < W; x++) {
      const fx = ((x + 0.5) * sw) / W - 0.5;
      const x0 = Math.floor(fx), wx = fx - x0;
      const xa = cl(x0, sw), xb = cl(x0 + 1, sw);
      const o = 3 * (y * W + x);
      for (let k = 0; k < 3; k++) {
        const p00 = src[3 * (ya * sw + xa) + k], p10 = src[3 * (ya * sw + xb) + k];
        const p01 = src[3 * (yb * sw + xa) + k], p11 = src[3 * (yb * sw + xb) + k];
        img[o + k] = (p00 * (1 - wx) * (1 - wy) + p10 * wx * (1 - wy)
                    + p01 * (1 - wx) * wy + p11 * wx * wy + 0.5) | 0;
      }
    }
  }
  return { img, W, H };
}

/** Rotate a basis about its own up axis. */
const turned = (B, a) => {
  const c = Math.cos(a), s = Math.sin(a);
  const f = [B.f[0] * c + B.r[0] * s, B.f[1] * c + B.r[1] * s, B.f[2] * c + B.r[2] * s];
  const r = [B.r[0] * c - B.f[0] * s, B.r[1] * c - B.f[1] * s, B.r[2] * c - B.f[2] * s];
  return { r, u: B.u, f };
};

const change = (a, b) => {
  let sum = 0, worst = 0;
  for (let i = 0; i < a.img.length; i++) {
    const d = Math.abs(a.img[i] - b.img[i]);
    sum += d; if (d > worst) worst = d;
  }
  return { mean: sum / a.img.length, worst };
};

const B0 = { r: [1, 0, 0], u: [0, 0, 1], f: [0, 1, 0] };
const QS = [0.25, 0.5, 1, 2];
console.log(`sky ${W}x${H}, one pixel = ${(PX * 1e3).toFixed(3)} mrad\n`);
const runs = {};
for (const [label, px] of [['floored', PX], ['spike', 0]]) {
  const base = skyOnly(B0, LOOK, px);
  runs[label] = QS.map((q) => change(base, skyOnly(turned(B0, q * PX), LOOK, px)));
  console.log(`  ${label.padEnd(8)}` + runs[label].map((d, i) =>
    `  ${QS[i]}px: mean ${d.mean.toFixed(3)} worst ${String(d.worst).padStart(3)}`).join(''));
}
// The same measurement on the sky as it now ships: drawn at half the
// viewport and stretched.  A quarter of the fragments, and the question is
// whether the field that comes out is still band-limited -- the stretch is a
// linear resample, which cannot ADD a frequency, but the star floor it is
// drawn under is twice as wide, which changes the picture.
{
  const base = skyScaled(B0, LOOK, 0.5);
  runs['half-res'] = QS.map((q) => change(base, skyScaled(turned(B0, q * PX), LOOK, 0.5)));
  console.log(`  ${'half-res'.padEnd(8)}` + runs['half-res'].map((d, i) =>
    `  ${QS[i]}px: mean ${d.mean.toFixed(3)} worst ${String(d.worst).padStart(3)}`).join(''));
}
const grow = (r) => r[2].mean / r[0].mean;        // 1 px against 1/4 px
const sat = (r) => r[0].worst / r[3].worst;       // 1/4 px against 2 px
console.log('');
console.log(`  a 4x bigger turn changes the floored sky`
  + ` ${grow(runs.floored).toFixed(2)}x as much, the spiky one`
  + ` ${grow(runs.spike).toFixed(2)}x`);
ok(grow(runs.floored) > 3.4 && grow(runs.floored) < 4.6,
   'the floored field answers motion in proportion to it',
   `${grow(runs.floored).toFixed(2)}x for a 4x turn`);
// An eight-fold bigger turn barely moves the spiky field's worst pixel: it
// was already at the top of its range for the smallest motion there is.
ok(sat(runs.spike) > 0.7,
   'the spiky field is already saturated at a quarter of a pixel -- it crawls',
   `worst pixel ${runs.spike[0].worst} at 1/4 px, ${runs.spike[3].worst} at 2 px`);
ok(sat(runs.floored) < 0.4,
   'and the floored one is not',
   `worst pixel ${runs.floored[0].worst} at 1/4 px, ${runs.floored[3].worst} at 2 px`);

// The half-resolution sky, on the same terms.  This is what justifies drawing
// the most expensive pass in the engine at a quarter of the fragments: it is
// allowed to look slightly different, it is not allowed to crawl.
const g = grow(runs['half-res']);
ok(g > 3.0 && g < 4.8,
   'the half-res sky answers motion in proportion to it too',
   `${g.toFixed(2)}x for a 4x turn`);
ok(sat(runs['half-res']) < 0.4,
   'and is no nearer saturation than the full-resolution one',
   `worst pixel ${runs['half-res'][0].worst} at 1/4 px, `
   + `${runs['half-res'][3].worst} at 2 px`);
// What it DOES cost, stated rather than hidden: the stars are twice as wide,
// so the picture is not the same picture.  Measured against the full-res sky
// at the same instant, with and without stars, so the number is the stars'
// own contribution and not the lobes'.
{
  const full = skyOnly(B0, LOOK, PX), half = skyScaled(B0, LOOK, 0.5);
  const fullNo = skyOnly(B0, { ...LOOK, starAmt: 0 }, PX);
  const halfNo = skyScaled(B0, { ...LOOK, starAmt: 0 }, 0.5);
  const lobes = change(fullNo, halfNo), all = change(full, half);
  console.log(`\n  half-res against full-res: the lobes and band differ by`
    + ` ${lobes.mean.toFixed(3)}/255 mean (worst ${lobes.worst}),`
    + ` the whole sky by ${all.mean.toFixed(3)}/255 (worst ${all.worst})`);
  ok(lobes.mean < 0.35,
     'the lobes and band survive the stretch -- they were band-limited already',
     `mean |d| ${lobes.mean.toFixed(3)}/255`);
  ok(all.mean > lobes.mean,
     'and the difference that is left is the stars, which are twice as wide',
     `${all.mean.toFixed(3)} against ${lobes.mean.toFixed(3)}/255`);
}

// Stars have to be THERE, not just steady: a field of nothing is perfectly
// steady too.
const lit = (im) => {
  let n = 0;
  for (let i = 0; i < im.img.length; i += 3) if (im.img[i + 2] > 60) n++;
  return n;
};
const withStars = skyOnly(B0, LOOK, PX);
const noStars = skyOnly(B0, { ...LOOK, starAmt: 0 }, PX);
console.log('');
ok(lit(withStars) > 20, 'there are stars in the frame',
   `${lit(withStars)} pixels over 60/255 against ${lit(noStars)} without them`);
ok(change(withStars, noStars).mean > 0.01, 'and turning them off changes the sky');
ok(change(noStars, skyOnly(B0, { ...LOOK, starAmt: 0 }, 0)).worst === 0,
   'with no stars the pixel floor has nothing to do');

// --------------------------------------------------------------- a picture
//
// Fly the real autopilot for a while, then photograph the same moment four
// ways, so the sky and the splat effects can be judged against each other
// rather than described.
const world = new FieldWorld({});
const cam = new VoidCamera(world, world.first());
let sd = 7;
const rng = () => ((sd = (1103515245 * sd + 12345) & 0x7fffffff) / 0x7fffffff);
const pilot = new Autopilot(world, cam, { rng });
const st = new AdaptiveStream(world.get(world.first()).att, 400000);
st.budgetBy = 'splats'; st.target = 120000;
let n = 0;
for (let i = 0; i < 420; i++) {
  world.update(1 / 30, cam); pilot.step(2); cam.reroot(); cam.retarget();
  n = world.expand(st, cam, view);
}
const home = world.get(cam.home);
const span = Math.max(home.att.R * 2, Math.hypot(
  cam.pos[0] - home.att.C[0], cam.pos[1] - home.att.C[1], cam.pos[2] - home.att.C[2]));
const shoot = (look, sky) => render({
  pos: st.pos, cov: st.cov, rgba: st.rgba, off: st.offset, count: n,
  fx: st.fx, look, sky,
  eye: cam.pos, B: { r: cam.r, u: cam.u, f: cam.f }, W, H,
  fogNear: span * 0.9, fogFar: span * 7.0 });

const shots = [shoot(null, null), shoot(LOOK, null),
               shoot(null, LOOK), shoot(LOOK, LOOK)];
mkdirSync('out', { recursive: true });
writeFileSync('out/space.png', png(tile(shots, 2)));
console.log('');
console.log(`  ${n} splats, phase ${pilot.phase}, depth ${cam.depth}`);
console.log('  out/space.png   plain | splat effects | sky | both');
ok(change(shots[0], shots[3]).mean > 1,
   'the full look is a different picture from the plain one',
   `mean |d| ${change(shots[0], shots[3]).mean.toFixed(2)}/255`);

console.log(fail ? `\n  ${fail} FAILED` : '\n  space ok');
process.exit(fail ? 1 : 0);
