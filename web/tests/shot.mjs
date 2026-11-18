// Render the adaptive walk's output to a PNG, so stage 2 and 3 can be looked
// at without a browser.
//
//   node tests/shot.mjs [set] [diveFrames] [out.png] [W] [H] [target]

import { writeFileSync, mkdirSync } from 'node:fs';
import { SETS, build } from '../js/maps.js';
import { Attractor, FlyCamera, AdaptiveStream, PALETTES } from '../js/adaptive.js';
import { render, png, basis } from '../js/cpu_raster.js';

export function shoot(spec, { dive = 0, W = 480, H = 360, target = 90000,
                              palette, fog = true, seed = 3, jitter } = {}) {
  // the set's own jitter unless the caller overrides it -- jitter is a map-set
  // parameter now, and `sierpinski` at 0.16 is not a Sierpinski gasket
  jitter = jitter ?? spec.jitter ?? 0.16;
  const b = build(spec);
  const att = new Attractor({
    maps: b.maps, trunkA: b.trunkA, sig0: b.sig0, mu0: b.mu0,
    omega: spec.omega ?? 0.28, jitter, seed,
    palette: PALETTES[palette || spec.palette || 'tree'],
  });
  const cam = new FlyCamera(att);
  cam.f = [att.C[0] - cam.pos[0], att.C[1] - cam.pos[1], att.C[2] - cam.pos[2]];
  cam.u = [0, 0, 1];
  cam.ortho();
  const st = new AdaptiveStream(att, Math.max(400000, target * 3));
  st.budgetBy = 'splats';
  st.target = target;
  const warm = dive > 0 ? dive : 25;
  for (let i = 0; i < warm; i++) {
    st.expand(cam, { focal: 1, tanX: 1, tanY: 1, znear: 1e-4, W, H });
    if (dive > 0) { cam.dive(0.05, 0.15); cam.reroot(); }
  }
  const focal = (H / 2) / Math.tan((55 * Math.PI) / 360);
  const tanY = Math.tan((55 * Math.PI) / 360);
  for (let i = 0; i < 12; i++)
    st.expand(cam, { focal, tanX: tanY * (W / H), tanY, znear: 1e-5, W, H });
  const n = st.expand(cam, { focal, tanX: tanY * (W / H), tanY, znear: 1e-5, W, H });
  const im = render({
    pos: st.pos, cov: st.cov, rgba: st.rgba, off: st.offset, count: n,
    eye: cam.pos, B: { r: cam.r, u: cam.u, f: cam.f }, W, H,
    fogNear: fog ? 5 : 1e9, fogFar: fog ? 55 : 2e9,
  });
  return { im, n, depth: cam.depth, scale: cam.scale, att };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  const name = process.argv[2] || 'tree';
  const dive = Number(process.argv[3] ?? 0);
  const out = process.argv[4] || `out/${name}.png`;
  const W = Number(process.argv[5] ?? 520), H = Number(process.argv[6] ?? 400);
  const target = Number(process.argv[7] ?? 90000);
  mkdirSync('out', { recursive: true });
  const r = shoot(SETS[name], { dive, W, H, target });
  writeFileSync(out, png(r.im));
  console.log(`${name}: ${r.n} splats, depth ${r.depth}, `
            + `zoom ${(1 / r.scale).toExponential(1)}x, ${r.im.drawn} drawn -> ${out}`);
}
