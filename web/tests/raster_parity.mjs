// Does the shader's projection agree with py/splat_raster.py?
//
// I cannot run WebGL here, so this transcribes the GLSL in js/shaders.js
// operation for operation into plain JS and rasterises with it.  The Python
// rasteriser was written independently and is what the stage-0 numbers were
// measured against; if the two images agree, the formulas in the shader are
// the formulas that were measured.  (It does not check GLSL syntax -- only a
// real GL context can do that.)

import { expand, orderFrame, makeScratch } from '../js/ifs_core.js';
import { writeFileSync } from 'node:fs';

const depth = Number(process.argv[2] ?? 6);
const RES = Number(process.argv[3] ?? 256);
const KAPPA = 0.35, LOWPASS = 0.3, CUT = 3.0;

const tree = expand(depth, 0.16, 3);
const N = tree.N, pos = tree.pos, cov = tree.cov, rgba = tree.rgba;

// camera: same convention as splat_raster.Camera
let ctr = [0, 0, 0];
for (let i = 0; i < N; i++) { ctr[0] += pos[3 * i]; ctr[1] += pos[3 * i + 1]; ctr[2] += pos[3 * i + 2]; }
ctr = ctr.map((x) => x / N);
let lo = [1e9, 1e9, 1e9], hi = [-1e9, -1e9, -1e9];
for (let i = 0; i < N; i++) for (let k = 0; k < 3; k++) {
  const x = pos[3 * i + k]; if (x < lo[k]) lo[k] = x; if (x > hi[k]) hi[k] = x;
}
const span = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
const dir = (() => { const d = [0.62, 0.41, -0.67]; const l = Math.hypot(...d); return d.map((x) => x / l); })();
const eye = [ctr[0] - dir[0] * span * 1.05, ctr[1] - dir[1] * span * 1.05, ctr[2] - dir[2] * span * 1.05];

const f3 = dir;
const upv = [0, 0, 1];
let r3 = [f3[1] * upv[2] - f3[2] * upv[1], f3[2] * upv[0] - f3[0] * upv[2], f3[0] * upv[1] - f3[1] * upv[0]];
const rl = Math.hypot(...r3); r3 = r3.map((x) => x / rl);
const u3 = [r3[1] * f3[2] - r3[2] * f3[1], r3[2] * f3[0] - r3[0] * f3[2], r3[0] * f3[1] - r3[1] * f3[0]];

const fovy = 50, W = RES, H = RES;
const F = (H / 2) / Math.tan((fovy * Math.PI) / 360);

const order = orderFrame(tree, dir, new Int32Array(N), makeScratch(tree));

const bg = [0.045, 0.05, 0.065];
const buf = new Float64Array(W * H * 3);
for (let i = 0; i < W * H; i++) { buf[3 * i] = bg[0]; buf[3 * i + 1] = bg[1]; buf[3 * i + 2] = bg[2]; }

for (let s = 0; s < N; s++) {
  const id = order[s];
  // ---- vertex shader, transcribed
  const dx = pos[3 * id] - eye[0], dy = pos[3 * id + 1] - eye[1], dz = pos[3 * id + 2] - eye[2];
  const vx = r3[0] * dx + r3[1] * dy + r3[2] * dz;
  const vy = u3[0] * dx + u3[1] * dy + u3[2] * dz;
  const vz = f3[0] * dx + f3[1] * dy + f3[2] * dz;
  if (vz < 0.02) continue;

  const c = cov.subarray(6 * id, 6 * id + 6);
  const px = KAPPA * vz / F, p2 = px * px;
  const S = [c[0] + p2, c[1], c[2], c[1], c[3] + p2, c[4], c[2], c[4], c[5] + p2];

  const zi = 1 / vz;
  // J rows: (F/z, 0, -F x/z^2), (0, -F/z, F y/z^2)
  const J = [F * zi, 0, -F * vx * zi * zi, 0, -F * zi, F * vy * zi * zi];
  // Wm = J * Rv  (2x3)
  const Rv = [r3[0], r3[1], r3[2], u3[0], u3[1], u3[2], f3[0], f3[1], f3[2]];
  const Wm = new Float64Array(6);
  for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++)
    Wm[i * 3 + j] = J[i * 3] * Rv[j] + J[i * 3 + 1] * Rv[3 + j] + J[i * 3 + 2] * Rv[6 + j];
  // S2 = Wm S Wm^T
  const WS = new Float64Array(6);
  for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++)
    WS[i * 3 + j] = Wm[i * 3] * S[j] + Wm[i * 3 + 1] * S[3 + j] + Wm[i * 3 + 2] * S[6 + j];
  let a = WS[0] * Wm[0] + WS[1] * Wm[1] + WS[2] * Wm[2];
  let b = WS[0] * Wm[3] + WS[1] * Wm[4] + WS[2] * Wm[5];
  let d = WS[3] * Wm[3] + WS[4] * Wm[4] + WS[5] * Wm[5];

  const det0 = a * d - b * b;
  a += LOWPASS; d += LOWPASS;
  const det1 = a * d - b * b;
  if (det1 <= 0) continue;
  const comp = Math.sqrt(Math.min(Math.max(det0, 0) / det1, 1));
  const con = [d / det1, -b / det1, a / det1];

  const mid = 0.5 * (a + d), rr = Math.sqrt(Math.max(mid * mid - det1, 0));
  const l1 = mid + rr, l2 = Math.max(mid - rr, 0.02);
  let ax1;
  if (Math.abs(b) < 1e-9) ax1 = a >= d ? [1, 0] : [0, 1];
  else { const n = Math.hypot(b, l1 - a); ax1 = [b / n, (l1 - a) / n]; }
  const ax2 = [ax1[1], -ax1[0]];
  const m1 = Math.min(CUT * Math.sqrt(l1), 2048), m2 = Math.min(CUT * Math.sqrt(l2), 2048);

  const cx = F * vx * zi + W / 2, cy = H / 2 - F * vy * zi;
  const rx = Math.abs(m1 * ax1[0]) + Math.abs(m2 * ax2[0]);
  const ry = Math.abs(m1 * ax1[1]) + Math.abs(m2 * ax2[1]);
  const x0 = Math.max(0, Math.floor(cx - rx)), x1 = Math.min(W - 1, Math.ceil(cx + rx));
  const y0 = Math.max(0, Math.floor(cy - ry)), y1 = Math.min(H - 1, Math.ceil(cy + ry));
  const al = (rgba[4 * id + 3] / 255) * comp;
  const cr = rgba[4 * id] / 255, cg = rgba[4 * id + 1] / 255, cbl = rgba[4 * id + 2] / 255;

  for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
    // ---- fragment shader, transcribed
    const ex = x + 0.5 - cx, ey = y + 0.5 - cy;
    const e = con[0] * ex * ex + 2 * con[1] * ex * ey + con[2] * ey * ey;
    if (e < 0) continue;
    const av = al * Math.exp(-0.5 * e);
    if (av < 0.0039) continue;
    const o = 3 * (y * W + x);
    buf[o] = buf[o] * (1 - av) + cr * av;
    buf[o + 1] = buf[o + 1] * (1 - av) + cg * av;
    buf[o + 2] = buf[o + 2] * (1 - av) + cbl * av;
  }
}

const img = new Float32Array(W * H * 3);
for (let i = 0; i < W * H * 3; i++) img[i] = Math.pow(Math.min(Math.max(buf[i], 0), 1), 1 / 1.1);
writeFileSync(process.argv[4] ?? '/tmp/js_raster.json', JSON.stringify({
  depth, RES, eye, dir, fovy, order: Array.from(order), img: Array.from(img),
}));
console.error(`js raster: ${N} splats at ${RES}px`);
