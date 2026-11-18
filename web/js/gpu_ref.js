// The JS twin of the WGSL kernel in gpu_kernel.js.
//
// Line for line, same names, same order of operations, float32 throughout via
// Math.fround -- so that checking THIS against ifs_core.js checks the kernel.
// It is slow by design: it is a reference, not a path anyone renders through.

import { h32, mix as mixh, stream as strm, rootHash } from './ifs_core.js';

const F = Math.fround;
const HEIGHT = 2.5, H_ATT = 1.0, T0 = 0.075, T2 = 0.4, MZ = HEIGHT / 2;
const SIG0 = [0.45 * 0.45, 0.45 * 0.45, (HEIGHT / 4) ** 2];
const FLOOR2 = (2e-4 * HEIGHT) ** 2;
const A_DISC = 3.61, B_DISC = 0.09;

const nrm = (h, k) => {
  const u1 = Math.max(strm(h, k), 1e-12), u2 = strm(h, k + 1);
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
};

/** Row-major 3x3 here; the WGSL is column-major but the maths is identical. */
function matmul(A, B) {
  const o = new Float64Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      o[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
  return o;
}

function jitterRot(h, amt) {
  if (amt <= 0) return new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  const z = 2 * strm(h, 0) - 1, phi = 2 * Math.PI * strm(h, 1);
  const r = Math.sqrt(Math.max(1 - z * z, 0));
  const ax = [r * Math.cos(phi), r * Math.sin(phi), z];
  const ang = amt * nrm(h, 2);
  const K = new Float64Array([0, -ax[2], ax[1], ax[2], 0, -ax[0], -ax[1], ax[0], 0]);
  const K2 = matmul(K, K);
  const s = Math.sin(ang), c = 1 - Math.cos(ang);
  const o = new Float64Array(9);
  for (let i = 0; i < 9; i++) o[i] = (i % 4 === 0 ? 1 : 0) + s * K[i] + c * K2[i];
  return o;
}

const clamp255 = (x) => Math.max(0, Math.min(255, Math.round(x * 255)));
const lerp = (a, b, t) => a + (b - a) * t;

function shadeWood(d, depth, h) {
  const f = d / Math.max(depth, 1), n = 1 + 0.06 * nrm(h, 10);
  return [clamp255(lerp(0.26, 0.47, f) * n), clamp255(lerp(0.17, 0.38, f) * n),
          clamp255(lerp(0.10, 0.28, f) * n), clamp255(0.99)];
}
function shadeLeaf(h) {
  const n = 1 + 0.22 * nrm(h, 12), g = 1 + 0.10 * nrm(h, 14);
  return [clamp255(0.17 * n), clamp255(0.41 * n * g), clamp255(0.13 * n), clamp255(0.55)];
}

function congruence(A, s) {
  const o = new Float64Array(6), ix = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]];
  for (let k = 0; k < 6; k++) {
    const [p, q] = ix[k];
    o[k] = A[p * 3] * A[q * 3] * s[0] + A[p * 3 + 1] * A[q * 3 + 1] * s[1]
         + A[p * 3 + 2] * A[q * 3 + 2] * s[2];
  }
  return o;
}

/** The closed form the kernel uses in place of an eigensolve. See gpu_kernel.js. */
export function discCovClosed(A) {
  const c3 = [A[2], A[5], A[8]];
  const s2 = c3[0] * c3[0] + c3[1] * c3[1] + c3[2] * c3[2];
  const inv = 1 / Math.sqrt(Math.max(s2, 1e-30));
  const v = [c3[0] * inv, c3[1] * inv, c3[2] * inv];
  const ev = [Math.max(s2 * SIG0[0], FLOOR2), Math.max(s2 * SIG0[1], FLOOR2),
              Math.max(s2 * SIG0[2], FLOOR2)];
  const g = Math.cbrt(Math.sqrt(ev[0]) * Math.sqrt(ev[1]) * Math.sqrt(ev[2]));
  const a2 = A_DISC * g * g, k = B_DISC * g * g - a2;
  return new Float64Array([a2 + k * v[0] * v[0], k * v[0] * v[1], k * v[0] * v[2],
                           a2 + k * v[1] * v[1], k * v[1] * v[2], a2 + k * v[2] * v[2]]);
}

const BRANCH_A = [];   // filled by setBranch()
export function setBranch(list) { BRANCH_A.length = 0; for (const m of list) BRANCH_A.push(m.A); }

/**
 * One invocation of the kernel: splat index `tid` -> its geometry and the
 * stream slot it occupies, for view direction `dir`.
 */
export function thread(tid, { depth, jitter, seed, dir, sizes, cents }) {
  let A = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  let b = [0, 0, 0];
  let h = rootHash(seed);
  let base = 0, idx = 0, k = tid, d = 0;

  for (;;) {
    const r = depth - d;
    const AW = new Float64Array([A[0] * T0, A[1] * T0, A[2] * T2,
                                 A[3] * T0, A[4] * T0, A[5] * T2,
                                 A[6] * T0, A[7] * T0, A[8] * T2]);
    const w = [A[2] * (T2 * MZ) + b[0], A[5] * (T2 * MZ) + b[1], A[8] * (T2 * MZ) + b[2]];
    const key = [0, 0, 0, 0, 0], size = [1, 0, 0, 0, 0];
    key[0] = w[0] * dir[0] + w[1] * dir[1] + w[2] * dir[2];

    if (r === 0) {
      const lp = [A[2] * MZ + b[0], A[5] * MZ + b[1], A[8] * MZ + b[2]];
      key[1] = lp[0] * dir[0] + lp[1] * dir[1] + lp[2] * dir[2];
      const rodFirst = key[0] >= key[1];
      if (k === 0)
        return { slot: base + (rodFirst ? 0 : 1), idx, pos: w,
                 cov: congruence(AW, SIG0), col: shadeWood(d, depth, h), level: d };
      return { slot: base + (rodFirst ? 1 : 0), idx: idx + 1, pos: lp,
               cov: discCovClosed(A), col: shadeLeaf(h), level: d };
    }

    const cA = [];
    const cb = [A[2] * H_ATT + b[0], A[5] * H_ATT + b[1], A[8] * H_ATT + b[2]];
    const cen = cents[r - 1], csz = sizes[r - 1];
    for (let i = 0; i < 4; i++) {
      const chh = mixh(h, i);
      const AE = matmul(jitterRot(chh, jitter), BRANCH_A[i]);
      const M = matmul(A, AE);
      cA.push(M);
      const px = M[0] * cen[0] + M[1] * cen[1] + M[2] * cen[2] + cb[0];
      const py = M[3] * cen[0] + M[4] * cen[1] + M[5] * cen[2] + cb[1];
      const pz = M[6] * cen[0] + M[7] * cen[1] + M[8] * cen[2] + cb[2];
      key[1 + i] = px * dir[0] + py * dir[1] + pz * dir[2];
      size[1 + i] = csz;
    }

    const o = [0, 1, 2, 3, 4];
    const cs = (i, j) => {
      if (key[i] < key[j]) {
        const tk = key[i]; key[i] = key[j]; key[j] = tk;
        const to = o[i]; o[i] = o[j]; o[j] = to;
      }
    };
    cs(0, 1); cs(3, 4); cs(2, 4); cs(2, 3); cs(0, 3); cs(0, 2); cs(1, 4); cs(1, 3); cs(1, 2);

    const off = [0, 0, 0, 0, 0];
    let acc = 0;
    for (let j = 0; j < 5; j++) { off[o[j]] = acc; acc += size[o[j]]; }

    if (k === 0)
      return { slot: base + off[0], idx, pos: w, cov: congruence(AW, SIG0),
               col: shadeWood(d, depth, h), level: d };
    const ci = Math.floor((k - 1) / csz);
    k = (k - 1) - ci * csz;
    base += off[1 + ci];
    idx += 1 + ci * csz;
    A = cA[ci];
    b = cb;
    h = mixh(h, ci);
    d += 1;
  }
}
