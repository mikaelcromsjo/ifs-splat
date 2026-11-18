// IFS tree splats: the generator and the ordering, in the browser.
//
// This is a port of py/ifs_core.py and has to agree with it splat for splat --
// tests/parity.mjs checks that it does.  The 32-bit hash is written so the two
// languages compute the same bits: Math.imul here, uint32 numpy there.
//
// Two things matter for everything built on top:
//
//   * every per-node random quantity is a pure function of the node's address,
//     so a subtree can be generated alone, at any time, and come out the same;
//   * ordering never sorts N things.  Each node sorts its m+1 children by
//     depth and writes them into slots, so the last level comes out already
//     back to front.

export const C0 = 0.28209479177387814;

// ------------------------------------------------------------------ the IFS

export const S_BR = 0.60;
export const H_ATT = 1.0;
export const HEIGHT = H_ATT / (1.0 - S_BR);          // 2.5

const rad = (d) => (d * Math.PI) / 180.0;

function rzM(a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [c, -s, 0, s, c, 0, 0, 0, 1];
}
function ryM(a) {
  const c = Math.cos(a), s = Math.sin(a);
  return [c, 0, s, 0, 1, 0, -s, 0, c];
}
export function mul3(A, B) {
  const o = new Array(9);
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) {
      let s = 0;
      for (let k = 0; k < 3; k++) s += A[i * 3 + k] * B[k * 3 + j];
      o[i * 3 + j] = s;
    }
  return o;
}
export function mv3(A, v) {
  return [
    A[0] * v[0] + A[1] * v[1] + A[2] * v[2],
    A[3] * v[0] + A[4] * v[1] + A[5] * v[2],
    A[6] * v[0] + A[7] * v[1] + A[8] * v[2],
  ];
}
function scale3(A, s) { return A.map((x) => x * s); }

function branch(tilt, roll, s, twist) {
  return {
    A: scale3(mul3(mul3(rzM(rad(roll)), ryM(rad(tilt))), rzM(rad(twist))), s),
    b: [0, 0, H_ATT],
  };
}

export const BRANCH = [
  branch(38.0, 0.0, S_BR, 25.0),
  branch(38.0, 120.0, S_BR, 25.0),
  branch(38.0, 240.0, S_BR, 25.0),
  branch(9.0, 60.0, 0.52, 40.0),          // leader shoot
];
export const M_BR = BRANCH.length;

export const TRUNK_A = [0.075, 0, 0, 0, 0.075, 0, 0, 0, 1.0 - S_BR];
export const MU0 = [0, 0, HEIGHT / 2];
export const SIG0 = [0.45 * 0.45, (0.45) * 0.45, (HEIGHT / 4) * (HEIGHT / 4)];
const SIG0D = [0.45 * 0.45, 0.45 * 0.45, (HEIGHT / 4) * (HEIGHT / 4)];
export const WOOD_MU_LOCAL = mv3(TRUNK_A, MU0);
export const SCALE_FLOOR = 2e-4 * HEIGHT;

export const KIND_WOOD = 0, KIND_LEAF = 1;

// -------------------------------------------------------- address hashing

export function h32(x) {
  x = x >>> 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x85ebca6b);
  x ^= x >>> 13;
  x = Math.imul(x, 0xc2b2ae35);
  x ^= x >>> 16;
  return x >>> 0;
}
export const mix = (h, v) => h32((h + Math.imul((v + 1) >>> 0, 0x9e3779b1)) >>> 0);
export const stream = (h, k) =>
  h32((h ^ Math.imul((k + 1) >>> 0, 0x85ebca77)) >>> 0) / 4294967296.0;
export const rootHash = (seed) => h32(seed >>> 0);

export function normal(h, k) {
  const u1 = Math.max(stream(h, k), 1e-12);
  const u2 = stream(h, k + 1);
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

/** Per-node random rotation, from the node's address hash alone. */
export function jitterRot(h, amt) {
  if (amt <= 0) return [1, 0, 0, 0, 1, 0, 0, 0, 1];
  const z = 2.0 * stream(h, 0) - 1.0;
  const phi = 2.0 * Math.PI * stream(h, 1);
  const r = Math.sqrt(Math.max(1.0 - z * z, 0.0));
  const ax = [r * Math.cos(phi), r * Math.sin(phi), z];
  const ang = amt * normal(h, 2);
  const K = [0, -ax[2], ax[1], ax[2], 0, -ax[0], -ax[1], ax[0], 0];
  const K2 = mul3(K, K);
  const s = Math.sin(ang), c = 1.0 - Math.cos(ang);
  const o = new Array(9);
  for (let i = 0; i < 9; i++) o[i] = (i % 4 === 0 ? 1 : 0) + s * K[i] + c * K2[i];
  return o;
}

// ------------------------------------------------- subtree size + centroid

export function subtreeSizes(depth) {
  const T = new Float64Array(depth + 1);
  T[0] = 2;
  for (let r = 1; r <= depth; r++) T[r] = 1 + M_BR * T[r - 1];
  return T;
}

/** C[r] = the exact mean of a depth-r subtree's splats, in the node's frame.
 *  Used as the subtree's depth representative: where its mass actually is. */
export function subtreeCentroids(depth) {
  const T = subtreeSizes(depth);
  const C = [[
    0.5 * (WOOD_MU_LOCAL[0] + MU0[0]),
    0.5 * (WOOD_MU_LOCAL[1] + MU0[1]),
    0.5 * (WOOD_MU_LOCAL[2] + MU0[2])]];
  for (let r = 1; r <= depth; r++) {
    const acc = WOOD_MU_LOCAL.slice();
    for (let i = 0; i < M_BR; i++) {
      const p = mv3(BRANCH[i].A, C[r - 1]);
      for (let k = 0; k < 3; k++) acc[k] += T[r - 1] * (p[k] + BRANCH[i].b[k]);
    }
    C.push(acc.map((x) => x / T[r]));
  }
  return C;
}

// -------------------------------------------------- symmetric 3x3 eigen

/** Jacobi eigendecomposition of a symmetric 3x3.  Writes eigenvalues into
 *  EW and the eigenvectors as COLUMNS of EV, and returns the index of the
 *  largest eigenvalue.  Allocation-free: this runs once per foliage disc, and
 *  at depth 8 that is 65k times per generation. */
const EW = new Float64Array(3);
const EV = new Float64Array(9);
const EA = new Float64Array(9);
const JPQ = new Int32Array([0, 1, 0, 2, 1, 2]);

export function eigh3(S) {
  for (let i = 0; i < 9; i++) { EA[i] = S[i]; EV[i] = i % 4 === 0 ? 1 : 0; }
  for (let sweep = 0; sweep < 12; sweep++) {
    if (EA[1] * EA[1] + EA[2] * EA[2] + EA[5] * EA[5] < 1e-30) break;
    for (let t2 = 0; t2 < 6; t2 += 2) {
      const p = JPQ[t2], q = JPQ[t2 + 1];
      const apq = EA[p * 3 + q];
      if (apq === 0) continue;
      const theta = (EA[q * 3 + q] - EA[p * 3 + p]) / (2 * apq);
      const sg = theta < 0 ? -1 : 1;
      const t = sg / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), sn = t * c;
      for (let k = 0; k < 3; k++) {
        const akp = EA[k * 3 + p], akq = EA[k * 3 + q];
        EA[k * 3 + p] = c * akp - sn * akq;
        EA[k * 3 + q] = sn * akp + c * akq;
      }
      for (let k = 0; k < 3; k++) {
        const apk = EA[p * 3 + k], aqk = EA[q * 3 + k];
        EA[p * 3 + k] = c * apk - sn * aqk;
        EA[q * 3 + k] = sn * apk + c * aqk;
        const vkp = EV[k * 3 + p], vkq = EV[k * 3 + q];
        EV[k * 3 + p] = c * vkp - sn * vkq;
        EV[k * 3 + q] = sn * vkp + c * vkq;
      }
    }
  }
  EW[0] = EA[0]; EW[1] = EA[4]; EW[2] = EA[8];
  let m = 0;
  if (EW[1] > EW[m]) m = 1;
  if (EW[2] > EW[m]) m = 2;
  return m;
}

// ------------------------------------------------------------- the expansion

/** Reshape a leaf Gaussian into a disc, flat along the branch axis.
 *  The two in-plane radii are equal, so the disc is rotationally symmetric
 *  about that axis and only the dominant eigenvector is needed:
 *      S_disc = a^2 (I - v v^T) + b^2 v v^T
 *  which is exactly the matrix numpy's V diag(disc^2) V^T produces. */
const ES = new Float64Array(9);

function discCov(c, out, o) {
  ES[0] = c[0]; ES[1] = c[1]; ES[2] = c[2];
  ES[3] = c[1]; ES[4] = c[3]; ES[5] = c[4];
  ES[6] = c[2]; ES[7] = c[4]; ES[8] = c[5];
  const m = eigh3(ES);
  const f = SCALE_FLOOR * SCALE_FLOOR;
  const g = Math.cbrt(Math.sqrt(Math.max(EW[0], f)) * Math.sqrt(Math.max(EW[1], f)) *
                      Math.sqrt(Math.max(EW[2], f)));
  const a2 = 3.61 * g * g, k = 0.09 * g * g - a2;
  const v0 = EV[m], v1 = EV[3 + m], v2 = EV[6 + m];
  out[o] = a2 + k * v0 * v0;
  out[o + 1] = k * v0 * v1;
  out[o + 2] = k * v0 * v2;
  out[o + 3] = a2 + k * v1 * v1;
  out[o + 4] = k * v1 * v2;
  out[o + 5] = a2 + k * v2 * v2;
}

const JR = new Float64Array(9);     // scratch: the jitter rotation
const AE = new Float64Array(9);     // scratch: R @ A_i
const CV = new Float64Array(6);     // scratch: one covariance

/** jitterRot, writing into a scratch instead of allocating. */
export function jitterRotInto(o, h, amt) {
  if (amt <= 0) { o[0] = o[4] = o[8] = 1; o[1] = o[2] = o[3] = o[5] = o[6] = o[7] = 0; return; }
  const z = 2.0 * stream(h, 0) - 1.0;
  const phi = 2.0 * Math.PI * stream(h, 1);
  const r = Math.sqrt(Math.max(1.0 - z * z, 0.0));
  const ax = r * Math.cos(phi), ay = r * Math.sin(phi), az = z;
  const ang = amt * normal(h, 2);
  const sn = Math.sin(ang), c = 1.0 - Math.cos(ang);
  // I + sin(a) K + (1-cos a) K^2, with K the cross-product matrix of the axis
  o[0] = 1 + c * (ax * ax - 1); o[1] = -sn * az + c * ax * ay; o[2] = sn * ay + c * ax * az;
  o[3] = sn * az + c * ay * ax; o[4] = 1 + c * (ay * ay - 1); o[5] = -sn * ax + c * ay * az;
  o[6] = -sn * ay + c * az * ax; o[7] = sn * ax + c * az * ay; o[8] = 1 + c * (az * az - 1);
}

/** out = M diag(s) M^T, upper triangle, M row-major at M[m..m+8]. */
function congruenceInto(out, o, M, m, s0, s1, s2) {
  const a = M[m], b = M[m + 1], c = M[m + 2];
  const d = M[m + 3], e = M[m + 4], f = M[m + 5];
  const g = M[m + 6], i = M[m + 7], j = M[m + 8];
  out[o] = a * a * s0 + b * b * s1 + c * c * s2;
  out[o + 1] = a * d * s0 + b * e * s1 + c * f * s2;
  out[o + 2] = a * g * s0 + b * i * s1 + c * j * s2;
  out[o + 3] = d * d * s0 + e * e * s1 + f * f * s2;
  out[o + 4] = d * g * s0 + e * i * s1 + f * j * s2;
  out[o + 5] = g * g * s0 + i * i * s1 + j * j * s2;
}

/**
 * Generate every splat, in canonical (address-lexicographic) order, plus the
 * per-node data the per-frame ordering needs.
 *
 *   pos    Float32Array(3N)
 *   cov    Float32Array(6N)   upper triangle: c00 c01 c02 c11 c12 c22
 *   rgba   Uint8Array(4N)
 *   level  Uint8Array(N)      recursion depth, for colour-by-depth later
 *   levels per level: the world position of each item's depth representative
 */
export function expand(depth, jitter = 0.0, seed = 7) {
  const T = subtreeSizes(depth);
  const C = subtreeCentroids(depth);
  const N = T[depth] | 0;

  const pos = new Float32Array(3 * N);
  const cov = new Float32Array(6 * N);
  const rgba = new Uint8Array(4 * N);
  const level = new Uint8Array(N);
  const levels = [];

  const t0 = TRUNK_A[0], t2 = TRUNK_A[8];
  const s0 = SIG0D[0], s1 = SIG0D[1], s2 = SIG0D[2];
  const mz = MU0[2];

  let n = 1;
  let A = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
  let b = new Float64Array(3);
  let h = new Uint32Array([rootHash(seed)]);
  let cb = new Int32Array(1);
  const AW = new Float64Array(9);

  for (let d = 0; d <= depth; d++) {
    const r = depth - d;
    const nItems = r === 0 ? 2 : M_BR + 1;
    const keyPts = new Float32Array(n * nItems * 3);
    const childSize = r === 0 ? 1 : T[r - 1];
    const nn = r === 0 ? 0 : n * M_BR;
    const nA = new Float64Array(9 * nn);
    const nb = new Float64Array(3 * nn);
    const nh = new Uint32Array(nn);
    const ncb = new Int32Array(nn);
    const cr = r > 0 ? C[r - 1] : null;

    for (let j = 0; j < n; j++) {
      const a0 = 9 * j, b0 = 3 * j, ci = cb[j], hj = h[j];
      const bx = b[b0], by = b[b0 + 1], bz = b[b0 + 2];

      // ---- the node's wood rod.  TRUNK_A is diagonal, so composing it in the
      // node's own frame just scales A's columns.
      for (let i = 0; i < 3; i++) {
        AW[i * 3] = A[a0 + i * 3] * t0;
        AW[i * 3 + 1] = A[a0 + i * 3 + 1] * t0;
        AW[i * 3 + 2] = A[a0 + i * 3 + 2] * t2;
      }
      const wx = AW[2] * mz + bx, wy = AW[5] * mz + by, wz = AW[8] * mz + bz;
      pos[3 * ci] = wx; pos[3 * ci + 1] = wy; pos[3 * ci + 2] = wz;
      congruenceInto(cov, 6 * ci, AW, 0, s0, s1, s2);
      level[ci] = d;
      shadeWood(rgba, ci, d, depth, hj);

      const kp = j * nItems * 3;
      keyPts[kp] = wx; keyPts[kp + 1] = wy; keyPts[kp + 2] = wz;

      if (r === 0) {
        // ---- leaf: the rod, plus one foliage disc; nothing to recurse into
        const li = ci + 1;
        const lx = A[a0 + 2] * mz + bx, ly = A[a0 + 5] * mz + by, lz = A[a0 + 8] * mz + bz;
        pos[3 * li] = lx; pos[3 * li + 1] = ly; pos[3 * li + 2] = lz;
        congruenceInto(CV, 0, A, a0, s0, s1, s2);
        discCov(CV, cov, 6 * li);
        level[li] = d;
        shadeLeaf(rgba, li, hj);
        keyPts[kp + 3] = lx; keyPts[kp + 4] = ly; keyPts[kp + 5] = lz;
        continue;
      }

      // ---- the m children.  A child's geometry depends only on its own
      // address, through the hash -- never on generation order.
      for (let i = 0; i < M_BR; i++) {
        const chh = mix(hj, i);
        const c0 = 9 * (j * M_BR + i), d0 = 3 * (j * M_BR + i);
        jitterRotInto(JR, chh, jitter);
        const BA = BRANCH[i].A;
        for (let p = 0; p < 3; p++)
          for (let q = 0; q < 3; q++)
            AE[p * 3 + q] = JR[p * 3] * BA[q] + JR[p * 3 + 1] * BA[3 + q] +
                            JR[p * 3 + 2] * BA[6 + q];
        for (let p = 0; p < 3; p++)
          for (let q = 0; q < 3; q++)
            nA[c0 + p * 3 + q] = A[a0 + p * 3] * AE[q] + A[a0 + p * 3 + 1] * AE[3 + q] +
                                 A[a0 + p * 3 + 2] * AE[6 + q];
        // BRANCH[i].b is (0,0,H_ATT), so A b_i is A's third column times H_ATT
        const cx = A[a0 + 2] * H_ATT + bx;
        const cy = A[a0 + 5] * H_ATT + by;
        const cz = A[a0 + 8] * H_ATT + bz;
        nb[d0] = cx; nb[d0 + 1] = cy; nb[d0 + 2] = cz;
        nh[j * M_BR + i] = chh;
        ncb[j * M_BR + i] = ci + 1 + i * childSize;

        // where this subtree's mass is, in world space.  Stage 1 expands once
        // on the CPU so this point is fixed and its depth is <p, dir>; stage 2
        // regenerates per frame and uses <c_i, M^T dir>, the same number.
        const q0 = cr[0], q1 = cr[1], q2 = cr[2];
        keyPts[kp + 3 + i * 3] =
          nA[c0] * q0 + nA[c0 + 1] * q1 + nA[c0 + 2] * q2 + cx;
        keyPts[kp + 4 + i * 3] =
          nA[c0 + 3] * q0 + nA[c0 + 4] * q1 + nA[c0 + 5] * q2 + cy;
        keyPts[kp + 5 + i * 3] =
          nA[c0 + 6] * q0 + nA[c0 + 7] * q1 + nA[c0 + 8] * q2 + cz;
      }
    }

    levels.push({ n, nItems, keyPts, cbase: cb, childSize });
    if (r === 0) break;
    A = nA; b = nb; h = nh; cb = ncb; n = nn;
  }

  return { N, depth, pos, cov, rgba, level, levels, T };
}

const clamp255 = (x) => Math.max(0, Math.min(255, Math.round(x * 255)));

function shadeWood(rgba, i, d, depth, h) {
  const f = d / Math.max(depth, 1);
  const n = 1.0 + 0.06 * normal(h, 10);
  rgba[4 * i] = clamp255(((1 - f) * 0.26 + f * 0.47) * n);
  rgba[4 * i + 1] = clamp255(((1 - f) * 0.17 + f * 0.38) * n);
  rgba[4 * i + 2] = clamp255(((1 - f) * 0.10 + f * 0.28) * n);
  rgba[4 * i + 3] = clamp255(0.99);
}

function shadeLeaf(rgba, i, h) {
  const n = 1.0 + 0.22 * normal(h, 12);
  const g = 1.0 + 0.10 * normal(h, 14);
  rgba[4 * i] = clamp255(0.17 * n);
  rgba[4 * i + 1] = clamp255(0.41 * n * g);
  rgba[4 * i + 2] = clamp255(0.13 * n);
  rgba[4 * i + 3] = clamp255(0.55);
}

// -------------------------------------------------------- ordered expansion

/**
 * Write the back-to-front draw order for view direction `dir` into `out`.
 *
 * Per node: five dot products, a nine-compare-swap sorting network, a
 * five-element prefix sum.  Nothing N-sized is ever sorted -- the stream comes
 * out ordered because every level was written in order.  This is the whole
 * claim the project rests on, and it is these forty lines.
 *
 * `dir` points away from the camera, so larger <p, dir> is farther and is
 * drawn first.
 */
export function orderFrame(tree, dir, out, scratch) {
  const levels = tree.levels, depth = tree.depth;
  const dx = dir[0], dy = dir[1], dz = dir[2];
  let ob = scratch.a, nob = scratch.b;
  ob[0] = 0;

  for (let d = 0; d <= depth; d++) {
    const L = levels[d];
    const n = L.n, kpA = L.keyPts, cbA = L.cbase, cs = L.childSize;

    if (L.nItems === 2) {
      // the last level: wood rod against foliage disc, one comparison
      for (let j = 0; j < n; j++) {
        const q = j * 6, base = ob[j], ci = cbA[j];
        const kw = kpA[q] * dx + kpA[q + 1] * dy + kpA[q + 2] * dz;
        const kl = kpA[q + 3] * dx + kpA[q + 4] * dy + kpA[q + 5] * dz;
        if (kw >= kl) { out[base] = ci; out[base + 1] = ci + 1; }
        else { out[base] = ci + 1; out[base + 1] = ci; }
      }
      break;
    }

    for (let j = 0; j < n; j++) {
      const q = j * 15;
      let k0 = kpA[q] * dx + kpA[q + 1] * dy + kpA[q + 2] * dz;
      let k1 = kpA[q + 3] * dx + kpA[q + 4] * dy + kpA[q + 5] * dz;
      let k2 = kpA[q + 6] * dx + kpA[q + 7] * dy + kpA[q + 8] * dz;
      let k3 = kpA[q + 9] * dx + kpA[q + 10] * dy + kpA[q + 11] * dz;
      let k4 = kpA[q + 12] * dx + kpA[q + 13] * dy + kpA[q + 14] * dz;
      let i0 = 0, i1 = 1, i2 = 2, i3 = 3, i4 = 4, t, u;
      // the optimal 5-element network: (0,1)(3,4)(2,4)(2,3)(0,3)(0,2)(1,4)(1,3)(1,2)
      if (k0 < k1) { t = k0; k0 = k1; k1 = t; u = i0; i0 = i1; i1 = u; }
      if (k3 < k4) { t = k3; k3 = k4; k4 = t; u = i3; i3 = i4; i4 = u; }
      if (k2 < k4) { t = k2; k2 = k4; k4 = t; u = i2; i2 = i4; i4 = u; }
      if (k2 < k3) { t = k2; k2 = k3; k3 = t; u = i2; i2 = i3; i3 = u; }
      if (k0 < k3) { t = k0; k0 = k3; k3 = t; u = i0; i0 = i3; i3 = u; }
      if (k0 < k2) { t = k0; k0 = k2; k2 = t; u = i0; i0 = i2; i2 = u; }
      if (k1 < k4) { t = k1; k1 = k4; k4 = t; u = i1; i1 = i4; i4 = u; }
      if (k1 < k3) { t = k1; k1 = k3; k3 = t; u = i1; i1 = i3; i3 = u; }
      if (k1 < k2) { t = k1; k1 = k2; k2 = t; u = i1; i1 = i2; i2 = u; }

      const base = ob[j], ci = cbA[j], c4 = j * 4;
      let off = base;
      if (i0 === 0) { out[off] = ci; off += 1; } else { nob[c4 + i0 - 1] = off; off += cs; }
      if (i1 === 0) { out[off] = ci; off += 1; } else { nob[c4 + i1 - 1] = off; off += cs; }
      if (i2 === 0) { out[off] = ci; off += 1; } else { nob[c4 + i2 - 1] = off; off += cs; }
      if (i3 === 0) { out[off] = ci; off += 1; } else { nob[c4 + i3 - 1] = off; off += cs; }
      if (i4 === 0) { out[off] = ci; } else { nob[c4 + i4 - 1] = off; }
    }
    const sw = ob; ob = nob; nob = sw;
  }
  return out;
}

export function makeScratch(tree) {
  const maxNodes = Math.pow(M_BR, tree.depth);
  return { a: new Int32Array(maxNodes), b: new Int32Array(maxNodes) };
}
