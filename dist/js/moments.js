// Moment-matched parents, in closed form.  See py/moments.py for the
// derivation; this is the same arithmetic, and tests/moment_parity.mjs checks
// the two agree.
//
// A node that is too small on screen is drawn as ONE Gaussian standing for its
// whole infinite subtree.  Give the node's own rod a share `omega` of the
// subtree's mass and split the rest equally between the m children; then the
// subtree's mean and covariance, in the node's own frame, are the solution of
//
//   C = w mu_w + ((1-w)/m) SUM (A_i C + b_i)
//   Q = w (S_w + (mu_w-C)(mu_w-C)^T) + ((1-w)/m) SUM [A_i Q A_i^T + d_i d_i^T]
//
// a 3x3 solve and a contraction.  Computed once per map set, never per node.

export function solve3(M, v) {
  const a = [M[0], M[1], M[2], v[0], M[3], M[4], M[5], v[1], M[6], M[7], M[8], v[2]];
  for (let c = 0; c < 3; c++) {
    let p = c;
    for (let r = c + 1; r < 3; r++) if (Math.abs(a[r * 4 + c]) > Math.abs(a[p * 4 + c])) p = r;
    if (p !== c) for (let k = 0; k < 4; k++) {
      const t = a[c * 4 + k]; a[c * 4 + k] = a[p * 4 + k]; a[p * 4 + k] = t;
    }
    const d = a[c * 4 + c];
    for (let k = c; k < 4; k++) a[c * 4 + k] /= d;
    for (let r = 0; r < 3; r++) {
      if (r === c) continue;
      const f = a[r * 4 + c];
      if (f === 0) continue;
      for (let k = c; k < 4; k++) a[r * 4 + k] -= f * a[c * 4 + k];
    }
  }
  return [a[3], a[7], a[11]];
}

const mm = (A, B) => {
  const o = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    o[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
  return o;
};
const mT = (A) => [A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]];
const mvv = (A, v) => [A[0] * v[0] + A[1] * v[1] + A[2] * v[2],
                       A[3] * v[0] + A[4] * v[1] + A[5] * v[2],
                       A[6] * v[0] + A[7] * v[1] + A[8] * v[2]];

/** Largest singular value of a 3x3, by power iteration on A^T A. */
const _G = new Float64Array(9);

/**
 * Largest singular value of a 3x3, by power iteration on A^T A.
 *
 * It stops when the estimate stops moving rather than running a fixed 64
 * sweeps: the ratio of the top two eigenvalues decides how fast that is, and
 * for the maps here it is a dozen sweeps, not sixty-four.  Called five times
 * per reparam, and reparam runs every few frames while an attractor morphs.
 */
export function opNorm(A) {
  const G = _G;
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++)
      G[i * 3 + j] = A[i] * A[j] + A[3 + i] * A[3 + j] + A[6 + i] * A[6 + j];
  let vx = 1, vy = 1, vz = 1, l = 0;
  for (let k = 0; k < 64; k++) {
    const wx = G[0] * vx + G[1] * vy + G[2] * vz;
    const wy = G[3] * vx + G[4] * vy + G[5] * vz;
    const wz = G[6] * vx + G[7] * vy + G[8] * vz;
    const n = Math.sqrt(wx * wx + wy * wy + wz * wz);
    if (n < 1e-300) return 0;
    vx = wx / n; vy = wy / n; vz = wz / n;
    if (k && Math.abs(n - l) < 1e-14 * n) { l = n; break; }
    l = n;
  }
  return Math.sqrt(l);
}

/**
 * maps: [{A:[9], b:[3]}], rod: {A:[9] (the trunk map), sig:[3] diag of SIG0,
 * mu:[3]}.  Returns {C, Q, R, omega, s[]}.
 */
const _AQ = new Float64Array(9);

export function invariantMoments(maps, rodA, sig0, mu0, omega = 0.28) {
  const m = maps.length, p = (1 - omega) / m;

  const muW = mvv(rodA, mu0);
  const SW = new Float64Array(9);                 // rodA diag(sig0) rodA^T
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    SW[i * 3 + j] = rodA[i * 3] * rodA[j * 3] * sig0[0] +
                    rodA[i * 3 + 1] * rodA[j * 3 + 1] * sig0[1] +
                    rodA[i * 3 + 2] * rodA[j * 3 + 2] * sig0[2];

  // ---- mean
  const Asum = new Float64Array(9), bsum = [0, 0, 0];
  for (const mp of maps) {
    for (let k = 0; k < 9; k++) Asum[k] += mp.A[k];
    for (let k = 0; k < 3; k++) bsum[k] += mp.b[k];
  }
  const Mx = new Float64Array(9);
  for (let k = 0; k < 9; k++) Mx[k] = (k % 4 === 0 ? 1 : 0) - p * Asum[k];
  const rhs = [0, 1, 2].map((k) => omega * muW[k] + p * bsum[k]);
  const C = solve3(Mx, rhs);

  // ---- covariance: iterate the contraction
  const d = maps.map((mp) => {
    const q = mvv(mp.A, C);
    return [q[0] + mp.b[0] - C[0], q[1] + mp.b[1] - C[1], q[2] + mp.b[2] - C[2]];
  });
  const base = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) {
    base[i * 3 + j] = omega * (SW[i * 3 + j] + (muW[i] - C[i]) * (muW[j] - C[j]));
    for (const x of d) base[i * 3 + j] += p * x[i] * x[j];
  }
  // Q = base + p * sum_i A_i Q A_i^T, iterated.  The contraction factor is
  // max s_i^2, so this converges geometrically and stops when it stops moving.
  //
  // Allocation-free on purpose: stage 3 morphs the maps every frame, so this
  // runs every frame, and the obvious version -- a fresh 3x3 out of mm() for
  // each map on each sweep -- spends more time in the allocator than in the
  // arithmetic.
  let Q = Float64Array.from(base);
  let Qn = new Float64Array(9);
  const AQ = _AQ;
  for (let it = 0; it < 400; it++) {
    Qn.set(base);
    for (let mi = 0; mi < m; mi++) {
      const A = maps[mi].A;
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++)
          AQ[i * 3 + j] = A[i * 3] * Q[j] + A[i * 3 + 1] * Q[3 + j] + A[i * 3 + 2] * Q[6 + j];
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++)                       // (AQ) A^T
          Qn[i * 3 + j] += p * (AQ[i * 3] * A[j * 3] + AQ[i * 3 + 1] * A[j * 3 + 1]
                              + AQ[i * 3 + 2] * A[j * 3 + 2]);
    }
    let mx = 0;
    for (let k = 0; k < 9; k++) {
      const e = Math.abs(Qn[k] - Q[k]);
      if (e > mx) mx = e;
    }
    const t = Q; Q = Qn; Qn = t;
    if (mx < 1e-13) break;
  }

  // ---- bounding sphere about C, covering every splat's 3-sigma ellipsoid
  const rodR = 3 * Math.sqrt(opNorm(SW)) + Math.hypot(muW[0] - C[0], muW[1] - C[1], muW[2] - C[2]);
  const s = maps.map((mp) => opNorm(mp.A));
  let R = rodR;
  for (let it = 0; it < 500; it++) {
    let Rn = rodR;
    for (let i = 0; i < m; i++)
      Rn = Math.max(Rn, Math.hypot(d[i][0], d[i][1], d[i][2]) + s[i] * R);
    if (Math.abs(Rn - R) < 1e-15 * Math.max(Rn, 1e-30)) { R = Rn; break; }
    R = Rn;
  }

  return { C, Q, R, omega, s, muW, SW, rodSig: [SW[0], SW[4], SW[8]] };
}

/** Average colour of a whole subtree, when the palette cycles every P levels.
 *  A P-dimensional fixed point: own rod at weight omega, the level below at
 *  (1-omega). */
/**
 * A node's colour is a fixed point too: it shows its own level's colour with
 * weight omega and its children's -- which are the next level's -- with the
 * rest.  Solving that around the cycle gives the colour a blob should be.
 *
 * Flat and allocation-free for the same reason as the covariance above: this
 * is on the morph path.  1e-9 is already four orders past what eight-bit
 * colour can hold.
 */
export function subtreeColours(palette, omega) {
  const P = palette.length, w = 1 - omega;
  const pal = new Float64Array(3 * P);
  for (let k = 0; k < P; k++)
    for (let j = 0; j < 3; j++) pal[3 * k + j] = omega * palette[k][j];
  let cur = new Float64Array(3 * P), nxt = new Float64Array(3 * P);
  for (let k = 0; k < 3 * P; k++) cur[k] = pal[k];
  for (let it = 0; it < 600; it++) {
    let mx = 0;
    for (let k = 0; k < P; k++) {
      const n3 = 3 * (((k + 1) % P));
      for (let j = 0; j < 3; j++) {
        const v = pal[3 * k + j] + w * cur[n3 + j];
        const e = Math.abs(v - cur[3 * k + j]);
        if (e > mx) mx = e;
        nxt[3 * k + j] = v;
      }
    }
    const t = cur; cur = nxt; nxt = t;
    if (mx < 1e-9) break;
  }
  const out = [];
  for (let k = 0; k < P; k++) out.push([cur[3 * k], cur[3 * k + 1], cur[3 * k + 2]]);
  return out;
}
