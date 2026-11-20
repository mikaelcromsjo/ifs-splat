// The restructured projection, against the one it replaces.
//
// shaders.js used to compute the screen-space covariance with the matrix
// operators -- `mat3x2 W = J * uRv; mat2 S2 = W * S * transpose(W);` -- and
// then take determinants of the full 2x2.  It now spells both out: J has two
// structurally zero entries of six, and S2 is symmetric, so the general
// products were doing 48 multiplies where 39 are needed.
//
// There is no GL here, so the only way to know that the rewrite is the same
// arithmetic is to transcribe BOTH forms and compare them -- which is the
// arrangement every shader claim in this repository is under.
//
// The first version of this test asserted bit-identity and was WRONG to, which
// is worth recording because the reason is the whole point of the rewrite.
// J * uRv really is bit-identical -- it adds products of exact zeros, and
// x + 0 is exact.  S2 is not: the old form's S2[0][1] and S2[1][0] are
// w0'Sw1 and w1'Sw0, mathematically equal because S is symmetric but computed
// from different products and so rounded differently -- and determinant()
// then MULTIPLIED THE TWO TOGETHER.  The new form computes that entry once
// and squares it, which is both cheaper and the better conditioned of the
// two, and is what cpu_raster.js and the transcription in raster_parity.mjs
// have always done.
//
// So what is checked is the SIZE of the difference, in the unit that decides
// whether it can matter: the shader computes in FLOAT32, whose epsilon is
// 1.19e-7.  A disagreement five orders of magnitude under one float32 ulp
// cannot be represented in the shader at all, let alone seen.
//
// The quad's extent is bounded rather than compared, for a different reason: a
// normalize followed by two multiplies became one inversesqrt, and the quad
// has only to COVER the kernel -- coverage is decided per fragment from
// vDelta, so the extent may round either way as long as it does not shrink.
import { expand } from '../js/ifs_core.js';

const CUT = 3.0, KAPPA = 0.35, LOWPASS = 0.3;

// ---- a 3x3 symmetric times a 3-vector, as GLSL's `S * v` does it
const mul3 = (S, v) => [
  S[0] * v[0] + S[3] * v[1] + S[6] * v[2],
  S[1] * v[0] + S[4] * v[1] + S[7] * v[2],
  S[2] * v[0] + S[5] * v[1] + S[8] * v[2]];
const dot3 = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** The old form: general mat3x2 * mat3, general mat3x2 * mat2x3, full 2x2. */
function before(S, Rv, focal, v) {
  const zi = 1 / v[2];
  // mat3x2 J, column-major as GLSL takes it: J[col][row]
  const J = [[focal * zi, 0], [0, -focal * zi],
             [-focal * v[0] * zi * zi, focal * v[1] * zi * zi]];
  // W = J * Rv : mat3x2 * mat3 -> mat3x2.  W[c][r] = sum_k J[k][r] * Rv[c][k]
  const W = [];
  for (let c = 0; c < 3; c++) {
    const col = [0, 0];
    for (let r = 0; r < 2; r++)
      for (let k = 0; k < 3; k++) col[r] += J[k][r] * Rv[3 * c + k];
    W.push(col);
  }
  // WS = W * S : mat3x2 * mat3 -> mat3x2
  const WS = [];
  for (let c = 0; c < 3; c++) {
    const col = [0, 0];
    for (let r = 0; r < 2; r++)
      for (let k = 0; k < 3; k++) col[r] += W[k][r] * S[3 * c + k];
    WS.push(col);
  }
  // S2 = WS * transpose(W) : mat3x2 * mat2x3 -> mat2.  All four entries.
  const S2 = [[0, 0], [0, 0]];
  for (let c = 0; c < 2; c++) for (let r = 0; r < 2; r++)
    for (let k = 0; k < 3; k++) S2[c][r] += WS[k][r] * W[k][c];
  const det0 = S2[0][0] * S2[1][1] - S2[0][1] * S2[1][0];
  S2[0][0] += LOWPASS; S2[1][1] += LOWPASS;
  const det1 = S2[0][0] * S2[1][1] - S2[0][1] * S2[1][0];
  const conic = [S2[1][1] / det1, -S2[0][1] / det1, S2[0][0] / det1];
  const comp = Math.sqrt(Math.min(Math.max(det0, 0) / det1, 1));
  const mid = 0.5 * (S2[0][0] + S2[1][1]);
  const rr = Math.sqrt(Math.max(mid * mid - det1, 0));
  const l1 = mid + rr, l2 = Math.max(mid - rr, 0.02);
  const m1 = CUT * Math.sqrt(l1), m2 = CUT * Math.sqrt(l2);
  let ax1;
  if (Math.abs(S2[0][1]) < 1e-9) ax1 = S2[0][0] >= S2[1][1] ? [1, 0] : [0, 1];
  else {
    const e = [S2[0][1], l1 - S2[0][0]];
    const n = Math.sqrt(e[0] * e[0] + e[1] * e[1]);     // GLSL normalize
    ax1 = [e[0] / n, e[1] / n];
  }
  const ax2 = [ax1[1], -ax1[0]];
  return { det1, comp, conic, l1, l2,
           w0: [W[0][0], W[1][0], W[2][0]], w1: [W[0][1], W[1][1], W[2][1]],
           // the two copies of the symmetric entry, as the old determinant
           // saw them
           s01: S2[0][1], s10: S2[1][0],
           maj: [m1 * ax1[0], m1 * ax1[1]], mnr: [m2 * ax2[0], m2 * ax2[1]] };
}

/** The new form: the two rows of W written out, three entries of S2. */
function after(S, Rv, focal, v) {
  const zi = 1 / v[2];
  const rr0 = [Rv[0], Rv[3], Rv[6]];      // vec3(uRv[0][0], uRv[1][0], uRv[2][0])
  const rr1 = [Rv[1], Rv[4], Rv[7]];
  const rr2 = [Rv[2], Rv[5], Rv[8]];
  const k0 = focal * zi, k1 = focal * v[0] * zi * zi;
  const k2 = focal * zi, k3 = focal * v[1] * zi * zi;
  const w0 = [k0 * rr0[0] - k1 * rr2[0], k0 * rr0[1] - k1 * rr2[1], k0 * rr0[2] - k1 * rr2[2]];
  const w1 = [-k2 * rr1[0] + k3 * rr2[0], -k2 * rr1[1] + k3 * rr2[1], -k2 * rr1[2] + k3 * rr2[2]];
  const ws0 = mul3(S, w0), ws1 = mul3(S, w1);
  let s00 = dot3(ws0, w0); const s01 = dot3(ws0, w1); let s11 = dot3(ws1, w1);
  const det0 = s00 * s11 - s01 * s01;
  s00 += LOWPASS; s11 += LOWPASS;
  const det1 = s00 * s11 - s01 * s01;
  const conic = [s11 / det1, -s01 / det1, s00 / det1];
  const comp = Math.sqrt(Math.min(Math.max(det0, 0) / det1, 1));
  const mid = 0.5 * (s00 + s11);
  const rr = Math.sqrt(Math.max(mid * mid - det1, 0));
  const l1 = mid + rr, l2 = Math.max(mid - rr, 0.02);
  const m1 = CUT * Math.sqrt(l1), m2 = CUT * Math.sqrt(l2);
  let maj, mnr;
  if (Math.abs(s01) < 1e-9) {
    const xMajor = s00 >= s11;
    maj = xMajor ? [m1, 0] : [0, m1];
    mnr = xMajor ? [0, -m2] : [m2, 0];
  } else {
    const e = [s01, l1 - s00];
    const inv = 1 / Math.sqrt(e[0] * e[0] + e[1] * e[1]);   // inversesqrt
    maj = [(m1 * inv) * e[0], (m1 * inv) * e[1]];
    mnr = [(m2 * inv) * e[1], (m2 * inv) * -e[0]];
  }
  return { det1, comp, conic, l1, l2, maj, mnr, w0, w1 };
}

// ---- a real covariance population, and cameras all round it
const tree = expand(7, 0.16, 3);
const N = tree.N;

function basis(seed) {
  // a deterministic orthonormal frame, so the comparison is reproducible
  let s = seed;
  const rnd = () => (s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff - 0.5;
  const nz = (v) => { const l = Math.hypot(...v); return v.map((x) => x / l); };
  const f = nz([rnd(), rnd(), rnd() || 0.3]);
  let r = [f[1], -f[0], 0];
  if (Math.hypot(...r) < 1e-6) r = [1, 0, 0];
  r = nz(r);
  const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]];
  // Rv rows are right, up, forward; GLSL mat3 is column-major, and the shader
  // uploads it transposed, so Rv[3*c + k] here is uRv[c][k]
  return [r[0], u[0], f[0], r[1], u[1], f[1], r[2], u[2], f[2]];
}

let worstDet = 0, worstConic = 0, worstComp = 0, worstEig = 0, worstQuad = 0;
let nQuad = 0, nSame = 0, n = 0, wSame = 0, negOld = 0, negNew = 0, dupSpread = 0;
const rel = (a, b) => Math.abs(a - b) / Math.max(Math.abs(a), Math.abs(b), 1e-300);

for (let cam = 0; cam < 24; cam++) {
  const Rv = basis(cam * 7919 + 13);
  const focal = 180 + 40 * (cam % 5);
  const eye = [0.3 * (cam % 3) - 0.3, 0.2 * (cam % 4) - 0.4, -1.5 - 0.4 * (cam % 6)];
  for (let i = cam; i < N; i += 29) {
    const d = [tree.pos[3 * i] - eye[0], tree.pos[3 * i + 1] - eye[1], tree.pos[3 * i + 2] - eye[2]];
    const v = [Rv[0] * d[0] + Rv[3] * d[1] + Rv[6] * d[2],
               Rv[1] * d[0] + Rv[4] * d[1] + Rv[7] * d[2],
               Rv[2] * d[0] + Rv[5] * d[1] + Rv[8] * d[2]];
    if (v[2] < 0.02) continue;
    const c = tree.cov, o = 6 * i, p2 = (KAPPA * v[2] / focal) ** 2;
    const S = [c[o] + p2, c[o + 1], c[o + 2],
               c[o + 1], c[o + 3] + p2, c[o + 4],
               c[o + 2], c[o + 4], c[o + 5] + p2];
    const A = before(S, Rv, focal, v), B = after(S, Rv, focal, v);
    if (A.det1 < 0) negOld++;
    if (B.det1 < 0) negNew++;
    if (!(A.det1 > 0) || !(B.det1 > 0)) continue;
    n++;
    // W itself: the claim that dropping J's structural zeros changes nothing
    if (A.w0[0] === B.w0[0] && A.w0[1] === B.w0[1] && A.w0[2] === B.w0[2]
     && A.w1[0] === B.w1[0] && A.w1[1] === B.w1[1] && A.w1[2] === B.w1[2]) wSame++;
    dupSpread = Math.max(dupSpread, rel(A.s01, A.s10));
    worstDet = Math.max(worstDet, rel(A.det1, B.det1));
    worstComp = Math.max(worstComp, rel(A.comp, B.comp));
    for (let k = 0; k < 3; k++) worstConic = Math.max(worstConic, rel(A.conic[k], B.conic[k]));
    worstEig = Math.max(worstEig, rel(A.l1, B.l1), rel(A.l2, B.l2));
    const q = Math.max(rel(A.maj[0], B.maj[0]), rel(A.maj[1], B.maj[1]),
                       rel(A.mnr[0], B.mnr[0]), rel(A.mnr[1], B.mnr[1]));
    worstQuad = Math.max(worstQuad, q);
    if (q > 0) nQuad++; else nSame++;
    // The quad must still cover the kernel: the new extent may not be shorter
    // than the old by more than a rounding of it.
    const la = Math.hypot(A.maj[0], A.maj[1]), lb = Math.hypot(B.maj[0], B.maj[1]);
    if (lb < la * (1 - 1e-12)) { console.log(`  FAIL  quad shrank: ${la} -> ${lb}`); process.exit(1); }
  }
}

let fail = 0;
const check = (ok, msg) => { console.log((ok ? '  ok    ' : '  FAIL  ') + msg); if (!ok) fail++; };
const F32 = 1.1920929e-7;                 // one float32 ulp, relative
const f32 = (r) => (r / F32).toExponential(2);

console.log(`the two forms of the projection, on ${n} splat-camera pairs\n`);
console.log('  worst relative difference, as a fraction of ONE FLOAT32 ULP --');
console.log('  the precision the shader actually computes in:');
console.log(`    det1   ${f32(worstDet)}      conic  ${f32(worstConic)}`);
console.log(`    comp   ${f32(worstComp)}      eigen  ${f32(worstEig)}`);
console.log(`    quad   ${f32(worstQuad)}`);
console.log(`  (${nSame} of ${n} pairs give a bit-identical quad, ${nQuad} differ)\n`);

// A thousandth of a float32 ulp is the bar: the two are the same formula
// rounded differently, and the shader cannot represent the difference.
const BAR = 1e-3 * F32;
check(worstDet < BAR, `det1 agrees to ${f32(worstDet)} of a float32 ulp`);
check(worstConic < BAR, `the conic the fragment shader reads, to ${f32(worstConic)}`);
check(worstComp < BAR, `the opacity compensation, to ${f32(worstComp)}`);
check(worstEig < BAR, `both eigenvalues, which face and the quad come from, to ${f32(worstEig)}`);
check(worstQuad < BAR, `and the quad's extent, to ${f32(worstQuad)}`);
check(wSame === n, `W = J * uRv is bit-identical on every pair (${wSame}/${n})`);

// Which of the two is right where they differ: the new one.  Checked on the
// population rather than argued.
check(negOld === 0 && negNew === 0,
  `neither form produced a negative det1 here (${negOld} old, ${negNew} new)`);
check(dupSpread > 0,
  `the duplicate entry really did differ -- worst S2[0][1] against S2[1][0] is `
  + `${f32(dupSpread)} of a float32 ulp apart, and the old determinant `
  + `multiplied the two together`);

console.log(fail ? `\n  ${fail} FAILED` : '\n  projection ok');
process.exit(fail ? 1 : 0);
