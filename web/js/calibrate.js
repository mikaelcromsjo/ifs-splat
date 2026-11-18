// What opacity should a moment-matched blob have?
//
// The blob's position and shape come from the maps in closed form -- mass,
// mean, covariance are a fixed point, and that part is exact.  Its OPACITY is
// not determined by those moments: a Gaussian with the right second moment can
// still be far too solid, and 0.95 (the obvious guess) turns a sparse branching
// subtree into an opaque lump.  Flying through one looks like fog.
//
// So measure it.  A node's actual matter is its rods -- blobs are only a
// stand-in for rods too small to see -- so:
//
//   1. render one node's rods, recursively, down to where they are sub-pixel;
//   2. render the single blob that would replace them, at opacity 1;
//   3. take the ratio of the light each blocks.
//
// That is the opacity at which cutting a subtree costs the image nothing,
// and it is a property of the map set, computed once.

import { render } from './cpu_raster.js';

/** Expand a node to `depth` levels, emitting rods only, into flat arrays. */
function rodsOnly(att, depth) {
  const T = [];
  for (let r = 0, n = 1; r <= depth; r++, n *= att.m) T.push(n);
  let total = 0;
  for (const n of T) total += n;
  const pos = new Float32Array(3 * total);
  const cov = new Float32Array(6 * total);
  const rgba = new Uint8Array(4 * total);
  let w = 0;
  const tA = att.trunkA, mu = att.mu0, S = att.sig0;

  const walk = (A, b, h, level) => {
    // the node's rod
    const AW = new Float64Array(9);
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        AW[i * 3 + j] = A[i * 3] * tA[j] + A[i * 3 + 1] * tA[3 + j] + A[i * 3 + 2] * tA[6 + j];
    const m = [AW[0] * mu[0] + AW[1] * mu[1] + AW[2] * mu[2] + b[0],
               AW[3] * mu[0] + AW[4] * mu[1] + AW[5] * mu[2] + b[1],
               AW[6] * mu[0] + AW[7] * mu[1] + AW[8] * mu[2] + b[2]];
    pos[3 * w] = m[0]; pos[3 * w + 1] = m[1]; pos[3 * w + 2] = m[2];
    const o = 6 * w, idx = [[0, 0], [0, 1], [0, 2], [1, 1], [1, 2], [2, 2]];
    for (let k = 0; k < 6; k++) {
      const [p, q] = idx[k];
      cov[o + k] = AW[p * 3] * AW[q * 3] * S[0] + AW[p * 3 + 1] * AW[q * 3 + 1] * S[1]
                 + AW[p * 3 + 2] * AW[q * 3 + 2] * S[2];
    }
    rgba[4 * w] = 255; rgba[4 * w + 1] = 255; rgba[4 * w + 2] = 255;
    rgba[4 * w + 3] = Math.round(att.alphaRod * 255);
    w++;
    if (level === depth) return;
    const CM = att.childMaps, MB = att.mapB;
    for (let i = 0; i < att.m; i++) {
      const chh = att.mixHash(h, i);
      const e = 9 * ((chh & att.tableMask) * att.m + i);
      const cA = new Float64Array(9);
      for (let p = 0; p < 3; p++)
        for (let q = 0; q < 3; q++)
          cA[p * 3 + q] = A[p * 3] * CM[e + q] + A[p * 3 + 1] * CM[e + 3 + q] +
                          A[p * 3 + 2] * CM[e + 6 + q];
      const B = [MB[3 * i], MB[3 * i + 1], MB[3 * i + 2]];
      const cb = [A[0] * B[0] + A[1] * B[1] + A[2] * B[2] + b[0],
                  A[3] * B[0] + A[4] * B[1] + A[5] * B[2] + b[1],
                  A[6] * B[0] + A[7] * B[1] + A[8] * B[2] + b[2]];
      walk(cA, cb, chh, level + 1);
    }
  };
  walk(new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]), [0, 0, 0], att.rootHash, 0);
  return { pos, cov, rgba, count: w };
}

/** Light blocked by an image rendered white-on-black: sum of luminance. */
function blocked(im) {
  let s = 0;
  for (let i = 0; i < im.img.length; i += 3) s += im.img[i];
  return s / 255;
}

/**
 * -> the opacity at which one blob blocks as much light as the subtree it
 * replaces.  `depth` only has to be deep enough that the next level is
 * sub-pixel; 7 is past that at this resolution.
 */
export function calibrateBlob(att, { depth = 7, px = 192, dist = 2.6 } = {}) {
  const sub = rodsOnly(att, depth);
  const W = px, H = px;
  const C = att.C;
  const eye = [C[0], C[1] - dist * att.R, C[2]];
  const B = { r: [1, 0, 0], u: [0, 0, 1], f: [0, 1, 0] };
  const opts = { eye, B, W, H, dilate: true, gamma: 1.0, bg: [0, 0, 0] };

  const truth = render({ ...sub, off: 0, ...opts });

  const blob = {
    pos: new Float32Array([C[0], C[1], C[2]]),
    cov: new Float32Array([att.Q[0], att.Q[1], att.Q[2], att.Q[4], att.Q[5], att.Q[8]]),
    rgba: new Uint8Array([255, 255, 255, 255]),
    count: 1, off: 0,
  };
  const one = render({ ...blob, ...opts });

  const a = blocked(truth) / Math.max(blocked(one), 1e-9);
  return { alpha: Math.max(0.02, Math.min(0.98, a)), raw: a,
           rods: sub.count, truth: blocked(truth), blob: blocked(one) };
}
