// Does the compute kernel produce the same world, and the same draw order, as
// the CPU expansion that stage 0 and stage 1 were verified against?
//
// No WebGPU runs here, so this checks the kernel's ARITHMETIC through its JS
// transcription.  That is the same guarantee stage 1's vertex shader has and
// it has the same hole in it: it says nothing about the bindings, the buffer
// layouts or whether the WGSL compiles.

import { expand, orderFrame, makeScratch, BRANCH, subtreeSizes,
         subtreeCentroids } from '../js/ifs_core.js';
import { thread, setBranch, discCovClosed } from '../js/gpu_ref.js';

const depth = Number(process.argv[2] ?? 5);
const jitter = 0.16, seed = 3;
const tree = expand(depth, jitter, seed);
const N = tree.N;
setBranch(BRANCH);
const sizes = Array.from(subtreeSizes(depth));
const cents = subtreeCentroids(depth);

const dirs = [[0.3, 0.8, 0.52], [-0.6, 0.2, 0.77], [0.57, 0.57, 0.59]];
const scratch = makeScratch(tree);
const ord = new Int32Array(N);

let worstPos = 0, worstCov = 0, worstCol = 0, slotBad = 0, idxBad = 0, discWorst = 0;
for (const raw of dirs) {
  const L = Math.hypot(...raw), dir = raw.map((v) => v / L);
  orderFrame(tree, dir, ord, scratch);
  const slotOf = new Int32Array(N);
  for (let s = 0; s < N; s++) slotOf[ord[s]] = s;

  for (let t = 0; t < N; t++) {
    const g = thread(t, { depth, jitter, seed, dir, sizes, cents });
    if (g.idx !== t) { idxBad++; continue; }
    if (g.slot !== slotOf[t]) slotBad++;
    for (let c = 0; c < 3; c++)
      worstPos = Math.max(worstPos, Math.abs(g.pos[c] - tree.pos[3 * t + c]));
    for (let c = 0; c < 6; c++) {
      const want = tree.cov[6 * t + c];
      const scale = Math.max(Math.abs(want), 1e-9);
      worstCov = Math.max(worstCov, Math.abs(g.cov[c] - want) / scale);
    }
    for (let c = 0; c < 4; c++)
      worstCol = Math.max(worstCol, Math.abs(g.col[c] - tree.rgba[4 * t + c]));
  }
}

// and the closed-form disc against the eigensolver it replaces
{
  const T = subtreeSizes(depth);
  for (let t = 0; t < N; t++) {
    if (tree.level[t] !== depth) continue;
    const g = thread(t, { depth, jitter, seed, dir: [0, 0, 1], sizes, cents });
    if (g.idx !== t) continue;
    for (let c = 0; c < 6; c++) {
      const want = tree.cov[6 * t + c];
      discWorst = Math.max(discWorst, Math.abs(g.cov[c] - want)
        / Math.max(Math.abs(want), 1e-12));
    }
  }
}

console.log(`depth ${depth}, ${N} splats, ${dirs.length} view directions\n`);
console.log(`  splat index    ${idxBad} wrong`);
console.log(`  draw slot      ${slotBad} wrong out of ${N * dirs.length}`);
console.log(`  position       worst absolute error ${worstPos.toExponential(2)}`);
console.log(`  covariance     worst relative error ${worstCov.toExponential(2)}`);
console.log(`  colour         worst difference ${worstCol} of 255`);
console.log(`  disc, closed form vs eigensolve   ${discWorst.toExponential(2)} relative`);
// The draw order and the colours must match EXACTLY -- they are integers and
// a single wrong slot is a splat drawn in the wrong place.  The geometry
// cannot: ifs_core stores the tree in Float32Array while this reference works
// in float64, so the comparison is against a value that has been rounded to
// float32 once.  One float32 ulp is 1.2e-7 relative, which is what both
// columns show, so the tolerance is that and not zero.
const EPS32 = 1.2e-7;
const ok = idxBad === 0 && slotBad === 0 && worstCol === 0
        && worstPos < 4 * EPS32 && worstCov < 4 * EPS32 && discWorst < 4 * EPS32;
console.log(`\n  ${ok ? 'KERNEL MATCHES THE CPU EXPANSION' : 'MISMATCH'}`);
process.exit(ok ? 0 : 1);
