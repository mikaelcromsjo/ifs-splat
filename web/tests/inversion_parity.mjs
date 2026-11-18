// The live counter in the proof panel has to agree with the offline one that
// produced the stage-0 verdict, or the panel is showing a different number
// from the one the claim rests on.
import { writeFileSync } from 'node:fs';
import { expand, orderFrame, makeScratch } from '../js/ifs_core.js';
import { ExactSorter } from '../js/sort_exact.js';
import { InversionCounter } from '../js/inversions.js';

const depth = Number(process.argv[2] ?? 6);
const tree = expand(depth, 0.16, 3);
const N = tree.N;
const dirs = [[0.3, 0.8, 0.52], [-0.6, 0.2, 0.77], [0.1, -0.95, 0.3],
              [0.57, 0.57, 0.59], [-0.2, -0.3, 0.93]];
const scratch = makeScratch(tree);
const ord = new Int32Array(N), ex = new Int32Array(N);
const sorter = new ExactSorter(N);
const alpha = new Float64Array(N);
for (let i = 0; i < N; i++) alpha[i] = tree.rgba[4 * i + 3] / 255;
const ic = new InversionCounter(N);

const rows = [];
for (const d of dirs) {
  const L = Math.hypot(...d), f = d.map((v) => v / L);
  orderFrame(tree, f, ord, scratch);
  sorter.sort(tree.pos, f, ex);
  const full = ic.measure(ord, ex, alpha, N);
  const s4 = ic.sampled(ord, ex, alpha, N, 4);
  const s16 = ic.sampled(ord, ex, alpha, N, 16);
  rows.push({ f, bad: full.bad, total: full.total, rate: full.rate,
              s4: s4.rate, s16: s16.rate });
  console.log(`  dir [${f.map((v) => v.toFixed(2)).join(' ')}]  rate ${(full.rate * 100).toFixed(3)}%`
    + `   1/4 sample ${(s4.rate * 100).toFixed(3)}%   1/16 sample ${(s16.rate * 100).toFixed(3)}%`);
}
writeFileSync('/tmp/js_inv.json', JSON.stringify({ depth, N, rows }));
console.log(`\n  ${N} splats, depth ${depth} -> /tmp/js_inv.json`);
