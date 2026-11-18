import { expand, orderFrame, makeScratch, M_BR } from '../js/ifs_core.js';
const depth = Number(process.argv[2] ?? 8);
let t = Date.now();
const tree = expand(depth, 0.16, 3);
console.log(`generate ${tree.N} splats: ${Date.now() - t} ms`);
const sc = makeScratch(tree), out = new Int32Array(tree.N);
const dirs = [];
for (let i = 0; i < 60; i++) {
  const a = (i / 60) * 6.28318;
  dirs.push([Math.cos(a), Math.sin(a), -0.3]);
}
for (let k = 0; k < 3; k++) for (const d of dirs) orderFrame(tree, d, out, sc);   // warm
t = Date.now();
for (let r = 0; r < 5; r++) for (const d of dirs) orderFrame(tree, d, out, sc);
console.log(`order: ${((Date.now() - t) / (5 * dirs.length)).toFixed(2)} ms/frame`);
console.log('nodes per level:', tree.levels.map((L) => L.n).join(' '));
// per-level cost
for (let d = 0; d <= depth; d++) {
  const sub = { levels: tree.levels.slice(0, d + 1), T: tree.T, depth: d };
  // not a valid tree; just time the cumulative loop by truncating
}
