// JS vs Python parity.  Dumps the JS tree as JSON for py/parity_check.py to diff.
import { expand, orderFrame, makeScratch, h32, mix, stream } from '../js/ifs_core.js';
import { writeFileSync } from 'node:fs';

const depth = Number(process.argv[2] ?? 6);
const jitter = Number(process.argv[3] ?? 0.16);
const seed = Number(process.argv[4] ?? 3);

const t0 = Date.now();
const tree = expand(depth, jitter, seed);
const tgen = Date.now() - t0;

const dirs = [];
for (let i = 0; i < 8; i++) {
  const u = (i + 0.5) / 8;
  const phi = Math.acos(1 - 2 * u), th = Math.PI * (1 + Math.sqrt(5)) * (i + 0.5);
  dirs.push([Math.sin(phi) * Math.cos(th), Math.sin(phi) * Math.sin(th), Math.cos(phi)]);
}
const scratch = makeScratch(tree);
const out = new Int32Array(tree.N);
const orders = [];
const t1 = Date.now();
for (const d of dirs) { orderFrame(tree, d, out, scratch); orders.push(Array.from(out)); }
const tord = (Date.now() - t1) / dirs.length;

writeFileSync(process.argv[5] ?? '/tmp/js_tree.json', JSON.stringify({
  depth, jitter, seed, N: tree.N,
  pos: Array.from(tree.pos), cov: Array.from(tree.cov),
  rgba: Array.from(tree.rgba), level: Array.from(tree.level),
  dirs, orders,
  levels: tree.levels.map((L) => ({ n: L.n, nItems: L.nItems,
                                    childSize: L.childSize,
                                    keyPts: Array.from(L.keyPts),
                                    cbase: Array.from(L.cbase) })),
  hashes: [h32(3), mix(h32(3), 2), stream(h32(3), 5)],
}));
console.error(`js: ${tree.N} splats, generate ${tgen} ms, order ${tord.toFixed(2)} ms/frame`);
