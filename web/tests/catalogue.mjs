// Every map set, and a walk along the spline between them, as one picture.
//
// The point of stage 5's library is that the world should stop looking like a
// wood.  That is a claim about appearance, so it is settled by looking.
//
//   node tests/catalogue.mjs

import { writeFileSync, mkdirSync } from 'node:fs';
import { SETS, SET_NAMES, morphPath } from '../js/maps.js';
import { shoot } from './shot.mjs';
import { png, tile } from '../js/cpu_raster.js';

const W = 240, H = 200, TARGET = 70000;
const SPECS = SET_NAMES.map((n) => SETS[n]);

console.log(`${SET_NAMES.length} sets, in loop order\n`);
const sets = [];
for (const n of SET_NAMES) {
  const s = SETS[n];
  sets.push(shoot(s, { W, H, target: TARGET, fog: false }).im);
  console.log(`  ${n.padEnd(11)} omega ${(s.omega ?? 0.28).toFixed(2)}`
            + `  jitter ${(s.jitter ?? 0.16).toFixed(2)}  palette ${s.palette}`
            + `  ${(s.omega ?? 0.28) === 0 ? '(no trunk -- not a tree)' : ''}`);
}
mkdirSync('out', { recursive: true });
writeFileSync('out/catalogue.png', png(tile(sets, 4)));

// and one segment of the spline, sampled, so the morph can be judged too
const seg = [];
const FROM = SET_NAMES.indexOf('tree');
for (let k = 0; k < 8; k++) {
  seg.push(shoot(morphPath(SPECS, FROM + k / 8), { W, H, target: TARGET, fog: false }).im);
}
writeFileSync('out/spline.png', png(tile(seg, 4)));
console.log(`\n  out/catalogue.png   all ${SET_NAMES.length} sets`);
console.log(`  out/spline.png      ${SET_NAMES[FROM]} -> ${SET_NAMES[(FROM + 1) % SET_NAMES.length]},`
          + ' eight steps along the spline');
