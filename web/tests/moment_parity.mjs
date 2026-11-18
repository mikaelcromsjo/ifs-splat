import { invariantMoments } from '../js/moments.js';
import { BRANCH, TRUNK_A, MU0, SIG0 } from '../js/ifs_core.js';
const SG = [SIG0[0], SIG0[1], SIG0[2]];
const M = invariantMoments(BRANCH.map((x) => ({ A: x.A, b: x.b })), TRUNK_A, SG, MU0,
                           Number(process.argv[2] ?? 0.28));
console.log(JSON.stringify({ C: M.C, Q: Array.from(M.Q), R: M.R, s: M.s }));
