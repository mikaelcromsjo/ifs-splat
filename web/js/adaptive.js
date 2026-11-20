// Stage 2: adaptive LOD, and the flight that never stops.
//
// Three things make the infinite zoom work, and they are all in this file.
//
// 1. THE CUT AND THE GENERATOR ARE THE SAME MECHANISM.  A node is expanded
//    while its subtree is bigger than a few pixels and drawn as one
//    moment-matched Gaussian when it is not.  There is no mesh, no stored
//    tree, no streaming: memory is O(visible), not O(world).
//
// 2. RE-ROOTING.  At depth 30 a node's affine has scale 0.6^30 ~ 2e-7, and
//    float32 world coordinates are long gone.  So there are no world
//    coordinates.  The camera is stored as (address, position inside that
//    node), and everything is expressed relative to the node the camera is
//    currently inside -- the trick deep Mandelbrot zooms use.  Descending one
//    level transforms the camera by A_i^-1, which is exact because A_i is a
//    similarity; no quantity ever grows or shrinks without bound.
//
// 3. ADDRESS-HASHED JITTER (in ifs_core.js).  Subtrees here are generated
//    independently and at different times, so a node's perturbation must be a
//    function of its address.  A sequential RNG would make the world shimmer
//    as things cross the LOD cut.
//
// Ordering is the same ordered expansion as stage 1, written as a depth-first
// walk in sorted-sibling order: that IS the back-to-front stream, and it
// handles varying subtree sizes for free, since it appends instead of
// computing slots.

import { mix, jitterRotInto, stream as jstream, normal as jnormal, h32, rootHash,
         BRANCH, TRUNK_A, MU0, SIG0, M_BR } from './ifs_core.js';
import { invariantMoments, subtreeColours, opNorm } from './moments.js';
import { makeBudget, budgetStep } from './budget.js';

const now = () => (typeof performance !== 'undefined' ? performance.now()
                                                     : Number(process.hrtime.bigint()) / 1e6);

// Colour by recursion depth: level k takes palette[k % P].  It costs nothing,
// it is the one attribute that is free at every scale, and it turns the
// fractal structure from decoration into something you can read -- when you
// fly in, the colour cycles, and the cycle IS the self-similarity.
//
// The ramp has to close on itself.  A normal gradient has a dark end and a
// bright end; here there is no deepest level to put the bright end at, so the
// last entry has to lead back into the first or the seam shows as a hard edge
// every P levels.
//
// Length is the knob that matters.  P is the period in LEVELS, and a parent's
// colour is the fixed point of mixing its own level with its children's, which
// averages about three adjacent entries.  The further a ramp travels per
// entry, the more that averaging bleaches it -- paletteBleach() below measures
// exactly how much, and the answer is that twelve entries costs about 8% of
// the saturation while sixteen entries spanning the whole hue wheel costs 30%.
// So: the further a palette travels, the longer it has to be.
//
// (The tree's olive canopy is NOT this effect -- that palette keeps 92% of its
// saturation.  Olive is brown rods and green leaves mixed in SPACE, below the
// resolution limit, which is a true answer rather than an artifact.)
const hsv = (h, s, v) => {
  h = ((h % 360) + 360) % 360 / 60;
  const c = v * s, x = c * (1 - Math.abs((h % 2) - 1)), m = v - c;
  const t = [[c, x, 0], [x, c, 0], [0, c, x], [0, x, c], [x, 0, c], [c, 0, x]][h | 0];
  return [t[0] + m, t[1] + m, t[2] + m];
};
// a full turn of the hue wheel closes on itself by construction
const wheel = (n, s, v, phase = 0) =>
  Array.from({ length: n }, (_, k) => hsv(phase + (360 * k) / n, s, v));

export const PALETTES = {
  tree: [[0.26, 0.17, 0.10], [0.31, 0.21, 0.13], [0.36, 0.26, 0.16],
         [0.33, 0.33, 0.17], [0.24, 0.38, 0.16], [0.17, 0.41, 0.13],
         [0.21, 0.33, 0.12]],

  // deep blue up through cyan to white and back down the other side
  spectral: [[0.05, 0.09, 0.28], [0.06, 0.16, 0.42], [0.08, 0.27, 0.56],
             [0.11, 0.40, 0.66], [0.19, 0.56, 0.74], [0.33, 0.71, 0.80],
             [0.55, 0.84, 0.88], [0.82, 0.94, 0.97], [0.92, 0.96, 0.99],
             [0.66, 0.82, 0.95], [0.38, 0.58, 0.86], [0.16, 0.29, 0.56]],

  // colder and tighter: glacier blue, very little hue travel, so it stays
  // saturated even where the blobs average it
  ice: [[0.08, 0.17, 0.30], [0.10, 0.25, 0.41], [0.13, 0.35, 0.52],
        [0.19, 0.47, 0.62], [0.29, 0.60, 0.72], [0.45, 0.74, 0.82],
        [0.66, 0.87, 0.92], [0.86, 0.96, 0.99], [0.60, 0.82, 0.93],
        [0.30, 0.52, 0.70], [0.15, 0.30, 0.48], [0.09, 0.21, 0.37]],

  // the one warm set, for contrast in the void: coal to white-hot
  ember: [[0.22, 0.05, 0.03], [0.36, 0.08, 0.04], [0.52, 0.13, 0.04],
          [0.68, 0.22, 0.05], [0.82, 0.35, 0.07], [0.92, 0.51, 0.13],
          [0.97, 0.68, 0.28], [0.99, 0.84, 0.54], [0.99, 0.93, 0.78],
          [0.86, 0.60, 0.36], [0.60, 0.26, 0.12], [0.35, 0.10, 0.05]],

  // soap film: a full turn of hue, kept pale so neighbouring levels stay
  // close and the fixed point does not grey out
  iridescent: wheel(24, 0.46, 0.92, 200),

  violet: [[0.14, 0.06, 0.28], [0.22, 0.08, 0.42], [0.33, 0.11, 0.56],
           [0.46, 0.17, 0.68], [0.60, 0.27, 0.78], [0.73, 0.41, 0.86],
           [0.85, 0.58, 0.92], [0.94, 0.76, 0.96], [0.97, 0.88, 0.98],
           [0.80, 0.62, 0.93], [0.54, 0.33, 0.80], [0.27, 0.13, 0.50]],
};

/** How much saturation a palette keeps once the blob fixed point averages it. */
export function paletteBleach(pal, omega = 0.28) {
  const sat = (c) => {
    const mx = Math.max(c[0], c[1], c[2]), mn = Math.min(c[0], c[1], c[2]);
    return mx < 1e-6 ? 0 : (mx - mn) / mx;
  };
  const blob = subtreeColours(pal, omega);
  const a = pal.reduce((s, c) => s + sat(c), 0) / pal.length;
  const b = blob.reduce((s, c) => s + sat(c), 0) / blob.length;
  return { leaf: a, blob: b, kept: b / Math.max(a, 1e-9) };
}

export class Attractor {
  constructor(opts = {}) {
    this.jitter = opts.jitter ?? 0.16;
    this.seed = opts.seed ?? 3;
    this.rootHash = rootHash(this.seed);
    this.m = (opts.maps || BRANCH).length;

    // A node's jitter rotation is a pure function of its address -- it is just
    // drawn from K of them instead of a continuum, which keeps the per-node
    // cost to a table lookup instead of four transcendentals.
    //
    // The rotations are NOT tabulated separately any more.  They are folded
    // straight into the premultiplied child maps in reparam(), for two
    // reasons.  It makes `jitter` a morphable parameter -- jitterRotInto
    // scales only the ANGLE, the axis comes from the hash alone, so a rigid
    // lattice can morph into an organic one, which it could not when the
    // rotations were baked at construction.  And it drops a 9*K scratch.
    //
    // K is 1024, not 4096.  reparam is dominated by this table and scales
    // with it -- 1.388 ms at 4096 against 0.330 ms at 1024 -- and that factor
    // of four is what pays for morphing continuously instead of in bursts.
    // The cost is that a node's jitter is drawn from 1024 rotations per map
    // rather than 4096; tests/jitterbits.mjs is the A/B that says whether an
    // eye can tell.
    // The table is allocated once at its LARGEST, and how much of it is in
    // use is a runtime dial.
    //
    // Everything a rebuild and a blend cost scales with K: the premultiply is
    // 9*m*K multiply-adds and the blend is 9*m*K lerps. But a slot four
    // pixels across expands a handful of nodes and draws its jitter from a
    // handful of entries, and rebuilding 1024 of them for it is work nobody
    // can see. Sizing K per slot by on-screen importance is what makes a
    // 60 Hz morph affordable: the object filling the screen gets the full
    // table, the specks get 64, and the specks are most of them.
    //
    // Allocating at the maximum and shrinking the ACTIVE length, rather than
    // reallocating, keeps this off the garbage collector -- which matters
    // here more than usual, since the thing being tuned is frame-time tails.
    this.maxTableBits = opts.maxJitterTableBits ?? (opts.jitterTableBits ?? 10);
    this.minTableBits = opts.minJitterTableBits ?? 6;
    this.tableBits = opts.jitterTableBits ?? 10;
    // The size a resize has ASKED for, applied by the next rebuild; equal to
    // the drawn size whenever nothing is pending. See setTableBits().
    this._wantBits = this.tableBits;
    const K = 1 << this.maxTableBits;
    this.tableMask = (1 << this.tableBits) - 1;
    // What IS tabulated is the part that does not depend on the jitter amount:
    // each entry's rotation axis and its standard normal deviate.  Those cost
    // a hash, a log, a sqrt and three trig calls apiece, and none of them
    // change when the jitter does -- only the ANGLE does, being amt * nrm.
    // Keeping them means a rebuild assembles the rotation from a sine and a
    // cosine instead of redrawing it, which is the difference between 0.63 ms
    // and 0.37 ms per reparam, on a budget that allows 1.5 ms a frame.
    this._axis = new Float64Array(4 * K);
    for (let k = 0; k < K; k++) {
      const h = h32(Math.imul(k, 2654435761) >>> 0);
      const z = 2.0 * jstream(h, 0) - 1.0;
      const phi = 2.0 * Math.PI * jstream(h, 1);
      const r = Math.sqrt(Math.max(1.0 - z * z, 0.0));
      this._axis[4 * k] = r * Math.cos(phi);
      this._axis[4 * k + 1] = r * Math.sin(phi);
      this._axis[4 * k + 2] = z;
      this._axis[4 * k + 3] = jnormal(h, 2);
    }
    // The assembled rotations, rebuilt only when the jitter amount actually
    // moves.  Shape morphs every rebuild; jitter usually does not, and
    // assembling 1024 rotations costs two thousand trig calls -- 0.2 of the
    // 0.57 ms.  Paying that only when it changes puts a steady-jitter rebuild
    // back at 0.37 ms while leaving jitter free to morph when a set wants it.
    this._jit = new Float64Array(9 * K);
    this._jitAmt = NaN;
    // Two tables, and the blend between them is done when the table is READ
    // rather than written into a third.
    //
    // A rebuild costs ~0.44 ms and the budget affords a couple per frame
    // across eight slots, so a slot's drawn shape changes 18 to 26 times a
    // second -- measured, the range being how much the budget affords on a
    // given machine -- while the display runs at 60: the same shape for two
    // or three frames, then a jump. That is a 20 Hz animation played at
    // 60 Hz, and it reads as judder however smooth the morph underneath is.
    // Measured over 30 s with eight slots: 58% of slot-frames drew exactly
    // what the frame before drew.
    //
    // What makes interpolating cheap is that the premultiply below is LINEAR
    // in the maps: childMaps[k][i] = J[k] * A[i], so
    //
    //     lerp(J*A0, J*A1) = J * lerp(A0, A1)
    //
    // -- blending two built tables is exactly premultiplying the blended
    // maps.
    //
    // The first version of this materialised the blend into a third table
    // every frame. It was correct, it fixed the judder, and it was far too
    // expensive: 0.075 ms of arithmetic per slot that moved the frame's 99th
    // percentile from 103 ms to 564 ms. Rewriting 288 KB evicts the cache
    // that tree expansion is working in, and the expansion then misses on
    // essentially every one of a hundred thousand random reads -- so the cost
    // landed in the expansion afterwards, where no amount of budgeting the
    // blend could find it.
    //
    // Doing the lerp in the traversal instead writes nothing at all. The two
    // tables stay read-only and resident; the hot loop reads 9 floats from
    // each instead of 9 from one and spends 9 fused multiply-adds. Nothing is
    // invalidated, so nothing has to be re-fetched. Measured on one frame of
    // 140k splats, drawn both ways: +3.8% on the expansion, and no tail.
    //
    // `_cmA` and `_cmB` are the two most recent builds -- consecutive samples
    // of the morph a known parameter distance apart -- and `childMaps` IS
    // `_cmB`, so every caller that does not interpolate (the GPU twin,
    // calibrate, the parity tests) reads exactly the table it always read.
    this._cmA = new Float64Array(9 * this.m * K);
    this._cmB = new Float64Array(9 * this.m * K);
    this.childMaps = this._cmB;
    // How far from _cmA to _cmB the drawn shape currently is. 1 means "the
    // newest build", which is what everything that never calls setBlend()
    // gets, and the hot loop skips the second table entirely at 1.
    //
    // Stored as 0.5 and then as 1, which looks like a mistake and is load
    // bearing. V8 picks a field's representation from the first value stored
    // in it: write the integer 1 here and the field is a small integer, so
    // the first real blend fraction -- 1.37, say -- changes its
    // representation, which transitions this object's hidden class and
    // throws away the optimised code of every method whose property lookups
    // depended on it. reparam() is that method, and it is 50 lookups and a
    // 36,864-iteration loop. Measured, with `this._bt = 1`: the first
    // rebuild after the blend starts costs 15 ms instead of 2.4, and it
    // stays there for six rebuilds until V8 re-optimises -- a visible stall
    // in the first second, caught by the "frames 0-3 are not a stall" check
    // in tests/field.mjs. Storing a fraction first makes the field a double
    // from the start, and 1 fits in a double.
    this._bt = 0.5;
    this._bt = 1;
    // How far setBlend() may extrapolate before the shape stops being
    // possible; see _btLimit(). A double for the same reason as _bt.
    this._btSafe = 0.5;
    this._btSafe = Infinity;
    // The same, for the bound in the other direction; see _btLimit().
    this._btSafeLo = -0.5;
    this._btSafeLo = -Infinity;
    // How many rebuilds a table resize is spread over.
    //
    // A node picks its rotation with `hash & tableMask`, so changing the mask
    // re-draws the jitter of every node whose index moves -- half of them for
    // one bit, nearly all of them for four. That is a different realisation
    // of the same set, not a different shape, and it was the largest single
    // movement anywhere in the demo: measured as splat positions,
    // tests/drift.mjs finds the median step on a resize frame at 9.8e-3 of
    // the object's radius against 5.0e-4 for an ordinary frame, and the worst
    // at 4.3e-2 -- 86x one frame's own motion, on whichever object had just
    // become the most important one on screen. Every one of the fourteen
    // largest steps in a 1500-frame flight was a resize.
    //
    // It is invisible to everything that measures the morph, because it is
    // not the morph: the drawn table and an exact build at the same
    // parameter re-draw the same way, so the error stays at 1e-5 while the
    // picture snaps. Only the STEP sees it, which is why nothing caught it.
    //
    // So the resize ramps instead of switching. _premulBlend() carries each
    // moving entry from the rotation the nodes were reading to the one they
    // will read, over this many rebuilds -- about a second at the rate the
    // budget affords -- and the blend pair smooths each of those steps across
    // the frames between them, exactly as it does for the morph itself. 1
    // restores the old behaviour, in one rebuild.
    this.migrateRebuilds = opts.migrateRebuilds ?? 4;
    // Entries [_migLo, K) are the ones moving; below _migLo nothing changes.
    // _migLo is the SMALLER of the two sizes either way round, so a growing
    // table blends its new entries up from their old partners and a
    // shrinking one blends its doomed entries down onto theirs.
    this._migLo = 0;
    // The weight the drawn table was built at: 0 at the far end, 1 settled.
    // A double from the first store, for the reason _bt is; see above.
    this._migW = 0.5;
    this._migW = 1;
    this._migStep = 0.5;
    this._migStep = 1;
    this._migGrow = true;
    this._migBits = this.tableBits;   // a shrink's target, applied at the end
    this._migDrop = false;            // one settled rebuild before the mask moves
    // Scratch for the blended rotation, so _premulBlend allocates nothing.
    this._jmix = new Float64Array(9);
    // Opt-in per attractor: a rebuild swaps which array is the newest build,
    // so `childMaps` moves under anything holding a reference to it, and an
    // object that is not morphing has nothing to interpolate anyway.
    this.interp = opts.interp ?? false;
    this._everBuilt = false;
    this.mapB = new Float64Array(3 * this.m);
    // The small derived state, snapshotted either side of a rebuild so that
    // it can be interpolated along with the table.
    //
    // It is small -- about sixty numbers against the table's thirty-odd
    // thousand -- but it cannot be left behind. Half of it is geometry the
    // traversal uses directly (the translations, the per-child scales, the
    // trunk map, the rod's moments) and the other half decides the level of
    // detail (C, R, qMax). A table that moves smoothly while the LOD metric
    // it is cut against jumps at 18 Hz is a different judder, not less of it.
    this._sN = 45 + 4 * this.m;
    this._sA = new Float64Array(this._sN);
    this._sB = new Float64Array(this._sN);

    this.reparam(opts);
  }

  /**
   * Assemble the jitter rotations for entries [k0, k1) at amount `amt`.
   *
   * Rodrigues, from the cached axis. Written out rather than calling
   * jitterRotInto so the axis is not redrawn; the expressions are character
   * for character the ones in ifs_core.js, because the two must agree -- the
   * reference expansion uses that one, and a test asserts this table equals
   * it exactly.
   *
   * A range rather than the whole table, because a table that GROWS needs
   * only its new entries filled; see reparam().
   */
  _assembleJit(k0, k1, amt) {
    const AX = this._axis, J = this._jit;
    for (let k = k0; k < k1; k++) {
      const r = 9 * k;
      if (amt <= 0) {
        J[r] = J[r + 4] = J[r + 8] = 1;
        J[r + 1] = J[r + 2] = J[r + 3] = J[r + 5] = J[r + 6] = J[r + 7] = 0;
      } else {
        const ax = AX[4 * k], ay = AX[4 * k + 1], az = AX[4 * k + 2];
        const ang = amt * AX[4 * k + 3];
        const sn = Math.sin(ang), c = 1.0 - Math.cos(ang);
        J[r] = 1 + c * (ax * ax - 1);
        J[r + 1] = -sn * az + c * ax * ay;
        J[r + 2] = sn * ay + c * ax * az;
        J[r + 3] = sn * az + c * ay * ax;
        J[r + 4] = 1 + c * (ay * ay - 1);
        J[r + 5] = -sn * ax + c * ay * az;
        J[r + 6] = -sn * ay + c * az * ax;
        J[r + 7] = sn * ax + c * az * ay;
        J[r + 8] = 1 + c * (az * az - 1);
      }
    }
    return this;
  }

  /**
   * Premultiply `maps` by the assembled rotations of entries [k0, k1) into
   * `dest`: the child's effective linear part, R(address) @ A_i, which is
   * what the traversal indexes instead of multiplying.
   */
  _premul(k0, k1, maps, dest) {
    const J = this._jit, m = this.m;
    for (let k = k0; k < k1; k++) {
      const r = 9 * k;
      for (let i = 0; i < m; i++) {
        const A = maps[i].A, o = 9 * (k * m + i);
        for (let p = 0; p < 3; p++) {
          const j0 = J[r + p * 3], j1 = J[r + p * 3 + 1], j2 = J[r + p * 3 + 2];
          dest[o + p * 3] = j0 * A[0] + j1 * A[3] + j2 * A[6];
          dest[o + p * 3 + 1] = j0 * A[1] + j1 * A[4] + j2 * A[7];
          dest[o + p * 3 + 2] = j0 * A[2] + j1 * A[5] + j2 * A[8];
        }
      }
    }
    return this;
  }

  /**
   * The same, for entries whose rotation is on its way from one table size
   * to another: the premultiply of a rotation blended between entry `k` and
   * its partner `k & srcMask`, at weight `w` on entry `k` itself.
   *
   * The blend is of the two ROTATION MATRICES, nine lerps, and not of their
   * axes and angles -- which would need a trig call per entry per rebuild,
   * and has a worse failure mode: two axes that happen to be nearly opposite
   * pass through a vector of no length, and a normalised nothing points
   * anywhere. Lerping the matrices has no such case. What it costs is that
   * the halfway matrix is not exactly a rotation: these are wobbles of a
   * tenth of a radian, so its determinant dips by about 1e-3 at w = 0.5 --
   * a 0.03% breath in the size of a splat, three orders under the 1.2e-3 of
   * its own radius that the splat moves in a frame anyway.
   *
   * At w = 0 this writes exactly what `_premul` would write for the partner
   * entry, and at w = 1 exactly what it would write for `k`. Both ends are
   * therefore bit-identical to a table that was never resized at all, which
   * is what makes the ramp start and finish without a step.
   */
  _premulBlend(k0, k1, srcMask, w, maps, dest) {
    const J = this._jit, m = this.m, B = this._jmix, v = 1 - w;
    for (let k = k0; k < k1; k++) {
      const r = 9 * k, q = 9 * (k & srcMask);
      for (let p = 0; p < 9; p++) B[p] = J[q + p] * v + J[r + p] * w;
      for (let i = 0; i < m; i++) {
        const A = maps[i].A, o = 9 * (k * m + i);
        for (let p = 0; p < 3; p++) {
          const j0 = B[p * 3], j1 = B[p * 3 + 1], j2 = B[p * 3 + 2];
          dest[o + p * 3] = j0 * A[0] + j1 * A[3] + j2 * A[6];
          dest[o + p * 3 + 1] = j0 * A[1] + j1 * A[4] + j2 * A[7];
          dest[o + p * 3 + 2] = j0 * A[2] + j1 * A[5] + j2 * A[8];
        }
      }
    }
    return this;
  }

  /**
   * Point this attractor at a different set of maps.
   *
   * Stage 3 morphs one attractor into another by interpolating map
   * PARAMETERS, which means the maps change every frame, which means
   * everything derived from them -- the moment fixed point, the bounding
   * radius, the premultiplied child table -- has to be rebuilt every frame
   * too.  Hence the split: the constructor allocates and tabulates the
   * jitter, this rebuilds the rest.  It is the per-frame cost of morphing,
   * and tests/morph.mjs measures it.
   */
  reparam(opts = {}) {
    // A requested resize takes effect HERE, with the build that fills the
    // table, rather than when it was asked for -- see setTableBits().
    //
    // And it keeps the blend's pair, which it used to throw away. Giving up
    // the pair stops a slot moving for a whole rebuild interval -- the two
    // tables are identical, so every fraction draws the same shape -- and the
    // pool resizes 150 times a minute. Measured as splat positions against an
    // exact build at the same parameter: inside 40 frames of a resize the
    // error ran to 9.4e-3 of the object's radius at p99 against 6.0e-5 for a
    // settled pair, 150x worse, and it is the object that has just become the
    // most important one on screen.
    //
    // Keeping it needs no stashed state, because of WHEN this runs. At the
    // top of reparam the newest build still sits in `_cmB`, and `this.maps`
    // and `this.jitter` are still the parameters it was built from -- the new
    // ones arrive from `opts` further down. So:
    //
    //   shrinking   nothing to do. Both tables hold (their own maps) x
    //               (rotation k) for every k, and the active entries are a
    //               subset, so entry k still pairs with entry k.
    //   growing     the entries above the old size were last written for some
    //               older build. Fill them for the build in `_cmB`, from the
    //               maps and rotations it used, and the pair is exact over
    //               the whole new table.
    //
    // A node's rotation index is `hash & mask`, so a resize changes which
    // rotation each node draws. Doing that in one frame is the largest
    // movement in the demo -- 86x one frame's own motion at its worst, see
    // `migrateRebuilds` -- so a resize starts a RAMP here instead, and only
    // one at a time: a size asked for while one is running waits in
    // `_wantBits` until it finishes, which is what it was already for.
    if (this._wantBits !== this.tableBits && this._migW >= 1 && !this._migDrop) {
      const oldK = this.tableMask + 1, newK = 1 << this._wantBits;
      if (!this._everBuilt) {
        // Never built: there is nothing on screen to be continuous with, and
        // the first-build path below fills both tables anyway.
        this.tableBits = this._wantBits;
        this.tableMask = newK - 1;
        this._jitAmt = NaN;
      } else if (this.migrateRebuilds <= 1) {
        // The ramp turned off: the size changes in one rebuild, which is what
        // this did before `migrateRebuilds` existed. Kept because a reference
        // build has to be able to be exactly the table of its own size --
        // tests/drift.mjs and tests/secant.mjs both compare the drawn table
        // against one -- and because it is the A/B that says what the ramp is
        // worth.
        this.tableBits = this._wantBits;
        this.tableMask = newK - 1;
        if (newK > oldK) {
          this._assembleJit(oldK, newK, this._jitAmt);
          this._premul(oldK, newK, this.maps, this._cmB);
        }
      } else {
        // The moving entries are the ones above the SMALLER of the two
        // sizes, whichever direction this is: a grow blends its new entries
        // up from the partners whose rotation their nodes have been reading,
        // and a shrink blends its doomed entries down onto those same
        // partners. `_migLo` is a power of two, so `_migLo - 1` is the mask
        // that names the partner.
        this._migLo = newK < oldK ? newK : oldK;
        this._migGrow = newK > oldK;
        this._migBits = this._wantBits;
        this._migStep = 1 / (this.migrateRebuilds > 1 ? this.migrateRebuilds : 1);
        this._migW = 0;
        this._migDrop = false;
        if (this._migGrow) {
          // A grow takes the new mask at once, because at w = 0 the new
          // entries are exact copies of the entries the nodes were already
          // reading: the mask moves and nothing on screen does.
          this.tableBits = this._wantBits;
          this.tableMask = newK - 1;
          // `_jit` is assembled at `_jitAmt`, the amount the build in `_cmB`
          // used; extend it over the new entries at that same amount and
          // `_jitAmt` stays true, so the premultiply below will not redo the
          // whole table unless the jitter itself has moved.
          this._assembleJit(oldK, newK, this._jitAmt);
          // Close the pair over the new entries, at w = 0, with the maps the
          // build in `_cmB` was made from -- it is about to become `_cmA` by
          // the swap below, and the older half of the pair has to cover the
          // whole new table or the blend reads a build seconds of morph old.
          this._premulBlend(oldK, newK, this._migLo - 1, 0, this.maps, this._cmB);
        }
        // A shrink keeps its size while it merges, and gives the mask up at
        // the end of this function, when both halves of the pair hold the
        // doomed entries as exact copies of their partners.
      }
    }
    // The newest build becomes the previous one: a pointer swap, nothing
    // copied.
    //
    // The first version folded the DRAWN shape into _cmA instead, reasoning
    // that starting the blend from what is on screen makes every rebuild
    // continuous by construction. It does, and it also destroys the thing
    // being measured. The two tables are read as two samples of the morph
    // curve a known parameter distance apart, and the line through them is
    // evaluated at the parameter the morph is at NOW. Fold the drawn shape in
    // and that distance collapses to nearly zero -- the drawn shape is
    // already at `now` -- so the line's slope becomes one tiny number divided
    // by another and the result is noise. Measured: 10% of frames still
    // stepped as far as they did with no interpolation at all.
    //
    // Two genuine consecutive builds, and the kink where one line hands over
    // to the next is second order in the rebuild interval -- the same order
    // as the error of reading a curve as a straight line in the first place.
    if (this.interp && this._everBuilt) {
      const t = this._cmA; this._cmA = this._cmB; this._cmB = t;
      // childMaps is the name the GPU twin, calibrate and the parity tests
      // know the newest build by, and the swap moves which array that is.
      this.childMaps = this._cmB;
      this._sA.set(this._sB);
    }
    if (opts.maps) {
      this.maps = opts.maps.map((m) => ({ A: Float64Array.from(m.A), b: m.b.slice() }));
      if (this.maps.length !== this.m)
        throw new Error(`reparam changed the map count ${this.m} -> ${this.maps.length}; `
                      + 'the jitter table and every address in flight assume it is fixed');
    }
    if (opts.trunkA) this.trunkA = Float64Array.from(opts.trunkA);
    if (opts.sig0) this.sig0 = opts.sig0.slice(0, 3);
    if (opts.mu0) this.mu0 = opts.mu0.slice();
    if (opts.omega !== undefined) this.omega = opts.omega;
    if (opts.jitter !== undefined) this.jitter = opts.jitter;
    if (opts.alphaRod !== undefined) this.alphaRod = opts.alphaRod;
    if (opts.alphaBlob !== undefined) this.alphaBlob = opts.alphaBlob;
    this.maps = this.maps || BRANCH.map((m) => ({ A: Float64Array.from(m.A), b: m.b.slice() }));
    this.trunkA = this.trunkA || Float64Array.from(TRUNK_A);
    this.sig0 = this.sig0 || SIG0.slice(0, 3);
    this.mu0 = this.mu0 || MU0.slice();
    this.omega = this.omega ?? 0.28;
    this.alphaRod = this.alphaRod ?? 0.99;
    this.alphaBlob = this.alphaBlob ?? 0.95;

    // The hot loop takes a shortcut that the moments do not, so the two can
    // silently describe different objects.  A rod's position is computed as
    // A * (0, 0, trunkA[8] * mu0[2]) + b and its covariance from trunkA's
    // diagonal alone -- three multiplies instead of a 3x3 compose, per node,
    // which is worth having.  invariantMoments, meanwhile, uses the full
    // trunkA and the full mu0.  They agree only while the trunk map is
    // diagonal and mu0 lies on z, which is true of every map set here and is
    // exactly the kind of assumption that is quietly broken by the next one.
    // So check it, with an error that says what to do about it.
    {
      const tA = this.trunkA;
      let off = 0;
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++) if (i !== j) off = Math.max(off, Math.abs(tA[i * 3 + j]));
      const xy = Math.max(Math.abs(this.mu0[0]), Math.abs(this.mu0[1]));
      if (off > 1e-9 || xy > 1e-9)
        throw new Error(
          `trunk map must be diagonal and mu0 must lie on z (off-diagonal ${off.toExponential(1)}`
          + `, mu0 x/y ${xy.toExponential(1)}). The traversal assumes both; give the trunk `
          + 'zero tilt/roll/twist and put any offset in mu0[2], or generalise emitRod() in '
          + 'adaptive.js and the rod branch of expand() in ifs_core.js together.');
    }

    const M = invariantMoments(this.maps, this.trunkA, this.sig0, this.mu0, this.omega);
    this.C = M.C; this.Q = M.Q; this.R = M.R; this.s = M.s;
    // The LOD metric is the size of the blob we would actually draw -- its
    // largest standard deviation -- not the bounding sphere, which is ~2.7x
    // larger here and would make us expand seven times as many nodes for the
    // same picture.  The sphere is still what the frustum test uses.
    this.qMax = Math.sqrt(opNorm(M.Q));
    this.rodMax = Math.sqrt(opNorm(M.SW));
    this.rodMu = M.muW;
    this.rodSig = M.rodSig;              // the trunk Gaussian is diagonal here
    this.rodFull = M.SW;

    this.setPalette(opts.palette || this.palette || PALETTES.tree);

    // The jittered child maps, all K*m of them, premultiplied.  A child's
    // effective linear part is R(its address) @ A_i, and both factors are
    // known here, so the traversal never multiplies them: it indexes them.
    // That is the difference between ~350 and ~210 multiplies per node.
    const K = this.tableMask + 1, amt = this.jitter;
    if (amt !== this._jitAmt) { this._assembleJit(0, K, amt); this._jitAmt = amt; }
    if (this._migW >= 1 && !this._migDrop) {
      this._premul(0, K, this.maps, this._cmB);
    } else {
      // One step of a resize ramp. `_migW` is the weight the table being
      // built here is at, so it advances BEFORE the build rather than after:
      // when this returns, `_migW` describes what is on screen, which is what
      // a test comparing the two can rely on.
      if (this._migW < 1) {
        this._migW += this._migStep;
        if (this._migW > 1) this._migW = 1;
      }
      const w = this._migGrow ? this._migW : 1 - this._migW;
      this._premul(0, this._migLo, this.maps, this._cmB);
      this._premulBlend(this._migLo, K, this._migLo - 1, w, this.maps, this._cmB);
      if (!this._migGrow && this._migW >= 1) {
        // The merge has finished: these entries now hold exactly what their
        // partners hold. One more rebuild at the same weight puts the OTHER
        // half of the pair there too -- and only then is dropping the mask
        // free, because the drawn shape is a blend of both halves and a node
        // whose index folds down onto its partner has to find the same
        // rotation in each. Dropping it one rebuild earlier moves the picture
        // by one ramp step, which is the whole thing this is avoiding.
        if (this._migDrop) {
          this.tableBits = this._migBits;
          this.tableMask = (1 << this._migBits) - 1;
          this._migDrop = false;
        } else {
          this._migDrop = true;
        }
      }
    }
    for (let i = 0; i < this.m; i++)
      for (let k = 0; k < 3; k++) this.mapB[3 * i + k] = this.maps[i].b[k];
    this.sArr = Float64Array.from(this.s);
    // Close the pair the blend reads, and bound how far it may be trusted.
    // WHICH two builds it reads was settled at the top of this function --
    // two genuine consecutive builds, not the drawn shape and this one; see
    // there for why the cheaper-looking choice turns the slope into noise.
    if (this.interp) {
      this._gather(this._sB);
      if (this._everBuilt) {
        this._btSafe = this._btLimit();
        // The fraction itself comes from the owner's parameter clock, in the
        // same update, after this returns.
      } else {
        // The first build has no previous state to come from, so it is its
        // own previous state -- otherwise frame one would interpolate out of
        // an uninitialised table, which is a field of zero matrices and looks
        // like every object imploding.
        const n = 9 * this.m * (this.tableMask + 1);
        const A = this._cmA, B = this._cmB;
        for (let i = 0; i < n; i++) A[i] = B[i];
        this._sA.set(this._sB);
        this._bt = 1;
        this._btSafeLo = -Infinity;
        // And the bound belongs to the pair it was computed from. `_btLimit`
        // reads `_sA` against `_sB`, so taking it before this collapse
        // bounds the blend by a pair that is being thrown away -- a pair
        // whose two halves are now identical, which cannot run any quantity
        // down to zero at any fraction whatsoever.
        this._btSafe = Infinity;
      }
    }
    this._everBuilt = true;
    return this;
  }

  /**
   * Resize the active jitter table.
   *
   * A node picks its rotation with `hash & tableMask`, so changing the mask
   * changes WHICH rotation each node gets -- the shape stays the same family
   * and the same size, but its branch jitter is a different draw. The claim
   * here used to be that this is a visible shimmer on something large and
   * invisible on something small, which is when it is allowed to happen.
   * The second half was wrong: the pool re-cuts tables by RANK, and rank
   * changes hands between the two objects that matter most on screen as
   * readily as between two specks. Measured as splat positions, a resize was
   * the largest movement in the demo -- 20x an ordinary frame's motion at the
   * median and 86x at the worst, on whichever object had just become the most
   * important one. So reparam() now ramps it over `migrateRebuilds` rebuilds
   * instead of switching it, and this is still where the request lands.
   *
   * So this does not resize anything: it records the size wanted, and the
   * next rebuild starts applying it. The mask and the table it indexes have
   * to change together, and only a rebuild can change the table -- while the
   * request is outstanding the slot keeps drawing the table it has, which is
   * consistent with the mask it has. A request that arrives while a ramp is
   * running waits here until that ramp finishes.
   *
   * Moving the mask at the moment of the request is not: a GROW makes the
   * traversal index entries above the ones the last build wrote, and those
   * hold whatever the slot's table held the last time it was that big --
   * another build of it, seconds of morph ago. The pool asks for the resize
   * in its ranking pass and the rebuild that would fill it is budgeted, so
   * the two are not the same frame whenever the budget is spent: measured
   * over 60 s of promotions and demotions, 11 to 47 of 150 resizes were drawn
   * for one or two frames before their rebuild landed -- a shimmer on
   * whichever object had just become the most important one on screen. The
   * spread is the budget, which is spent in real milliseconds, so how often
   * this happened depended on how busy the machine was.
   * (A SHRINK was always safe, the active entries being a subset of what the
   * last build wrote, and it costs nothing to make both wait.)
   *
   * The saving is unaffected: it was never in the mask, it is in the rebuild
   * premultiplying K entries instead of 1024, and that happens when the bits
   * land.
   */
  setTableBits(b) {
    const want = b < this.minTableBits ? this.minTableBits
               : b > this.maxTableBits ? this.maxTableBits : b;
    if (want === this._wantBits) return false;
    this._wantBits = want;
    // Asking for the size already on screen cancels a pending resize, and
    // needs no rebuild of its own.
    return want !== this.tableBits;
  }

  setPalette(pal) {
    this.palette = pal;
    this.blobCol = subtreeColours(pal, this.omega);
    this.P = pal.length;
    return this;
  }

  mixHash(h, i) { return mix(h, i); }

  /** Where child i of the node with hash h keeps its effective linear part. */
  childOffset(h, i) {
    return 9 * ((mix(h, i) & this.tableMask) * this.m + i);
  }

  /** Effective linear part of child i of a node with hash h, copied out.
   *  Used by the camera, which does this a handful of times per frame. */
  childLinear(out, h, i) {
    const o = this.childOffset(h, i);
    const B = this._cmB, t = this._bt;
    if (t === 1) { for (let k = 0; k < 9; k++) out[k] = B[o + k]; return; }
    // The camera has to see the same shape the renderer draws, or it dives
    // into a structure that is not the one on screen.
    const A = this._cmA;
    for (let k = 0; k < 9; k++) out[k] = A[o + k] + (B[o + k] - A[o + k]) * t;
  }

  /**
   * Gather the small derived state into `out`, or write it back lerped.
   *
   * One flat layout, used three ways: snapshot what is on screen before a
   * rebuild, snapshot what the rebuild produced, and interpolate between the
   * two every frame. Writing it out by hand keeps the three in step -- a
   * field added to one and forgotten in another would show up as a shape
   * that interpolates in some of its parts and jumps in the rest.
   */
  _gather(out) {
    let o = 0;
    // The layout, because _btLimit() indexes it directly: trunkA 0, sig0 9,
    // mu0 12, C 15, Q 18, rodMu 27, rodSig 30, rodFull 33, mapB 42,
    // sArr 42+3m, then R, qMax, rodMax.
    const put = (a, n) => { for (let i = 0; i < n; i++) out[o++] = a[i]; };
    put(this.trunkA, 9); put(this.sig0, 3); put(this.mu0, 3);
    put(this.C, 3); put(this.Q, 9); put(this.rodMu, 3); put(this.rodSig, 3);
    put(this.rodFull, 9);
    put(this.mapB, 3 * this.m); put(this.sArr, this.m);
    out[o++] = this.R; out[o++] = this.qMax; out[o++] = this.rodMax;
    return out;
  }

  _scatter(t) {
    const A = this._sA, B = this._sB;
    let o = 0;
    const take = (a, n) => {
      for (let i = 0; i < n; i++, o++) a[i] = A[o] + (B[o] - A[o]) * t;
    };
    take(this.trunkA, 9); take(this.sig0, 3); take(this.mu0, 3);
    take(this.C, 3); take(this.Q, 9); take(this.rodMu, 3); take(this.rodSig, 3);
    take(this.rodFull, 9);
    take(this.mapB, 3 * this.m); take(this.sArr, this.m);
    this.R = A[o] + (B[o] - A[o]) * t; o++;
    this.qMax = A[o] + (B[o] - A[o]) * t; o++;
    this.rodMax = A[o] + (B[o] - A[o]) * t; o++;
    // `s` is the plain-array twin the camera indexes; sArr is the traversal's.
    for (let i = 0; i < this.m; i++) this.s[i] = this.sArr[i];
    // And the camera reads translations out of maps[i].b while the traversal
    // reads mapB. Two names for one quantity: if only one of them is
    // interpolated, the camera dives at a structure that is not the one being
    // drawn, which is the hardest kind of wrongness to see and the easiest to
    // feel.
    for (let i = 0; i < this.m; i++) {
      const b = this.maps[i].b, o = 3 * i;
      b[0] = this.mapB[o]; b[1] = this.mapB[o + 1]; b[2] = this.mapB[o + 2];
    }
  }

  /**
   * How far the blend may run before it would make the shape impossible.
   *
   * Interpolation between two valid shapes is safe for free: every quantity
   * is a convex combination of two values, so a variance that is positive in
   * both builds is positive all the way across. Extrapolation is not, and
   * this interpolation extrapolates by design -- it carries on past the
   * newest build to reach the parameter the morph is actually at. Measured
   * over 200 s without this: sigma0 reached **-0.0204**, a child scale
   * -0.047, and R -0.42. A negative variance is the exact fault that
   * `tests/breath.mjs` was written for when a clamp was hiding it in the
   * spline, and it would be no better arriving from here.
   *
   * So the bound comes from the data. For each quantity that must stay
   * positive, the fraction at which the straight line through the two builds
   * would reach zero is A / (A - B), and the blend is allowed 90% of the
   * nearest of them. Both builds are fixed between rebuilds, so this is
   * computed once per rebuild rather than per frame, and in the ordinary case
   * -- nothing heading for zero -- it does not bind at all.
   */
  _btLimit() {
    const A = this._sA, B = this._sB, m = this.m;
    let lim = Infinity, lo = -Infinity;
    // Both ways along the line, because the blend now runs both ways. Past
    // the newer build a SHRINKING quantity is the one that reaches zero;
    // before the older one it is a GROWING one, at the mirror of the same
    // formula. The bound is kept to 90% of the nearer crossing in each
    // direction -- for the negative bound that is 0.9x a negative number,
    // which moves it towards zero, the safe way.
    const bound = (i) => {
      const a = A[i], b = B[i];
      if (b === a) return;                   // flat: cannot reach zero either way
      const t0 = a / (a - b);                // where the line crosses zero
      if (b < a) { if (t0 > 0 && t0 < lim) lim = t0; }
      else if (t0 < 0 && t0 > lo) lo = t0;
    };
    for (let j = 0; j < 3; j++) { bound(9 + j); bound(30 + j); }   // sig0, rodSig
    bound(18); bound(22); bound(26);                              // diag Q
    bound(33); bound(37); bound(41);                              // diag of the rod
    const sb = 42 + 3 * m;
    for (let j = 0; j < m; j++) bound(sb + j);                    // child scales
    const tail = 42 + 4 * m;
    bound(tail); bound(tail + 1); bound(tail + 2);                // R, qMax, rodMax
    this._btSafeLo = lo === -Infinity ? -Infinity : lo * 0.9;
    return lim === Infinity ? Infinity : lim * 0.9;
  }

  /**
   * Put the drawn shape a fraction `t` of the way from the previous rebuild
   * to the newest one, at display rate.
   *
   * `t` may exceed 1, and that is the point. Clamping it at 1 was the first
   * version's remaining judder: the fraction is a parameter position divided
   * by the distance between two builds, the next build arrives when the
   * BUDGET allows rather than on time, and every frame in between sat frozen
   * at the newest build -- 35 Hz of movement instead of 60. Carrying on past
   * 1 along the same straight line is a linear extrapolation of a curve that
   * is already being read linearly between its samples; `tMax` bounds how far
   * it may be trusted.
   *
   * Costs about sixty lerps. The table is not touched: see the constructor.
   */
  setBlend(t, tMax = 2.5, tMin = -1.5) {
    if (!this.interp) return this;
    const top = tMax < this._btSafe ? tMax : this._btSafe;
    // Negative is allowed, and has to be.
    //
    // The owner builds about one rebuild interval ahead so that the pair
    // brackets the present (see field.js). That makes the hand-over from one
    // secant to the next land at the sample they share -- exactly continuous
    // -- and the fraction is 0 there. So the floor sits right where every
    // rebuild lands, and a gap that comes in shorter than the estimate asks
    // for a fraction slightly below it. Flooring at 0 froze the shape for
    // those frames and then released it: measured, 22 slot-frames of 10,552
    // stepped more than 8x the true motion, every one of them at exactly
    // 0.00. Carrying the line a little way back the other side of its older
    // sample is the same second-order read as carrying it past the newer one,
    // and it is continuous.
    const bot = tMin > this._btSafeLo ? tMin : this._btSafeLo;
    this._bt = t < bot ? bot : t > top ? top : t;
    this._scatter(this._bt);
    return this;
  }

  /**
   * Turn read-time interpolation on or off on a LIVE attractor, which is a
   * different thing from setting `interp` and has to be, in both directions.
   *
   * Off is not just "stop lerping". The drawn shape may be anywhere between
   * the two builds, or past the newer one -- the blend extrapolates -- and
   * the small state (moments, radius, map offsets) was scattered to match.
   * Leaving that in place and clearing the flag would freeze the shape at a
   * fraction of a rebuild that is about to be overwritten, and leave the
   * moments describing a shape the table no longer draws. So both go to the
   * newest build.
   *
   * On is the one with teeth. While interpolation is off, reparam() stops
   * swapping the tables, so `_cmA` is whatever build happened to be current
   * when it was switched off -- one rebuild ago, or ten thousand, which is a
   * completely different shape. Starting to lerp from that would walk the
   * drawn shape back through a stale morph, and extrapolating from it can
   * push variances negative (the reason `_btLimit` exists). Making the pair
   * degenerate first means the blend is a no-op until the next rebuild gives
   * it a real pair, and from there it is ordinary two-build interpolation.
   *
   * Both directions end in the same few lines, and the fold is the one that
   * makes this a switch rather than a jolt: the blend is
   * FOLDED into the newest build before the pair is collapsed. Collapsing
   * without folding sets the drawn shape to `_cmB`, and `_cmB` is not what
   * is on screen -- the blend extrapolates, so the shape sits typically a
   * third of a rebuild interval PAST it. The picture would snap backwards at
   * the click, which was measured (8 of 8 slots moved) before the fold was
   * added. Folding costs one pass over the table, once, on a UI action.
   */
  setInterp(on) {
    // `_sB` is a snapshot taken under this very flag -- reparam() gathers it
    // only while interpolating -- so while interpolation is OFF every rebuild
    // moves the live moments, radius, child scales and translations and
    // leaves the snapshot behind. Collapsing the pair onto a stale `_sB` and
    // scattering it is then a silent clobber of the shape itself: measured
    // across an off-then-on click with ten seconds of morphing in between,
    // all 8 slots jumped, R by up to 1.37 and sigma0 by 8x, while the drawn
    // table stayed the newest build. So the geometry, its level of detail and
    // the camera's idea of where its children are all described a build that
    // had left the screen seconds before, until that slot's next rebuild
    // re-gathered -- and with morphing off, for good.
    //
    // Re-gathering here makes `_scatter(1)` below reassert what is actually
    // on screen. It is also what keeps a never-interpolated attractor from
    // scattering a table of ZEROS: `_sB` starts zeroed, and an attractor
    // built with `interp: false` has never had it filled.
    if (!this.interp) this._gather(this._sB);
    const t = this._bt;
    if (t !== 1) {
      const A = this._cmA, B = this._cmB;
      for (let i = 0; i < B.length; i++) B[i] = A[i] + (B[i] - A[i]) * t;
      const a = this._sA, b = this._sB;
      for (let i = 0; i < b.length; i++) b[i] = a[i] + (b[i] - a[i]) * t;
    }
    this._cmA.set(this._cmB);
    this._sA.set(this._sB);
    this._bt = 1;
    this._btSafe = Infinity;
    this._btSafeLo = -Infinity;
    this._scatter(1);
    this.interp = !!on;
    return this;
  }
}

// ----------------------------------------------------------------- camera

/**
 * The camera, stored as "which node am I inside, and where in it".
 *
 * Moving at a constant speed in this frame is an exponential approach in world
 * terms, because the frame rescales as you descend -- which is exactly the
 * flight the demo wants, and it falls out rather than being animated.
 */
export class FlyCamera {
  constructor(att) {
    this.att = att;
    this.path = [];                       // digits from the true root
    this.hashes = [att.rootHash];         // hash chain alongside it
    this.pos = [att.C[0], att.C[1] - att.R * 2.2, att.C[2]];
    this.r = [1, 0, 0];
    this.u = [0, 0, 1];
    this.f = [0, 1, 0];
    this.reroots = 0;
    this.descendMargin = 0.80;            // descend only when well inside
    this.ascendMargin = 1.25;             // ascend only when well outside
    // How much of the child choice is "which one is on the boundary" rather
    // than "which one is biggest and most nearly ahead", and how far out
    // along that boundary the aim point is pushed, in units of the child's
    // own radius.  Both 0 reproduces the old dive exactly.
    this.edgeBias = 0.55;
    this.edgeAim = 0.52;
    // How far sideways the dive slides per unit of forward travel, along the
    // same outward direction. 0 flies straight at the aim point.
    //
    // OFF, and the reason is measured rather than assumed. Curving the path
    // as well as the aim sounded like more of a good thing; over a 70-second
    // dive-only flight at 0.30 it made the washed-out rate WORSE, 33.7%
    // against 30.0% (tests/legible.mjs). Aiming at the boundary is what
    // helps; sliding along it as well mostly costs forward progress, and the
    // camera spends longer at each depth with the same thing in front of it.
    // Kept, switched off, with the number that says why.
    this.edgeSlide = 0;
    // How much better a challenger has to score before the dive changes its
    // mind about which child it is falling into.
    this.stick = 2.2;
    // Banking: radians of lean per radian/second of sideways turn, the cap,
    // and how long the lean takes to follow a change in the turn.
    this.bankGain = 0.55;
    this.bankMax = 0.45;
    this.bankTau = 0.9;
    this._roll = 0;
  }

  /** world scale of one unit in the current frame: 0.6^depth-ish */
  get scale() {
    let k = 1;
    for (let i = 0; i < this.path.length; i++) k *= this.att.s[this.path[i]];
    return k;
  }
  get depth() { return this.path.length; }

  orient(yaw, pitch) {
    // rotate the basis about u by yaw, then about r by pitch
    const rot = (v, axis, a) => {
      const c = Math.cos(a), s = Math.sin(a);
      const d = axis[0] * v[0] + axis[1] * v[1] + axis[2] * v[2];
      const cx = [axis[1] * v[2] - axis[2] * v[1], axis[2] * v[0] - axis[0] * v[2],
                  axis[0] * v[1] - axis[1] * v[0]];
      return [v[0] * c + cx[0] * s + axis[0] * d * (1 - c),
              v[1] * c + cx[1] * s + axis[1] * d * (1 - c),
              v[2] * c + cx[2] * s + axis[2] * d * (1 - c)];
    };
    if (yaw) { this.f = rot(this.f, this.u, yaw); this.r = rot(this.r, this.u, yaw); }
    if (pitch) { this.f = rot(this.f, this.r, pitch); this.u = rot(this.u, this.r, pitch); }
    this.ortho();
  }

  ortho() {
    const n = (v) => { const l = Math.hypot(v[0], v[1], v[2]); return [v[0] / l, v[1] / l, v[2] / l]; };
    this.f = n(this.f);
    this.r = n([this.f[1] * this.u[2] - this.f[2] * this.u[1],
                this.f[2] * this.u[0] - this.f[0] * this.u[2],
                this.f[0] * this.u[1] - this.f[1] * this.u[0]]);
    this.u = [this.r[1] * this.f[2] - this.r[2] * this.f[1],
              this.r[2] * this.f[0] - this.r[0] * this.f[2],
              this.r[0] * this.f[1] - this.r[1] * this.f[0]];
  }

  move(dr, du, df) {
    for (let k = 0; k < 3; k++)
      this.pos[k] += this.r[k] * dr + this.u[k] * du + this.f[k] * df;
  }

  /**
   * Autopilot: pick the child subtree most nearly ahead, steer towards it and
   * move in.  Because the frame rescales every time the camera descends, a
   * constant speed here is an exponential approach in world terms -- the
   * flight that never arrives.  This is the hero shot, and it is a dozen
   * lines, because the LOD cut and the generator are the same mechanism.
   */
  dive(speed = 0.04, turn = 0.085, ticks = 1) {
    const att = this.att, C = att.C;
    const h = this.hashes[this.hashes.length - 1];
    const m = att.m;
    // One scratch block, grown once: this runs every frame.
    const S = (this._sc && this._sc.length >= 8 * m) ? this._sc
            : (this._sc = new Float64Array(8 * m));
    // Which child to fall into?  Not simply the one straight ahead: in a
    // sparse attractor -- the dust, the sheet -- the child straight ahead is
    // often the small one, and steering at it puts the camera in a gap with
    // nothing on screen for twenty seconds.  Score each child by how big it
    // will LOOK, its radius over its distance, and prefer forward only as a
    // tie-break.  Aim at mass, and the screen stays full.
    //
    // Aiming at mass alone is what put a quarter of the flight inside a
    // featureless wash (tests/legible.mjs, and the stage-4 note in IDEAS.md).
    // The mass of an attractor is its INTERIOR, and the interior of a
    // Gaussian field at close range has no silhouette left in it: a correct
    // picture of being inside foliage, and nothing to look at.  What reads is
    // the boundary -- structure against void.  So each child also carries how
    // far OUT of the node it sits, |c_i - C| against the furthest sibling's,
    // and the score leans toward the outer ones.  They are the children with
    // emptiness on one side, which is what makes an edge an edge.
    //
    // Scale-free on purpose: the measure is each child's rank against its own
    // siblings this frame, not an absolute distance, so it needs no retuning
    // when a map set contracts harder or the camera is at a different depth
    // (the same argument as the pool's rank-keyed table sizes in field.js).
    let maxOut = 0;
    for (let i = 0; i < m; i++) {
      att.childLinear(AE, h, i);
      const b = att.maps[i].b;
      const lx = AE[0] * C[0] + AE[1] * C[1] + AE[2] * C[2] + b[0];
      const ly = AE[3] * C[0] + AE[4] * C[1] + AE[5] * C[2] + b[1];
      const lz = AE[6] * C[0] + AE[7] * C[1] + AE[8] * C[2] + b[2];
      const cx = lx - this.pos[0], cy = ly - this.pos[1], cz = lz - this.pos[2];
      const l = Math.hypot(cx, cy, cz);
      const o = 8 * i;
      S[o] = cx; S[o + 1] = cy; S[o + 2] = cz;
      // Outward: away from the node's own centre of mass, which is where the
      // boundary is.  In the node's local frame, not the camera's.
      const ox = lx - C[0], oy = ly - C[1], oz = lz - C[2];
      const orad = Math.hypot(ox, oy, oz);
      S[o + 3] = ox; S[o + 4] = oy; S[o + 5] = oz; S[o + 6] = orad;
      if (orad > maxOut) maxOut = orad;
      if (l < 1e-12) { S[o + 7] = -1; continue; }
      const d = (cx * this.f[0] + cy * this.f[1] + cz * this.f[2]) / l;
      const apparent = (att.s[i] * att.qMax) / l;
      S[o + 7] = apparent * (0.35 + 0.65 * (0.5 + 0.5 * d));
    }
    let best = -1, bestScore = -1, prevScore = -1;
    for (let i = 0; i < m; i++) {
      const o = 8 * i;
      if (S[o + 7] < 0) continue;
      const edge = maxOut > 1e-12 ? S[o + 6] / maxOut : 0;
      const score = S[o + 7] * (1 - this.edgeBias + this.edgeBias * edge);
      S[o + 7] = score;
      if (score > bestScore) { bestScore = score; best = i; }
      if (i === this._diveChild) prevScore = score;
    }
    // Stay with the child already being aimed at unless another is clearly
    // better.
    //
    // Re-deciding freely every frame is where the flight's worst lurches came
    // from: two children often score within a percent of each other, the
    // winner alternates between frames, and the camera is handed a target in
    // a quite different direction each time. Measured over two minutes, 84
    // frames turned the view by more than 20 degrees during a dive, the worst
    // by 168.
    //
    // A challenger must be `stick` times better, and `stick` is large (2.2).
    // At 1.25 the camera still changed its mind 45 times in two minutes, and
    // each change is a new heading in a different part of the subtree -- the
    // single most noticeable discontinuity left in the flight. Children's
    // scores are close by nature (that is what self-similarity means), so a
    // small margin is no margin at all.
    if (this._diveHash !== h) { this._diveHash = h; this._diveChild = -1; }
    if (this._diveChild >= 0 && prevScore > 0 && bestScore < prevScore * this.stick)
      best = this._diveChild;
    else this._diveChild = best;
    if (best < 0) { this.move(0, 0, speed); return; }
    // Aim at the child's outer FACE, not its centre.
    //
    // Flying at a centre means flying into the middle of the mass, and the
    // middle of the mass is the wash.  The aim point is pushed back out along
    // the same outward direction the score prefers, by `edgeAim` of the
    // child's own radius, so the approach grazes the boundary: structure on
    // one side of the frame, void on the other, all the way in.
    //
    // It has to stay inside `descendMargin`, or the camera would hover on the
    // shell and never get in -- the descent test is distance from the child's
    // CENTRE under s_i * R * descendMargin, so an aim point further out than
    // that is a dive that stops going deeper.
    const o = 8 * best;
    let tx = S[o], ty = S[o + 1], tz = S[o + 2];
    const orad = S[o + 6];
    if (orad > 1e-12) {
      const amt = Math.min(this.edgeAim, this.descendMargin * 0.7)
                * att.s[best] * att.R / orad;
      tx += S[o + 3] * amt; ty += S[o + 4] * amt; tz += S[o + 5] * amt;
    }
    const dist = Math.hypot(tx, ty, tz) || 1e-12;
    // Where it is actually aiming, for tests/pilot.mjs to check against the
    // maps. Written into a preallocated triple rather than returned, because
    // this runs every frame and the stage-4 lesson was that an allocation per
    // frame per object is what cost whole frames to the collector.
    const am = this._aimv || (this._aimv = new Float64Array(4));
    am[0] = tx; am[1] = ty; am[2] = tz; am[3] = best;
    // Steer with a critically damped follower rather than a fixed fraction of
    // the error. A fixed fraction eases OUT of a turn but never into one: the
    // rate is proportional to the error, so the moment the target moves the
    // turn rate jumps with it. Carrying an angular velocity means the view
    // leans into the turn and settles out of it, and its direction is never
    // discontinuous.
    //
    // `ticks` is the step in 60fps ticks, and it is a parameter rather than a
    // constant 1/60 because it used to be constant: the follower integrated a
    // sixtieth of a second however long the frame actually was, so the turn
    // was twice as slow at 30fps as at 60 and every headless measurement was
    // of a flight the browser never flew.
    const v = this._fv || (this._fv = [0, 0, 0]);
    const stiff = turn * 60, damp = 2 * Math.sqrt(stiff), hs = ticks / 60;
    const d0 = tx / dist, d1 = ty / dist, d2 = tz / dist;
    v[0] += ((d0 - this.f[0]) * stiff - v[0] * damp) * hs;
    v[1] += ((d1 - this.f[1]) * stiff - v[1] * damp) * hs;
    v[2] += ((d2 - this.f[2]) * stiff - v[2] * damp) * hs;
    this.f = [this.f[0] + v[0] * hs, this.f[1] + v[1] * hs, this.f[2] + v[2] * hs];
    this.ortho();
    // Lean into the turn.  The sideways part of the angular velocity is
    // already here, and it is what an aircraft banks on.
    this.bank(v[0] * this.r[0] + v[1] * this.r[1] + v[2] * this.r[2], hs);
    // and close a FRACTION of the remaining distance, never a fixed amount:
    // that cannot overshoot into the void, and once the camera is close
    // enough the re-root takes over and the whole thing repeats one level
    // down.  Geometric approach in, geometric rescale on arrival -- which is
    // why the flight has no end.
    //
    // The fraction is 1 - exp(-speed), not speed, and that is what makes the
    // flight the same flight at any frame rate. `speed` is documented as a
    // time constant rather than a velocity; written as a bare fraction it was
    // not one, because closing 5% twice does not leave the same distance as
    // closing 10% once (0.9025 against 0.9000). Compounded over a minute that
    // is a factor of two in where the camera ends up, which is why 30fps and
    // 60fps were different flights. Exponentiating makes two half-steps
    // exactly one whole one, and for small steps it is the same number to
    // within a couple of percent.
    //
    // Scaled by how much of the heading actually points at the aim point --
    // the same guard VoidCamera.cruise has, and for the same reason, which
    // turns out to apply here too. Without it a camera that is facing away
    // from its target closes a fraction of a distance that its own motion is
    // increasing, and that is a geometric series the wrong way round: it
    // accelerates into the void. Left running, a bare dive reaches 3e14 local
    // units and the screen has been black since the first few seconds. The
    // full flight hid it, because the autopilot's climb and retarget phases
    // keep hauling the camera back; a dive on its own never recovers.
    //
    // It is also the softer thing to do: the camera slows through a turn and
    // picks the speed back up as it comes out of it, instead of carrying full
    // speed sideways.
    const align = Math.max(0, (tx * this.f[0] + ty * this.f[1] + tz * this.f[2]) / dist);
    const step = -dist * Math.expm1(-speed) * align;
    this.move(0, 0, step);
    // Skim, rather than fall straight in.
    //
    // Aiming at the outer face bends the approach; this bends the path as
    // well, by sliding the camera sideways along the same outward direction
    // while it closes. The forward component is untouched -- the descent
    // costs what it always did -- but the trajectory curves around the
    // structure instead of through it, which is the difference between
    // flying past a cliff and flying into it.
    //
    // Only the part perpendicular to the heading is used, because the
    // parallel part is already in `step` and would just be a speed change.
    if (this.edgeSlide > 0 && orad > 1e-12) {
      const ux = S[o + 3] / orad, uy = S[o + 4] / orad, uz = S[o + 5] / orad;
      const dp = ux * this.f[0] + uy * this.f[1] + uz * this.f[2];
      let px = ux - dp * this.f[0], py = uy - dp * this.f[1], pz = uz - dp * this.f[2];
      const pl = Math.hypot(px, py, pz);
      if (pl > 1e-9) {
        const k = (step * this.edgeSlide) / pl;
        this.pos[0] += px * k; this.pos[1] += py * k; this.pos[2] += pz * k;
      }
    }
  }

  /**
   * Lean into a turn, the way anything flying does.
   *
   * There is no world up to roll against -- the camera's frame is rescaled
   * every time it descends and nothing outside the current node survives the
   * trip -- so roll is carried as an OFFSET rather than measured as an angle:
   * `_roll` is how far the camera is currently leaning, the target is
   * proportional to how fast it is turning sideways, and when the turn ends
   * the target goes back to zero and so does the lean.  Nothing accumulates
   * and nothing has to be unwound, which is what makes it safe to apply to a
   * basis that re-roots underneath it.
   *
   * `rate` is the component of the heading's angular velocity along `r`;
   * positive is a turn to the right, and a turn to the right rolls right,
   * which tips `u` toward `r` -- the rotation of u about f by a positive
   * angle, since ortho() builds r as f x u.
   */
  bank(rate, h) {
    const max = this.bankMax, want = Math.max(-max, Math.min(max, this.bankGain * rate));
    const k = Math.min(1, h / this.bankTau);
    const d = (want - this._roll) * k;
    if (!(Math.abs(d) > 1e-7)) return;
    this._roll += d;
    const c = Math.cos(d), s = Math.sin(d), f = this.f, u = this.u;
    const cx = f[1] * u[2] - f[2] * u[1],
          cy = f[2] * u[0] - f[0] * u[2],
          cz = f[0] * u[1] - f[1] * u[0];
    this.u = [u[0] * c + cx * s, u[1] * c + cy * s, u[2] * c + cz * s];
    this.ortho();
  }

  /** Move the frame so the camera stays inside the node it is in. */
  reroot(maxSteps = 8) {
    const att = this.att, C = att.C, R = att.R;
    for (let step = 0; step < maxSteps; step++) {
      // ---- descend: is the camera well inside one of the children?
      let best = -1, bestD = Infinity;
      for (let i = 0; i < att.m; i++) {
        att.childLinear(AE, this.hashes[this.hashes.length - 1], i);
        const b = att.maps[i].b;
        const cx = AE[0] * C[0] + AE[1] * C[1] + AE[2] * C[2] + b[0];
        const cy = AE[3] * C[0] + AE[4] * C[1] + AE[5] * C[2] + b[1];
        const cz = AE[6] * C[0] + AE[7] * C[1] + AE[8] * C[2] + b[2];
        const d = Math.hypot(this.pos[0] - cx, this.pos[1] - cy, this.pos[2] - cz);
        if (d < att.s[i] * R * this.descendMargin && d < bestD) { bestD = d; best = i; }
      }
      if (best >= 0) {
        const h = mix(this.hashes[this.hashes.length - 1], best);
        att.childLinear(AE, this.hashes[this.hashes.length - 1], best);
        const s = att.s[best], b = att.maps[best].b;
        // A_i^-1 = (1/s) R^T, applied to (pos - b_i)
        const px = this.pos[0] - b[0], py = this.pos[1] - b[1], pz = this.pos[2] - b[2];
        const inv = 1 / (s * s);
        this.pos = [(AE[0] * px + AE[3] * py + AE[6] * pz) * inv,
                    (AE[1] * px + AE[4] * py + AE[7] * pz) * inv,
                    (AE[2] * px + AE[5] * py + AE[8] * pz) * inv];
        const rotT = (v) => [(AE[0] * v[0] + AE[3] * v[1] + AE[6] * v[2]) / s,
                             (AE[1] * v[0] + AE[4] * v[1] + AE[7] * v[2]) / s,
                             (AE[2] * v[0] + AE[5] * v[1] + AE[8] * v[2]) / s];
        this.r = rotT(this.r); this.u = rotT(this.u); this.f = rotT(this.f);
        this.ortho();
        this.path.push(best);
        this.hashes.push(h);
        this.reroots++;
        continue;
      }
      // ---- ascend: has the camera left this node altogether?
      const d = Math.hypot(this.pos[0] - C[0], this.pos[1] - C[1], this.pos[2] - C[2]);
      if (this.path.length > 0 && d > R * this.ascendMargin) {
        const i = this.path.pop();
        this.hashes.pop();
        att.childLinear(AE, this.hashes[this.hashes.length - 1], i);
        const b = att.maps[i].b;
        const p = this.pos;
        this.pos = [AE[0] * p[0] + AE[1] * p[1] + AE[2] * p[2] + b[0],
                    AE[3] * p[0] + AE[4] * p[1] + AE[5] * p[2] + b[1],
                    AE[6] * p[0] + AE[7] * p[1] + AE[8] * p[2] + b[2]];
        const s = att.s[i];
        const rot = (v) => [(AE[0] * v[0] + AE[1] * v[1] + AE[2] * v[2]) / s,
                            (AE[3] * v[0] + AE[4] * v[1] + AE[5] * v[2]) / s,
                            (AE[6] * v[0] + AE[7] * v[1] + AE[8] * v[2]) / s];
        this.r = rot(this.r); this.u = rot(this.u); this.f = rot(this.f);
        this.ortho();
        this.reroots++;
        continue;
      }
      break;
    }
  }

  /** The true root's affine, expressed in the camera's current frame. */
  rootFrame(out) {
    const att = this.att;
    // Ping-pong between two scratch matrices instead of allocating one per
    // level.  At depth 46 the allocating version made ninety-two short-lived
    // arrays per call and this is called twice a frame, which the collector
    // eventually answers with a pause in the middle of a frame.  The returned
    // A and b are these buffers: valid until the next rootFrame() call, which
    // every caller satisfies by using the result immediately.
    let A = RF0, An = RF1;
    A[0] = 1; A[1] = 0; A[2] = 0;
    A[3] = 0; A[4] = 1; A[5] = 0;
    A[6] = 0; A[7] = 0; A[8] = 1;
    const b = RFB;
    b[0] = 0; b[1] = 0; b[2] = 0;
    let kap = 1;
    for (let k = this.path.length - 1; k >= 0; k--) {
      const i = this.path[k], s = att.s[i];
      att.childLinear(AE, this.hashes[k], i);
      // A_i is a similarity s*R, so A_i^-1 = (1/s^2) A_i^T.  The new frame is
      // old @ A_i^-1 -- note the TRANSPOSE: composing with A_i itself instead
      // sends the whole world off to infinity the moment the camera descends
      // one level, and leaves the screen empty.
      const inv = 1 / (s * s);
      for (let p = 0; p < 3; p++)
        for (let q = 0; q < 3; q++)
          An[p * 3 + q] = (A[p * 3] * AE[q * 3] + A[p * 3 + 1] * AE[q * 3 + 1] +
                           A[p * 3 + 2] * AE[q * 3 + 2]) * inv;
      const bi = att.maps[i].b;
      const b0 = b[0] - (An[0] * bi[0] + An[1] * bi[1] + An[2] * bi[2]);
      const b1 = b[1] - (An[3] * bi[0] + An[4] * bi[1] + An[5] * bi[2]);
      const b2 = b[2] - (An[6] * bi[0] + An[7] * bi[1] + An[8] * bi[2]);
      b[0] = b0; b[1] = b1; b[2] = b2;
      const sw = A; A = An; An = sw;
      kap /= s;
    }
    out.A = A; out.b = b; out.kap = kap;
    return out;
  }
}

const AE = new Float64Array(9);
const RF0 = new Float64Array(9), RF1 = new Float64Array(9), RFB = new Float64Array(3);

// ------------------------------------------------------------- the stream

export class AdaptiveStream {
  constructor(att, maxSplats = 400000) {
    this.att = att;
    this.max = maxSplats;
    // One allocation, four views into it, at the same indices they always
    // had: pos[3*i], cov[6*i], rgba[4*i], fx[i].  Nothing that reads them
    // changes, and that is the point -- cpu_raster.js and the parity tests
    // index these arrays directly and compare them bit for bit.
    //
    // NOT interleaved per splat, which was the first idea and is a
    // pessimisation here.  The GPU reads these as four instanced streams with
    // divisor 1, so consecutive instances read consecutive addresses within
    // each stream: a 64-byte line from pos serves 5.3 instances, from cov
    // 2.7, and the whole set costs 41/64 = 0.64 lines an instance.  One
    // 44-byte interleaved record -- 41 bytes padded to a 4-byte stride for
    // the float views -- costs 0.69.  Structure-of-arrays is the right layout
    // for a full sequential sweep, and that is the only access pattern there
    // is here.
    //
    // What one allocation does buy: the whole frame's splats are one
    // ArrayBuffer, so a worker can hand it over with a single transfer
    // instead of four, and a ring slot is one object rather than four.
    const STRIDE = 3 * 4 + 6 * 4 + 4 + 1;        // 41 bytes a splat
    this.bytes = new ArrayBuffer(STRIDE * maxSplats);
    this.pos = new Float32Array(this.bytes, 0, 3 * maxSplats);
    this.cov = new Float32Array(this.bytes, 12 * maxSplats, 6 * maxSplats);
    this.rgba = new Uint8Array(this.bytes, 36 * maxSplats, 4 * maxSplats);
    // Which look-effects each splat's OBJECT permits, one byte per splat.
    //
    // A separate array rather than bits stolen from rgba's alpha. Alpha is
    // the fog and opacity channel and every existing measurement depends on
    // its full range, and more importantly pos/cov/rgba are what
    // parity_check.py and gpu_transcribe compare against the Python
    // reference bit for bit. Adding a field beside them cannot disturb that;
    // repacking one of them would have put a cosmetic feature in a position
    // to break the generator's correctness proof.
    this.fx = new Uint8Array(this.bytes, 40 * maxSplats, maxSplats);
    this.curFx = 0xff;   // what the object being expanded right now allows
    // A stable identity per splat, so a test can tell "this blob faded" from
    // "this blob vanished".  Not uploaded; the renderer never needs it.
    this.id = new Uint32Array(maxSplats);
    this.count = 0;
    this.nodes = 0;
    this.maxLevel = 96;
    // whose colours the emitters read -- the attractor's own unless add() was
    // handed a tint, see setTint()
    this.pal = att.palette; this.blobPal = att.blobCol;
    this._tintPal = null; this._tintBlob = null;
    // thresholds in units of the blob's own 1-sigma screen radius, in pixels:
    // below `lo` a subtree is one blob, above `hi` it is fully expanded, and in
    // between both are drawn and cross-faded.  A band, not a hard cut, is what
    // stops nodes thrashing at the boundary -- it subsumes hysteresis, because
    // nothing ever flips discontinuously in the first place.
    this.lo = 0.9;
    this.hi = 3.2;
    // The near side of the same idea, for the node's OWN lump.
    //
    // `lo`/`hi` above decide when a subtree is too small to be worth
    // expanding. Nothing decided when a lump was too BIG to be worth drawing,
    // so every visited node painted its own Gaussian at any size, including
    // one that filled the screen -- and a screen-filling Gaussian is a smooth
    // wash with no structure in it, laid over the finer lumps of its own
    // children, which the walk is already drawing underneath.
    //
    // So: fade the lump out as its projected size grows, and what is left is
    // the cloud of its children. That is not invented detail, it is the same
    // attractor one level down, jittered per node by hash -- which is why it
    // can only move `tests/fidelity.mjs` the right way.
    //
    // In fractions of the viewport height, not pixels, so a phone and a
    // desktop fade at the same apparent size. Full below rodFadeLo, gone
    // above rodFadeHi.
    this.rodFade = true;
    this.rodFadeLo = 0.10;
    this.rodFadeHi = 0.40;
    this.lodScale = 4.0;                      // feedback, to hold the budget
    // A TIME budget, not a splat budget.  How many nodes a machine can expand
    // per frame varies by an order of magnitude between a phone and a desktop,
    // and the right answer is "as much detail as fits in the frame" on both.
    // The cross-fade band is what makes a moving threshold safe: nothing ever
    // flips, it only slides.
    this.targetMs = 6.0;
    this.target = Math.min(200000, maxSplats * 0.6);   // used if budgetBy==='splats'
    this.budgetBy = 'time';
    this.lastMs = 0;
    this.budgetHit = false;
    // The controller itself lives in budget.js, and these three are what the
    // viewer fills in each frame so that it can see the GPU as well as the
    // walk.  Left null they are simply absent from the error term, which is
    // what every headless test wants: tour, legible and fidelity have no
    // vsync and no timer query, and must keep measuring the walk-only
    // controller they have always measured.
    this.cpuMs = null;      // the frame's JS time, submission not completion
    this.gpuMs = null;      // EXT_disjoint_timer_query_webgl2, if there is one
    this.interval = null;   // ms between frame starts
    this.useFrame = true;
    this.bud = makeBudget({ lodScale: 4.0 });
    // Front-to-back occlusion.  The attractor's projected area grows with
    // depth -- 4^k subtrees each covering (0.6^k)^2 is 1.44^k -- so overdraw
    // rises without bound as the LOD cut descends, and almost all of it is
    // buried inside an opaque canopy.  Walking near-to-far with a coarse
    // transmittance buffer lets a whole subtree be dropped the moment every
    // tile it could touch is already opaque.  This is the one optimisation
    // that a global per-frame sort cannot do at all: a sort has to produce
    // all N before anything can be drawn.
    this.occlude = false;   // measured: see tests/flight.mjs
    this.tile = 16;
    this.tEps = 0.012;          // below this transmittance a tile is closed
    this.tiles = null;
    this.occSkipped = 0;
    const D = 100;
    this._CA = new Float64Array(9 * 4 * D);   // child linear parts, per level
    this._CB = new Float64Array(3 * 4 * D);   // child translations
    this._CC = new Float64Array(3 * 4 * D);   // child subtree centres
    this._KEY = new Float64Array(5 * D);      // sort keys, per level
    this._ORD = new Int32Array(5 * D);
    this._CHA = new Float64Array(9 * D);      // the child affine handed down
    this._CHB = new Float64Array(3 * D);
    this._CHH = new Uint32Array(4 * D);       // child hashes, computed once
  }

  /** view: {focal, tanX, tanY, znear, W, H} ; cam: FlyCamera */
  /**
   * Start a frame.  Splits out from expand() because stage 3 puts several
   * attractors in one void and they all have to land in ONE stream, in one
   * draw order: the budget, the transmittance tiles and the write head are
   * properties of the frame, not of any one attractor.
   */
  begin(cam, view) {
    this.count = 0; this.nodes = 0; this.budgetHit = false; this.occSkipped = 0;
    // emitted back to front, so the walk goes front to back and writes
    // downwards: the valid stream is [head, max)
    this.head = this.max;
    const nx = Math.ceil(view.W / this.tile), ny = Math.ceil(view.H / this.tile);
    if (!this.tiles || this.nx !== nx || this.ny !== ny) {
      this.nx = nx; this.ny = ny;
      this.tiles = new Float32Array(nx * ny);
    }
    this.tiles.fill(1);
    this.invTileArea = 1 / (this.tile * this.tile);
    this.cam = cam; this.view = view;
    this.secX = Math.sqrt(1 + view.tanX * view.tanX);
    this.secY = Math.sqrt(1 + view.tanY * view.tanY);
    this.W = view.W; this.H = view.H;
    this._t0 = now();
    return this;
  }

  /**
   * Point the emitters at this object's colours.
   *
   * A field of cells shares a small pool of Attractor objects, so two cells
   * showing the same pool slot share its palette as well -- and a palette is
   * the most legible thing about an object at a distance, so without this they
   * read as literal copies however differently they are jittered.
   *
   * The tint is applied to the PALETTE, once per object, not to each splat.
   * A palette is at most a couple of dozen entries and a frame is up to three
   * hundred thousand splats, so per-splat would be four orders of magnitude
   * more work for the same answer.  It also avoids a trap: `rgba` is a
   * Uint8Array, whose assignment truncates modulo 256 rather than clamping, so
   * a tint above 1 on a near-white palette entry (0.95 * 1.25 * 255 = 302)
   * would wrap round to 46 and the brightest splats would come out dark.
   * Clamping twenty-four entries is free; clamping every splat is not.
   *
   * A neutral or absent tint keeps the attractor's own arrays by reference, so
   * the single-object path is untouched -- tests/parity.mjs and
   * py/parity_check.py compare colours bit-for-bit and must stay exact.
   */
  setTint(att, tint) {
    if (!tint || (tint[0] === 1 && tint[1] === 1 && tint[2] === 1)) {
      this.pal = att.palette; this.blobPal = att.blobCol;
      return;
    }
    const P = att.P;
    if (!this._tintPal || this._tintPal.length < P) {
      this._tintPal = []; this._tintBlob = [];
      for (let k = 0; k < P; k++) { this._tintPal.push([0, 0, 0]); this._tintBlob.push([0, 0, 0]); }
    }
    for (let k = 0; k < P; k++) {
      const a = att.palette[k], b = att.blobCol[k];
      const ta = this._tintPal[k], tb = this._tintBlob[k];
      for (let c = 0; c < 3; c++) {
        ta[c] = Math.min(1, a[c] * tint[c]);
        tb[c] = Math.min(1, b[c] * tint[c]);
      }
    }
    this.pal = this._tintPal; this.blobPal = this._tintBlob;
  }

  /**
   * Walk one attractor into the frame, rooted by the affine `frame`
   * {A, b, kap} that places its root in the camera's working coordinates.
   *
   * Call these NEAREST FIRST.  Within a subtree the draw order is exact, but
   * between two separate attractors nothing orders them except the order they
   * are added, and the stream is written backwards, so the first one added
   * ends up drawn last -- in front.
   */
  add(att, frame, hash, tint) {
    this.att = att;
    this.setTint(att, tint);
    const R0 = this._CHA, T0 = this._CHB;
    for (let k = 0; k < 9; k++) R0[k] = frame.A[k];
    T0[0] = frame.b[0]; T0[1] = frame.b[1]; T0[2] = frame.b[2];
    this.visit(R0, 0, T0, 0, frame.kap, hash, 0, 1.0, 0);
    return this;
  }

  /**
   * Close the frame and hold the budget by nudging the LOD threshold -- a
   * little per frame, so the cross-fade absorbs the change.  The response is
   * proportional to how far off we are, so a cold start converges in a few
   * frames instead of a few hundred, but it is damped enough not to pump.
   */
  end() {
    const c = this.count;
    this.offset = this.head;
    this.lastMs = now() - this._t0;
    if (this.budgetBy !== 'frozen') {
      // 'both' is what a user-facing quality dial needs. A time budget alone
      // says nothing about how many splats come out of it -- a walk that
      // emits cheap splats fast can spend 6 ms and hand the GPU a quarter of
      // a million of them -- so turning the dial down shortened the walk
      // without reliably shortening the draw. Taking the worse of the two
      // errors lets whichever budget binds first drive the threshold: the
      // clock on a slow machine, the splat count on a fast one.
      // The error term, the damping and the bounds are all in budget.js now,
      // because the frame term needed a clock and an observed refresh period
      // and this method is not the place for either. `lodScale` stays a plain
      // field rather than becoming an accessor on the budget state: it is read
      // once per node in visit(), the hottest loop in the program, and a
      // property read through a getter there is not worth the tidiness.
      const b = this.bud;
      b.targetMs = this.targetMs; b.target = this.target;
      b.budgetBy = this.budgetBy; b.useFrame = this.useFrame;
      b.lodScale = this.lodScale;
      budgetStep(b, { walkMs: this.lastMs, count: c, cpuMs: this.cpuMs,
                      gpuMs: this.gpuMs, interval: this.interval,
                      hit: this.budgetHit });
      this.lodScale = b.lodScale;
    }
    return c;
  }

  /** One attractor, the stage-2 case. */
  expand(cam, view) {
    this.begin(cam, view);
    this.add(this.att, cam.rootFrame({}), cam.hashes[0], null);
    return this.end();
  }

  /** `AB`/`ai` and `BB`/`bi0` are a buffer and an offset, not a slice: taking
   *  a subarray per child allocates four typed-array views per node, which is
   *  most of the cost of the whole traversal. */
  visit(AB, ai, BB, bi0, kap, h, level, fade, sp) {
    if (this.budgetHit) return;
    const att = this.att, cam = this.cam, view = this.view;
    if (this.head - 2 < 0) { this.budgetHit = true; return; }
    if (sp >= 98) return;
    this.nodes++;

    const C = att.C;
    const a0 = ai, b0 = bi0;
    const A00 = AB[a0], A01 = AB[a0 + 1], A02 = AB[a0 + 2];
    const A10 = AB[a0 + 3], A11 = AB[a0 + 4], A12 = AB[a0 + 5];
    const A20 = AB[a0 + 6], A21 = AB[a0 + 7], A22 = AB[a0 + 8];
    const bx = BB[b0], by = BB[b0 + 1], bz = BB[b0 + 2];
    const cx = A00 * C[0] + A01 * C[1] + A02 * C[2] + bx - cam.pos[0];
    const cy = A10 * C[0] + A11 * C[1] + A12 * C[2] + by - cam.pos[1];
    const cz = A20 * C[0] + A21 * C[1] + A22 * C[2] + bz - cam.pos[2];
    const Rw = kap * att.R;

    const z = cx * cam.f[0] + cy * cam.f[1] + cz * cam.f[2];
    if (z + Rw < view.znear) return;
    const x = cx * cam.r[0] + cy * cam.r[1] + cz * cam.r[2];
    const y = cx * cam.u[0] + cy * cam.u[1] + cz * cam.u[2];
    const mx = Rw * this.secX, my = Rw * this.secY;
    if (x - z * view.tanX > mx || -x - z * view.tanX > mx) return;
    if (y - z * view.tanY > my || -y - z * view.tanY > my) return;

    // the blob's own 1-sigma radius on screen, in pixels
    const sigma = view.focal * kap * att.qMax / Math.max(z, view.znear);

    // ---- is every tile this subtree's mass can reach already opaque?
    // Tested against the subtree's own 2.5-sigma extent, not its bounding
    // sphere: the sphere is ~2.7x larger here and has to contain every splat's
    // 3-sigma tail, so insisting all of *its* tiles be closed almost never
    // fires.  2.5 sigma holds essentially all the mass that could show.
    if (this.occlude && z > view.znear &&
        this.closed(x, y, z, 2.5 * sigma)) { this.occSkipped++; return; }
    const lo = this.lo * this.lodScale, hi = this.hi * this.lodScale;
    let t = (sigma - lo) / (hi - lo);
    t = t < 0 ? 0 : t > 1 ? 1 : t;
    if (level >= this.maxLevel) t = 0;

    this.curHash = h;
    if (t < 1) this.emitBlob(AB, a0, cx, cy, cz, level, fade * (1 - t),
                             x, y, z, sigma);
    if (t <= 0) return;
    const ft = fade * t;

    // ---- the node's own rod
    const tA = att.trunkA, mu = att.mu0;
    const rx = A02 * tA[8] * mu[2] + bx;
    const ry = A12 * tA[8] * mu[2] + by;
    const rz = A22 * tA[8] * mu[2] + bz;
    const kRod = (rx - cam.pos[0]) * cam.f[0] + (ry - cam.pos[1]) * cam.f[1] +
                 (rz - cam.pos[2]) * cam.f[2];

    // How big is this node's own lump about to be drawn?  Same quantity the
    // occlusion test uses below: its 1-sigma radius on screen.  A lump whose
    // centre has gone behind the eye lands at znear and so fades out whole,
    // which is the case the near band in the shader can only soften.
    let rodF = 1;
    if (this.rodFade) {
      const rs = view.focal * kap * att.rodMax / Math.max(kRod, view.znear);
      const lo = this.rodFadeLo * view.H, hi = this.rodFadeHi * view.H;
      rodF = (hi - rs) / (hi - lo);
      rodF = rodF < 0 ? 0 : rodF > 1 ? 1 : rodF;
    }

    // ---- the m children, and where their mass is
    const CA = this._CA, CB = this._CB, CC = this._CC;
    const ca = 36 * sp, cb = 12 * sp;
    // scratch is indexed by recursion level: this function calls itself from
    // inside the loop that reads `ord`, so sharing one buffer would corrupt it
    const key = this._KEY, ord = this._ORD, ko = 5 * sp;
    const CM = att._cmB, MB = att.mapB, mm = att.m, tm = att.tableMask;
    // The interpolated read. `bt` is constant for the whole walk, so the
    // branch is perfectly predicted; at 1 -- every caller that does not
    // interpolate, and every slot that is up to date -- the second table is
    // never touched and this costs a comparison per child.
    const CMA = att._cmA, bt = att._bt;
    const fx = cam.f[0], fy = cam.f[1], fz = cam.f[2];
    const px0 = cam.pos[0], py0 = cam.pos[1], pz0 = cam.pos[2];
    const CH = this._CHH, hh = 4 * sp;
    key[ko] = kRod; ord[ko] = 0;
    for (let i = 0; i < mm; i++) {
      const chh = mix(h, i);
      CH[hh + i] = chh;
      const e = 9 * ((chh & tm) * mm + i);
      const o = ca + 9 * i;
      let E0 = CM[e], E1 = CM[e + 1], E2 = CM[e + 2];
      let E3 = CM[e + 3], E4 = CM[e + 4], E5 = CM[e + 5];
      let E6 = CM[e + 6], E7 = CM[e + 7], E8 = CM[e + 8];
      // Not `bt < 1`: the fraction usually sits BETWEEN 1 and 2, because
      // this is an extrapolation from the two most recent builds up to the
      // parameter the morph is at right now, not an interpolation that lags
      // one rebuild behind it. Testing for `< 1` silently drew the newest
      // build and nothing else -- the judder back, with the cost paid.
      if (bt !== 1) {
        // p, not a: `a0` is this node's own affine, four lines up.
        const p0 = CMA[e], p1 = CMA[e + 1], p2 = CMA[e + 2];
        const p3 = CMA[e + 3], p4 = CMA[e + 4], p5 = CMA[e + 5];
        const p6 = CMA[e + 6], p7 = CMA[e + 7], p8 = CMA[e + 8];
        E0 = p0 + (E0 - p0) * bt; E1 = p1 + (E1 - p1) * bt; E2 = p2 + (E2 - p2) * bt;
        E3 = p3 + (E3 - p3) * bt; E4 = p4 + (E4 - p4) * bt; E5 = p5 + (E5 - p5) * bt;
        E6 = p6 + (E6 - p6) * bt; E7 = p7 + (E7 - p7) * bt; E8 = p8 + (E8 - p8) * bt;
      }
      CA[o] = A00 * E0 + A01 * E3 + A02 * E6;
      CA[o + 1] = A00 * E1 + A01 * E4 + A02 * E7;
      CA[o + 2] = A00 * E2 + A01 * E5 + A02 * E8;
      CA[o + 3] = A10 * E0 + A11 * E3 + A12 * E6;
      CA[o + 4] = A10 * E1 + A11 * E4 + A12 * E7;
      CA[o + 5] = A10 * E2 + A11 * E5 + A12 * E8;
      CA[o + 6] = A20 * E0 + A21 * E3 + A22 * E6;
      CA[o + 7] = A20 * E1 + A21 * E4 + A22 * E7;
      CA[o + 8] = A20 * E2 + A21 * E5 + A22 * E8;
      const b3 = 3 * i, d = cb + 3 * i;
      const B0 = MB[b3], B1 = MB[b3 + 1], B2 = MB[b3 + 2];
      CB[d] = A00 * B0 + A01 * B1 + A02 * B2 + bx;
      CB[d + 1] = A10 * B0 + A11 * B1 + A12 * B2 + by;
      CB[d + 2] = A20 * B0 + A21 * B1 + A22 * B2 + bz;
      CC[d] = CA[o] * C[0] + CA[o + 1] * C[1] + CA[o + 2] * C[2] + CB[d];
      CC[d + 1] = CA[o + 3] * C[0] + CA[o + 4] * C[1] + CA[o + 5] * C[2] + CB[d + 1];
      CC[d + 2] = CA[o + 6] * C[0] + CA[o + 7] * C[1] + CA[o + 8] * C[2] + CB[d + 2];
      key[ko + 1 + i] = (CC[d] - px0) * fx + (CC[d + 1] - py0) * fy +
                        (CC[d + 2] - pz0) * fz;
      ord[ko + 1 + i] = 1 + i;
    }
    sort5Asc(key, ord, ko);          // near to far; the stream is written backwards

    // ---- emit far to near: that IS the ordered expansion, written depth-first
    const chA = this._CHA, chB = this._CHB, ha = 9 * sp, hb = 3 * sp;
    for (let k = 0; k < 5; k++) {
      const it = ord[ko + k];
      if (it === 0) {
        if (rodF > 0) {
          this.curHash = h;
          this.emitRod(AB, a0, rx, ry, rz, level, ft * rodF, kap);
        }
      } else {
        const i = it - 1, o = ca + 9 * i, d = cb + 3 * i;
        for (let q = 0; q < 9; q++) chA[ha + q] = CA[o + q];
        chB[hb] = CB[d]; chB[hb + 1] = CB[d + 1]; chB[hb + 2] = CB[d + 2];
        this.visit(chA, ha, chB, hb, kap * att.sArr[i], CH[hh + i], level + 1, ft, sp + 1);
      }
    }
  }

  /** Is every screen tile this sphere can touch already opaque?  Only small
   *  nodes are tested: a large one is near, and near things are not hidden. */
  closed(x, y, z, rp) {
    const v = this.view, T = this.tiles, ts = this.tile;
    const sx = v.focal * x / z + this.W * 0.5, sy = this.H * 0.5 - v.focal * y / z;
    let x0 = ((sx - rp) / ts) | 0, x1 = ((sx + rp) / ts) | 0;
    let y0 = ((sy - rp) / ts) | 0, y1 = ((sy + rp) / ts) | 0;
    if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
    if (x1 >= this.nx) x1 = this.nx - 1;
    if (y1 >= this.ny) y1 = this.ny - 1;
    if (x1 < x0 || y1 < y0) return false;
    if ((x1 - x0 + 1) * (y1 - y0 + 1) > 64) return false;      // too big to be hidden
    const eps = this.tEps;
    for (let ty = y0; ty <= y1; ty++) {
      const row = ty * this.nx;
      for (let tx = x0; tx <= x1; tx++) if (T[row + tx] > eps) return false;
    }
    return true;
  }

  /** Account for a splat we just emitted, in the coarse transmittance buffer.
   *  A splat covering a fraction of a tile closes that fraction of it, which
   *  is the mean-field estimate the per-pixel blend converges to. */
  occludeWith(x, y, z, sigp, a) {
    const v = this.view, T = this.tiles, ts = this.tile;
    if (z < v.znear || a <= 0) return;
    const sx = v.focal * x / z + this.W * 0.5, sy = this.H * 0.5 - v.focal * y / z;
    const rp = Math.max(sigp, 0.5);
    let x0 = ((sx - rp) / ts) | 0, x1 = ((sx + rp) / ts) | 0;
    let y0 = ((sy - rp) / ts) | 0, y1 = ((sy + rp) / ts) | 0;
    if (x0 < 0) x0 = 0; if (y0 < 0) y0 = 0;
    if (x1 >= this.nx) x1 = this.nx - 1;
    if (y1 >= this.ny) y1 = this.ny - 1;
    if (x1 < x0 || y1 < y0) return;
    const area = Math.PI * rp * rp;
    const nt = (x1 - x0 + 1) * (y1 - y0 + 1);
    const frac = Math.min(1, area / (nt * this.tile * this.tile));
    const k = 1 - a * frac;
    for (let ty = y0; ty <= y1; ty++) {
      const row = ty * this.nx;
      for (let tx = x0; tx <= x1; tx++) T[row + tx] *= k;
    }
  }

  emitRod(AB, a0, px, py, pz, level, fade, kap) {
    const n = --this.head;
    this.count++;
    const att = this.att, tA = att.trunkA;
    this.pos[3 * n] = px; this.pos[3 * n + 1] = py; this.pos[3 * n + 2] = pz;
    // cov = (A T) SIG0 (A T)^T, and T SIG0 T^T is diagonal for this trunk map
    const o = 6 * n;
    const a = AB[a0] * tA[0], b1 = AB[a0 + 1] * tA[4], c = AB[a0 + 2] * tA[8];
    const d = AB[a0 + 3] * tA[0], e = AB[a0 + 4] * tA[4], f = AB[a0 + 5] * tA[8];
    const g = AB[a0 + 6] * tA[0], i2 = AB[a0 + 7] * tA[4], j = AB[a0 + 8] * tA[8];
    const s0 = att.sig0[0], s1 = att.sig0[1], s2 = att.sig0[2];
    this.cov[o] = a * a * s0 + b1 * b1 * s1 + c * c * s2;
    this.cov[o + 1] = a * d * s0 + b1 * e * s1 + c * f * s2;
    this.cov[o + 2] = a * g * s0 + b1 * i2 * s1 + c * j * s2;
    this.cov[o + 3] = d * d * s0 + e * e * s1 + f * f * s2;
    this.cov[o + 4] = d * g * s0 + e * i2 * s1 + f * j * s2;
    this.cov[o + 5] = g * g * s0 + i2 * i2 * s1 + j * j * s2;
    this.id[n] = h32(Math.imul(this.curHash, 2) >>> 0);
    this.fx[n] = this.curFx;
    const col = this.pal[level % att.P], q = 4 * n;
    this.rgba[q] = col[0] * 255; this.rgba[q + 1] = col[1] * 255;
    this.rgba[q + 2] = col[2] * 255;
    const al = att.alphaRod * fade;
    this.rgba[q + 3] = Math.max(0, Math.min(255, al * 255)) | 0;
    if (this.occlude) {
      const cam = this.cam, v = this.view;
      const dx = px - cam.pos[0], dy = py - cam.pos[1], dz = pz - cam.pos[2];
      const z = dx * cam.f[0] + dy * cam.f[1] + dz * cam.f[2];
      if (z > v.znear)
        this.occludeWith(dx * cam.r[0] + dy * cam.r[1] + dz * cam.r[2],
                         dx * cam.u[0] + dy * cam.u[1] + dz * cam.u[2], z,
                         v.focal * kap * att.rodMax / z, al);
    }
  }

  emitBlob(AB, a0, cx, cy, cz, level, fade, x, y, z, sigma) {
    const n = --this.head;
    this.count++;
    const att = this.att, Q = att.Q;
    this.pos[3 * n] = cx + this.cam.pos[0];
    this.pos[3 * n + 1] = cy + this.cam.pos[1];
    this.pos[3 * n + 2] = cz + this.cam.pos[2];
    // cov = A Q A^T
    const a00 = AB[a0], a01 = AB[a0 + 1], a02 = AB[a0 + 2];
    const a10 = AB[a0 + 3], a11 = AB[a0 + 4], a12 = AB[a0 + 5];
    const a20 = AB[a0 + 6], a21 = AB[a0 + 7], a22 = AB[a0 + 8];
    const q0 = Q[0], q1 = Q[1], q2 = Q[2], q4 = Q[4], q5 = Q[5], q8 = Q[8];
    const r0 = a00 * q0 + a01 * q1 + a02 * q2;
    const r1 = a00 * q1 + a01 * q4 + a02 * q5;
    const r2 = a00 * q2 + a01 * q5 + a02 * q8;
    const s0 = a10 * q0 + a11 * q1 + a12 * q2;
    const s1 = a10 * q1 + a11 * q4 + a12 * q5;
    const s2 = a10 * q2 + a11 * q5 + a12 * q8;
    const u0 = a20 * q0 + a21 * q1 + a22 * q2;
    const u1 = a20 * q1 + a21 * q4 + a22 * q5;
    const u2 = a20 * q2 + a21 * q5 + a22 * q8;
    const o = 6 * n;
    this.cov[o] = r0 * a00 + r1 * a01 + r2 * a02;
    this.cov[o + 1] = r0 * a10 + r1 * a11 + r2 * a12;
    this.cov[o + 2] = r0 * a20 + r1 * a21 + r2 * a22;
    this.cov[o + 3] = s0 * a10 + s1 * a11 + s2 * a12;
    this.cov[o + 4] = s0 * a20 + s1 * a21 + s2 * a22;
    this.cov[o + 5] = u0 * a20 + u1 * a21 + u2 * a22;
    this.id[n] = h32((Math.imul(this.curHash, 2) + 1) >>> 0);
    this.fx[n] = this.curFx;
    const col = this.blobPal[level % att.P], q = 4 * n;
    this.rgba[q] = col[0] * 255; this.rgba[q + 1] = col[1] * 255;
    this.rgba[q + 2] = col[2] * 255;
    const al = att.alphaBlob * fade;
    this.rgba[q + 3] = Math.max(0, Math.min(255, al * 255)) | 0;
    if (this.occlude) this.occludeWith(x, y, z, sigma, al);
  }
}



/** The optimal 5-element network, 9 compare-swaps, on a slice at offset `b`.
 *  Ascending, because the walk goes near to far and writes the stream
 *  backwards; reversed, that is exactly the back-to-front ordered expansion. */
function sort5Asc(k, o, b) {
  let t, u, x, y;
  for (let i = 0; i < 18; i += 2) {
    x = b + NET[i]; y = b + NET[i + 1];
    if (k[x] > k[y]) {
      t = k[x]; k[x] = k[y]; k[y] = t;
      u = o[x]; o[x] = o[y]; o[y] = u;
    }
  }
}
const NET = new Int32Array([0, 1, 3, 4, 2, 4, 2, 3, 0, 3, 0, 2, 1, 4, 1, 3, 1, 2]);
