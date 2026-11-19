// The endless field: the void as a pure function of integer cell coordinates.
//
// Stage 3's void is seven attractors on a ring (`ringLayout` in world.js).  It
// is a fixed set, so it has an edge, and flying past the edge leaves a black
// screen.  This replaces it with a lattice: space is cut into cubes of side
// L, and everything a cube contains -- whether it holds an attractor at all,
// where, how big, which way up, what colour, and which realisation of the maps
// -- is a pure function of (ix, iy, iz) and a seed.  Nothing is stored, so
// flying forever costs nothing and eviction is free: come back and the same
// hash gives the same object, bit for bit.
//
// The whole design rests on one arithmetic fact, worked out below in
// "separation": every object is contained in the sphere INSCRIBED IN ITS OWN
// CELL, whatever rho is.  So two objects in different cells can never
// intersect.  That is not cosmetic.  Draw order between two attractors comes
// only from the order they are streamed into the AdaptiveStream -- there is no
// sort across objects -- which is correct exactly while their bounding spheres
// are disjoint.  Let two spheres overlap and the picture shows a flat sorting
// seam slicing through both of them.

import { h32, mix, stream } from './ifs_core.js';
import { Attractor, PALETTES } from './adaptive.js';
import { SETS, SET_NAMES, build, morph, morphPath,
         crPalette, palBuffer, palStopsAt } from './maps.js';

/**
 * The conservative, morph-invariant reach of an attractor in its own local
 * frame: the radius about its LOCAL ORIGIN of a sphere containing everything
 * it will ever draw, for unit scale.
 *
 * It is not att.R.  att.R is the radius about the centroid att.C, and both C
 * and R move as the shape morphs, so a bound taken from this frame's R is not
 * a bound on the shape it is turning into -- and the field places objects once
 * and then morphs them forever.  |C| + R about the origin is invariant to that
 * because the origin does not move.
 *
 * Measured over BOTH ways a shape can move:
 *
 *   all straight morph pairs, 11 sets x 110 pairs x 11 steps   3.75369
 *   the closed Catmull-Rom path, 11 segments x 500 steps       3.91915
 *
 * The spline is the larger, and it is the one that matters, because the
 * spline is what the pool actually flies.  Catmull-Rom leaves the hull of its
 * control points by design -- that overshoot is most of what makes the motion
 * look alive, and it is deliberately kept -- so the path reaches shapes no
 * straight morph between two sets ever visits.  Left at the straight-morph
 * figure the field would size its objects for a reach they exceed, two
 * neighbours would intersect, and the draw order between them -- valid only
 * while their spheres are disjoint -- would show as a seam cutting through
 * both.
 *
 * How much larger depends on the ORDER of the loop, which is worth knowing
 * before editing SET_NAMES.  An earlier ordering put `spiral` next to `shell`,
 * two of the three highest-contraction sets, and the overshoot between them
 * reached 5.12 -- 36% over -- because R grows like 1/(1 - contraction) and
 * the last few percent are most of the size.  Interleaving crystalline sets
 * between the organic ones separated them and brought it back to 3.92.
 *
 * 3.95 is 3.91915 with 0.8% of margin.  tests/field.mjs check 2 re-measures
 * it over the spline and fails loudly if a new set, or a new ORDER, pushes
 * past.
 */
export const REACH_MAX = 3.95;
// Re-measured after `tree` was taken out of the cycle: max |C|+R over the
// whole spline is now 3.7628 (at u = 3.02), down from the 3.95 this bound was
// set to. Left at 3.95 rather than tightened: a smaller bound lets every
// object be proportionally LARGER in its cell, which changes the density of
// the field, and that is a look decision rather than a consequence of
// dropping one set. The bound only has to be valid, and it is.

/**
 * The control points of the closed morph loop, in a fixed order.  Built once:
 * morphPath() is called on every rebuild and must not be allocating an array
 * of the same seven objects each time.
 */
const SPECS = SET_NAMES.map((n) => SETS[n]);

// Which stream index draws what.  They are spelled out because adding a draw
// in the middle would silently reshuffle every cell in the field: the whole
// point is that today's field and next year's field are the same field.
const K_ROLE = 0;         // empty / subdivides / holds an object
const K_JX = 1, K_JY = 2, K_JZ = 3;
const K_U = 4;            // size
const K_YAW = 5, K_TILT = 6;
const K_HUE = 7, K_AMP = 8;
// motion: drift amplitude, rate and phase per axis, then the size pulse and
// the spin.  Appended at the end, never inserted, so every field that existed
// before this stage draws exactly what it drew before.
const K_DX = 9, K_DY = 10, K_DZ = 11;
const K_WX = 12, K_WY = 13, K_WZ = 14;
const K_PX = 15, K_PY = 16, K_PZ = 17;
const K_PULSE = 18, K_PULSEP = 19, K_SPIN = 20;
// Which look-effects this object asks for. Drawn per cell from the same hash
// stream as everything else, so a field shows a mix rather than one uniform
// treatment -- and so it is DETERMINISTIC: the same cell asks for the same
// effects every time it is generated, rather than flickering as cells are
// evicted and regenerated under a moving camera.
const K_FXIRID = 21, K_FXRIM = 22;
// Bits, matching uFxMask in the shader and FX in cpu_raster.
export const FX_IRID = 1;
export const FX_RIM = 2;
// How much of the field opts into each. Not 1.0: an effect that is simply
// everywhere stops reading as a property of an object and becomes a property
// of the renderer, and the contrast between a rimmed object and a plain one
// beside it is most of what makes either visible.
const P_IRID = 0.55, P_RIM = 0.65;

/**
 * How much of the slack between an object and its cell wall may be spent on
 * drifting, rather than on sitting still somewhere off-centre.
 *
 * This is the one number that decides whether motion is free. The separation
 * proof bounds the ENVELOPE, not the instantaneous position: an object whose
 * every reachable position lies inside its own cell cannot ever intersect a
 * neighbour, no matter where in that range it is at any moment. So the
 * requirement is |base| + amplitude <= 1 in units of the wander half-width,
 * and the amplitude is drawn from what is left after the base, never added
 * on top of it. At 1.0 an object dead-centre in its cell could swing the full
 * half-width; 0.55 keeps some stillness in the field so the motion reads as
 * drift rather than as everything jittering at once.
 */
const DRIFT_FRACTION = 0.55;
const PULSE_DEPTH = 0.30;        // how much of its size an object breathes
const DRIFT_HZ = [0.013, 0.021]; // slow: a drift you notice only by watching
const SPIN_HZ = 0.018;

/**
 * Fold a cell's integer coordinates and the seed into one uint32, using the
 * same h32/mix chain that addresses splats in ifs_core.js.
 *
 * `>>> 0` is what makes negative coordinates work: mix() multiplies by an odd
 * constant, which is a bijection on uint32, so -1 and +4294967295 are the same
 * cell -- unreachable in practice (that is 43 billion world units out) and
 * harmless if it ever were.  mix chains rather than sums, so the order of the
 * three axes matters and (3,1,2) is not (1,2,3): a commutative fold would put
 * identical objects on every permutation of a coordinate triple, which reads
 * as a diagonal mirror plane through the field.
 */
export function cellHash(ix, iy, iz, seed = 1) {
  let h = h32(seed >>> 0);
  h = mix(h, ix >>> 0);
  h = mix(h, iy >>> 0);
  h = mix(h, iz >>> 0);
  return h32(h);
}

/** Map key for a cell.  Strings, so the field has no coordinate ceiling: a
 *  bit-packed int would cap the flight at whatever bit budget it chose, and
 *  "fly forever" is the feature.
 *
 *  Level 0 keys are bare, exactly as before the hierarchy existed, so that
 *  anything addressing a root cell -- and the parseKey round trip -- is
 *  unchanged.  Deeper levels are prefixed, because (1,2,3) at level 0 and
 *  (1,2,3) at level 2 are different cells in different places. */
export const cellKey = (ix, iy, iz, level = 0) =>
  (level ? `${level}:${ix},${iy},${iz}` : `${ix},${iy},${iz}`);

// --------------------------------------------------------- the hierarchy
//
// A flat grid of equal cells, each independently occupied, is a lattice: the
// spacing is one number, the sizes are one number give or take, and the eye
// reads the regularity long before it reads the objects.  A universe has
// clusters, clusters of clusters, real voids, and a size range spanning
// orders of magnitude.
//
// So a cell does one of three things, decided by its own hash:
//
//     empty        nothing, ever, and nothing below it either -- this is
//                  what opens real voids
//     subdivides   into SUBDIV^3 children of side L/SUBDIV, recursively
//     holds        exactly one object, sized for ITS level
//
// A cell that holds an object does NOT subdivide.  That one rule is what
// keeps the separation proof -- see "separation" below -- and it makes the
// proof shorter than the flat one rather than longer.
//
// The size distribution comes free: an object at level l is bounded by
// L/SUBDIV^l, so sizes fall geometrically while the number of cells able to
// hold them rises geometrically.  That is a power law, and nobody had to
// choose its exponent; it is SUBDIV and the subdivision probabilities.
const SUBDIV = 2;                       // children per axis: 8 per cell
const MAX_LEVEL = 2;                    // sides L, L/2, L/4
const P_SUB = [0.14, 0.10, 0];          // chance of subdividing, by level

/**
 * The most objects one level-0 cell may contribute.
 *
 * The recursion is a branching process with mean offspring SUBDIV^3 * P_SUB,
 * which is 1.12 at level 0 -- near enough to 1 that the TYPICAL cell is cheap
 * and the occasional one is a dense cluster, which is the whole point. But
 * "occasional" is doing real work there: the tail is what makes a cluster,
 * and an unbounded tail is a frame that takes a second. MAX_LEVEL already
 * bounds it at 1 + 8 + 64 = 73; this bounds it somewhere useful. The walk
 * order is fixed, so truncating is as deterministic as everything else here.
 */
const CAP_PER_ROOT = 24;

/** The hash of a cell at a given level.  The level is folded into the seed:
 *  without it, cell (1,2,3) at level 0 and cell (1,2,3) at level 2 -- two
 *  different places -- would draw the identical object. */
export const levelHash = (ix, iy, iz, seed, level) =>
  cellHash(ix, iy, iz, (seed + Math.imul(level, 0x9E3779B1)) >>> 0);

export const ROLE_EMPTY = 0, ROLE_SUBDIVIDE = 1, ROLE_OBJECT = 2;

/**
 * What a cell does.  One draw from one stream index, so the three outcomes
 * are exclusive by construction and cannot drift apart.
 */
export function cellRole(h, level, density) {
  const r = stream(h, K_ROLE);
  const ps = level < MAX_LEVEL ? P_SUB[level] : 0;
  if (r < ps) return ROLE_SUBDIVIDE;
  if (r < ps + density) return ROLE_OBJECT;
  return ROLE_EMPTY;
}

/** Inverse of cellKey, for pin() -- never on a per-frame path. */
export function parseKey(key) {
  const p = key.split(',');
  if (p.length !== 3) return null;
  const a = +p[0], b = +p[1], c = +p[2];
  return Number.isInteger(a) && Number.isInteger(b) && Number.isInteger(c) ? [a, b, c] : null;
}

/** The occupancy rule, in one place so the fast enumeration in FieldLayout and
 *  cellContents can never disagree about which cells exist.  A level-0 cell is
 *  worth walking if it is anything other than empty -- a cell that subdivides
 *  holds no object itself but may hold a crowd of them below. */
export const cellOccupied = (h, density) =>
  cellRole(h, 0, density) !== ROLE_EMPTY;

/**
 * Everything in cell (ix, iy, iz), or null if it is empty.  Pure: no clock, no
 * camera, no call-order dependence, no state anywhere.
 *
 * The returned record is frozen because it is handed to the renderer every
 * frame and the determinism guarantee is only as good as the weakest writer:
 * one `rec.scale *= fade` in some future LOD experiment and the field stops
 * being the same field on the way back.
 */
export function cellContents(ix, iy, iz, opts = {}) {
  const level = opts.level ?? 0;
  const L0 = opts.L ?? 10;
  // Every length in this function is the length of THIS cell, not of a root
  // cell.  That is the whole of what makes an object at level 2 a quarter the
  // size and a quarter the wander of one at level 0, and it is also what
  // keeps the separation arithmetic true per level.
  const L = L0 / SUBDIV ** level;
  const density = opts.density ?? 0.22;
  const seed = opts.seed ?? 1;
  // rho splits a cell between how big an object may BE and how far it may
  // WANDER: hw + rho*L = L/2 exactly, for every rho.  It was briefly raised
  // to 0.33 to claw back size lost when REACH_MAX looked like it had to go to
  // 5.20; reordering the morph loop brought REACH_MAX back to 3.95 and made
  // that unnecessary, so it is 0.25 again -- the wander is worth more than
  // the size here, because an evenly spaced field is the thing that reads as
  // a lattice rather than a universe.
  const rho = opts.rho ?? 0.25;
  const uMin = opts.uMin ?? 0.35;
  const slots = opts.slots ?? 8;

  const h = levelHash(ix, iy, iz, seed, level);
  if (cellRole(h, level, density) !== ROLE_OBJECT) return null;

  // Placement.  The jitter half-width h = L*(0.5 - rho) and the reach cap
  // rho*L are two halves of one number: h + rho*L = L/2 exactly, for every
  // rho.  See "separation" below.
  const hw = L * (0.5 - rho);
  const bx = 2 * stream(h, K_JX) - 1;
  const by = 2 * stream(h, K_JY) - 1;
  const bz = 2 * stream(h, K_JZ) - 1;
  const t = [
    (ix + 0.5) * L + bx * hw,
    (iy + 0.5) * L + by * hw,
    (iz + 0.5) * L + bz * hw,
  ];

  // Size.  u is uniform-with-a-lean rather than uniform: v^0.625 has mean
  // 1/1.625 = 0.615, so the mean u lands at 0.750 of full size instead of
  // 0.675.  Straight uniform gives a field that is mostly small things with
  // the occasional big one, which reads as debris; leaning on it keeps most
  // cells filled enough to be objects while still leaving the small ones.
  // Nothing here can exceed u = 1, which is what the separation bound assumes.
  const u = uMin + (1 - uMin) * Math.pow(stream(h, K_U), 0.625);
  const scale = (rho * L / REACH_MAX) * u;
  // PULSE_DEPTH is taken OUT of u, never added to it: an object breathes
  // between (1 - depth)*u and u of the cap, so the largest it can ever be is
  // the size the separation bound already assumes. Growing about the static
  // size instead would put every object over its reach at the top of its
  // breath, which is the one way motion could break the proof.
  const uLo = u * (1 - PULSE_DEPTH);

  // Orientation.  tilt is uniform in COS, not in angle: Placed's rotation is
  // Rz(yaw)*Ry(tilt), so the object's up-axis ends up at polar angle `tilt`,
  // and uniform-in-angle piles those axes at the two poles -- a field where
  // every third attractor stands upright, which is exactly how you notice the
  // lattice is axis-aligned.
  const yaw = 360 * stream(h, K_YAW);
  const tilt = (Math.acos(1 - 2 * stream(h, K_TILT)) * 180) / Math.PI;

  // Tint.  Its only job is to stop two cells sharing a pool slot from reading
  // as the same object, and that must not cost anything the picture is already
  // using colour for.  So: a zero-sum hue triple, with its Rec.709 luminance
  // subtracted out and the result normalised to unit max, scaled by an
  // amplitude <= 0.24.  Luminance is then exactly 1 by construction, so the
  // tint rotates hue and nothing else -- the fog distance cue and the palette's
  // depth ramp are untouched, and no channel can leave [0.76, 1.24].
  const th = 2 * Math.PI * stream(h, K_HUE);
  const e0 = Math.cos(th);
  const e1 = Math.cos(th - 2.0943951023931953);
  const e2 = Math.cos(th + 2.0943951023931953);
  const lum = 0.2126 * e0 + 0.7152 * e1 + 0.0722 * e2;
  const f0 = e0 - lum, f1 = e1 - lum, f2 = e2 - lum;
  const nrm = Math.max(Math.abs(f0), Math.abs(f1), Math.abs(f2)) || 1;
  const amp = 0.10 + 0.14 * stream(h, K_AMP);
  const tint = [1 + (amp * f0) / nrm, 1 + (amp * f1) / nrm, 1 + (amp * f2) / nrm];

  // slot and root hash come from the same uint32.  They are correlated -- every
  // cell on slot 3 has h = 3 (mod slots) -- and that is fine, because the
  // traversal's first act on a root hash is mix(h, i), and h32's avalanche
  // destroys a 3-bit congruence completely.  What matters is that `hash`
  // differs per cell, so two cells on one slot are different REALISATIONS of
  // the same maps: same shape, different branch jitter.
  return Object.freeze({
    ix, iy, iz, level,
    key: cellKey(ix, iy, iz, level),
    t: Object.freeze(t),
    scale,
    yaw,
    tilt,
    slot: h % slots,
    tint: Object.freeze(tint),
    fx: (stream(h, K_FXIRID) < P_IRID ? FX_IRID : 0)
      | (stream(h, K_FXRIM) < P_RIM ? FX_RIM : 0),
    hash: h,
    reach: scale * REACH_MAX,

    // --- motion -------------------------------------------------------
    // The record stays frozen and pure. What it carries is the DESCRIPTION
    // of the motion, not a position at some time: `t`, `scale`, `yaw` above
    // remain the values at time zero, so eviction, `first()` and every
    // existing consumer keep working unchanged, and the renderer asks for an
    // instant when it needs one (see Placed.animate in world.js).
    tc: Object.freeze([(ix + 0.5) * L, (iy + 0.5) * L, (iz + 0.5) * L]),
    hw,
    cap: rho * L / REACH_MAX,
    u, uLo,
    // base offset and drift amplitude, in units of hw, with |base| + amp <= 1
    // by construction -- the amplitude is what is LEFT after the base
    anim: Object.freeze([
      bx, by, bz,
      (1 - Math.abs(bx)) * DRIFT_FRACTION * stream(h, K_DX),
      (1 - Math.abs(by)) * DRIFT_FRACTION * stream(h, K_DY),
      (1 - Math.abs(bz)) * DRIFT_FRACTION * stream(h, K_DZ),
      TWO_PI * (DRIFT_HZ[0] + (DRIFT_HZ[1] - DRIFT_HZ[0]) * stream(h, K_WX)),
      TWO_PI * (DRIFT_HZ[0] + (DRIFT_HZ[1] - DRIFT_HZ[0]) * stream(h, K_WY)),
      TWO_PI * (DRIFT_HZ[0] + (DRIFT_HZ[1] - DRIFT_HZ[0]) * stream(h, K_WZ)),
      TWO_PI * stream(h, K_PX), TWO_PI * stream(h, K_PY), TWO_PI * stream(h, K_PZ),
    ]),
    pulseW: TWO_PI * (0.5 * DRIFT_HZ[0] + DRIFT_HZ[1] * stream(h, K_PULSE)),
    pulseP: TWO_PI * stream(h, K_PULSEP),
    spinW: TWO_PI * SPIN_HZ * (2 * stream(h, K_SPIN) - 1),
  });
}

const TWO_PI = 2 * Math.PI;

/**
 * Every object in the subtree under level-0 cell (ix, iy, iz), appended to
 * `out`.  Pure, like cellContents: no clock, no camera, no call order.
 *
 * The walk order is fixed (z, then y, then x, depth first), which is what
 * makes CAP_PER_ROOT deterministic -- truncating a fixed order always
 * truncates the same way, so a cell that is over the cap still produces the
 * same objects every time it is asked.
 */
export function cellObjects(ix, iy, iz, opts = {}, out = []) {
  // The cap counts what THIS root cell contributed, not what is in `out`.
  // Measuring it against out.length instead works fine when the caller hands
  // over a fresh array per cell and silently truncates the entire field once
  // the caller accumulates -- 24 objects across 24,389 root cells, which
  // looks like an empty universe rather than like a bug.
  _walk(ix, iy, iz, 0, opts, opts.seed ?? 1, opts.density ?? 0.22, out, out.length);
  return out;
}

function _walk(ix, iy, iz, level, opts, seed, density, out, base) {
  if (out.length - base >= CAP_PER_ROOT) return;
  const role = cellRole(levelHash(ix, iy, iz, seed, level), level, density);
  if (role === ROLE_EMPTY) return;
  if (role === ROLE_OBJECT) {
    _lvl.level = level;
    _lvl.L = opts.L ?? 10;
    _lvl.density = density;
    _lvl.seed = seed;
    _lvl.rho = opts.rho ?? 0.25;
    _lvl.uMin = opts.uMin ?? 0.35;
    _lvl.slots = opts.slots ?? 8;
    const rec = cellContents(ix, iy, iz, _lvl);
    if (rec) out.push(rec);
    return;
  }
  const b = SUBDIV, cx = b * ix, cy = b * iy, cz = b * iz;
  for (let dz = 0; dz < b; dz++)
    for (let dy = 0; dy < b; dy++)
      for (let dx = 0; dx < b; dx++)
        _walk(cx + dx, cy + dy, cz + dz, level + 1, opts, seed, density, out, base);
}

// One options object for the whole walk: cellContents only reads it, and a
// fresh literal per visited cell would be thousands of objects per boundary
// cross, which is the allocation pattern that cost 33 ms/frame once already.
const _lvl = {};

// --------------------------------------------------------------- separation
//
// Why 26 neighbours are enough, and why the bound holds at all.
//
// An object in cell c is a sphere about t of radius `reach`.  By construction
//
//     |t - centre(c)|_inf  <=  hw      = L*(0.5 - rho)
//     reach                 = rho*L*u  <= rho*L          (u <= 1)
//
// so every point of the sphere is within hw + rho*L = L/2 of the cell centre,
// in the infinity norm AND therefore in the 2-norm along each axis.  L/2 is
// the inradius of the cell.  Every object is inside its own cell, full stop,
// and that is true for any rho in [0, 0.5] -- the two halves of the budget are
// defined so that they always add to exactly L/2.
//
// Consequences:
//   * Cells at Chebyshev distance >= 2 have centres >= 2L apart along some
//     axis; each sphere stays within L/2 of its centre, so the gap is >= L.
//     Only the 26 touching neighbours can ever be close, so a sweep that
//     checks those 26 has checked everything.
//   * For two touching cells, centres differ by >= L along some axis, each t
//     moves <= hw along it, so |t_i - t_j| >= L - 2*hw = 2*rho*L, while
//     reach_i + reach_j <= 2*rho*L.  Margin >= 0, with equality only if both
//     cells drew u = 1 AND both jitters hit the end stop toward each other.
//     tests/field.mjs reports the smallest margin it actually finds, so an
//     edit that erodes the budget shows up as a shrinking number instead of
//     waiting to become a visible seam.

const smooth = (t) => t * t * (3 - 2 * t);
const now = () => (typeof performance !== 'undefined' ? performance.now() : 0);

/**
 * A fixed set of live, morphing Attractors that the whole field draws from.
 *
 * The field has tens of objects on screen; stage 3 had seven.  World.update
 * (world.js:100) rebuilds every item at rebuildHz, and Attractor.reparam()
 * costs ~1 ms, so seven items measured 1.78 ms/frame -- and forty would be
 * ~10 ms/frame of pure rebuild on top of the traversal, which is the demo.
 *
 * A pool decouples the two: however many cells are resident, there are only
 * `slots` attractors, so the rebuild cost is bounded by the pool and not by
 * the field.  Two cells on one slot still look like different objects, because
 * they get different root hashes (different branch jitter), different tints,
 * sizes and orientations.
 *
 * On top of that the pool holds a per-frame time budget, because `slots`
 * rebuilds in one frame is still ~8 ms.  update() refuses to START a rebuild
 * it does not expect to fit (see the controller below) and picks the work up
 * next frame, highest priority first and most overdue within a priority.
 * Deferring a rebuild is free in the picture: at a 14 s morph and 12 Hz, one
 * rebuild advances the shape by 1/168 of the way, and the LOD cross-fade is
 * already smoothing more than that.
 *
 * What that costs, measured over 2000 frames at the defaults: 1.08 ms/frame
 * mean, 0.93 rebuilds/frame, and an EFFECTIVE 7.0 Hz per slot rather than the
 * 12 Hz asked for -- 8 slots at 12 Hz is 1.6 rebuilds/frame and 1.5 ms buys
 * one.  1.7 slot-frames sit deferred at any moment.  That is the budget doing
 * its job rather than a fault, but it does mean rebuildHz is now advisory: to
 * actually get 12 Hz the budget has to go to ~2.5 ms, and to stop queueing
 * work that is always deferred, rebuildHz should come down to 7.
 */
export class ShapePool {
  constructor(opts = {}) {
    this.slots = opts.slots ?? 8;
    this.morphSeconds = opts.morphSeconds ?? 14;
    this.holdSeconds = opts.holdSeconds ?? 10;
    // Continuous morphing along a closed spline, at a breathing rate.
    //
    // `setSeconds` is how long one segment of the loop takes at rate 1, and
    // the breath multiplies that rate between breathMin and breathMax on a
    // raised cosine -- smooth, so the rate itself has no corner.
    //
    // The breath is not decoration.  Dropping the old hold-then-morph cycle
    // for a constant rate would mean the world is never any PARTICULAR
    // fractal, just perpetual mush; the hold was the only thing making a
    // shape legible.  A rate that falls to breathMin restores that -- the
    // shape nearly stops, you read it, it moves on -- without the corner in
    // the path that a hold puts there.  It also costs less: quantisation
    // below is in units of distance travelled, so a slow breath asks for
    // fewer rebuilds, and the legible moments are the cheap ones.
    this.continuous = opts.continuous ?? true;
    //
    // Slower, and with a shallower breath: 16 s a segment against 9, and a
    // rate that tops out at 1.15 instead of 1.90.
    //
    // The step a rebuild takes is (dt * breath / setSeconds) of a segment of
    // the loop, so those two numbers ARE the step size -- the thing that
    // decides whether a shape is seen changing or seen to have changed.
    // Measured over 7200 frames of a pool, the step per rebuild went from a
    // mean of 1/171 of a segment to 1/463, and -- the number that matters,
    // because it is the worst a viewer sees -- from a p99 of 1/71 to 1/194.
    // It also costs less rather than more, because the quantum below is
    // distance travelled: a slower morph asks for fewer rebuilds, and the
    // breath's slow moments ask for almost none. tests/breath.mjs puts the
    // whole pool at 0.58 rebuilds/frame and 0.44 ms against a 1.5 ms budget.
    this.setSeconds = opts.setSeconds ?? 16;
    this.breathSeconds = opts.breathSeconds ?? 40;
    this.breathMin = opts.breathMin ?? 0.10;
    this.breathMax = opts.breathMax ?? 1.15;
    this.breath = 1;
    // Draw the shape between its last two rebuilds rather than holding the
    // newest one still until the next arrives. See Attractor.setBlend.
    //
    // ON. It was off, and the reason it was off was real: the first version
    // materialised the blended table every frame, which is 288 KB of writes
    // per slot that cost 5 ms of whole-frame time in the expansion that read
    // it afterwards. The blend is now done when the table is READ, writes
    // nothing, and the measurement is in Attractor's constructor.
    //
    // What it buys, measured: a slot's drawn shape changes 18 times a second
    // without it, against a 60 Hz display -- an 18 fps animation, which is
    // what judder is. The budget cannot fix that by rebuilding harder; eight
    // slots at 60 Hz is 3.5 ms a frame of reparam against a 1.5 ms budget.
    this.interpolate = opts.interpolate ?? true;
    // How far past the newest build the blend may be extrapolated before it
    // is held. See Attractor.setBlend: rebuilds arrive when the budget
    // allows, not on time, and clamping at 1 put a third of the frames back
    // at a standstill.
    this.blendMax = opts.blendMax ?? 2.5;
    // How far BEFORE the older of the two builds the blend may be read.
    //
    // Symmetric with blendMax, which allows 1.5 spans past the newer sample,
    // and for the same reason. The pair brackets the present, so a rebuild
    // lands at a fraction of 0 and the fraction a shorter-than-estimated gap
    // asks for is 1 - gap/lead below it -- a gap half the estimate asks for
    // -1. A floor of -0.5 was reached on 9 slot-frames of 10,552 and every
    // one of them was a jump, because a floor is a freeze and the release is
    // a step: this is a budget-driven schedule, so irregular gaps are the
    // normal case, not the exception. See Attractor.setBlend.
    this.blendMin = opts.blendMin ?? -1.5;
    // The jitter table is what makes both a rebuild and a blend cost what
    // they cost: everything scales with 9*m*K. tests/jitterbits.mjs says 256
    // entries (x127 reuse) still show no visible repetition, so paying for
    // 1024 buys nothing an eye can collect.
    this.tableBits = opts.jitterTableBits ?? 10;
    // Jitter table size by RANK, not by an absolute priority threshold.
    //
    // Rank is scale-free: `priority` is whatever on-screen importance the
    // caller measures, in whatever units, and a table keyed off its absolute
    // value would need retuning the moment that measure changed. Ranking the
    // slots against each other needs no units at all.
    //
    // The sizes below sum to 2,432 entries against the 8,192 a uniform 1024
    // would spend -- 3.4x less rebuild and blend work for a field that looks
    // the same, because the slots that lost entries are the ones a few pixels
    // across. tests/jitterbits.mjs is the evidence that even 256 is
    // indistinguishable.
    this.bitsByRank = opts.bitsByRank ?? [10, 9, 8, 8, 7, 7, 6, 6];
    // How often the ranking is redone. Every frame would re-cut the tables
    // as objects jostle for position, and a resize costs a forced rebuild.
    this.rankEvery = opts.rankEvery ?? 24;
    // Start a full interval out, not at zero. Firing on the very first update
    // resizes slots that were only just built, and a resize forces a rebuild:
    // frame 0 then started six of them at once and took 14.5 ms, which
    // tests/field.mjs correctly calls a stall. Nothing needs re-ranking
    // before the camera has moved.
    this._rankTick = this.rankEvery;
    // At most one slot changes size per pass.
    //
    // The budget already limits how many rebuilds a frame may START, but a
    // forced resize is not a scheduling request -- it invalidates the table a
    // slot is drawing from, so the rebuild cannot be deferred without the
    // slot showing a table that no longer matches its mask. Resizing one at a
    // time keeps that from ever being more than one slot's worth of work, and
    // costs nothing: ranks drift slowly, so a pass almost never has more than
    // one slot wanting a different size anyway.
    this.resizePerPass = opts.resizePerPass ?? 1;
    // Rebuilds a re-cut table is ramped over; see Attractor.migrateRebuilds.
    // Here so that a test can turn the ramp off (1) for the whole pool.
    this.migrateRebuilds = opts.migrateRebuilds ?? 4;
    this._rank = new Int32Array(this.slots);
    // The quantum, in rebuilds per second of travel. It is not the rate any
    // slot achieves -- the budget is -- it is the finest step the schedule
    // will ask for. At 12 it was the binding constraint while half the
    // budget went unspent, which is the wrong way round: the budget should
    // be what limits this, because the budget is the thing that is actually
    // scarce.
    // 'auto' tracks the frame rate: the schedule asks for one rebuild per
    // slot per rendered frame, and the budget decides how many of those
    // actually happen. That is the right way round. A fixed number has to be
    // guessed against a frame rate nobody knows at build time -- 12 was the
    // binding constraint while half the budget went unspent, and 30 is only
    // correct for a 30 Hz display. Asking for the finest step the display can
    // show, and letting the scarce thing do the limiting, needs no guess and
    // follows a machine that turns out to be faster or slower than this one.
    this.rebuildHz = opts.rebuildHz ?? 'auto';
    this._hz = 30;        // the effective rate, measured when 'auto'
    this._dtEwma = 0;
    this.hzMin = opts.hzMin ?? 8;
    this.hzMax = opts.hzMax ?? 120;
    this.seed = opts.seed ?? 101;
    this.jitter = opts.jitter ?? 0.16;
    this.rebuildBudgetMs = opts.rebuildBudgetMs ?? 1.5;
    this.morphing = opts.morphing ?? true;

    this.shapes = new Array(this.slots);
    for (let k = 0; k < this.slots; k++) {
      // Each slot starts on a different set and on its own phase offset, so
      // the field never changes shape all at once -- a void where everything
      // morphs in step looks mechanical (the same reason world.js:108 offsets
      // its items).
      const name = SET_NAMES[k % SET_NAMES.length];
      const spec = SETS[name];
      const b = build(spec);
      // Each slot owns its palette buffer, because setPalette keeps the
      // array it is handed rather than copying it: one shared scratch would
      // have every slot drawing from whichever slot rebuilt last.
      const palBuf = palBuffer();
      const p0 = PALETTES[spec.palette] || PALETTES.spectral;
      // Resampled to PAL_STOPS even at construction, so the stop count does
      // not change between the first frame and the first rebuild -- that
      // would be one recolour of everything at startup.
      crPalette(p0, p0, p0, p0, 0, palBuf);
      const att = new Attractor({
        maps: b.maps, trunkA: b.trunkA, sig0: b.sig0, mu0: b.mu0,
        omega: spec.omega ?? 0.28, seed: this.seed + k, jitter: this.jitter,
        palette: palBuf,
        jitterTableBits: this.tableBits,
        migrateRebuilds: this.migrateRebuilds,
        // Opt this attractor into read-time interpolation: it is what makes
        // a rebuild fold the displayed shape into the previous table instead
        // of discarding it.
        interp: this.interpolate,
      });
      // uA is where the drawn shape sits on the morph spline, uB where the
      // newest build sits. Equal at construction: nothing to interpolate yet.
      this.shapes[k] = { att, name, from: name, to: name, phase: 0,
                         lastBuilt: -1, builtAt: 0, rebuilds: 0, palBuf,
                         uA: (k * SET_NAMES.length) / this.slots,
                         uB: (k * SET_NAMES.length) / this.slots };
    }

    this._clock = 0;
    this.reparamMs = 0;
    this.rebuildsThisFrame = 0;
    this.deferred = 0;
    this.lastMs = 0;
    // scratch, allocated once: update() runs every frame and must not add to
    // the GC's work.  _due doubles as the worklist.
    this._due = new Uint8Array(this.slots);
    this._pFrom = new Array(this.slots).fill('');
    this._pTo = new Array(this.slots).fill('');
    this._pPhase = new Float64Array(this.slots);
    this._pQuant = new Float64Array(this.slots);
    this._pU = new Float64Array(this.slots);
    this._want = new Int32Array(this.slots);   // blend candidates, reused
    this._u = 0;

    // The two rates the HUD shows, measured rather than assumed.
    //
    // "How often does the shape on screen change" is neither the frame rate
    // nor the rebuild rate, and the difference between those two is the
    // whole point of interpolating: a slot can be rebuilt 3 times a second
    // and still move every frame, or move 3 times a second and judder. Only
    // the pool can tell them apart, because only the pool knows whether a
    // slot's drawn shape moved -- it moved if it was rebuilt, or if its
    // blend fraction did. That is the same event tests/breath.mjs counts
    // offline; this counts it live, for a number on the panel.
    //
    // Over a window rather than per frame: 1/dt at 60 Hz is noise, and a
    // rebuild rate near 3 Hz needs a window to be a rate at all.
    this.shapeHz = 0;          // drawn-shape changes per second, per slot
    this.rebuildHzNow = 0;     // rebuilds per second, per slot
    this.meterSeconds = opts.meterSeconds ?? 0.5;
    this._accChg = 0; this._accReb = 0; this._accDt = 0;
    this._btPrev = new Float64Array(this.slots);
    this._builtK = new Uint8Array(this.slots);

    // Warm the rebuild path here, in the constructor, where a pause is already
    // expected (building the slots costs ~143 ms of jitter tables anyway).
    //
    // Measured before this existed: frame 1 of the animation cost 52.3 ms for
    // a SINGLE rebuild whose steady-state cost is 1.19 ms, and frames 0 and 2
    // cost 28.0 and 20.8 ms.  That is V8 compiling morph/build/reparam, not
    // this class being slow -- but it lands on the first few frames anyone
    // sees, which is the worst possible place for a 50 ms hitch.
    //
    // It has to go through the real path or the JIT compiles nothing useful:
    // a morph midpoint (so lerpMap, mapMatrix and the full reparam all run),
    // then back to the slot's own pure set so nothing is left mid-morph and
    // the bookkeeping below is untouched.
    //
    // One pass of 8 slots is not enough -- V8 promotes to TurboFan on a call
    // count, and after 16 reparams frames 1-3 still measured 4.4-5.2 ms
    // against a settled 0.9 ms.  So repeat until a pass stops getting cheaper
    // by more than 15%, capped at 6 passes.  That is the point at which the
    // code is compiled; carrying on past it buys nothing and only delays the
    // first frame.  The last pass's median is the honest per-rebuild cost, so
    // it seeds the controller already converged.
    this.reparamEst = 0;
    this.worstAdmit = 0;      // most any non-first admission ever claimed it would cost
    this.warmupMs = 0;
    this.warmupPasses = 0;
    {
      const w0 = now();
      let prev = Infinity, med = Infinity;
      const warm = [];
      for (let pass = 0; pass < 6; pass++) {
        warm.length = 0;
        for (let k = 0; k < this.slots; k++) {
          const sh = this.shapes[k];
          // Warm the path the steady state actually takes -- a point part way
          // along the spline, not a straight morph.  Warming the wrong one
          // leaves morphPath and crMap uncompiled, which is the whole hitch
          // this warm-up exists to remove.
          {
            const m = morphPath(SPECS, k + 0.5);
            const bm = build(m);
            sh.att.reparam({ maps: bm.maps, trunkA: bm.trunkA, sig0: bm.sig0, mu0: bm.mu0,
                             omega: m.omega ?? 0.28, jitter: m.jitter ?? 0.16 });
          }
          const r0 = now();
          const base = SETS[sh.name];
          const bb = build(base);
          sh.att.reparam({ maps: bb.maps, trunkA: bb.trunkA, sig0: bb.sig0, mu0: bb.mu0,
                           omega: base.omega ?? 0.28,
                           palette: PALETTES[base.palette] || sh.att.palette });
          warm.push(now() - r0);
        }
        // median, not mean: the first slot of a pass is still part-cold.
        warm.sort((a, b) => a - b);
        med = warm[warm.length >> 1];
        this.warmupPasses = pass + 1;
        if (med > prev * 0.85) break;
        prev = med;
      }
      this.reparamEst = med < Infinity ? med : 1.2;
      this.warmupMs = now() - w0;
      // and reset the morph bookkeeping, so the warm-up is invisible to the
      // schedule: every slot is its own pure set at phase 0, exactly as it was
      // when it was constructed, and update() will not re-book any of it.
      for (let k = 0; k < this.slots; k++) {
        const sh = this.shapes[k];
        sh.from = sh.name; sh.to = sh.name; sh.phase = 0;
        sh.lastBuilt = -1; sh.builtAt = 0; sh.rebuilds = 0; sh.gapEst = 0;
        // Both tables hold the warm-up's last build, so there is nothing to
        // interpolate between until the first real rebuild: equal parameters,
        // fraction pinned at 1, the newest build drawn.
        sh.uA = sh.uB = this._u + (k * SET_NAMES.length) / this.slots;
        sh.att._cmA.set(sh.att._cmB); sh.att._sA.set(sh.att._sB);
        sh.att._bt = 1;
        // And the first real rebuild must not extrapolate AWAY from this one.
        // The warm-up leaves each slot on a pure set at phase 0; the first
        // rebuild puts it somewhere on the spline, which is a different shape,
        // and a fraction of 1.33 through that pair lands a third of the way
        // past a shape that was never meant to be reached. Measured, over the
        // first two seconds: sigma0 at -0.0204, a child scale at -0.047 -- a
        // negative variance. Clearing this makes that rebuild its own previous
        // build, so interpolation starts with the second one, between two
        // builds that are a rebuild apart like every other pair.
        sh.att._everBuilt = false;
      }
    }
  }

  /** `{att, name, phase}` for a slot -- what a cell on that slot draws. */
  shape(slot) {
    const sh = this.shapes[slot % this.slots];
    return { att: sh.att, name: sh.name, phase: sh.phase };
  }

  /**
   * Advance the morphs.  `dt` in seconds.  `priority[k]` is the largest
   * on-screen importance any cell on slot k has this frame -- the caller knows
   * that, this class does not and must not.  Omitted means all slots equal,
   * which degrades to round-robin by overdue-ness.
   */
  /**
   * Count this frame's drawn-shape changes and, every `meterSeconds`, turn
   * the counts into the two rates. Called on every path out of update(),
   * including the one where morphing is off -- a meter that stops being
   * updated reads as whatever it last said, which is the one number a
   * frozen morph must not show.
   */
  /**
   * Start the meters again from nothing. Called by both switches, because a
   * rate measured across a switch is a rate for neither side of it: the
   * window that straddles the click would report the old behaviour for up to
   * half a second after the new one started, which is exactly the half
   * second someone who just clicked is looking at the number.
   */
  _meterReset() {
    this.shapeHz = 0; this.rebuildHzNow = 0;
    this._accChg = 0; this._accReb = 0; this._accDt = 0;
    for (let k = 0; k < this.slots; k++) {
      this._btPrev[k] = this.shapes[k].att._bt;
      this._builtK[k] = 0;
    }
    return this;
  }

  _meter(dt) {
    let chg = 0;
    for (let k = 0; k < this.slots; k++) {
      const bt = this.shapes[k].att._bt;
      if (this._builtK[k] || bt !== this._btPrev[k]) chg++;
      this._btPrev[k] = bt;
      this._builtK[k] = 0;
    }
    this._accChg += chg;
    this._accDt += dt;
    if (this._accDt >= this.meterSeconds) {
      this.shapeHz = this._accChg / this.slots / this._accDt;
      this.rebuildHzNow = this._accReb / this.slots / this._accDt;
      this._accChg = 0; this._accReb = 0; this._accDt = 0;
    }
  }

  /**
   * Interpolation on or off, live, for every slot at once.
   *
   * Not a plain assignment: each attractor has to reconcile its own blend
   * state first, and Attractor.setInterp says why. The pool's own half is
   * the parameter pair -- `uA` is where the drawn shape was when the newest
   * build landed, and after setInterp the drawn shape IS the newest build,
   * so uA has to come with it. Leaving it behind would hand the next frame a
   * fraction measured from a shape that is no longer on screen.
   */
  setInterpolate(on) {
    this.interpolate = !!on;
    for (let k = 0; k < this.slots; k++) {
      const sh = this.shapes[k];
      sh.att.setInterp(this.interpolate);
      sh.uA = sh.uB;
    }
    this._meterReset();
    return this;
  }

  /**
   * Morphing on or off. Off holds every slot at the shape it is drawing now:
   * the parameter stops advancing, nothing is rebuilt, and the blend stops
   * being touched, so the frame is identical to the last one morph made.
   * Nothing is thrown away -- turning it back on carries on from there.
   */
  setMorphing(on) {
    this.morphing = !!on;
    this._meterReset();
    return this;
  }

  update(dt, priority) {
    const t0 = now();
    this._clock += dt;
    if (this.rebuildHz === 'auto') {
      // Smoothed, because an unsmoothed 1/dt makes the quantum itself jitter
      // with the frame time: a single slow frame would coarsen the schedule
      // and a single fast one would re-sample a morph that had not moved.
      this._dtEwma = this._dtEwma > 0 ? this._dtEwma + (dt - this._dtEwma) * 0.05 : dt;
      const hz = this._dtEwma > 0 ? 1 / this._dtEwma : 30;
      this._hz = hz < this.hzMin ? this.hzMin : hz > this.hzMax ? this.hzMax : hz;
    } else {
      this._hz = this.rebuildHz;
    }
    this.reparamMs = 0;
    this.rebuildsThisFrame = 0;
    this.deferred = 0;
    if (!this.morphing || SET_NAMES.length < 2) {
      this._meter(dt);
      this.lastMs = now() - t0;
      return;
    }

    const N = SET_NAMES.length;
    const due = this._due;
    let nDue = 0;

    // The breath: a raised cosine between breathMin and breathMax, so the
    // rate is smooth and strictly positive -- the morph slows almost to rest
    // and picks up again, but never stops dead and never reverses.
    this.breath = this.breathMin + (this.breathMax - this.breathMin)
                * 0.5 * (1 - Math.cos((2 * Math.PI * this._clock) / this.breathSeconds));
    this._u += (dt * this.breath) / this.setSeconds;

    // Re-rank the slots and resize their tables. Before pass 1, so a slot
    // whose table just changed shape is marked due and rebuilt in the same
    // frame rather than drawing a half-stale table until its next turn.
    if (priority && --this._rankTick <= 0) {
      this._rankTick = this.rankEvery;
      const rank = this._rank;
      for (let k = 0; k < this.slots; k++) rank[k] = k;
      // slots is 8; an insertion sort is smaller and allocates nothing
      for (let i = 1; i < this.slots; i++) {
        const v = rank[i], pv = priority[v];
        let j = i - 1;
        while (j >= 0 && priority[rank[j]] < pv) { rank[j + 1] = rank[j]; j--; }
        rank[j + 1] = v;
      }
      let resized = 0;
      for (let r = 0; r < this.slots && resized < this.resizePerPass; r++) {
        const sh = this.shapes[rank[r]];
        const bits = this.bitsByRank[r < this.bitsByRank.length
          ? r : this.bitsByRank.length - 1];
        // Asked for, not forced. Marking the slot due here (`lastBuilt = -1`)
        // used to be how the new size reached the screen promptly, and it is
        // now the only thing left that makes the blend jump: it rebuilds a
        // slot that is nowhere near due, so the two samples end up a quarter
        // of a quantum apart while the older one still sits a whole quantum
        // AHEAD of the present. The fraction that asks for is -3.06,
        // measured, which the floor then clamps -- a freeze, and the release
        // is a step of 9x to 26x the true motion.
        //
        // It is also unnecessary. The resize is applied by reparam() now, so
        // it rides along with the slot's next ordinary rebuild, which is at
        // most one quantum away and is exactly what setTableBits() says it
        // does. The dial is about what a rebuild costs on average; waiting a
        // few frames for one costs nothing measurable.
        if (sh.att.setTableBits(bits)) resized++;
      }
    }

    // Pass 1: how far along the loop is each slot, and has it moved far
    // enough to be worth a rebuild?  The quantum is distance travelled, not
    // time elapsed, so a slot that is barely moving is not rebuilt at all.
    for (let k = 0; k < this.slots; k++) {
      const sh = this.shapes[k];
      // Slots are spread around the whole loop, so the field shows several
      // different shapes at once rather than one shape everywhere.
      const u = this._u + (k * N) / this.slots;
      const quant = Math.round(u * this.setSeconds * this._hz);
      if (quant === sh.lastBuilt) { due[k] = 0; continue; }
      this._pU[k] = u; this._pQuant[k] = quant;
      due[k] = 1; nDue++;
    }

    // Pass 2: spend the budget.  Linear scan rather than a sort -- slots is 8,
    // so this is at most 64 comparisons and it allocates nothing, where a
    // comparator sort allocates an index array every frame.
    while (nDue > 0) {
      // Test the budget BEFORE starting a rebuild, against what one costs.
      // Testing it after is not a budget: at 1.19 ms a rebuild and a 1.5 ms
      // budget, one rebuild leaves 1.19 < 1.5, a second starts, and the frame
      // lands at 2.4 ms -- 1.6x over, every frame, in steady state.  The
      // estimate is an EWMA seeded from the constructor's warm-up, so it is
      // already right on frame 0 and tracks a machine that is slower or a
      // map set that is dearer to build.
      //
      // The first rebuild of a frame always goes ahead even if the estimate
      // alone exceeds the budget.  Otherwise a budget set below the cost of
      // one rebuild stops the field morphing altogether, which is a deadlock
      // that looks like a frozen scene rather than like a misconfiguration.
      if (this.rebuildsThisFrame > 0 &&
          this.reparamMs + this.reparamEst > this.rebuildBudgetMs) break;
      // What the policy believed this admission would cost, at the moment it
      // made the decision.  Recorded rather than re-derived by the test: the
      // estimate moves as rebuilds get cheaper or dearer, so a test comparing
      // the frame's rebuild COUNT against the estimate at the END of a run
      // fails whenever actual cost dips below it -- which says nothing about
      // the policy.  This is the policy's own claim, and it is exactly the
      // thing that must never exceed the budget.
      if (this.rebuildsThisFrame > 0) {
        const claim = this.reparamMs + this.reparamEst;
        if (claim > this.worstAdmit) this.worstAdmit = claim;
      }
      // Importance TIMES staleness, not importance then staleness.
      //
      // Sorting strictly by on-screen priority and using age only to break
      // ties lets the most important slot win every frame it is due, and a
      // slot that is merely quite important never gets rebuilt at all while
      // that lasts. Measured over a minute: a median hold of 4 frames but a
      // worst of 28 -- nearly half a second frozen, and then a tenth of a
      // whole morph segment arriving in a single frame. The average looked
      // fine and the tail was the thing anyone would actually see.
      //
      // Multiplying by age makes starvation impossible: whatever a slot's
      // priority, its score grows without bound until it is served. The 0.02
      // floor keeps a slot that is currently off-screen ticking over slowly,
      // so it is not a decade out of date on the frame it comes into view.
      let best = -1, bestS = -Infinity, bestP = 0, bestO = 0;
      for (let k = 0; k < this.slots; k++) {
        if (!due[k]) continue;
        const p = priority ? priority[k] : 1;
        const o = this._clock - this.shapes[k].builtAt;
        const score = (p + 0.02) * o;
        if (score > bestS) { best = k; bestS = score; bestP = p; bestO = o; }
      }
      if (best < 0) break;
      due[best] = 0; nDue--;

      const sh = this.shapes[best];
      const u = this._pU[best];
      // Build AHEAD of the morph, by about one rebuild interval.
      //
      // This is the difference between a blend that interpolates and one that
      // only ever extrapolates, and it is the whole of the pop.
      //
      // Building at `u` -- where the morph is NOW -- makes the newest sample
      // the present, so the blend can only ever read the line beyond its own
      // endpoint. Worse, it puts the hand-over from one line to the next at
      // the far tip of the old one, which is exactly where the two disagree:
      // on the rebuild frame the fraction is (u - uA)/(u - uA) = 1 exactly,
      // so whatever the old line had extrapolated to is discarded and the
      // shape is set to the new build in one frame. Measured against an exact
      // build at the same parameter, over 10,552 slot-frames of splat
      // positions walked down fixed addresses: 17 frames stepped more than 8x
      // as far as the true shape moved, up to 36x, and every single one of
      // them was a rebuild frame with the fraction at 1.00. That is the
      // sawtooth -- drift out along the secant, snap back on the rebuild --
      // and it is a pop on 3.6% of rebuilds, which at 8 slots and 2.7
      // rebuilds a second is something popping somewhere most seconds.
      //
      // Building at `u + one interval` instead puts the pair either side of
      // the present. The fraction runs 0 -> 1 between rebuilds and reaches 1
      // just as the next build lands, so the hand-over happens at the sample
      // the two lines SHARE, where they agree exactly, instead of at the tip
      // where they do not. The drawn parameter is unchanged -- the secant is
      // still read at `u` -- so nothing lags; only the spacing of the two
      // samples moves, and a secant read between its samples is both smaller
      // in error and bounded, where one read past them is neither.
      //
      // The lead is the schedule's own QUANTUM, which is exact, and not the
      // measured gap, which is not.
      //
      // A slot is due when `round(u * setSeconds * hz)` changes, so due
      // events are exactly `1/(setSeconds*hz)` of parameter apart and no
      // actual gap can be shorter than that -- only longer, when the budget
      // defers one. Lead by the quantum and the fraction starts at
      // `1 - quantum/gap`, which is 0 when the schedule kept up and positive
      // when it did not: it can never put the older sample ahead of the
      // present.
      //
      // The smoothed measured gap was the first thing tried here and it is
      // the wrong quantity in the way that matters: it is an average of an
      // irregular series, so it is larger than the gap about half the time,
      // and each of those asks for a negative fraction. Measured, with the
      // floor at -0.5 and then at -1.5: 9 and then 5 slot-frames of 10,552
      // sat exactly on the floor, and every one of them was a jump of 9x to
      // 26x the true motion -- a floor is a freeze, and the release is a
      // step. The quantum does not have that failure mode at all.
      //
      // Lateness is unchanged: if the next rebuild misses its slot the
      // fraction carries on past 1 and extrapolates as it always did,
      // bounded by `blendMax` and `_btSafe`.
      const lead = 1 / (this.setSeconds * this._hz);
      const pu = u + lead;
      const r0 = now();
      const spec = morphPath(SPECS, pu);
      const b = build(spec);
      // The palette on the same spline as the geometry, but at the DRAWN
      // parameter rather than the built one: unlike the table, the palette is
      // not blended as it is read -- a rebuild replaces it outright -- so the
      // colour on screen should be the colour of where the morph is now, not
      // of where the geometry sample was taken. `palT` is just the fractional
      // position, which is `frac` below, so this costs no second spline.
      // SPECS is in SET_NAMES order and palStopsAt returns the same four sets
      // morphPath splines between, so the colour and the shape are never a
      // segment apart.
      const ps = palStopsAt(SPECS, u);
      const pal = crPalette(
        PALETTES[ps[0].palette] || PALETTES.spectral,
        PALETTES[ps[1].palette] || PALETTES.spectral,
        PALETTES[ps[2].palette] || PALETTES.spectral,
        PALETTES[ps[3].palette] || PALETTES.spectral,
        u - Math.floor(u), sh.palBuf);
      sh.att.reparam({
        maps: b.maps, trunkA: b.trunkA, sig0: b.sig0, mu0: b.mu0,
        omega: spec.omega ?? 0.28,
        // jitter rides the spline like any other parameter now, which is what
        // lets a rigid crystal morph into an organic tangle
        jitter: spec.jitter ?? 0.16,
        // The spline blend, not PALETTES[spec.palette]. Looking the name up
        // was the bug: morphPath's `palette` is whichever of the two sets is
        // nearer, so it flipped at t = 0.5 and recoloured the whole object in
        // one frame while its geometry carried on smoothly.
        palette: pal,
      });
      const frac = u - Math.floor(u);
      const i1 = ((Math.floor(u) % N) + N) % N;
      sh.from = SET_NAMES[i1]; sh.to = SET_NAMES[(i1 + 1) % N]; sh.phase = frac;
      sh.lastBuilt = this._pQuant[best];
      // How long this slot actually waited, smoothed. The schedule is
      // budget-driven and irregular, so the interval has to be learnt rather
      // than assumed: assuming 1/rebuildHz would run the blend to 1 early
      // and hold there, which is the judder back again with extra steps.
      // Where the drawn shape is on the spline right now, and where the
      // build that just landed is. The blend runs between these two, so the
      // interval it covers is the one that actually happened rather than the
      // one that was expected.
      // The two tables are now the builds at uA and uB, and the drawn shape
      // is the line through them read at wherever the morph has got to --
      // which is now BETWEEN them rather than beyond both.
      sh.uA = sh.uB;
      sh.uB = pu;
      const gap = this._clock - sh.builtAt;
      sh.gapEst = sh.gapEst > 0 ? sh.gapEst + (gap - sh.gapEst) * 0.25 : gap;
      sh.builtAt = this._clock;
      sh.name = frac < 0.5 ? sh.from : sh.to;
      sh.rebuilds++;
      this._builtK[best] = 1;
      this._accReb++;
      const cost = now() - r0;
      this.reparamMs += cost;
      this.rebuildsThisFrame++;
      // 0.25 keeps the estimate inside ~2% of a settled cost within a dozen
      // rebuilds, while a single slow frame (a GC landing mid-reparam) moves
      // it by a quarter rather than setting it.
      this.reparamEst += (cost - this.reparamEst) * 0.25;
    }
    // Advance every slot's blend, every frame, from the PARAMETER rather
    // than from a stopwatch.
    //
    // The fraction wanted is "how far has this slot's morph parameter moved
    // between the two builds it is interpolating", and the pool knows that
    // exactly: `uA` is where the displayed shape was when the newest build
    // landed and `uB` is where that build is. Dividing elapsed time by a
    // smoothed estimate of the rebuild interval -- what this did before --
    // is the same quantity guessed at, and the guess is low often enough
    // that the blend reached 1 and sat there for a third of the frames.
    //
    // Costs about sixty lerps per slot and touches no table: the traversal
    // does the interpolation as it reads. So every slot gets it, not the two
    // the old budget of places could afford.
    if (this.interpolate) {
      for (let k = 0; k < this.slots; k++) {
        const sh = this.shapes[k];
        const u = this._u + (k * N) / this.slots;
        const span = sh.uB - sh.uA;
        sh.att.setBlend(span > 1e-12 ? (u - sh.uA) / span : 1,
                        this.blendMax, this.blendMin);
      }
    }
    this.deferred = nDue;
    this._meter(dt);
    this.lastMs = now() - t0;
  }
}

/**
 * Which cells are currently real, and nothing else.
 *
 * No camera, no matrices, no draw order: this decides what exists and where,
 * and the next step composes it into World, which is where frames and sorting
 * belong.
 *
 * Residency is a pure function of WHICH CELL the camera is in, not of where it
 * is inside that cell.  That is what makes the boundary-cross early-out exact
 * rather than approximate: enumerating (2*ceil(genRadius/L)+1)^3 cells every
 * frame is 2197 hashes at the default settings, for a set that provably cannot
 * have changed.  The price is that the effective generation radius wobbles by
 * the camera's offset within its cell, at most L*sqrt(3)/2 = 8.7 of 55 units;
 * the eviction hysteresis band (55 -> 66) is wider than that, so nothing
 * thrashes.
 */
export class FieldLayout {
  constructor(pool, opts = {}) {
    this.pool = pool || null;
    this.L = opts.L ?? 10;
    // 0.22, not 0.35: a cell now spends part of its probability on
    // subdividing, and the hierarchy multiplies what is left.  See the
    // expected-objects arithmetic by P_SUB.
    this.density = opts.density ?? 0.22;
    this.seed = opts.seed ?? 1;
    this.rho = opts.rho ?? 0.25;   // see FieldLayout
    this.uMin = opts.uMin ?? 0.35;
    this.slots = opts.slots ?? (pool ? pool.slots : 8);
    this.genRadius = opts.genRadius ?? 55;
    this.evictFactor = opts.evictFactor ?? 1.2;

    this.cells = new Map();
    this.pinned = new Set();

    // One opts object, reused for every cellContents call: it is read-only in
    // there, and a fresh literal per candidate cell would be ~700 objects per
    // boundary cross.
    this._cellOpts = { L: this.L, density: this.density, seed: this.seed,
                       rho: this.rho, uMin: this.uMin, slots: this.slots };

    this.ci = 0; this.cj = 0; this.ck = 0;
    this._have = false;

    this.lastEnumeration = 0;     // ms spent in the last re-enumeration
    this.materialised = 0;        // this update
    this.evicted = 0;             // this update
    this.enumerations = 0;        // cumulative: how often the box was walked
    this.updates = 0;             // cumulative: how often update() was called
    this.totalMaterialised = 0;
    this.totalEvicted = 0;

    // In cell units, squared, so the inner loop is integer arithmetic.
    this._rad = Math.ceil(this.genRadius / this.L);
    this._gen2 = (this.genRadius / this.L) ** 2;
    this._evi2 = ((this.genRadius * this.evictFactor) / this.L) ** 2;
    this._evictR2 = (this.genRadius * this.evictFactor) ** 2;   // world units
    this._batch = [];        // one root cell's objects, reused
  }

  get size() { return this.cells.size; }
  get(key) { return this.cells.get(key); }
  /** Iterable of the live records.  Not a copy -- do not mutate while walking. */
  resident() { return this.cells.values(); }

  /** The key of the cell a world point is in.  The caller pins this one for
   *  the camera: it is the only cell the camera can have descended into, and
   *  evicting it would leave the camera holding an address into an object that
   *  no longer exists. */
  keyAt(p) {
    const L = this.L;
    return cellKey(Math.floor(p[0] / L), Math.floor(p[1] / L), Math.floor(p[2] / L));
  }

  /** The live attractor a record draws, or null without a pool. */
  shapeFor(rec) { return this.pool ? this.pool.shape(rec.slot) : null; }

  /**
   * Bring residency up to date for a camera at `worldPos`.  Returns true if it
   * actually re-enumerated (i.e. the camera crossed a cell boundary).
   */
  update(p) {
    this.updates++;
    this.materialised = 0;
    this.evicted = 0;
    const L = this.L;
    const ci = Math.floor(p[0] / L), cj = Math.floor(p[1] / L), ck = Math.floor(p[2] / L);
    if (this._have && ci === this.ci && cj === this.cj && ck === this.ck) return false;
    this.ci = ci; this.cj = cj; this.ck = ck; this._have = true;
    this.enumerations++;

    const t0 = now();
    const r = this._rad, g2 = this._gen2, seed = this.seed, density = this.density;
    const cells = this.cells, copts = this._cellOpts;
    const batch = this._batch;
    for (let dz = -r; dz <= r; dz++) {
      for (let dy = -r; dy <= r; dy++) {
        for (let dx = -r; dx <= r; dx++) {
          if (dx * dx + dy * dy + dz * dz > g2) continue;
          const ix = ci + dx, iy = cj + dy, iz = ck + dz;
          // Test the root cell before walking it.  Most are empty, and an
          // empty cell must leave no trace at all -- not even a string.
          if (!cellOccupied(cellHash(ix, iy, iz, seed), density)) continue;
          // A root cell is no longer one object: it is a subtree, and may
          // yield none, one, or a cluster.
          batch.length = 0;
          cellObjects(ix, iy, iz, copts, batch);
          for (let i = 0; i < batch.length; i++) {
            const rec = batch[i];
            if (cells.has(rec.key)) continue;
            cells.set(rec.key, rec);
            this.materialised++;
          }
        }
      }
    }

    // Eviction in WORLD units, against the object's own centre.  The flat
    // field could compare cell indices because every object was a root cell;
    // an object at level 2 has coordinates in quarter-size units, and
    // comparing those against the camera's root-cell index would evict the
    // small ones almost immediately and keep distant ones forever.
    const e2 = this._evictR2, pinned = this.pinned;
    for (const rec of cells.values()) {
      const dx = rec.t[0] - p[0], dy = rec.t[1] - p[1], dz = rec.t[2] - p[2];
      if (dx * dx + dy * dy + dz * dz <= e2) continue;
      if (pinned.has(rec.key)) continue;
      cells.delete(rec.key);
      this.evicted++;
    }

    this.lastEnumeration = now() - t0;
    this.totalMaterialised += this.materialised;
    this.totalEvicted += this.evicted;
    return true;
  }

  /** Pin a cell against eviction, materialising it if it is not resident.
   *  Returns the record, or null if that cell is simply empty. */
  pin(key) {
    this.pinned.add(key);
    let rec = this.cells.get(key);
    if (!rec) {
      const c = parseKey(key);
      if (c) rec = cellContents(c[0], c[1], c[2], this._cellOpts);
      if (rec) this.cells.set(key, rec);
    }
    return rec || null;
  }

  unpin(key) { return this.pinned.delete(key); }
  isPinned(key) { return this.pinned.has(key); }
}
