// Map sets, described by parameters rather than by matrices.
//
// Stage 3 morphs one attractor into another, continuously, and every splat has
// to move smoothly while it happens.  Interpolating raw 3x3 matrices does not
// give that: halfway between two rotations is a shrunken, sheared thing, and
// it can pass through a singular matrix on the way.  So a map is stored as the
// numbers a person would choose -- a contraction, a tilt, a roll, a twist, an
// attach point -- and interpolation happens there, where every intermediate
// value is a valid map.
//
//     A = s * Rz(roll) * Ry(tilt) * Rz(twist) * diag(aniso)
//
// With aniso = [1,1,1] that is a similarity, which is worth keeping wherever
// possible: a similarity's accumulated affine has ||A|| = product of the s's,
// a single multiply, and the LOD cut and the frustum test are both one
// multiply because of it.

const D2R = Math.PI / 180;

function rz(a) { const c = Math.cos(a), s = Math.sin(a);
  return [c, -s, 0, s, c, 0, 0, 0, 1]; }
function ry(a) { const c = Math.cos(a), s = Math.sin(a);
  return [c, 0, s, 0, 1, 0, -s, 0, c]; }
function mul(A, B) {
  const o = new Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    o[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
  return o;
}

/** One parametric map -> its 3x3 linear part. */
export function mapMatrix(m) {
  const an = m.aniso || [1, 1, 1];
  let A = mul(mul(rz((m.roll || 0) * D2R), ry((m.tilt || 0) * D2R)),
              rz((m.twist || 0) * D2R));
  const o = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    o[i * 3 + j] = m.s * A[i * 3 + j] * an[j];
  return o;
}

export function isSimilarity(m) {
  const a = m.aniso;
  return !a || (Math.abs(a[0] - a[1]) < 1e-12 && Math.abs(a[1] - a[2]) < 1e-12);
}

/** A spec -> the {A, b} list the Attractor wants, plus the terminal map. */
export function build(spec) {
  return {
    maps: spec.maps.map((m) => ({ A: mapMatrix(m), b: (m.b || [0, 0, 1]).slice() })),
    trunkA: mapMatrix(spec.trunk),
    sig0: spec.sig0.slice(),
    mu0: spec.mu0.slice(),
    similarity: spec.maps.every(isSimilarity),
  };
}

const lerp = (a, b, t) => a + (b - a) * t;

/**
 * The shortest signed step from angle `a` to angle `b`, in degrees.
 *
 * The tie-break matters more than it looks.  At exactly 180 apart there is no
 * shorter way round -- both are equal -- and the plain formula resolves the
 * tie the same way whichever direction it is asked about, so delta(a,b) and
 * delta(b,a) both come back -180 instead of being opposites.  Pairwise that
 * is harmless.  On the closed spline it is not: the tangent at the end of one
 * segment is built from delta(a,b) and the tangent at the start of the next
 * from delta(b,a), so a tie makes the two disagree and puts a corner in a
 * path whose entire purpose is not to have one.  `roll` on map 3 steps
 * exactly 180 twice in this set (dust->spine 60->240, spine->tree 240->60),
 * which is how this was found: waypoints 1-5 were smooth to 0.02% and the two
 * either side of the wrap sat at 42% and 13% no matter how fine the step.
 *
 * Resolving the tie by value order makes delta antisymmetric, so the two
 * tangents agree.  Which of the two equal ways round it picks is arbitrary
 * and invisible at the endpoints; being consistent about it is what counts.
 */
function delta(a, b) {
  const d = ((b - a) % 360 + 540) % 360 - 180;
  if (Math.abs(Math.abs(d) - 180) < 1e-12) {
    // Resolve the tie on the RESIDUES, never on the raw values.  crAngle
    // unwraps its controls into a running chain, so by the time this is asked
    // about the same pair from the next segment's point of view, one of them
    // may have had a multiple of 360 added to it.  `b > a` on raw values then
    // answers differently for what is geometrically the same question, the
    // antisymmetry this tie-break exists to provide is lost again, and the
    // corner comes back -- which is exactly what `whirl`, with its rolls at
    // 0/90/180/270, did to the first version of this.
    const an = ((a % 360) + 360) % 360, bn = ((b % 360) + 360) % 360;
    return bn > an ? 180 : -180;
  }
  return d;
}

/** Angles are periodic: interpolate along the short way round. */
function lerpAngle(a, b, t) {
  return a + delta(a, b) * t;
}

function lerpMap(x, y, t) {
  return {
    s: lerp(x.s, y.s, t),
    tilt: lerpAngle(x.tilt || 0, y.tilt || 0, t),
    roll: lerpAngle(x.roll || 0, y.roll || 0, t),
    twist: lerpAngle(x.twist || 0, y.twist || 0, t),
    aniso: [0, 1, 2].map((k) => lerp((x.aniso || [1, 1, 1])[k], (y.aniso || [1, 1, 1])[k], t)),
    b: [0, 1, 2].map((k) => lerp((x.b || [0, 0, 1])[k], (y.b || [0, 0, 1])[k], t)),
  };
}

/**
 * Interpolate two specs.  They must have the same number of maps -- which is
 * why every set below has four: an attractor can morph into any other one,
 * and a sixteen-second crossfade between any pair is a whole space of shapes
 * rather than six fixed ones.
 */
export function morph(a, b, t) {
  if (a.maps.length !== b.maps.length)
    throw new Error(`cannot morph ${a.name} (${a.maps.length} maps) into `
                  + `${b.name} (${b.maps.length})`);
  return {
    name: `${a.name} -> ${b.name}`,
    maps: a.maps.map((m, i) => lerpMap(m, b.maps[i], t)),
    trunk: lerpMap(a.trunk, b.trunk, t),
    sig0: [0, 1, 2].map((k) => lerp(a.sig0[k], b.sig0[k], t)),
    mu0: [0, 1, 2].map((k) => lerp(a.mu0[k], b.mu0[k], t)),
    palette: t < 0.5 ? a.palette : b.palette,
    omega: lerp(a.omega ?? 0.28, b.omega ?? 0.28, t),
    jitter: lerp(a.jitter ?? 0.16, b.jitter ?? 0.16, t),
  };
}

// ------------------------------------------------- the continuous path

/** Catmull-Rom through four control values. */
const cr = (p0, p1, p2, p3, t) => {
  const t2 = t * t, t3 = t2 * t;
  return 0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
};

/**
 * The same on an angle.  Unwrap the four controls into one continuous run
 * before splining: 350 and 10 degrees are twenty apart, and a spline told they
 * are three hundred and forty apart sends the map spinning the wrong way
 * through the whole turn.
 */
function crAngle(p0, p1, p2, p3, t) {
  const near = (from, a) => from + delta(from, a);
  const q1 = p1, q0 = near(q1, p0), q2 = near(q1, p2), q3 = near(q2, p3);
  return cr(q0, q1, q2, q3, t);
}

const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

/**
 * Catmull-Rom in log space, for quantities that must stay strictly positive:
 * scales, standard deviations, anisotropy.
 *
 * The spline overshoots the hull of its controls by design -- that overshoot
 * is most of what makes the motion look alive -- but for a positive quantity
 * an overshoot DOWNWARD is not liveliness, it is a sign error.  sig0 runs
 * from 0.0016 to 0.39 across these sets, and splining it linearly sends it to
 * -0.021 on the way down: a negative variance, which is a degenerate splat,
 * not a small one.
 *
 * Interpolating the logarithm instead is positive by construction, and it is
 * the right geometry anyway -- halfway between 0.0016 and 0.39 should be
 * their geometric mean, 0.025, not the arithmetic 0.196 that spends nearly
 * the whole transit looking like the larger one.  exp of a C1 function is C1,
 * so this costs nothing in smoothness, which is the point of the stage.
 *
 * The alternative, clamping, is what this replaces: a clamp whose bounds come
 * from the segment being travelled CHANGES at every waypoint, which puts a
 * corner back exactly where the spline was introduced to remove one.
 */
const crPos = (p0, p1, p2, p3, t) => Math.exp(
  cr(Math.log(Math.max(p0, 1e-12)), Math.log(Math.max(p1, 1e-12)),
     Math.log(Math.max(p2, 1e-12)), Math.log(Math.max(p3, 1e-12)), t));

/**
 * Catmull-Rom for quantities that must stay >= 0 and ARE legitimately zero.
 *
 * omega and jitter are both like this: the four crystalline sets have no
 * trunk at all (omega = 0) and sierpinski has no jitter at all, and zero is
 * the meaningful value, not a floor.  Log space cannot represent it.  A clamp
 * at zero can, but a clamp is flat while it is active and sloped the instant
 * it is not, so every set sitting exactly ON the clamp gets a corner -- which
 * is what the four new sets did, each one putting a velocity jump into the
 * path at its own waypoint.
 *
 * Splining the square root and squaring the result is non-negative by
 * construction with no clamp anywhere: d(w^2)/dt = 2*w*w' is continuous
 * wherever w' is, including at w = 0.  It also makes a trunk grow in from
 * nothing gently rather than linearly, which is the nicer reading of what
 * omega means.
 */
const crNonNeg = (p0, p1, p2, p3, t) => {
  const w = cr(Math.sqrt(Math.max(p0, 0)), Math.sqrt(Math.max(p1, 0)),
               Math.sqrt(Math.max(p2, 0)), Math.sqrt(Math.max(p3, 0)), t);
  return w * w;
};

function crMap(x0, x1, x2, x3, t) {
  const an = (m) => m.aniso || [1, 1, 1];
  const bb = (m) => m.b || [0, 0, 1];
  return {
    // Scale in log space: positive by construction, and the overshoot that
    // remains is bounded well under contraction 1 (0.83 over the whole loop,
    // against controls that reach 0.80).  tests/mapset.mjs re-measures it.
    s: crPos(x0.s, x1.s, x2.s, x3.s, t),
    tilt: crAngle(x0.tilt || 0, x1.tilt || 0, x2.tilt || 0, x3.tilt || 0, t),
    roll: crAngle(x0.roll || 0, x1.roll || 0, x2.roll || 0, x3.roll || 0, t),
    twist: crAngle(x0.twist || 0, x1.twist || 0, x2.twist || 0, x3.twist || 0, t),
    aniso: [0, 1, 2].map((k) => crPos(an(x0)[k], an(x1)[k], an(x2)[k], an(x3)[k], t)),
    // Translations may be negative and mean it, so these stay linear.
    b: [0, 1, 2].map((k) => cr(bb(x0)[k], bb(x1)[k], bb(x2)[k], bb(x3)[k], t)),
  };
}

/**
 * One point on a closed loop through every set.
 *
 * `u` is a continuous position, in units of one set: u = 3.5 is halfway from
 * specs[3] to specs[4], and u wraps.  Unlike morph(), which walks a chain of
 * straight lines and so has a corner at every waypoint, this is C1 everywhere
 * -- the shape never changes direction abruptly, and the world never stops
 * moving.  What makes a shape legible at a waypoint is no longer a pause in
 * the path but a slowing of the BREATH that drives it (see field.js), which is
 * the same effect without the corner.
 */
// How many stops a blended palette has.
//
// Fixed, and that is the point. Palettes in PALETTES are different lengths --
// `tree` has 7 stops, `spectral` has 10 -- and a splat's colour is
// `pal[level % pal.length]`. So if the length changed as one set morphed into
// the next, every splat in the tree would be reassigned a different stop at
// that moment: a hard recolour of the whole object, no matter how smoothly the
// colours themselves were interpolated. Resampling every palette to one
// length removes that failure mode entirely rather than managing it.
export const PAL_STOPS = 16;

/** Sample a palette of any length at u in [0,1], linearly. */
function palAt(pal, u, out) {
  const x = u * (pal.length - 1);
  const i = Math.min(pal.length - 2, Math.max(0, Math.floor(x)));
  const f = Math.min(1, Math.max(0, x - i));
  const a = pal[i], b = pal[i + 1];
  out[0] = a[0] + (b[0] - a[0]) * f;
  out[1] = a[1] + (b[1] - a[1]) * f;
  out[2] = a[2] + (b[2] - a[2]) * f;
  return out;
}

const _pa = [0, 0, 0], _pb = [0, 0, 0], _pc = [0, 0, 0], _pd = [0, 0, 0];

/**
 * Catmull-Rom between four palettes, into `out` (PAL_STOPS x [r,g,b]).
 *
 * The palette travels the same spline as the geometry, for the same reason:
 * `t < 0.5 ? B.palette : C.palette` -- which is what this replaces -- is a
 * step function. It swapped the whole colour scheme in a single frame at the
 * midpoint of every morph, while the shape around it moved continuously. That
 * is the project's recurring defect in its purest form: the spline was never
 * the problem, the guard wrapped around it was.
 *
 * `out` is caller-owned so that each pool slot keeps its own buffer. Sharing
 * one scratch would have every slot overwrite the palette the others are
 * drawing from.
 */
export function crPalette(A, B, C, D, t, out) {
  for (let k = 0; k < PAL_STOPS; k++) {
    const u = k / (PAL_STOPS - 1);
    palAt(A, u, _pa); palAt(B, u, _pb); palAt(C, u, _pc); palAt(D, u, _pd);
    const o = out[k];
    for (let ch = 0; ch < 3; ch++) {
      // Clamped to the gamut. Catmull-Rom can overshoot a little, and a
      // negative or >1 channel is not a colour. This is a clip, so in
      // principle it is a corner -- but it can only engage where a channel is
      // already at 0 or 1 and the overshoot is a fraction of a percent, which
      // is below the 1/255 the framebuffer can represent.
      const v = cr(_pa[ch], _pb[ch], _pc[ch], _pd[ch], t);
      o[ch] = v < 0 ? 0 : v > 1 ? 1 : v;
    }
  }
  return out;
}

/** A fresh buffer of the right shape for crPalette. */
export function palBuffer() {
  const out = new Array(PAL_STOPS);
  for (let k = 0; k < PAL_STOPS; k++) out[k] = [0, 0, 0];
  return out;
}

/** The four palette names the spline is between at u, in A,B,C,D order. */
export function palStopsAt(specs, u) {
  const n = specs.length, f = Math.floor(u);
  const i1 = ((f % n) + n) % n;
  return [specs[(i1 - 1 + n) % n], specs[i1], specs[(i1 + 1) % n], specs[(i1 + 2) % n]];
}

export function morphPath(specs, u) {
  const n = specs.length;
  if (n < 4) throw new Error(`a closed spline needs at least 4 sets, got ${n}`);
  const f = Math.floor(u), t = u - f;
  const i1 = ((f % n) + n) % n;
  const i0 = (i1 - 1 + n) % n, i2 = (i1 + 1) % n, i3 = (i1 + 2) % n;
  const A = specs[i0], B = specs[i1], C = specs[i2], D = specs[i3];
  return {
    name: `${B.name} -> ${C.name}`,
    maps: B.maps.map((_, i) => crMap(A.maps[i], B.maps[i], C.maps[i], D.maps[i], t)),
    trunk: crMap(A.trunk, B.trunk, C.trunk, D.trunk, t),
    sig0: [0, 1, 2].map((k) => crPos(A.sig0[k], B.sig0[k], C.sig0[k], D.sig0[k], t)),
    mu0: [0, 1, 2].map((k) => cr(A.mu0[k], B.mu0[k], C.mu0[k], D.mu0[k], t)),
    // The NEAREST name, for the HUD and for debugging only. The colour the
    // renderer actually uses is the spline blend -- see crPalette, and
    // field.js where it is applied. Reading this as the palette is what made
    // the colours jump at every midpoint.
    palette: t < 0.5 ? B.palette : C.palette,
    palT: t,
    omega: crNonNeg(A.omega ?? 0.28, B.omega ?? 0.28, C.omega ?? 0.28, D.omega ?? 0.28, t),
    jitter: crNonNeg(A.jitter ?? 0.16, B.jitter ?? 0.16, C.jitter ?? 0.16, D.jitter ?? 0.16, t),
  };
}

// --------------------------------------------------------------- the sets

const H = 2.5;                       // the reference height SIG0 is built from
const SIG = [0.2025, 0.2025, 0.390625];
const MU = [0, 0, H / 2];

export const SETS = {
  // the original: a tree.  Three branches and a leader shoot.
  tree: {
    name: 'tree',
    maps: [
      { s: 0.60, tilt: 38, roll: 0, twist: 25, b: [0, 0, 1] },
      { s: 0.60, tilt: 38, roll: 120, twist: 25, b: [0, 0, 1] },
      { s: 0.60, tilt: 38, roll: 240, twist: 25, b: [0, 0, 1] },
      { s: 0.52, tilt: 9, roll: 60, twist: 40, b: [0, 0, 1] },
    ],
    trunk: { s: 1, aniso: [0.075, 0.075, 0.4], tilt: 0, roll: 0, twist: 0 },
    sig0: SIG, mu0: MU, palette: 'tree', omega: 0.28,
  },

  // a volute: a rod that turns a fixed angle and lifts a fixed amount every
  // segment, so the chain of rods is a curl -- of curls, of curls.  It winds
  // about -y, the axis the camera looks down, so the curl is seen face on.
  spiral: {
    name: 'spiral',
    maps: [
      { s: 0.78, tilt: -26, roll: 0, twist: 14, b: [0.74, -0.16, 0] },
      { s: 0.42, tilt: -26, roll: 0, twist: 14, b: [0.40, -0.08, 0] },
      { s: 0.30, tilt: -104, roll: 0, twist: 24, b: [0.58, -0.03, 0] },
      { s: 0.24, tilt: 84, roll: 0, twist: 24, b: [0.28, -0.03, 0] },
    ],
    trunk: { s: 1, aniso: [1, 1, 1], tilt: 0, roll: 0, twist: 0 },
    sig0: [0.055, 0.0016, 0.0016], mu0: [0, 0, 0],
    palette: 'violet', omega: 0.30,
  },

  // a horn: the same winding, but what is wound is a plate rather than a rod.
  // The flatness is in the plate, not in the maps: an anisotropic map is not a
  // similarity, and the camera's re-root inverts a map assuming it is one.
  shell: {
    name: 'shell',
    maps: [
      { s: 0.80, tilt: -24, roll: 0, twist: 8, b: [0.74, -0.26, 0] },
      { s: 0.40, tilt: -24, roll: 0, twist: 8, b: [0.41, -0.12, 0] },
      { s: 0.26, tilt: -110, roll: 0, twist: 10, b: [0.57, -0.05, 0] },
      { s: 0.20, tilt: 70, roll: 0, twist: 10, b: [0.26, -0.04, 0] },
    ],
    trunk: { s: 1, aniso: [1, 1, 1], tilt: 0, roll: 0, twist: 0 },
    sig0: [0.050, 0.0003, 0.024], mu0: [0, 0, 0],
    palette: 'ember', omega: 0.38,
  },

  // a crystal: half-scale cells on a corner of a cube, and nothing but square
  // turns, so every beam ends up on an axis.  The terminal shape is a beam
  // rather than a blob: a scaffold has void in it, and void is what survives
  // the dive.
  lattice: {
    name: 'lattice',
    maps: [
      { s: 0.44, tilt: 0, roll: 0, twist: 0, b: [0, 0, 0] },
      { s: 0.44, tilt: 90, roll: 0, twist: 0, b: [0.5, 0, 0] },
      { s: 0.44, tilt: 90, roll: 90, twist: 0, b: [0, 0.5, 0] },
      { s: 0.44, tilt: 0, roll: 90, twist: 0, b: [0, 0, 0.5] },
    ],
    trunk: { s: 1, aniso: [1, 1, 1], tilt: 0, roll: 0, twist: 0 },
    sig0: [0.00025, 0.00025, 0.013], mu0: [0, 0, 0.22],
    palette: 'ice', omega: 0.15,
  },

  // a frond: Barnsley's four maps.  Nothing here is flat except the leaflet
  // the trunk map draws -- the sheet is flat because every map turns almost
  // entirely within it, and crumpled because the pinnae do not quite.
  plane: {
    name: 'plane',
    maps: [
      { s: 0.74, tilt: -5, roll: 0, twist: 3, b: [0.30, -0.02, 0] },
      { s: 0.35, tilt: -76, roll: 0, twist: 18, b: [0.16, -0.03, 0.05] },
      { s: 0.35, tilt: 76, roll: 0, twist: -18, b: [0.16, -0.03, -0.05] },
      { s: 0.24, tilt: 176, roll: 0, twist: 0, b: [0.02, -0.03, 0] },
    ],
    trunk: { s: 1, aniso: [1, 1, 1], tilt: 0, roll: 0, twist: 0 },
    sig0: [0.010, 0.0002, 0.005], mu0: [0, 0, 0],
    palette: 'spectral', omega: 0.26,
  },

  // a dust: four small copies, thrown far enough apart that the set never
  // joins up -- clumps of clumps of clumps, and void in between.
  dust: {
    name: 'dust',
    maps: [
      { s: 0.42, tilt: 20, roll: 0, twist: 0, b: [0.70, 0, 0.55] },
      { s: 0.42, tilt: 125, roll: 120, twist: 40, b: [-0.35, 0.60, -0.55] },
      { s: 0.40, tilt: 70, roll: 240, twist: 80, b: [-0.35, -0.60, 0.55] },
      { s: 0.38, tilt: 160, roll: 60, twist: 20, b: [0, 0, -0.75] },
    ],
    trunk: { s: 1, aniso: [0.12, 0.12, 0.12], tilt: 0, roll: 0, twist: 0 },
    sig0: [0.2025, 0.2025, 0.2025], mu0: [0, 0, 0],
    palette: 'iridescent', omega: 0.30,
  },

  // a spine: a straight tapering shaft with whorls of three thorns, each
  // whorl turned a fixed angle from the one below.
  spine: {
    name: 'spine',
    maps: [
      { s: 0.80, tilt: 0, roll: 0, twist: 47, b: [0, 0, 0.72] },
      { s: 0.32, tilt: 96, roll: 0, twist: 0, b: [0, 0, 0.62] },
      { s: 0.32, tilt: 96, roll: 120, twist: 0, b: [0, 0, 0.62] },
      { s: 0.32, tilt: 96, roll: 240, twist: 0, b: [0, 0, 0.62] },
    ],
    trunk: { s: 1, aniso: [0.045, 0.045, 0.34], tilt: 0, roll: 0, twist: 0 },
    sig0: SIG, mu0: MU, palette: 'ember', omega: 0.22,
  },

  // ---------------------------------------------------------------------
  // Not trees.
  //
  // Everything above is a rod with things growing off it: omega, the trunk's
  // mass fraction, runs 0.15 to 0.38 across all seven, and every one of them
  // uses the same jitter of 0.16.  That is why the field reads as a wood no
  // matter which set a cell draws.  omega = 0 removes the rod entirely and
  // leaves the bare attractor, and jitter = 0 removes the per-node wobble and
  // leaves a crystal.  Neither needed an engine change -- the knobs were
  // already there and had simply never been turned.
  //
  // These four exist to be the other end of both knobs, so that the spline
  // between a tree and a Sierpinski gasket is a real journey rather than a
  // choice between seven shades of shrub.

  // The Sierpinski tetrahedron: four halvings onto the vertices of a
  // tetrahedron.  The canonical non-tree fractal, and exact -- contraction
  // 0.5, R = sqrt(3), no trunk, no jitter.
  sierpinski: {
    name: 'sierpinski',
    maps: [
      { s: 0.5, tilt: 0, roll: 0, twist: 0, b: [0.5, 0.5, 0.5] },
      { s: 0.5, tilt: 0, roll: 0, twist: 0, b: [0.5, -0.5, -0.5] },
      { s: 0.5, tilt: 0, roll: 0, twist: 0, b: [-0.5, 0.5, -0.5] },
      { s: 0.5, tilt: 0, roll: 0, twist: 0, b: [-0.5, -0.5, 0.5] },
    ],
    trunk: { s: 1, aniso: [1, 1, 1], tilt: 0, roll: 0, twist: 0 },
    sig0: [0.012, 0.012, 0.012], mu0: [0, 0, 0],
    palette: 'ice', omega: 0, jitter: 0,
  },

  // The same skeleton, every map turned: still crystalline, but chiral, and
  // it no longer closes into flat faces.
  whirl: {
    name: 'whirl',
    maps: [
      { s: 0.52, tilt: 35, roll: 0, twist: 55, b: [0.5, 0.5, 0.5] },
      { s: 0.52, tilt: 35, roll: 90, twist: 55, b: [0.5, -0.5, -0.5] },
      { s: 0.52, tilt: 35, roll: 180, twist: 55, b: [-0.5, 0.5, -0.5] },
      { s: 0.52, tilt: 35, roll: 270, twist: 55, b: [-0.5, -0.5, 0.5] },
    ],
    trunk: { s: 1, aniso: [1, 1, 1], tilt: 0, roll: 0, twist: 0 },
    sig0: [0.014, 0.014, 0.014], mu0: [0, 0, 0],
    palette: 'violet', omega: 0, jitter: 0.04,
  },

  // A folded curve in the dragon family: two large folds at right angles,
  // plus two small ones that lift it out of its plane.
  fold: {
    name: 'fold',
    maps: [
      { s: 0.70, tilt: 0, roll: 45, twist: 0, b: [0.28, 0, 0] },
      { s: 0.70, tilt: 0, roll: 135, twist: 0, b: [-0.28, 0, 0] },
      { s: 0.38, tilt: 72, roll: 0, twist: 0, b: [0, 0.24, 0.18] },
      { s: 0.38, tilt: -72, roll: 180, twist: 0, b: [0, -0.24, -0.18] },
    ],
    trunk: { s: 1, aniso: [1, 1, 1], tilt: 0, roll: 0, twist: 0 },
    sig0: [0.016, 0.016, 0.016], mu0: [0, 0, 0],
    palette: 'ember', omega: 0, jitter: 0.02,
  },

  // Four maps on the corners of a square, turned on their sides, so the thing
  // closes on itself into a cage of struts rather than spreading outward.
  cage: {
    name: 'cage',
    maps: [
      { s: 0.46, tilt: 90, roll: 0, twist: 0, b: [0.34, 0.34, 0] },
      { s: 0.46, tilt: 90, roll: 90, twist: 0, b: [-0.34, 0.34, 0] },
      { s: 0.46, tilt: 90, roll: 180, twist: 0, b: [-0.34, -0.34, 0] },
      { s: 0.46, tilt: 90, roll: 270, twist: 0, b: [0.34, -0.34, 0] },
    ],
    trunk: { s: 1, aniso: [1, 1, 1], tilt: 0, roll: 0, twist: 0 },
    sig0: [0.013, 0.013, 0.013], mu0: [0, 0, 0],
    palette: 'spectral', omega: 0, jitter: 0.06,
  },
};

/**
 * The order of the closed morph loop.
 *
 * Not Object.keys order: the loop alternates organic and crystalline, so
 * every segment of the spline is a journey between opposites -- a tree
 * hardening into a gasket, a cage unwinding into a volute -- rather than a
 * drift between neighbours that already look alike.  Each set still appears
 * exactly once, and the list is checked against SETS below so a set added to
 * one and not the other is a loud error rather than a silently missing shape.
 */
// The sets the world morphs through, in the order it visits them.
//
// `tree` is deliberately NOT here. It is still defined in SETS, and must stay
// defined: py/parity_check.py and the stage-0 measurements are all built on
// the tree, so deleting it would throw away the reference the whole engine is
// checked against. Excluding it from this list removes it from the world
// without touching that reference.
export const SET_NAMES = [
  'sierpinski', 'spiral', 'cage', 'shell', 'whirl',
  'lattice', 'fold', 'plane', 'dust', 'spine',
];
// Sets that exist but are not in the cycle. Listed explicitly so that the
// check below still catches a set added to SETS and forgotten here, which is
// the mistake the check was written for -- a bare subset test would silently
// accept it.
export const SETS_NOT_IN_CYCLE = ['tree'];
{
  const have = Object.keys(SETS).sort().join(',');
  const want = [...SET_NAMES, ...SETS_NOT_IN_CYCLE].sort().join(',');
  if (have !== want)
    throw new Error(`SET_NAMES and SETS disagree:\n  SETS      ${have}\n  SET_NAMES ${want}`);
  if (SET_NAMES.length < 4)
    throw new Error(`a closed spline needs at least 4 sets, got ${SET_NAMES.length}`);
}
