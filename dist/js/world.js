// Stage 3: several attractors in a dark void, and a camera that can leave one
// and fly to another.
//
// The hard part is that stage 2's camera does not have a world position.  It
// has an ADDRESS -- which node of which attractor it is inside -- plus a
// position within that node, and that is the whole point: it is what lets the
// zoom run past the end of float32.  A world position would defeat it at about
// depth 25.
//
// So the void is built the other way round.  Every attractor keeps a placement
// (where it sits, how big, which way up).  The camera keeps its address inside
// ONE of them, the home.  To draw another attractor, its root is pushed
// through
//
//     j's local  ->  world  ->  home's local  ->  the camera's frame
//
// and the last of those three is the re-rooting transform that already exists.
// Nothing ever needs a world position for the camera, and the other attractors
// fall out of the frustum and into the fog by themselves as it descends --
// at depth 20 they are genuinely a million local units away, which is the
// right answer and costs nothing to compute.

import { Attractor, FlyCamera, AdaptiveStream, PALETTES } from './adaptive.js';
import { SETS, SET_NAMES, build, morph } from './maps.js';
import { ShapePool, FieldLayout, REACH_MAX as REACH_MAX_W } from './field.js';

const DEG = Math.PI / 180;

function rotY(a) { const c = Math.cos(a), s = Math.sin(a);
  return [c, 0, s, 0, 1, 0, -s, 0, c]; }
function rotZ(a) { const c = Math.cos(a), s = Math.sin(a);
  return [c, -s, 0, s, c, 0, 0, 0, 1]; }
function mm3(A, B) {
  const o = new Float64Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++)
    o[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
  return o;
}
const mv3 = (A, v) => [A[0] * v[0] + A[1] * v[1] + A[2] * v[2],
                       A[3] * v[0] + A[4] * v[1] + A[5] * v[2],
                       A[6] * v[0] + A[7] * v[1] + A[8] * v[2]];
const mtv3 = (A, v) => [A[0] * v[0] + A[3] * v[1] + A[6] * v[2],
                        A[1] * v[0] + A[4] * v[1] + A[7] * v[2],
                        A[2] * v[0] + A[5] * v[1] + A[8] * v[2]];

/** One attractor in the void: its maps, where it sits, and what it is becoming. */
export class Placed {
  /**
   * `att` shares an existing Attractor instead of building one -- how a field
   * cell works.  The expensive, morphing part of an attractor (the moment
   * fixed point, the child table) then costs once per POOL SLOT rather than
   * once per object, which is what makes a field of them affordable at all.
   *
   * What keeps two cells on the same slot from being literal copies is
   * `hash`: it is the root of the address-hashed jitter, so the same map set
   * walked from a different root hash is a different random REALISATION of
   * the same shape -- different branch angles the whole way down -- for no
   * extra memory and no extra rebuild.  `tint` and the placement do the rest.
   */
  constructor(name, { pos, scale = 1, yaw = 0, tilt = 0, palette, seed = 3,
                      jitter = 0.16, att = null, pool = null, slot = 0,
                      hash = null, tint = null, reach = null, cell = null,
                      fx = 0xff } = {}) {
    this._name = name;
    this.pool = pool; this.slot = slot;
    if (att) {
      this.att = att;
    } else {
      const spec = SETS[name];
      const b = build(spec);
      this.att = new Attractor({
        maps: b.maps, trunkA: b.trunkA, sig0: b.sig0, mu0: b.mu0,
        omega: spec.omega ?? 0.28, seed, jitter,
        palette: PALETTES[palette || spec.palette] || PALETTES.spectral,
      });
    }
    this.t = pos.slice();
    this.s = scale;
    this.R = mm3(rotZ(yaw * DEG), rotY(tilt * DEG));
    this.hash = hash === null ? this.att.rootHash : hash;
    this.tint = tint;
    this.fx = fx;        // which look-effects this object opts into
    // Conservative and morph-invariant, about `t` rather than about the
    // centroid: see field.js.  Used for separation and eviction only --
    // culling and draw order use the tight (C, R) sphere, which is smaller
    // but moves and resizes as the shape morphs.
    this.reach = reach === null ? (this.att.R + Math.hypot(...this.att.C)) * scale : reach;
    // The field record this was made from, or null for the fixed ring. It is
    // what animate() reads; it is frozen and shared, and nothing here writes
    // to it.
    this.cell = cell;
    this.yaw0 = yaw; this.tilt0 = tilt;
    // morph state
    this.from = name; this.to = name; this.phase = 0; this.lastBuilt = -1;
  }

  /**
   * Move, resize and turn this object to where it is at `time` seconds.
   *
   * Allocation-free, and called for every resident object every frame: the
   * stage-4 lesson was a 33 ms frame caused by `frames()` allocating a matrix
   * and four vectors per object per frame, so this writes into `this.t` and
   * `this.R` in place and never builds an intermediate.
   *
   * Nothing here can break the separation guarantee, and that is by
   * construction rather than by luck. The proof bounds the ENVELOPE of an
   * object -- the union of every position and size it can ever take -- and
   * `anim` was drawn so that |base| + amplitude <= 1 in units of the cell's
   * wander half-width, while the size breathes DOWNWARD from the cap. So the
   * envelope is exactly the sphere the static field already fitted inside the
   * cell. Rotation is free: reach is measured about `t` and is rotation
   * invariant.
   */
  animate(time) {
    const c = this.cell;
    if (!c) return;
    const a = c.anim, hw = c.hw, tc = c.tc;
    this.t[0] = tc[0] + hw * (a[0] + a[3] * Math.sin(a[6] * time + a[9]));
    this.t[1] = tc[1] + hw * (a[1] + a[4] * Math.sin(a[7] * time + a[10]));
    this.t[2] = tc[2] + hw * (a[2] + a[5] * Math.sin(a[8] * time + a[11]));
    // u -> uLo -> u, never above u
    const g = c.uLo + (c.u - c.uLo) * 0.5 * (1 + Math.sin(c.pulseW * time + c.pulseP));
    this.s = c.cap * g;
    this.reach = this.s * REACH_MAX_W;
    const yaw = (this.yaw0 * DEG) + c.spinW * time, tilt = this.tilt0 * DEG;
    const cy = Math.cos(yaw), sy = Math.sin(yaw);
    const ct = Math.cos(tilt), st = Math.sin(tilt);
    // Rz(yaw) * Ry(tilt), written out rather than two allocations and a
    // multiply
    const R = this.R;
    R[0] = cy * ct; R[1] = -sy; R[2] = cy * st;
    R[3] = sy * ct; R[4] = cy;  R[5] = sy * st;
    R[6] = -st;     R[7] = 0;   R[8] = ct;
  }

  // A pooled cell's shape is whatever its slot currently is, and that changes
  // under it as the pool morphs, so the name cannot be a stored string.
  get name() { return this.pool ? this.pool.shape(this.slot).name : this._name; }
  set name(v) { this._name = v; }

  /** Place a point of this attractor's local space into world space. */
  toWorld(p) {
    const q = mv3(this.R, p);
    return [q[0] * this.s + this.t[0], q[1] * this.s + this.t[1], q[2] * this.s + this.t[2]];
  }

  /** Its centre and radius in world units -- what the void is laid out with. */
  worldCentre() { return this.toWorld(this.att.C); }
  worldRadius() { return this.att.R * this.s; }
}

export class World {
  /**
   * @param layout  [{name, pos, scale, yaw, tilt, palette}]
   * @param opts    {morphSeconds, holdSeconds, rebuildHz}
   */
  constructor(layout, opts = {}) {
    this.items = layout.map((L) => new Placed(L.name, L));
    // The fixed ring's own hold-then-morph cycle, kept in step with the
    // pool's slower spline (field.js): longer morph, longer hold.
    this.morphSeconds = opts.morphSeconds ?? 24;
    this.holdSeconds = opts.holdSeconds ?? 14;
    // Rebuilding an attractor's moments and child table costs about a
    // millisecond, so it does not happen every frame.  At a twenty-four
    // second morph, twelve rebuilds a second advances the shape by one part
    // in 288 each time, which is far below what the cross-fade already
    // smooths over.
    this.rebuildHz = opts.rebuildHz ?? 12;
    this.morphing = opts.morphing ?? true;
    this.reparamMs = 0;
    this._clock = 0;
  }

  get length() { return this.items.length; }

  // The id-addressed interface.  An endless field has no array to index, so
  // everything that used to say items[j] goes through these five instead, and
  // an id is whatever the world says it is -- an array index here, a cell key
  // in FieldWorld.  Nothing downstream may assume an id is a number.
  get(id) { return this.items[id]; }
  first() { return 0; }
  each(fn) { for (let i = 0; i < this.items.length; i++) fn(i, this.items[i]); }
  rehome(/* fromId, toId */) {}
  pickTarget(rng, fromId) {
    const n = this.items.length;
    return (fromId + 1 + ((rng() * (n - 1)) | 0)) % n;
  }

  /**
   * Advance the morphs.  `dt` in seconds.  Only touches an attractor when its
   * shape has moved enough to be worth the rebuild.
   */
  update(dt) {
    this._clock += dt;
    if (!this.morphing || SET_NAMES.length < 2) return;
    const period = this.morphSeconds + this.holdSeconds;
    let ms = 0;
    this.items.forEach((it, k) => {
      // each attractor runs the same cycle, offset so they do not all change
      // together -- a void where everything morphs in step looks mechanical
      const u = (this._clock + (k * period) / this.items.length) % period;
      const phase = u < this.holdSeconds ? 0
                  : (u - this.holdSeconds) / this.morphSeconds;
      const step = Math.floor((this._clock + (k * period) / this.items.length) / period);
      const from = SET_NAMES[(k + step) % SET_NAMES.length];
      const to = SET_NAMES[(k + step + 1) % SET_NAMES.length];
      const quant = Math.round(phase * this.morphSeconds * this.rebuildHz);
      if (from === it.from && to === it.to && quant === it.lastBuilt) return;
      it.from = from; it.to = to; it.phase = phase; it.lastBuilt = quant;
      if (phase === 0 && from === it.name) return;       // already built
      const t0 = (typeof performance !== 'undefined' ? performance.now() : 0);
      const spec = phase === 0 ? SETS[from] : morph(SETS[from], SETS[to], smooth(phase));
      const b = build(spec);
      it.att.reparam({ maps: b.maps, trunkA: b.trunkA, sig0: b.sig0, mu0: b.mu0,
                       omega: spec.omega ?? 0.28,
                       // Follow the shape.  Without this an item that has
                       // morphed into a `shell` keeps the palette it was
                       // built with, so the void's colours drift out of step
                       // with its geometry over a cycle.  Free: reparam calls
                       // setPalette either way (adaptive.js:207), and morph()
                       // already interpolates the palette name.
                       palette: PALETTES[spec.palette] || it.att.palette });
      it.name = phase < 0.5 ? from : to;
      if (typeof performance !== 'undefined') ms += performance.now() - t0;
    });
    this.reparamMs = ms;
  }

  /**
   * The affine that places item j's root in the camera's working coordinates,
   * for every j, nearest first.  `cam` must be a VoidCamera homed in this
   * world.
   */
  frames(cam, out = [], view = null) {
    out.length = 0;
    const pool = this._slots || (this._slots = []);
    const rf = cam.rootFrame(this._rf2 || (this._rf2 = {}));
    const home = this.items[cam.home];
    for (let j = 0; j < this.items.length; j++) {
      const it = this.items[j];
      out.push(placeFrame(frameSlot(pool, j), it, home, rf, cam, j, j === cam.home));
    }
    out.sort((p, q) => p.d - q.d);                        // nearest first
    return out;
  }

  /** One frame of the whole void into one stream, in one draw order. */
  expand(stream, cam, view) {
    stream.begin(cam, view);
    const fr = this.frames(cam, this._fr || (this._fr = []), view);
    for (const f of fr) { stream.curFx = f.fx; stream.add(f.att, f, f.hash, f.tint); }
    this.visible = fr.length;
    return stream.end();
  }
}

const _Ag = new Float64Array(9), _bg = new Float64Array(3);

/**
 * Compose one object's root frame in the camera's working coordinates, into a
 * record that is reused across frames.
 *
 * Shared by both worlds on purpose.  This is the chain that has to be right
 * for re-rooting to survive -- tests/void.mjs checks it to 2e-16 -- and two
 * copies of it would be two things to keep right.
 *
 * Allocation-free, and that is not premature.  Written the obvious way it
 * allocates a Float64Array(9), four temporary vectors and an object literal
 * per visible object per frame.  At seven objects that is noise; at the forty
 * a field puts in view it is some two thousand short-lived allocations a
 * second, and the garbage collector answers with pauses -- a settled frame
 * measured at 33 ms in tests/tour.mjs, against a rebuild cost whose p99 over
 * every morph pair is 2.06 ms.  The spike was never the arithmetic.
 */
function placeFrame(out, it, home, rf, cam, id, isHome) {
  const A = out.A, b = out.b;
  let kap;
  if (isHome) {
    for (let k = 0; k < 9; k++) A[k] = rf.A[k];
    b[0] = rf.b[0]; b[1] = rf.b[1]; b[2] = rf.b[2];
    kap = rf.kap;
  } else {
    // its local -> world -> home's local
    const g = it.s / home.s;
    const hR = home.R, iR = it.R;
    // Ag = home.R^T * it.R, written out rather than composed from transpose()
    // and mm3(), each of which would allocate
    for (let i = 0; i < 3; i++)
      for (let j = 0; j < 3; j++)
        _Ag[i * 3 + j] = hR[i] * iR[j] + hR[3 + i] * iR[3 + j] + hR[6 + i] * iR[6 + j];
    const dx = it.t[0] - home.t[0], dy = it.t[1] - home.t[1], dz = it.t[2] - home.t[2];
    for (let i = 0; i < 3; i++)
      _bg[i] = (hR[i] * dx + hR[3 + i] * dy + hR[6 + i] * dz) / home.s;
    // then home's local -> camera
    for (let p = 0; p < 3; p++) {
      const r0 = rf.A[p * 3], r1 = rf.A[p * 3 + 1], r2 = rf.A[p * 3 + 2];
      for (let q = 0; q < 3; q++)
        A[p * 3 + q] = g * (r0 * _Ag[q] + r1 * _Ag[3 + q] + r2 * _Ag[6 + q]);
      b[p] = r0 * _bg[0] + r1 * _bg[1] + r2 * _bg[2] + rf.b[p];
    }
    kap = rf.kap * g;
  }
  const C = it.att.C;
  const cx = A[0] * C[0] + A[1] * C[1] + A[2] * C[2] + b[0] - cam.pos[0];
  const cy = A[3] * C[0] + A[4] * C[1] + A[5] * C[2] + b[1] - cam.pos[1];
  const cz = A[6] * C[0] + A[7] * C[1] + A[8] * C[2] + b[2] - cam.pos[2];
  const Rw = kap * it.att.R;
  out.att = it.att; out.kap = kap; out.item = it; out.id = id;
  out.hash = isHome ? cam.hashes[0] : it.hash; out.tint = it.tint;
  out.fx = it.fx;
  out.cx = cx; out.cy = cy; out.cz = cz; out.Rw = Rw;
  out.d = Math.hypot(cx, cy, cz) - Rw;
  return out;
}

/** The reusable records themselves: grown, never freed. */
function frameSlot(pool, k) {
  let r = pool[k];
  if (!r) { r = { A: new Float64Array(9), b: new Float64Array(3) }; pool[k] = r; }
  return r;
}

/** Is this object's bounding sphere anywhere in the frustum? */
function inView(f, cam, view, secX, secY) {
  const z = f.cx * cam.f[0] + f.cy * cam.f[1] + f.cz * cam.f[2];
  if (z + f.Rw < view.znear) return false;
  const x = f.cx * cam.r[0] + f.cy * cam.r[1] + f.cz * cam.r[2];
  const y = f.cx * cam.u[0] + f.cy * cam.u[1] + f.cz * cam.u[2];
  const mx = f.Rw * secX, my = f.Rw * secY;
  if (x - z * view.tanX > mx || -x - z * view.tanX > mx) return false;
  if (y - z * view.tanY > my || -y - z * view.tanY > my) return false;
  return true;
}

/**
 * An endless field of attractors, generated on the fly from the cell grid in
 * field.js.
 *
 * Satisfies the same id-addressed interface as World, so the camera and the
 * autopilot do not know which kind of world they are flying through.  An id
 * here is a cell key, and the set of valid ids changes as you move.
 */
export class FieldWorld {
  constructor(opts = {}) {
    // Ask for the rate the budget can actually serve.
    //
    // Eight slots at the inherited 12 Hz is 0.93 rebuilds a frame against a
    // 1.5 ms budget that affords exactly one, so the pool achieved 7.0 Hz and
    // kept 1.73 slot-frames permanently queued -- work scheduled only to be
    // deferred, and a `deferred` counter that never reads zero and so never
    // means anything.  Asking for 7 makes demand match capacity.
    //
    // Nothing is lost visually: 7 Hz over a fourteen-second morph is ninety
    // eight steps, and the cross-fade already smooths over far coarser than
    // that.  Raising the budget instead would buy 12 Hz for 2.5 ms of every
    // frame, which is the wrong trade -- expansion needs that time more.
    this.pool = opts.pool || new ShapePool({ rebuildHz: 7, ...opts });
    this.layout = opts.layout || new FieldLayout(this.pool, opts);
    this.items = new Map();                 // cell key -> Placed
    this.reparamMs = 0;
    this.visible = 0;
    // `morphing` and `interpolate` live on the pool, which is where the
    // machinery is; these are accessors onto it (below) rather than a second
    // copy of the flag. A world that kept its own and gated the pool's
    // update() would be the obvious way to write it and the wrong one: the
    // pool would then stop being called at all, and its rate meters -- the
    // numbers the panel shows -- would hold whatever they said when the
    // morph was switched off instead of falling to zero.
    this.morphing = opts.morphing ?? true;
    // Objects drift, breathe and turn. Separate from `morphing`, which is the
    // shapes changing: a field can do either without the other, and the tests
    // that measure draw order want both off so the thing they measure holds
    // still.
    this.moving = opts.moving ?? true;
    this._clock = 0;
    this._fr = [];
    this._prio = new Float64Array(this.pool.slots);
    this._centre = [0, 0, 0];
    this.layout.update(this._centre);
    this._sync();
  }

  get length() { return this.items.size; }
  get(id) { return this.items.get(id); }
  each(fn) { for (const [k, v] of this.items) fn(k, v); }
  rehome(from, to) {
    if (from !== null && from !== undefined) this.layout.unpin(from);
    if (to !== null && to !== undefined) this.layout.pin(to);
  }

  /** The resident cell nearest the field's origin -- where a flight starts. */
  first() {
    let best = null, bestD = Infinity;
    for (const [k, it] of this.items) {
      const d = Math.hypot(it.t[0], it.t[1], it.t[2]);
      if (d < bestD) { bestD = d; best = k; }
    }
    if (best === null) throw new Error('the field generated no cells at the origin; '
                                     + 'density or genRadius is too low');
    return best;
  }

  /** Somewhere else to fly to: a resident cell at a comfortable crossing. */
  pickTarget(rng, fromId) {
    const here = this._centre;
    let best = fromId, bestScore = -Infinity;
    for (const [k, it] of this.items) {
      if (k === fromId) continue;
      const d = Math.hypot(it.t[0] - here[0], it.t[1] - here[1], it.t[2] - here[2]);
      // Prefer a few cells away, not the nearest: crossing to a neighbour you
      // are already inside the fog of is not a crossing, and the far edge of
      // the generated region may be evicted before you arrive.
      // 0.32, was 0.45.  The crossing is the one stretch of the flight with
      // nothing on screen, and a slower autopilot made it twice as long.  A
      // nearer target shortens it without making the arrival abrupt -- it is
      // still well outside the fog, which is what the 0.45 was protecting.
      const want = this.layout.genRadius * 0.32;
      const score = -Math.abs(d - want) + rng() * want * 0.35;
      if (score > bestScore) { bestScore = score; best = k; }
    }
    return best;
  }

  /**
   * Residency, then morphs.  `cam` may be null before a camera exists.
   *
   * Priority comes from LAST frame's visibility, so a shape that nobody can
   * see closely is rebuilt last.  One frame of lag is invisible against a
   * fourteen-second morph and it saves walking the field twice.
   */
  // The two morph switches, forwarded to the pool. Assignment reads as a
  // plain field at the call site and is a method call underneath, because
  // turning interpolation on or off mid-flight is not a flag flip: see
  // ShapePool.setInterpolate and Attractor.setInterp.
  get morphing() { return this.pool.morphing; }
  set morphing(on) { this.pool.setMorphing(on); }
  get interpolate() { return this.pool.interpolate; }
  set interpolate(on) { this.pool.setInterpolate(on); }
  // What the panel shows: how often the drawn shape actually changes, and
  // how often it is rebuilt. Per slot, per second, measured.
  get shapeHz() { return this.pool.shapeHz; }
  get rebuildHzNow() { return this.pool.rebuildHzNow; }

  update(dt, cam) {
    this._clock += dt;
    if (cam) this._centre = cam.worldPos();
    this.layout.update(this._centre);
    this._sync();
    this.pool.update(dt, this._prio);
    // After _sync, so an object that materialised this frame is placed at the
    // current time rather than at time zero -- otherwise a cell entering the
    // generation radius would appear at its t=0 position and then jump.
    if (this.moving) {
      const t = this._clock;
      for (const it of this.items.values()) it.animate(t);
    }
    this.reparamMs = this.pool.reparamMs;
  }

  _sync() {
    const L = this.layout;
    for (const cell of L.resident()) {
      if (this.items.has(cell.key)) continue;
      const sh = this.pool.shape(cell.slot);
      this.items.set(cell.key, new Placed(sh.name, {
        pos: cell.t, scale: cell.scale, yaw: cell.yaw, tilt: cell.tilt,
        att: sh.att, pool: this.pool, slot: cell.slot,
        hash: cell.hash, tint: cell.tint, reach: cell.reach, cell, fx: cell.fx,
      }));
    }
    if (this.items.size !== L.size) {
      for (const k of this.items.keys()) if (!L.get(k)) this.items.delete(k);
    }
  }

  frames(cam, out = [], view = null) {
    out.length = 0;
    const pool = this._slots || (this._slots = []);
    const rf = cam.rootFrame(this._rf2 || (this._rf2 = {}));
    const home = this.items.get(cam.home);
    const secX = view ? Math.sqrt(1 + view.tanX * view.tanX) : 0;
    const secY = view ? Math.sqrt(1 + view.tanY * view.tanY) : 0;
    this._prio.fill(0);
    let slot = 0;
    for (const [k, it] of this.items) {
      const f = placeFrame(frameSlot(pool, slot), it, home, rf, cam, k, k === cam.home);
      // Cull whole objects before the sort, not after: residency is hundreds
      // of cells and the view holds tens, and an object that is not in the
      // frustum should cost one dot product rather than a sort slot and a
      // walk.
      if (view && !inView(f, cam, view, secX, secY)) continue;
      slot++;                       // only advance on a keeper, so the reused
      out.push(f);                  // records stay packed and bounded by view
      const ap = f.Rw / Math.max(f.d + f.Rw, 1e-9);
      if (ap > this._prio[it.slot]) this._prio[it.slot] = ap;
    }
    out.sort((p, q) => p.d - q.d);                        // nearest first
    return out;
  }

  expand(stream, cam, view) {
    stream.begin(cam, view);
    const fr = this.frames(cam, this._fr, view);
    for (const f of fr) { stream.curFx = f.fx; stream.add(f.att, f, f.hash, f.tint); }
    this.visible = fr.length;
    return stream.end();
  }
}

const smooth = (t) => t * t * (3 - 2 * t);
function transpose(A) {
  return new Float64Array([A[0], A[3], A[6], A[1], A[4], A[7], A[2], A[5], A[8]]);
}

/** A FlyCamera that knows which attractor it is homed in, and can change. */
export class VoidCamera extends FlyCamera {
  constructor(world, home = world.first()) {
    super(world.get(home).att);
    this.world = world;
    this.home = home;
    this.hashes = [world.get(home).hash];
    world.rehome(null, home);
  }

  /**
   * Hand the camera over to whichever attractor it is now nearest, measured in
   * units of that attractor's own radius so a small one does not win just by
   * being compact.  Only legal at the top: once the camera has descended into
   * a node its position means something only inside that node.
   */
  retarget() {
    if (this.path.length) return false;
    const W = this.world, here = W.get(this.home);
    // Inlined rather than using toWorld()/worldCentre(), which each allocate.
    // This runs every frame over every resident cell -- some three hundred and
    // forty of them in a field -- and two small arrays apiece is seven hundred
    // allocations a frame, which is the same garbage that was costing whole
    // frames in frames() before it was made allocation-free.
    const p = this.pos, hR = here.R, hs = here.s, ht = here.t;
    const wx = (hR[0] * p[0] + hR[1] * p[1] + hR[2] * p[2]) * hs + ht[0];
    const wy = (hR[3] * p[0] + hR[4] * p[1] + hR[5] * p[2]) * hs + ht[1];
    const wz = (hR[6] * p[0] + hR[7] * p[1] + hR[8] * p[2]) * hs + ht[2];
    let best = this.home, bestD = Infinity;
    W.each((id, it) => {
      const C = it.att.C, R = it.R, s = it.s, t = it.t;
      const cx = (R[0] * C[0] + R[1] * C[1] + R[2] * C[2]) * s + t[0];
      const cy = (R[3] * C[0] + R[4] * C[1] + R[5] * C[2]) * s + t[1];
      const cz = (R[6] * C[0] + R[7] * C[1] + R[8] * C[2]) * s + t[2];
      const d = Math.hypot(wx - cx, wy - cy, wz - cz) / (it.att.R * s);
      if (d < bestD) { bestD = d; best = id; }
    });
    if (best === this.home) return false;
    const there = W.get(best);
    // world -> the new home's local
    const rel = [wx - there.t[0], wy - there.t[1], wz - there.t[2]];
    this.pos = mtv3(there.R, rel).map((v) => v / there.s);
    const Rc = mm3(transpose(there.R), here.R);
    this.r = mv3(Rc, this.r); this.u = mv3(Rc, this.u); this.f = mv3(Rc, this.f);
    this.ortho();
    W.rehome(this.home, best);
    this.home = best;
    this.att = there.att;
    // rootFrame() always walks back to the TRUE root, so the walk that draws
    // this object starts from hashes[0].  If that is not the cell's own root
    // hash, the camera dives through one realisation while the screen shows
    // another: the structure ahead is not the structure you fly into.
    this.hashes = [there.hash];
    return true;
  }

  /** Where the camera is in world units -- for the HUD and for autopilot. */
  worldPos() { return this.world.get(this.home).toWorld(this.pos); }

  /** Where attractor j's centre is, in the camera's own working coordinates. */
  bearing(j) {
    const W = this.world, here = W.get(this.home), tgt = W.get(j).worldCentre();
    const loc = mtv3(here.R, [tgt[0] - here.t[0], tgt[1] - here.t[1], tgt[2] - here.t[2]])
      .map((v) => v / here.s);
    const d = [loc[0] - this.pos[0], loc[1] - this.pos[1], loc[2] - this.pos[2]];
    const dist = Math.hypot(d[0], d[1], d[2]) || 1e-12;
    return { dir: [d[0] / dist, d[1] / dist, d[2] / dist], dist,
             near: W.get(j).worldRadius() / here.s * 2.2 };
  }

  /** Point straight at it, no turn. */
  faceToward(j) {
    this.f = this.bearing(j).dir;
    this.ortho();
  }

  /**
   * Turn toward `dir` with a critically damped follower, and return how
   * closely the camera is now pointing at it.
   *
   * The camera used to be steered by assignment: faceToward() set the heading
   * outright at the start of every crossing, and cruise() lerped toward the
   * bearing at a fixed fraction per frame. Both change direction
   * discontinuously. The lerp eases OUT of a turn but never into one -- the
   * turn rate is proportional to the error, so the instant the target changes
   * the error jumps and so does the rate. Assignment is worse: the whole turn
   * happens between two frames.
   *
   * Carrying an angular velocity and accelerating it toward the heading fixes
   * both ends. The camera leans into a turn and settles out of it, and there
   * is no instant at which its direction is discontinuous -- the same
   * property the morph spline was given, one level up. Critically damped
   * (`damp = 2*sqrt(stiff)`) is the fastest approach that cannot overshoot,
   * and overshoot here reads as the camera wobbling about its target.
   */
  steer(dir, dt, stiff = 9) {
    const v = this._fv || (this._fv = [0, 0, 0]);
    const damp = 2 * Math.sqrt(stiff);
    const f = this.f;
    for (let i = 0; i < 3; i++) {
      const a = (dir[i] - f[i]) * stiff - v[i] * damp;
      v[i] += a * dt;
      f[i] += v[i] * dt;
    }
    this.ortho();
    return Math.max(0, f[0] * dir[0] + f[1] * dir[1] + f[2] * dir[2]);
  }

  /**
   * Aim at an attractor and close on it; true once it is near.
   *
   * The step is scaled by how much of the heading actually points at the
   * target.  Without that factor a camera that has just climbed out of
   * something -- and so is facing away from everything -- flies BACKWARDS at a
   * quarter of the remaining distance per frame, which is a geometric series
   * the wrong way round: twenty seconds of that and the void is four hundred
   * thousand units behind you and the screen is black.  Measured, not
   * theorised: tests/tour.mjs went from 2875 blank frames to none.
   */
  cruise(j, speed, ticks = 1) {
    const { dir, dist, near } = this.bearing(j);
    // Gentler than the default: a crossing is long and there is no reason to
    // arrive pointing exactly at the target early in it.  Gentler again now
    // (2.4, was 4.0): the crossing is the one stretch with room to turn, and
    // a slow turn across open void is the shot.
    //
    // `ticks` rather than a hardcoded 1/60: the follower used to integrate a
    // sixtieth of a second however long the frame actually was, so every
    // headless measurement at 30fps turned at half the rate the browser does.
    const align = this.steer(dir, ticks / 60, 2.4);
    const v = this._fv;
    if (v) this.bank(v[0] * this.r[0] + v[1] * this.r[1] + v[2] * this.r[2], ticks / 60);
    // 1 - exp(-speed), so that two half-steps are one whole step however
    // long a frame is -- see the note in FlyCamera.dive.
    this.move(0, 0, Math.min(-Math.expm1(-speed), 0.25) * dist * align);
    return dist < near;
  }
}

/** Seven attractors on a slow spiral through the void, biggest last. */
export function ringLayout(names = SET_NAMES, { radius = 26, rise = 9 } = {}) {
  const n = names.length;
  return names.map((name, k) => {
    const a = (2 * Math.PI * k) / n;
    const r = radius * (0.72 + 0.4 * ((k * 7) % n) / n);
    return {
      name,
      pos: [r * Math.cos(a), r * Math.sin(a), rise * Math.sin(2.4 * a + 1.1)],
      scale: 1.0 + 0.45 * Math.sin(1.7 * k + 0.4),
      yaw: (k * 137.5) % 360,
      tilt: 18 * Math.sin(2.1 * k),
      palette: SETS[name].palette,
      seed: 3 + k,
    };
  });
}

/**
 * The shape of the demo, as a state machine: cross the void to an attractor,
 * fall into it while it refines forever, then climb back out and pick another.
 *
 * Climbing is diving with the sign flipped.  Re-rooting ascends by itself as
 * the camera leaves a node, so there is no separate way back up that could be
 * got wrong -- which matters, because a way back up that drifted would show
 * as the attractor sliding off screen after the first round trip.
 */
export class Autopilot {
  constructor(world, cam, opts = {}) {
    this.world = world; this.cam = cam;
    this.phase = 'cruise'; this.target = world.first(); this.t = 0;
    // Slower, and longer in each phase.
    //
    // The flight is meant to be something to watch, not somewhere to get to.
    // Every number here used to be set by how long it took to reach the depth
    // limit, which is the wrong thing to optimise: the interesting part is the
    // approach, and at the old speed it went past. Longer phases also mean
    // fewer phase CHANGES, and a phase change is the most abrupt thing the
    // camera ever does.
    this.cruiseSeconds = opts.cruiseSeconds ?? 44;
    this.diveSeconds = opts.diveSeconds ?? 72;
    this.climbSeconds = opts.climbSeconds ?? 8;
    // The forward rate, as a fraction of the distance remaining. Geometric,
    // so this is a time constant and not a velocity: halving it halves the
    // rate of descent at every scale equally, which is the only way to slow a
    // flight whose scale changes by factors of ten.
    //
    // 0.0062, was 0.0105.  The dive is a geometric approach, so this is the
    // only honest way to slow it: every scale is slowed by the same factor
    // and the flight keeps its shape.  Slow enough that structure resolving
    // out of a blob takes several seconds, which is the part worth watching.
    this.speed = opts.speed ?? 0.0062;
    // Deeper, now that the dive follows the boundary instead of burying
    // itself in the mass: there is something to see down there.  In practice
    // `diveSeconds` is what ends a dive at this speed; the cap is the
    // running-time backstop, not the aesthetic one.
    // 56.  Deeper than the old 46, and not deeper still: washed-out frames
    // sit at a mean depth of 34 against 19 for legible ones, so depth does
    // cost legibility even with the dive following the boundary.  At this
    // speed `diveSeconds` is usually what ends a dive anyway.
    this.maxDepth = opts.maxDepth ?? 56;
    this.rng = opts.rng || Math.random;
    // Signed, smoothed travel. +1 is going in, -1 is backing out, and the
    // transition passes through 0 rather than flipping between frames.
    this._drive = 1;
    this.turnSeconds = opts.turnSeconds ?? 2.4;
    // How far the climb slides sideways per unit of backward travel.
    //
    // OFF, measured. Arcing out instead of reversing back up the line it came
    // in on is the better-looking manoeuvre in isolation, and over 200 s of
    // flight it costs more than it buys: washed-out frames 37.6% against
    // 30.5% with no arc, nearly-empty 11.6% against 8.3%, and the following
    // DIVE washed 41.0% against 31.6% -- the arc leaves the camera off to one
    // side of everything, and the dive that starts from there has further to
    // go before it is inside anything. The sideways direction is still
    // computed below, so this is one number to put back.
    this.arc = opts.arc ?? 0;
    this._arcR = 0; this._arcU = 0; this._arcSet = false;
  }

  /** `dt` in 60fps ticks, to match the rest of the viewer. */
  step(dt, speed = this.speed) {
    const cam = this.cam, W = this.world;
    this.t += dt / 60;
    // Ease the direction of travel rather than reversing it in one frame.
    // Diving and climbing are opposite motions along the same axis, and
    // switching between them by changing the sign of a speed is the most
    // visible discontinuity in the whole flight: the camera is falling into
    // something at full speed and is suddenly rising out of it. Passing
    // through zero gives a moment of stillness at the bottom of the dive,
    // which is also the moment worth looking at.
    const want = this.phase === 'climb' ? -1 : 1;
    const k = Math.min(1, (dt / 60) / this.turnSeconds);
    this._drive += (want - this._drive) * k;
    if (this.phase === 'cruise') {
      // The target can be evicted mid-crossing. It was resident when it was
      // picked, but a crossing takes many seconds, the field re-enumerates as
      // the camera moves, and a cell behind the camera leaves the eviction
      // radius -- at which point every bearing taken on it reads a property
      // of undefined. The flat field hid this because its cells were large
      // and few; a hierarchy evicts clusters of small ones far more often.
      if (!W.get(this.target)) {
        this.target = W.pickTarget(this.rng, this.target);
        if (!W.get(this.target)) { this.phase = 'dive'; this.t = 0; return this.phase; }
      }
      // 2.4, was 1.6.  Slower is better where there is something to look at,
      // and the crossing is the one stretch where there is not: 80-99% of
      // cruise frames have next to nothing on screen (tests/legible.mjs), and
      // halving the autopilot's speed had doubled how long that lasted.  The
      // approach and the dive keep the slower speed; the empty part does not.
      if (cam.cruise(this.target, speed * 2.4 * dt, dt) || this.t > this.cruiseSeconds)
        { this.phase = 'dive'; this.t = 0; }
      return this.phase;
    } else if (this.phase === 'dive') {
      // A softer follower: 0.14, was 0.25.
      //
      // The old one had to be tight because the aim point was a child's
      // CENTRE, and in a flat set the camera would drift off the sheet while
      // still inside its bounding sphere and end up looking along an empty
      // plane. The aim point is now on the boundary and the score prefers
      // the children that have one, so the heading has something to hold on
      // to and the turn can be slow.
      cam.dive(speed * dt * Math.max(0, this._drive), 0.14, dt);
      // Stop descending at maxDepth.  Not a precision limit -- re-rooted
      // float32 is still exact at depth 125 -- but a running-time one: the
      // climb back out costs a few frames per level, so a dive that goes
      // twice as deep costs twice as long coming back, with nothing new to
      // see because every level looks like the last one.
      if (this.t > this.diveSeconds || cam.depth >= this.maxDepth) {
        this.phase = 'climb'; this.t = 0;
        if (!this._arcSet) {
          const a = this.rng() * 2 * Math.PI;
          this._arcR = Math.cos(a) * this.arc; this._arcU = Math.sin(a) * this.arc;
          this._arcSet = true;
        }
      }
    } else {
      // Climb out BACKWARDS, without turning.
      //
      // Turning to face outward and flying off is the obvious way and it is
      // wrong twice over.  Visually, everything the camera was looking at is
      // now behind it, and everything in front is a hundred local units away
      // and solid fog, so the screen goes black for the whole ascent.  And
      // mechanically it is far too slow: re-rooting ascends a level only when
      // the camera leaves the node, which takes about sixteen frames at this
      // speed while the dive descends a level in two, so the depth runs away
      // -- a hundred and twenty-five levels deep and still climbing.
      //
      // Reversing instead keeps the structure centred the entire way, and the
      // shot becomes the hero zoom played backwards: detail merging into
      // detail, the whole attractor reassembling out of what you were just
      // inside.  The speed scales with the node's own radius, so it ascends
      // at a constant number of frames per level no matter how deep it is.
      const att = W.get(cam.home).att;
      const C = att.C;
      const L = Math.hypot(cam.pos[0] - C[0], cam.pos[1] - C[1], cam.pos[2] - C[2]);
      const back = speed * 11 * dt * Math.max(att.R, L) * Math.max(0, -this._drive);
      // Arc out, rather than reverse back down the line it came in on.
      //
      // Straight back out is the hero zoom played backwards, and measured it
      // is also the least legible stretch of the whole flight: 81% of climb
      // frames were a featureless wash against 21% of dive frames in the same
      // run (tests/legible.mjs). The reason is geometric rather than a matter
      // of taste -- backing out along the entry path keeps the camera in the
      // middle of the structure it just flew into, and the middle of a
      // structure has no silhouette in it at any depth.
      //
      // Sliding sideways while backing out swings that structure across the
      // frame and brings the void in behind it. One direction for the whole
      // climb, drawn once, so it reads as a single long curve rather than as
      // wandering; and it is lateral only, so the ascent still costs the same
      // number of frames per level that it always did.
      cam.move(this._arcR * back, this._arcU * back, -back);
      if (cam.depth === 0 && this.t > this.climbSeconds) {
        this._arcSet = false;
        this.phase = 'cruise'; this.t = 0;
        this.target = W.pickTarget(this.rng, this.target);
        // Deliberately NOT faceToward() any more. Snapping the heading to the
        // new target is a cut, and the whole crossing is the one stretch of
        // the flight with room to turn gracefully. cruise() steers there with
        // the damped follower instead, over about a second.
      }
    }
    return this.phase;
  }
}
