// The scene tests/shard.mjs shards, built identically on every thread.
//
// Both the main thread and each worker call setup() with the same arguments
// and must end up with the same world, the same camera and the same
// attractors -- otherwise the comparison is between two different pictures and
// says nothing.  That is why this is a module rather than a message: sending
// the frame records would mean sending Attractor objects, which hold typed
// arrays and methods and do not survive postMessage.
//
// The ring world is the right vehicle and the field world is not.  World's
// morph is driven by a clock accumulated from dt and `performance.now()`
// appears in it only to MEASURE reparamMs (world.js:211), so the same dt
// sequence gives the same state on any thread.  FieldWorld's ShapePool
// budgets its rebuilds against the wall clock (field.js:1051-1106) and warms
// up under a time budget, so two replicas of it diverge by design -- which is
// the honest reason the browser's sharded path needs the pool's tables in
// shared memory rather than a replica per thread.
import { World, VoidCamera, ringLayout } from '../js/world.js';
import { SET_NAMES } from '../js/maps.js';

export const W = 960, H = 540, FOV = 55;
export const view = (() => {
  const tanY = Math.tan((FOV * Math.PI) / 360);
  return { focal: (H / 2) / tanY, tanX: tanY * (W / H), tanY, znear: 1e-5, W, H };
})();

/** `steps` ticks of the morph at a fixed dt, then a fixed camera pose. */
export function setup(steps = 40, dt = 1 / 60) {
  const world = new World(ringLayout(SET_NAMES, {}), {
    morphing: true, morphSeconds: 14, holdSeconds: 10,
  });
  for (let i = 0; i < steps; i++) world.update(dt);
  const id = world.first();
  const cam = new VoidCamera(world, id);
  const h = world.get(id), c = h.worldCentre(), r = h.worldRadius();
  // A fixed pose outside the ring, looking at it: no autopilot, because
  // Autopilot takes its rng from Math.random unless handed one (world.js:715)
  // and this test is about the walk, not the flight.
  const back = [c[0] * 1.9 - 4, c[1] * 1.9 - 4, c[2] + r * 2.4];
  cam.pos = [(back[0] - h.t[0]) / h.s, (back[1] - h.t[1]) / h.s,
             (back[2] - h.t[2]) / h.s];
  const nz = (v) => { const l = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / l, v[1] / l, v[2] / l]; };
  cam.f = nz([c[0] - back[0], c[1] - back[1], c[2] - back[2]]);
  cam.u = [0, 0, 1];
  cam.ortho();
  cam.reroot(); cam.retarget();
  return { world, cam, view };
}

/** The frames list, nearest first -- the order the concatenation must keep. */
export function frameList(world, cam) {
  return world.frames(cam, [], view).map((f) => ({ ...f }));
}

/** The cost proxy: apparent size, which is what the LOD threshold is compared
 *  against, so it is the best cheap predictor of how much walking an object
 *  will take. FieldWorld.frames() already computes it for its own priorities. */
export const costOf = (f) => f.Rw / Math.max(f.d + f.Rw, 1e-9);

/** Walk a contiguous run of the frames list into `st`, nearest first.
 *
 *  `into` collects how many splats each object emitted, which is the cost
 *  proxy worth having: apparent size is a formula and gets it badly wrong
 *  (see tests/shard.mjs), while last frame's count per object is a
 *  measurement, and the camera moves by a fraction of a percent a frame. */
export function walkRun(st, frames, i0, i1, into = null) {
  for (let i = i0; i < i1; i++) {
    const f = frames[i];
    st.curFx = f.fx;
    const before = st.count;
    st.add(f.att, f, f.hash, f.tint);
    if (into) into[i] = st.count - before;
  }
}
