// The moment, as a link.
//
// "It went jumpy just then" cannot be chased from a frame number, because
// this flight cannot be re-run: the autopilot integrates real frame times
// and tests/pilot.mjs establishes that the dive is chaotic -- one part in a
// million of speed puts the heading 60 degrees away in three seconds. So the
// identifier has to carry the state rather than point at it.
//
// Carrying the state sounds expensive and is not, for one reason: every
// heavy array in the engine is a pure function of a few parameters. A slot's
// drawn shape is two premultiplied child tables read at a fraction, 288 KB
// each, and both of them are
//
//     premul(build(morphPath(SPECS, u)), jitter table at `bits`, seed + k)
//
// or, while a resize is migrating, that same build with the jitter rotations
// blended at weight `_migW` between entry and partner -- which is again a
// function of (u, bits, migLo, migW, grow). Ten numbers a slot. The jitter
// axis table is derived from the entry index alone, so the seed fixes it.
// The palette is a function of u. The field's cells are a hash of where the
// camera is. What is left to record is about three hundred numbers, which is
// a kilobyte of JSON and fits in a URL fragment.
//
// What the fragment therefore is: an exact description of one frame of the
// simulation, not a measurement of it. tests/mark.mjs rebuilds the slots
// from it through the same reparam() the pool calls, checks the state it
// reaches against the state recorded here -- every rebuild in the window,
// mask and migration weight included -- and only then draws anything.
//
// Three things it does not carry, stated here because each one bounds what a
// replay can claim:
//
//   1. Reaction time. The key is pressed about three tenths of a second
//      after something is seen, so the frame in `slots` is the aftermath.
//      `tail` is the hundred and fifty milliseconds before it, per slot, per
//      frame, with every rebuild that landed in it -- that is the window the
//      jump is actually in.
//   2. The renderer. Nothing in this repository has ever run a shader; a
//      replay draws with js/cpu_raster.js, the line-by-line twin the parity
//      tests hold to the Python reference. A faithful twin, not your frame.
//   3. Forward motion. The morph replays exactly under a fixed 60 Hz step,
//      because dt is all the pool consumes. The autopilot does not: it reads
//      real frame times, so running on past the snapshot diverges.
const r = (v, n) => (Number.isFinite(v) ? Number(v.toFixed(n)) : 0);
const p4 = (v) => (Number.isFinite(v) && v !== 0 ? Number(v.toPrecision(4)) : 0);

/**
 * The state of the simulation, as plain data.
 *
 * Read-only: it touches nothing, so it cannot destabilise the demo it is
 * describing. Everything in it is either a parameter a rebuild is a function
 * of, or a number that says what the frame looked like so a replay can be
 * checked against it.
 */
export function snapshot(viewer) {
  const w = viewer.world, pool = w && w.pool;
  if (!pool) return null;
  const L = viewer.jumpLog, cam = viewer.fly;
  const S = pool.slots;
  const slots = [];
  for (let k = 0; k < S; k++) {
    const sh = pool.shapes[k], att = sh.att;
    slots.push({
      // The pair the blend is reading, and where in it the frame sits.
      uA: r(sh.uA, 9), uB: r(sh.uB, 9), bt: r(att._bt, 6),
      // The mask, what it has been asked to become, and how far through the
      // migration between them this table is.
      bits: att.tableBits, want: att._wantBits,
      migW: r(att._migW, 6), migGrow: att._migGrow ? 1 : 0,
      migLo: att._migLo, migBits: att._migBits, migDrop: att._migDrop ? 1 : 0,
      // Bookkeeping: not needed to draw the frame, needed to carry on from it.
      rebuilds: sh.rebuilds, gapEst: p4(sh.gapEst), lastBuilt: sh.lastBuilt,
      builtAt: r(sh.builtAt, 6),
      // And what the frame was, so a replay can be told it is wrong.
      R: p4(att.R), btSafe: p4(att._btSafe), btSafeLo: p4(att._btSafeLo),
      from: sh.from, to: sh.to, phase: r(sh.phase, 6),
      // `L.last` only exists once the probe has seen a frame, and J is
      // pressable before that.
      step: L && L.last ? p4(L.last[k]) : 0,
      med: L && L.med ? p4(L.med[k]) : 0,
    });
  }
  const tail = { f: [], dt: [], n: [], bt: [], step: [], reb: [] };
  if (L && L.tN) {
    tail.f = L.tailFrames();
    tail.dt = L.tailDt().map((v) => r(v, 2));
    tail.n = L.tailSplats().map((v) => Math.round(v));
    const n = tail.f.length;
    for (let k = 0; k < S; k++) {
      const t = L._trail(k, n);
      tail.bt.push(t.map((x) => r(x.bt, 5)));
      tail.step.push(t.map((x) => p4(x.step)));
    }
    const f0 = tail.f[0];
    tail.reb = L.rebs.filter((x) => x.f >= f0)
      .map((x) => [x.f, x.k, r(x.u, 9), x.want, x.bits, r(x.w, 6), x.g, x.lo, x.drop]);
  }
  return {
    v: 1,
    build: (L && L.build) || 'dev',
    frame: L ? L.frame : 0,
    t: L ? r(L.clock, 3) : 0,
    // Everything about the pool that a rebuild is a function of. Recorded
    // rather than assumed from the defaults: a replay that silently used a
    // different setSeconds would reproduce a different morph and look right.
    pool: {
      seed: pool.seed, slots: S, setSeconds: pool.setSeconds,
      breathSeconds: pool.breathSeconds, breathMin: pool.breathMin,
      breathMax: pool.breathMax, rebuildHz: pool.rebuildHz, hz: p4(pool._hz),
      u: r(pool._u, 9), clock: r(pool._clock, 6), breath: r(pool.breath, 6),
      interpolate: pool.interpolate ? 1 : 0, morphing: pool.morphing ? 1 : 0,
      blendMax: pool.blendMax, blendMin: pool.blendMin,
      jitter: pool.jitter, migrateRebuilds: pool.migrateRebuilds,
      bitsByRank: pool.bitsByRank.slice(), rankEvery: pool.rankEvery,
      resizePerPass: pool.resizePerPass, tableBits: pool.tableBits,
      rebuildBudgetMs: pool.rebuildBudgetMs,
    },
    slots,
    // The camera, in its own working frame: `home` is the cell it belongs
    // to, `path` the descent inside that cell, and the hash chain alongside
    // it -- which decides which realisation the camera is flying through, so
    // it is not derivable from the path alone.
    cam: cam ? {
      home: String(cam.home), path: (cam.path || []).slice(),
      hashes: (cam.hashes || []).map((h) => h >>> 0),
      pos: (cam.pos || []).map((v) => p4(v)),
      r: (cam.r || []).map((v) => r(v, 9)),
      u: (cam.u || []).map((v) => r(v, 9)),
      f: (cam.f || []).map((v) => r(v, 9)),
      depth: cam.depth ?? (cam.path || []).length, scale: p4(cam.scale),
      roll: r(cam._roll ?? 0, 6), reroots: cam.reroots ?? 0,
    } : null,
    world: { clock: r(w._clock ?? 0, 6), resident: w.length ?? 0,
             visible: w.visible ?? 0,
             centre: (w._centre || []).map((v) => p4(v)) },
    // What the viewer was set to. None of it changes the morph; all of it
    // changes the picture, so a replay drawn at another detail level is
    // answering a different question.
    view: { detail: viewer.detail, dissolve: viewer.dissolve ? 1 : 0,
            dilate: viewer.dilate ? 1 : 0, fovy: viewer.opts.fovy,
            autopilot: viewer.autopilot ? 1 : 0,
            phase: viewer.stats.phase ?? '', splats: viewer.stats.splats ?? 0,
            fps: p4(viewer.stats.fps), look: viewer.look.mask,
            fogNear: p4(viewer.fogNear), fogFar: p4(viewer.fogFar),
            W: viewer.canvas ? viewer.canvas.width : 0,
            H: viewer.canvas ? viewer.canvas.height : 0 },
    tail,
    probe: L ? { worst: p4(L.worst), slot: L.worstSlot, frame: L.worstFrame,
                 jumps: L.jumps, ratio: L.ratio, ms: p4(L.probeMs) } : null,
  };
}

// base64url, in chunks: String.fromCharCode.apply on a long array overflows
// the argument list, which is the kind of failure that only shows up on the
// one machine with a big window.
function b64url(bytes) {
  let s = '';
  for (let i = 0; i < bytes.length; i += 0x8000)
    s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
  return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
}

/**
 * The fragment: `m1` gzipped, `m0` plain, both base64url of the JSON.
 *
 * Gzip is not an optimisation here, it is what keeps the link pasteable: the
 * tail is ninety frames of eight mostly-smooth numbers, which is exactly
 * what a compressor is good at -- about 11 KB of JSON down to under 3 KB.
 * CompressionStream is not everywhere, so the plain form is a real path and
 * is tagged rather than guessed at by the reader.
 */
export async function encode(obj) {
  const json = JSON.stringify(obj);
  const raw = new TextEncoder().encode(json);
  if (typeof CompressionStream === 'function') {
    try {
      const cs = new CompressionStream('gzip');
      const wr = cs.writable.getWriter();
      wr.write(raw); wr.close();
      const buf = await new Response(cs.readable).arrayBuffer();
      return 'm1=' + b64url(new Uint8Array(buf));
    } catch (_) { /* fall through to the plain form */ }
  }
  return 'm0=' + b64url(raw);
}

/**
 * The inverse, taking a whole link, a bare fragment, or the base64 alone.
 *
 * Kept here next to `encode` rather than in the test that uses it: the two
 * are one format, and a format whose halves live in different files drifts.
 * It also means a page can read a link it is given, which is what a
 * `restore()` would be built on.
 */
export async function decode(text) {
  const t = String(text).trim();
  const m = t.match(/m([01])=([A-Za-z0-9_-]+)/);
  if (!m) throw new Error('no m0=/m1= fragment in that text');
  const b64 = m[2].replace(/-/g, '+').replace(/_/g, '/');
  const bin = atob(b64 + '='.repeat((4 - (b64.length % 4)) % 4));
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  if (m[1] === '0') return JSON.parse(new TextDecoder().decode(bytes));
  const ds = new DecompressionStream('gzip');
  const wr = ds.writable.getWriter();
  wr.write(bytes); wr.close();
  const buf = await new Response(ds.readable).arrayBuffer();
  return JSON.parse(new TextDecoder().decode(buf));
}

/** The whole link, which is what gets pasted back. */
export async function link(viewer, base) {
  const snap = snapshot(viewer);
  if (!snap) return null;
  const frag = await encode(snap);
  const url = (base || (typeof location !== 'undefined' ? location.href : ''))
    .split('#')[0];
  return { snap, frag, url: `${url}#${frag}` };
}

export default snapshot;
