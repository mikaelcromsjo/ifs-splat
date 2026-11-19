// Where did it jump? -- an identifier a viewer can hand back.
//
// Everything in tests/ measures the morph on this machine, headless, over a
// flight it drives itself. None of it can answer "it went jumpy just then",
// because "just then" is a moment in somebody else's browser on a flight
// that cannot be re-run: the autopilot integrates real frame times, and
// tests/pilot.mjs establishes that the dive is chaotic -- one part in a
// million of speed puts the heading 60 degrees away in three seconds. A
// frame number on its own is therefore not an identifier of anything.
//
// So the page has to do the measuring itself, at the moment it happens, and
// print something short enough to paste back. This is that: the probe from
// tests/drift.mjs -- fixed addresses walked down the drawn tables at the
// drawn fraction, in units of the object's own radius -- running live on
// every slot, every frame, with each slot's own median step as the yardstick.
// A JUMP is a frame that moves a slot's splats much further than that slot's
// ordinary frame does. The log carries the state that produced it (fraction,
// span, table size, migration weight, whether a rebuild landed), which is
// what tests/mark.mjs needs to rebuild the shape offline and draw it.
//
// It watches four other things as well, because they look exactly like a
// jump to an eye and are not the morph at all, and the first job of a
// diagnosis is to say which mechanism it was:
//
//   HITCH   the frame took far longer than the others. At 60 Hz a 200 ms
//           frame delivers twelve frames of motion at once; nothing in the
//           morph is wrong and everything looks wrong.
//   POP     the splat count moved by a quarter in one frame -- the budget or
//           the LOD changing WHAT is drawn, not where it is.
//   REROOT  the camera changed depth, which rescales the whole frame.
//   FIELD   cells entered or left the resident set.
//
// And MARK: the viewer's own "I saw it now", on a key. That one is the
// reason the rest exists -- the detector's thresholds are guesses about what
// is visible, a human eye is not, and the log around the keypress is the
// evidence either way. If the worst morph step anywhere near the mark is 2x
// a median step, the morph is not what was seen, and that is as useful an
// answer as a jump would have been.
//
// Cost, measured by tests/mark.mjs on this machine: 8 addresses at depth 3
// over 8 slots is 192 node compositions a frame, 0.019 ms, against the
// 1.5 ms the pool's rebuild budget alone is allowed. It allocates nothing
// after construction.
import { mix } from './ifs_core.js';

export const DEPTH = 3, NADDR = 8;

// The addresses, derived from their own index: the same splats every frame
// and the same splats in every copy of the demo, so a step is the morph
// moving and not a different splat. Same construction as tests/drift.mjs,
// which can afford 24 of them at depth 4 because it is not inside a frame.
const ADDR = [];
for (let a = 0; a < NADDR; a++) {
  let x = (a * 2654435761) >>> 0;
  const d = [];
  for (let k = 0; k < DEPTH; k++) {
    x = (x * 1664525 + 1013904223) >>> 0;
    d.push(x % 4);
  }
  ADDR.push(d);
}

const e2 = (v) => (Number.isFinite(v) ? v.toExponential(1) : '-');
const f2 = (v) => (Number.isFinite(v) ? v.toFixed(2) : '-');

export class JumpLog {
  constructor(opts = {}) {
    // How much further than its own median a slot may move in one frame
    // before it is worth a line in the log.
    //
    // 8, for the same reason tests/drift.mjs uses 8 against the true motion:
    // with the resize ramped the worst step in a 1500-frame flight is 5x the
    // median and the p99 is 4.8x, so 8 sits above the ordinary distribution
    // and far below a real fault -- switching a resize instead of ramping it
    // reads 86x. A threshold inside the ordinary distribution would fill the
    // log with frames nobody saw, which would make the log useless in the
    // one way that matters: the viewer has to be able to read it.
    this.ratio = opts.ratio ?? 8;
    this.hitchRatio = opts.hitchRatio ?? 3;
    this.hitchMs = opts.hitchMs ?? 40;       // and absolutely slow, not just relatively
    this.popFrac = opts.popFrac ?? 0.25;
    this.window = opts.window ?? 128;        // steps kept per slot for the median
    this.warm = opts.warm ?? 60;             // samples before a median means anything
    this.trail = opts.trail ?? 6;            // frames of history printed with a jump
    // How many frames of per-slot history a snapshot carries.
    //
    // 90 is a second and a half at 60 Hz, and the number it has to cover is
    // human reaction time: the key is pressed about three tenths of a second
    // -- eighteen frames -- after something is seen, so a snapshot of the
    // present frame alone is a snapshot of the aftermath. At 7 rebuilds a
    // second a window this long also holds about ten rebuilds per slot,
    // which is what lets js/snapshot.js replay the pair hand-overs rather
    // than just the fraction sweeping between two fixed builds.
    this.tail = opts.tail ?? 90;
    this.maxRebs = opts.maxRebs ?? 400;
    this.max = opts.max ?? 40;               // events kept; the oldest go first
    this.build = opts.build ?? 'dev';
    this.slots = 0;
    this.frame = 0;
    this.clock = 0;
    this.events = [];
    this.dropped = 0;
    this.worst = 0; this.worstSlot = -1; this.worstFrame = -1;
    this.jumps = 0;
    this.probeMs = 0;
    this._seq = 0;
    // scratch, allocated once -- this runs inside the frame
    this._A = new Float64Array(9);
    this._nA = new Float64Array(9);
    this._E = new Float64Array(9);
    this._b = new Float64Array(3);
    this._nb = new Float64Array(3);
    this._sort = new Float64Array(this.window);
    this._dts = new Float64Array(this.window);
    this._nDt = 0;
    this._lastN = -1; this._lastDepth = -1; this._lastRes = -1;
  }

  /** Size everything to a pool. Called on the first tick, and again if the
   *  pool is replaced (restarting the flight builds a new world). */
  _attach(pool) {
    const S = pool.slots;
    this.slots = S;
    this.pos = new Float64Array(3 * NADDR * S);
    this.prev = new Float64Array(3 * NADDR * S);
    this.have = new Uint8Array(S);
    this.steps = new Float64Array(this.window * S);
    this.nStep = new Int32Array(S);
    this.med = new Float64Array(S);
    // The tail rings: one entry per frame per slot, oldest overwritten.
    // Float32 is deliberate -- these are quantised again on the way into a
    // URL, and the thing they describe is a distance of about 1e-3 of a
    // radius, which single precision carries to seven figures.
    this.tStep = new Float32Array(this.tail * S);
    this.tBt = new Float32Array(this.tail * S);
    this.tFlag = new Uint8Array(this.tail * S);      // 1 built, 2 migrating
    this.tFrame = new Int32Array(this.tail);         // which frame each slot of the ring is
    // The frame's own cost and size, in the same ring, because the two
    // things most often mistaken for a jump are a slow frame and a change in
    // what is drawn. One number each per frame, and they let the question be
    // settled offline from the link alone.
    this.tDt = new Float32Array(this.tail);
    this.tNs = new Float32Array(this.tail);
    this.tHead = -1;
    this.tN = 0;
    // Every rebuild in living memory, with the state it produced. This is
    // what makes an offline reconstruction exact rather than plausible: a
    // table is a pure function of its build parameter, its mask and the
    // migration weight it was built at, so replaying this list through the
    // same reparam() reproduces the two tables the browser was reading --
    // and tests/mark.mjs checks every one of these records against the state
    // its replay actually reaches.
    this.rebs = [];
    this.last = new Float64Array(S);                 // most recent step
    this.lastRatio = new Float64Array(S);
    // A rebuild is read off the slot's own counter rather than pool._builtK,
    // which the pool's rate meter has already cleared by the time a frame
    // gets here.
    this.reb = new Int32Array(S);
    for (let k = 0; k < S; k++) this.reb[k] = pool.shapes[k].rebuilds;
    this.built = new Uint8Array(S);
    // The three most recent build parameters per slot, oldest first.
    //
    // uA and uB alone describe the pair being read NOW, which is enough to
    // rebuild the shape at any fraction -- but not enough to reproduce the
    // step taken ON a rebuild frame, where the frame before was read from
    // the previous pair (u0, u1). Those are exactly the frames the hand-over
    // lands on, so they are the ones worth being able to re-draw.
    this.uh = new Float64Array(3 * S);
    for (let k = 0; k < S; k++) {
      const sh = pool.shapes[k];
      this.uh[3 * k] = sh.uA; this.uh[3 * k + 1] = sh.uA; this.uh[3 * k + 2] = sh.uB;
    }
    return this;
  }

  /**
   * One slot's drawn splat positions, down the fixed addresses.
   *
   * This is tests/drift.mjs's `where()` with the allocation taken out: the
   * node centre CA/CB/CC in Attractor.visit(), by hand, reading the blended
   * table entry at whatever fraction the slot is drawing. `att.C`, `att.R`
   * and `att.mapB` are already the drawn values -- Attractor._scatter writes
   * the interpolated derived state into them every frame -- so the whole
   * position is the one on screen, not a mixture of drawn and newest.
   */
  probe(att, out, o0 = 0) {
    const CM = att._cmB, CMA = att._cmA, MB = att.mapB, m = att.m;
    const tm = att.tableMask, C = att.C;
    const bt = att.interp ? att._bt : 1;
    const A = this._A, nA = this._nA, E = this._E, b = this._b, nb = this._nb;
    for (let a = 0; a < NADDR; a++) {
      A[0] = A[4] = A[8] = 1; A[1] = A[2] = A[3] = A[5] = A[6] = A[7] = 0;
      b[0] = b[1] = b[2] = 0;
      let h = att.rootHash;
      const addr = ADDR[a];
      for (let d = 0; d < addr.length; d++) {
        const i = addr[d] % m;
        const e = 9 * ((mix(h, i) & tm) * m + i);
        if (bt === 1) for (let k = 0; k < 9; k++) E[k] = CM[e + k];
        else for (let k = 0; k < 9; k++) {
          const p = CMA[e + k]; E[k] = p + (CM[e + k] - p) * bt;
        }
        for (let r = 0; r < 3; r++) {
          nb[r] = b[r];
          for (let c = 0; c < 3; c++) {
            nb[r] += A[3 * r + c] * MB[3 * i + c];
            let s = 0;
            for (let k = 0; k < 3; k++) s += A[3 * r + k] * E[3 * k + c];
            nA[3 * r + c] = s;
          }
        }
        for (let k = 0; k < 9; k++) A[k] = nA[k];
        b[0] = nb[0]; b[1] = nb[1]; b[2] = nb[2];
        h = mix(h, i);
      }
      const o = o0 + 3 * a;
      for (let r = 0; r < 3; r++)
        out[o + r] = A[3 * r] * C[0] + A[3 * r + 1] * C[1] + A[3 * r + 2] * C[2] + b[r];
    }
  }

  /** Mean distance between two probe sets, in units of the object's radius,
   *  so slots of different sizes can share one threshold. */
  dist(a, b, o0, R) {
    let s = 0;
    for (let i = 0; i < NADDR; i++) {
      const o = o0 + 3 * i;
      s += Math.hypot(a[o] - b[o], a[o + 1] - b[o + 1], a[o + 2] - b[o + 2]);
    }
    return s / NADDR / (R || 1);
  }

  _median(k) {
    const n = Math.min(this.nStep[k], this.window);
    if (n < this.warm) return 0;
    const s = this._sort.subarray(0, n);
    s.set(this.steps.subarray(k * this.window, k * this.window + n));
    s.sort();
    return s[n >> 1];
  }

  _push(ev) {
    ev.i = ++this._seq;
    this.events.push(ev);
    // The oldest go first: the interesting moment is the one being looked at
    // now, and a log that stops recording once it is full would miss it.
    while (this.events.length > this.max) { this.events.shift(); this.dropped++; }
    return ev;
  }

  /** The state of one slot, as the log prints it and tests/mark.mjs reads it. */
  _state(pool, k) {
    const sh = pool.shapes[k], att = sh.att;
    return {
      k, step: this.last[k], x: this.lastRatio[k], med: this.med[k],
      bt: att.interp ? att._bt : 1, uA: sh.uA, uB: sh.uB,
      span: sh.uB - sh.uA, bits: att.tableBits,
      mig: att._migW, drop: att._migDrop ? 1 : 0,
      built: this.built[k], u0: this.uh[3 * k], u1: this.uh[3 * k + 1],
      from: sh.from, to: sh.to, phase: sh.phase,
    };
  }

  /**
   * One frame. `env` is what the viewer knows and the pool does not: the
   * frame's own cost, how many splats were drawn, how deep the camera is,
   * and how much of the field is resident.
   */
  tick(dt, pool, env = {}) {
    if (!pool || !pool.shapes) return;
    if (this.slots !== pool.slots || !this.pos) this._attach(pool);
    const t0 = (typeof performance !== 'undefined' ? performance.now() : Date.now());
    this.frame++;
    // One snapshot of the frame's own numbers, shared by every event this
    // frame records and never mutated afterwards -- an event that kept a
    // reference to the caller's live object would print the state of
    // whatever frame the log was finally read on.
    const E = { dt: dt * 1000, splats: env.splats ?? -1, depth: env.depth ?? -1,
                visible: env.visible ?? -1, resident: env.resident ?? -1 };
    this.clock += dt;
    const S = this.slots, W = this.window, T = this.tail;
    const head = this.tHead = (this.tHead + 1) % T;
    this.tFrame[head] = this.frame;
    this.tDt[head] = dt * 1000;
    this.tNs[head] = env.splats ?? 0;
    if (this.tN < T) this.tN++;

    for (let k = 0; k < S; k++) {
      const att = pool.shapes[k].att, o0 = 3 * NADDR * k;
      this.probe(att, this.pos, o0);
      const mig = att._migW < 1 || att._migDrop;
      const nr = pool.shapes[k].rebuilds;
      const built = nr !== this.reb[k] ? 1 : 0;
      this.reb[k] = nr; this.built[k] = built;
      if (built) {
        const sh = pool.shapes[k], o = 3 * k;
        this.uh[o] = this.uh[o + 1]; this.uh[o + 1] = sh.uA; this.uh[o + 2] = sh.uB;
        // `want` is what setTableBits was asked for, which is not always what
        // the table is yet: a shrink keeps its mask for one more rebuild so
        // that both halves of the blend's pair agree about which entries
        // exist. A replay has to ask the same question, not assert the answer.
        this.rebs.push({ f: this.frame, k, u: sh.uB, want: att._wantBits,
                         bits: att.tableBits, w: att._migW,
                         g: att._migGrow ? 1 : 0, lo: att._migLo,
                         drop: att._migDrop ? 1 : 0 });
        while (this.rebs.length > this.maxRebs) this.rebs.shift();
      }
      let step = 0, ratio = 0;
      if (this.have[k]) {
        step = this.dist(this.pos, this.prev, o0, att.R);
        this.steps[k * W + (this.nStep[k] % W)] = step;
        this.nStep[k]++;
        // The median is the yardstick and it is a sort, so it is refreshed
        // one slot per frame round-robin rather than eight times a frame.
        if ((this.frame + k) % S === 0 || this.med[k] === 0) this.med[k] = this._median(k);
        const m = this.med[k];
        ratio = m > 0 ? step / m : 0;
        if (ratio > this.worst) {
          this.worst = ratio; this.worstSlot = k; this.worstFrame = this.frame;
        }
      }
      this.last[k] = step; this.lastRatio[k] = ratio;
      this.tStep[k * T + head] = step;
      this.tBt[k * T + head] = att.interp ? att._bt : 1;
      this.tFlag[k * T + head] = built | (mig ? 2 : 0);
      this.prev.set(this.pos.subarray(o0, o0 + 3 * NADDR), o0);
      this.have[k] = 1;
    }

    // The morph's own events, after every slot has been probed, so a jump
    // can print what the rest of the pool was doing in the same frame.
    for (let k = 0; k < S; k++) {
      if (this.lastRatio[k] >= this.ratio && this.nStep[k] > this.warm) {
        this.jumps++;
        this._push({ kind: 'MORPH', f: this.frame, t: this.clock, env: E,
                     s: this._state(pool, k), trail: this._trail(k) });
      }
    }

    // ...and the four things that are not the morph.
    const ms = dt * 1000;
    this._dts[this._nDt % W] = ms; this._nDt++;
    const nDt = Math.min(this._nDt, W);
    if (nDt >= this.warm) {
      const s = this._sort.subarray(0, nDt);
      s.set(this._dts.subarray(0, nDt));
      s.sort();
      const medDt = s[nDt >> 1];
      if (ms > this.hitchMs && ms > this.hitchRatio * medDt)
        this._push({ kind: 'HITCH', f: this.frame, t: this.clock, env: E,
                     ms, medDt, x: ms / medDt });
    }
    const n = env.splats ?? -1;
    if (this._lastN > 1000 && n > 0 &&
        Math.abs(n - this._lastN) > this.popFrac * this._lastN)
      this._push({ kind: 'POP', f: this.frame, t: this.clock, env: E,
                   was: this._lastN, now: n });
    if (this._lastDepth >= 0 && env.depth !== this._lastDepth)
      this._push({ kind: 'REROOT', f: this.frame, t: this.clock, env: E,
                   was: this._lastDepth, now: env.depth });
    if (this._lastRes >= 0 && env.resident !== this._lastRes)
      this._push({ kind: 'FIELD', f: this.frame, t: this.clock, env: E,
                   was: this._lastRes, now: env.resident });
    this._lastN = n; this._lastDepth = env.depth ?? -1;
    this._lastRes = env.resident ?? -1;

    this.probeMs += ((typeof performance !== 'undefined'
      ? performance.now() : Date.now()) - t0 - this.probeMs) * 0.05;
  }

  /** The last `n` frames of one slot, oldest first: what the log prints next
   *  to a jump, and what js/snapshot.js packs for the whole window. */
  _trail(k, n = this.trail) {
    const T = this.tail, have = Math.min(this.tN, n), out = [];
    for (let i = have - 1; i >= 0; i--) {
      const j = ((this.tHead - i) % T + T) % T;
      out.push({ f: this.tFrame[j], step: this.tStep[k * T + j],
                 bt: this.tBt[k * T + j], flag: this.tFlag[k * T + j] });
    }
    return out;
  }

  /** The frame numbers the tail covers, oldest first. */
  tailFrames() { return this._ring(this.tFrame); }
  /** The frame cost and splat count over the same window. */
  tailDt() { return this._ring(this.tDt); }
  tailSplats() { return this._ring(this.tNs); }

  _ring(a) {
    const T = this.tail, have = Math.min(this.tN, T), out = [];
    for (let i = have - 1; i >= 0; i--) out.push(a[((this.tHead - i) % T + T) % T]);
    return out;
  }

  /**
   * The viewer's own mark. Records every slot's current state, not just the
   * worst one: whatever was seen, it was somewhere on the screen, and which
   * slot it was is exactly what is not known at this point.
   */
  mark(pool, env = {}, note = '') {
    const E = { dt: env.dt ?? 0, splats: env.splats ?? -1, depth: env.depth ?? -1,
                visible: env.visible ?? -1, resident: env.resident ?? -1 };
    const all = [];
    for (let k = 0; k < this.slots; k++) all.push(this._state(pool, k));
    return this._push({ kind: 'MARK', f: this.frame, t: this.clock, env: E,
                        note, all });
  }

  /** The log as text: what gets pasted back. Self-contained -- a paste with
   *  no explanation around it still says what every column is. */
  text(pool) {
    const L = [];
    const P = pool || null;
    L.push(`ifs-splat jump log — build ${this.build}`);
    L.push(`  frame ${this.frame}, ${this.clock.toFixed(1)} s of flight, `
         + `probe ${this.probeMs.toFixed(3)} ms/frame`);
    if (P) L.push(`  pool: ${P.slots} slots, rebuild ${P.rebuildHz} Hz, `
      + `morph ${P.morphing ? 'on' : 'OFF'}, interpolate `
      + `${P.interpolate ? 'on' : 'OFF'}, u ${P._u.toFixed(5)}, `
      + `breath ${P.breath.toFixed(3)}, bits ${P.shapes.map((s) => s.att.tableBits).join('')}`);
    L.push(`  worst morph step anywhere in the flight: `
         + `${this.worst.toFixed(1)}x the slot's own median `
         + `(slot ${this.worstSlot}, frame ${this.worstFrame})`);
    L.push(`  ${this.jumps} over the ${this.ratio}x threshold`
         + (this.dropped ? `, ${this.dropped} older events dropped` : ''));
    L.push(`  step is in units of the object's radius; x is step/median;`
         + ` B a rebuild landed, M a table migration is running`);
    L.push('');
    for (const ev of this.events) {
      const e = ev.env || {};
      const tail = `| dt ${(e.dt ?? 0).toFixed(1)}ms n ${e.splats ?? '-'}`
                 + ` d ${e.depth ?? '-'} vis ${e.visible ?? '-'}/${e.resident ?? '-'}`;
      if (ev.kind === 'MORPH') {
        const s = ev.s;
        L.push(`#${ev.i} f${ev.f} t${ev.t.toFixed(2)} MORPH s${s.k}`
          + ` ${s.from}>${s.to} ${s.phase.toFixed(2)}`
          + ` step ${e2(s.step)} ${s.x.toFixed(1)}x med ${e2(s.med)}`
          + ` bt ${f2(s.bt)} span ${e2(s.span)} bits ${s.bits}`
          + ` mig ${f2(s.mig)}${s.drop ? ' drop' : ''}${s.built ? ' B' : ''} ${tail}`);
        L.push(`    uA ${s.uA.toFixed(6)} uB ${s.uB.toFixed(6)}`
          + ` u0 ${s.u0.toFixed(6)} trail `
          + ev.trail.map((x) => `${e2(x.step)}@${f2(x.bt)}`
            + (x.flag & 1 ? 'B' : '') + (x.flag & 2 ? 'M' : '')).join(' '));
      } else if (ev.kind === 'HITCH') {
        L.push(`#${ev.i} f${ev.f} t${ev.t.toFixed(2)} HITCH ${ev.ms.toFixed(1)} ms`
          + ` = ${ev.x.toFixed(1)}x the median frame (${ev.medDt.toFixed(1)} ms) ${tail}`);
      } else if (ev.kind === 'MARK') {
        L.push(`#${ev.i} f${ev.f} t${ev.t.toFixed(2)} MARK${ev.note ? ' ' + ev.note : ''} ${tail}`);
        for (const s of ev.all)
          L.push(`    s${s.k} ${s.from}>${s.to} ${s.phase.toFixed(2)}`
            + ` step ${e2(s.step)} ${s.x.toFixed(1)}x bt ${f2(s.bt)}`
            + ` span ${e2(s.span)} bits ${s.bits} mig ${f2(s.mig)}`
            + `${s.drop ? ' drop' : ''}${s.built ? ' B' : ''}`
            + ` uA ${s.uA.toFixed(6)} uB ${s.uB.toFixed(6)}`);
      } else {
        L.push(`#${ev.i} f${ev.f} t${ev.t.toFixed(2)} ${ev.kind}`
          + ` ${ev.was} -> ${ev.now} ${tail}`);
      }
    }
    if (!this.events.length) L.push('  (nothing logged yet)');
    return L.join('\n');
  }
}

export default JumpLog;
