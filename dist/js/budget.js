// The detail controller, and the clock it needs.
//
// WHY THIS FILE EXISTS
//
// AdaptiveStream.end() has always held a budget by nudging `lodScale`, the
// multiplier on the LOD's pixel thresholds.  Its error term was the WALK's own
// time -- `lastMs / targetMs` -- plus, later, the splat count.  Both are
// measured on the CPU, inside the expansion.
//
// Which means the controller could not see the GPU at all.  Every saving on
// that side of the fence -- a cheaper vertex shader, a smaller upload, a
// half-resolution background -- produced exactly nothing visible: the walk
// still spent its 6 ms, still emitted the same splats, and the frame just
// finished earlier with the slack thrown away.  Optimise the draw and the
// picture does not change.  That is the fault this fixes.
//
// So the error term gains a FRAME term, and the frame is measured two ways,
// because neither alone is enough:
//
//   * `EXT_disjoint_timer_query_webgl2`, when the driver has it.  This is the
//     only honest measure of GPU work.  It is asynchronous -- a query's result
//     lands a frame or two after the frame it timed -- which is fine for a
//     controller already damped to move slowly, and it is two-sided: spend
//     less on the GPU and the number falls, so the slack is spent on detail.
//
//   * the interval between frame starts, otherwise.  A vsync-locked frame
//     holds at the refresh period no matter how much GPU headroom is left, so
//     this CANNOT say how much room there is -- only that there is none left,
//     by stretching past the refresh period once frames start being missed.
//     It is therefore wired as a brake only: it can take detail away, never
//     grant it.  With no timer query and frames on time, the controller is
//     exactly what it was before this file existed.
//
// The refresh period is not asked for, it is observed: the running minimum of
// the intervals, which is the one statistic a vsync-locked display hands you
// for free.  60 Hz gives 16.7, 120 Hz gives 8.3, and a machine that never
// makes its refresh gives whatever it does make, which is the right target to
// aim under anyway.
//
// All of it is pure, and that is deliberate.  No GL, no clock, no stream: the
// state goes in, the next `lodScale` comes out, and tests/budget.mjs drives it
// with synthetic load -- a step change, a slow GPU, a vsync cliff -- and
// checks that it converges, does not pump, and spends what it is given.  None
// of that can be tested in a browser from here, and the arithmetic is the part
// that can be got wrong quietly.

/** How much of a refresh period a frame may use before the controller pushes
 *  back.  Not 1.0: a frame that exactly fills its period has no room for the
 *  compositor, the page, or the next frame being slightly worse, and a
 *  controller aiming at the cliff edge lives on the cliff edge. */
export const FRAME_FILL = 0.85;

/** A stretched interval below this multiple of the refresh period is not
 *  treated as a miss.  Frames jitter; a single long one is not a signal. */
export const MISS_AT = 1.15;

/** The brake fires on MISSED FRAMES, one step per miss, and not off the
 *  averaged interval.
 *
 *  Driving it from the average was the first attempt and it wound up: the
 *  average lags by its own time constant, so the brake went on pushing for
 *  fifteen frames after the frames themselves were landing on time, and the
 *  threshold moved 36% peak to peak in steady state.  A miss is a discrete
 *  event and the response to it is a discrete step. */

/** How fast the remembered cliff leaks downward, per frame.  The brake learns
 *  the threshold at which this machine started missing frames and refuses to
 *  go back under it; without a leak that memory would be permanent, and a
 *  scene that got simpler, or a tab that got its GPU back, could never regain
 *  detail.  0.997 a frame is about 17% a second, and because the floor is
 *  what the walk term runs into on its way down, that leak also sets the
 *  speed at which detail is allowed to come back. */
export const CLIFF_LEAK = 0.997;

/** How far one braking frame may push the threshold up.  The ordinary bound is
 *  30% a frame, which is right when the measurement is trustworthy -- but the
 *  brake's is not: it fires off a lagging average of a quantised signal, so it
 *  overshoots, and that overshoot is the amplitude of the ripple the cliff
 *  memory exists to remove.  A gentler step takes a few more frames to shed a
 *  bad overload, on a display that is already dropping frames, in exchange for
 *  a threshold that does not visibly pulse the rest of the time. */
export const BRAKE_RISE = 1.06;

/** The leak on the frame fit, and a direct trade against how fast the
 *  controller answers a step change in what a splat costs.  Measured on a 6x
 *  GPU slowdown at frame 200 (tests/budget.mjs scenario 6), frames to get back
 *  inside the budget: 1/300 -> 106, 1/200 -> 75, 1/120 -> 40, 1/80 -> 19.  The
 *  fitted floor moves the other way, and only in the regime where the term has
 *  already stood down.  1/120 is two seconds at 60 Hz. */
export const FIT_A = 1 / 120;

/** Frames before the fit is believed at all. */
export const FIT_N = 90;

/** The least relative spread in the splat count (sd/mean) that identifies the
 *  slope.  Below this the count has barely moved and the fit is noise. */
export const SPAN_MIN = 0.02;

/** The least of the budget that must be the splats' before the frame term is
 *  believed, in ms.  Below this the frame is almost all floor and the ratio is
 *  noise. */
export const MIN_HEAD = 1.5;

/** The most the controller may ever coarsen the threshold.  0.9 x 64 = 58 px
 *  is a coarse scene but still a recognisable one; the old clamp of 1000 was
 *  900 px, which is a single splat and no scene at all.  A bound here is what
 *  makes the worst case survivable rather than blank. */
export const LS_MAX = 64;

/** Bounds on the observed refresh period, in ms.  Outside these the number is
 *  not a refresh period -- it is a stall, or a clock with no vsync at all. */
export const REFRESH_LO = 5.0, REFRESH_HI = 40.0;

/** The window of recent intervals, how many are needed before it is read, how
 *  often it is read, how near a multiple an interval must be to count as one,
 *  and the fraction of the window a candidate must explain. */
// IV_MIN is 4 and not 24 because the early intervals are the only ones that
// ever show the true period on a machine that cannot keep up: the scene starts
// coarse and those first frames land on time, and by frame twenty the detail
// has grown and every interval is a multiple.  The window's answer can only
// LOWER the remembered period, and a wrong low value cannot get in anyway --
// it would have to divide 70% of the intervals.
export const IV_WIN = 120, IV_MIN = 4, IV_EVERY = 5;
// 0.7, not 0.9: a quarter of the callbacks arriving early is not rare, and a
// candidate has to survive that to be found at all.
export const IV_TOL = 0.12, IV_EXPLAIN = 0.7;

/**
 * The refresh period from a window of frame intervals.
 *
 * A VSYNC-LOCKED INTERVAL IS AN INTEGER MULTIPLE OF THE PERIOD.  That one fact
 * is the whole estimator, and not having used it is the second bug this
 * controller shipped.  The period was the running minimum of the intervals,
 * which assumes rAF only ever runs LATE.  It does not: after a long frame the
 * browser catches up and delivers the next callback almost immediately, so a
 * 60 Hz display hands out the occasional 6 ms gap.  A minimum believes it, the
 * budget becomes 0.85 x 6 ms, and the controller spends the session chasing
 * 200 fps -- taking detail away every frame to get there.  A scene that starts
 * as a fractal and shrinks away, which is how it was reported.
 *
 * A 6 ms gap cannot be a 60 Hz period, and the window says so: 16.67 is not a
 * multiple of 6.  So the candidates are the intervals actually observed, and
 * the answer is the LARGEST of them that explains the rest as multiples of
 * itself -- which is the greatest common divisor, give or take the tolerance.
 * A plain low percentile was the first attempt and it is not enough either: it
 * is robust to the early callbacks but it cannot tell a 30 Hz display from a
 * 60 Hz one missing every frame, and the long memory below is what settles
 * that, from whatever on-time frames were seen while the scene was still
 * coarse.
 *
 * Returns 0 when nothing explains the window, and the caller keeps what it
 * had: a window with no majority has nothing to say.
 */
export function periodOf(ring, n) {
  // Candidates, largest first, bucketed to a quarter of a millisecond so that
  // a hundred near-identical intervals are one candidate and not a hundred.
  const seen = new Set(), cand = [];
  for (let i = 0; i < n; i++) {
    const b = Math.round(ring[i] * 4);
    // Bucketed so a hundred near-identical intervals are one candidate, but
    // the candidate is the interval ITSELF and not the bucket's centre: a
    // 60 Hz period is 16.667 ms and rounding it to 16.75 is a half-percent
    // error in every budget that follows from it.
    if (!seen.has(b)) { seen.add(b); cand.push(ring[i]); }
  }
  cand.sort((a, b) => b - a);
  const need = n * IV_EXPLAIN;
  for (const c of cand) {
    if (c < REFRESH_LO) break;
    let ok = 0;
    for (let i = 0; i < n; i++) {
      const r = ring[i] / c, k = Math.round(r);
      if (k >= 1 && Math.abs(r - k) <= IV_TOL) ok++;
    }
    if (ok >= need) return c;
  }
  return 0;
}

export function makeBudget(opts = {}) {
  return {
    // what the rung asks for: the walk's own ceilings
    targetMs: opts.targetMs ?? 6.0,
    target: opts.target ?? 200000,
    // 'time' | 'splats' | 'both' | 'frozen', exactly as before
    budgetBy: opts.budgetBy ?? 'both',
    // the frame term, which is the point of this file.  Switchable so that a
    // test -- and tests/tour.mjs, which has no GL and no vsync -- can ask for
    // the old behaviour and get it bit for bit.
    useFrame: opts.useFrame ?? true,
    fill: opts.fill ?? FRAME_FILL,
    lodScale: opts.lodScale ?? 4.0,
    // observed, never configured
    refreshMs: 0,
    ivRing: new Float64Array(IV_WIN), ivAt: 0, ivN: 0, ivPinned: false,
    floorMs: 0,
    fitMx: 0, fitMy: 0, fitCxx: 0, fitCxy: 0, fitN: 0, fitK: 0,
    emaFrame: 0,
    emaAlpha: opts.emaAlpha ?? 0.12,
    // The cliff the brake remembers.  Only ever used on the fallback path:
    // with a timer query the frame term is continuous and needs no trial.
    cliffLs: 0,
    // what the last step decided, for the HUD
    err: 1, binding: 'none', frameMs: 0, gpuMs: 0,
  };
}

/**
 * One step of the controller.
 *
 * `m` is the frame's measurements:
 *   walkMs   the expansion's own time
 *   count    splats emitted
 *   cpuMs    the frame's JS time -- submission, not completion
 *   gpuMs    from a timer query, or null/undefined if there is none
 *   interval ms since the previous frame started, or null
 *   hit      the stream ran out of buffer
 *
 * Returns the state, mutated, with `lodScale` advanced.
 */
export function budgetStep(st, m) {
  // The refresh period: the smallest interval seen, with a slow leak upwards
  // so that moving to a slower display, or a tab throttled and restored, is
  // not remembered for ever as "it was 16.7 ms once".
  const iv = m.interval;
  if (iv != null && iv >= REFRESH_LO && iv <= REFRESH_HI) {
    // A LOW PERCENTILE of the recent intervals, not the least of them.
    //
    // The running minimum was the first version and it is the second bug this
    // controller shipped.  requestAnimationFrame does not only run late, it
    // sometimes runs EARLY: after a long frame the browser catches up and
    // delivers the next callback almost immediately, so a 60 Hz display hands
    // out the occasional 6 ms gap.  A minimum believes it, the budget becomes
    // 0.85 x 6 ms, and the controller spends the rest of the session chasing
    // 200 fps -- taking detail away every frame to get there, which is a
    // scene that starts as a fractal and shrinks away.  One such gap was
    // enough to cut a 255k-splat scene to 88k and leave it oscillating
    // twenty seconds later.
    //
    // A percentile is robust at both ends, and the two ends are not alike: a
    // missed frame makes the interval a MULTIPLE of the period, while
    // coalescing makes it a fraction.  The 25th discards the early ones
    // without being dragged up by the late ones, and a 120-frame window
    // follows a real change of display inside two seconds instead of the
    // thirty-four a leaking minimum took.
    st.ivRing[st.ivAt++ % IV_WIN] = iv;
    if (st.ivN < IV_WIN) st.ivN++;
    if (st.ivPinned) { /* pinned by ?refresh= : believe the caller, not rAF */ }
    else if (st.ivN >= IV_MIN && st.ivAt % IV_EVERY === 0) {
      const c = periodOf(st.ivRing, st.ivN);
      if (c > 0) st.refreshMs = st.refreshMs > 0
        ? Math.min(c, st.refreshMs * 1.0005)
        : c;
    }
  }
  if (iv != null) {
    st.emaFrame = st.emaFrame > 0
      ? st.emaFrame + st.emaAlpha * (iv - st.emaFrame)
      : iv;
  }

  if (st.budgetBy === 'frozen') { st.err = 1; st.binding = 'frozen'; return st; }

  const errT = st.targetMs > 0 ? m.walkMs / st.targetMs : 0;
  const errN = st.target > 0 ? m.count / Math.max(st.target, 1) : 0;

  // ---- the frame term
  let errF = 0, how = 'none';
  if (st.useFrame) {
    const refresh = st.refreshMs > 0 ? st.refreshMs : 16.667;
    const budget = refresh * st.fill;
    if (m.gpuMs != null && m.gpuMs >= 0) {
      // Both sides of the fence, against a fraction of the refresh period.
      // Two-sided: this is the term that converts a cheaper draw into detail.
      // The SUM, which is conservative and probably too much so: the CPU and
      // the GPU pipeline across frames -- this frame's JS runs while the last
      // frame's draw is still going, which is what the ring of upload buffers
      // exists to allow -- so the sustained rate is nearer 1/max(cpu, gpu),
      // and adding them overstates the frame by the whole overlap.  Switching
      // to max() measured 33% more detail at the same frame rate here.  It is
      // NOT done, because it moves the operating point far enough that the
      // splat ceiling and the missed-frame question both come back, and those
      // cannot be settled without a GPU.  It is a separate change, and it is
      // the next one worth making.
      st.frameMs = (m.cpuMs ?? 0) + m.gpuMs;
      st.gpuMs = m.gpuMs;
      // ...but only against the part of the frame the detail can actually
      // move.  THE FIRST VERSION OF THIS DIVIDED THE WHOLE FRAME BY THE
      // BUDGET, and that is wrong in a way that destroys the scene: the sky,
      // the pool rebuild, the uniforms and the present cost the same whether
      // there are 300k splats or one, so when that floor alone exceeds the
      // budget there is no number of splats that satisfies the term.  The
      // controller then pushes the threshold to its clamp and the remembered
      // cliff pins it there: one splat, 900 px across, and a detail key that
      // cannot win because the loop rewrites the threshold every frame.  It
      // shipped, and it is what a live machine showed within the hour.
      //
      // So the frame is SEPARATED into the part that follows the splat count
      // and the part that does not, by fitting
      //
      //     frameMs  ~  floor + k * count
      //
      // over a leaky window.  Taking the least frame ever seen as the floor
      // was the obvious cheap answer and it does not work: the minimum is
      // whatever the detail happened to cost when the page opened, not the
      // floor, so the headroom collapses to nothing and the term switches
      // itself off -- safe, but it gives up the one thing it was added for.
      //
      // A fit needs the count to move to identify the slope, which it does:
      // the controller is nudging every frame.  When it has not moved enough
      // to be sure (cxx below COV_MIN), k is unidentified and the term says
      // so rather than guessing.
      const a = FIT_A;
      const dx = m.count - st.fitMx, dy = st.frameMs - st.fitMy;
      st.fitMx += a * dx;
      st.fitMy += a * dy;
      st.fitCxx = (1 - a) * (st.fitCxx + a * dx * dx);
      st.fitCxy = (1 - a) * (st.fitCxy + a * dx * dy);
      st.fitN++;
      const span = st.fitMx > 0 ? Math.sqrt(st.fitCxx) / st.fitMx : 0;
      // A loop that is doing its job stops exciting its own plant: once the
      // threshold settles the count barely moves, the slope is no longer
      // identified, and a term that needed a live fit every frame would
      // switch itself off exactly when it was working.  So the slope is
      // LEARNED while the count is moving and HELD while it is not.  It only
      // has to be relearned if what a splat costs actually changes, and that
      // change is itself what moves the count again.
      if (st.fitN >= FIT_N && span >= SPAN_MIN && st.fitCxx > 0)
        st.fitK = Math.max(0, st.fitCxy / st.fitCxx);
      const k = st.fitK;
      st.floorMs = st.frameMs - k * m.count;
      const head = budget - st.floorMs;
      if (k > 0) {
        // The splats' own share of the frame, against the share of the budget
        // that is theirs -- but never against less than MIN_HEAD of it.
        //
        // Standing the term down entirely when the floor ate the budget was
        // the first attempt, and it fails the other way: with nothing but the
        // walk clock to answer to, the detail runs free.  On a 120 Hz display
        // whose floor is 6.5 ms of a 7.0 ms budget that came out as 1.2
        // million splats and a 42 ms frame, eight times over.  So the splats
        // always get a share and the share is always bounded: at least
        // MIN_HEAD, at most the real headroom.  Neither an empty screen nor
        // an unbounded one.
        errF = (k * m.count) / Math.max(head, MIN_HEAD);
        how = head >= MIN_HEAD ? 'gpu' : 'floor';
      } else {
        how = 'warmup';
      }
    } else {
      // No timer query.  All that is left is the interval, and a vsync-locked
      // frame holds at the period however much headroom is left -- so this can
      // only ever say "there is none", by a frame arriving late.  A brake,
      // then, with two parts:
      //
      //   one STEP PER MISSED FRAME.  Not a function of the averaged interval:
      //     see MISS_AT.  This is additive-increase, multiplicative-decrease,
      //     and the thing being avoided is integral windup.
      //   a remembered CLIFF, because without it the loop is: grant detail
      //     until a frame is missed, take it away, grant it again, for ever.
      //     Learning the threshold that broke, and leaking it slowly
      //     downward, turns that oscillation into a slow creep towards
      //     whatever this machine can actually do.
      st.frameMs = st.emaFrame;
      st.gpuMs = 0;
      if (iv != null && iv > refresh * MISS_AT) {
        errF = iv / refresh;             // > 1 by construction
        how = 'interval';
      }
    }
  }

  const errWalk = st.budgetBy === 'time' ? errT
                : st.budgetBy === 'splats' ? errN
                : Math.max(errT, errN);
  const err = Math.max(errWalk, errF);
  st.err = err;
  st.binding = err === errF && errF > 0 ? how
             : errT >= errN ? 'time' : 'splats';

  // The response, unchanged from what AdaptiveStream.end() has always done:
  // proportional in the log, bounded per frame, and three times quicker to
  // give detail up than to take it back.  A causal controller cannot be right
  // about a view that changes faster than it can track; what it can promise is
  // a bound on how wrong one frame gets.
  const g = Math.pow(Math.max(err, 1e-3), m.hit ? 0.5 : 0.22);
  const lo = m.hit ? 1.0 : 0.80;
  const hi = m.hit ? 3.0 : how === 'interval' ? BRAKE_RISE : 1.30;
  st.lodScale *= Math.max(lo, Math.min(hi, g));

  // The remembered cliff, and the floor it puts under the threshold.  Only
  // the brake sets it: with a timer query there is a continuous measure of
  // the frame and nothing to learn by trial.
  st.cliffLs *= CLIFF_LEAK;
  if (how === 'interval') st.cliffLs = Math.max(st.cliffLs, st.lodScale);
  if (st.cliffLs > 0) st.lodScale = Math.max(st.lodScale, st.cliffLs);

  st.lodScale = Math.max(0.05, Math.min(LS_MAX, st.lodScale));
  return st;
}

/**
 * The GPU's own clock: EXT_disjoint_timer_query_webgl2, wrapped so that the
 * caller never has to think about it.
 *
 * Timer queries are asynchronous and must not be polled in the frame that
 * issued them -- doing so is a stall, which is the opposite of the point.  So
 * this keeps a ring of queries, begins one per frame, and collects whichever
 * older one has become available.  `ms` is therefore a frame or two stale.
 * The controller is damped over tens of frames, so that is immaterial.
 *
 * `disjoint` is the extension's way of saying the GPU was interrupted and
 * every number in flight is rubbish -- a context switch, a power state
 * change.  The whole ring is thrown away when it is set, because a wrong
 * reading here is a wrong lodScale, which is visible.
 *
 * NEVER EXECUTED HERE.  There is no GL on the machine this was written on.
 * The usage is the documented one and the failure path is total: no extension,
 * or any throw at all, and `ms` stays null, which the controller reads as "no
 * timer query" and falls back to the interval brake.
 */
export class GpuClock {
  constructor(gl, depth = 4) {
    this.gl = gl; this.ms = null; this.ok = false;
    this.ext = null; this.ring = []; this.depth = depth; this.active = null;
    try {
      this.ext = gl.getExtension('EXT_disjoint_timer_query_webgl2');
      this.ok = !!this.ext;
    } catch (_) { this.ok = false; }
  }

  /** Open a query for this frame, if the last one has been closed. */
  begin() {
    if (!this.ok || this.active) return;
    try {
      const q = this.ring.length < this.depth ? this.gl.createQuery() : null;
      if (!q) return;
      this.gl.beginQuery(this.ext.TIME_ELAPSED_EXT, q);
      this.active = q;
    } catch (_) { this.ok = false; this.active = null; }
  }

  /** Close this frame's query and collect any older result that is ready. */
  end() {
    if (!this.ok) return this.ms;
    const gl = this.gl;
    try {
      if (this.active) {
        gl.endQuery(this.ext.TIME_ELAPSED_EXT);
        this.ring.push(this.active);
        this.active = null;
      }
      if (gl.getParameter(this.ext.GPU_DISJOINT_EXT)) {
        for (const q of this.ring) gl.deleteQuery(q);
        this.ring.length = 0;
        this.ms = null;
        return this.ms;
      }
      while (this.ring.length) {
        const q = this.ring[0];
        if (!gl.getQueryParameter(q, gl.QUERY_RESULT_AVAILABLE)) break;
        const ns = gl.getQueryParameter(q, gl.QUERY_RESULT);
        gl.deleteQuery(q);
        this.ring.shift();
        this.ms = ns / 1e6;
      }
    } catch (_) {
      this.ok = false; this.ms = null;
      for (const q of this.ring) { try { gl.deleteQuery(q); } catch (_) {} }
      this.ring.length = 0;
    }
    return this.ms;
  }
}
