// Splitting one frame's walk across several threads, without losing the draw
// order that the whole engine rests on.
//
// WHY THIS CAN BE DONE AT ALL
//
// The expansion emits back-to-front by construction and there is no sort
// anywhere -- that is the engine's central claim.  Parallelising a walk
// normally destroys exactly that.  Here it does not, and the reason is a
// proof that was already needed for something else: a cell either holds an
// object or subdivides, never both, so every pair of object-holding cells is
// disjoint whatever levels they are at (983,142 pair tests, 0 intersecting --
// tests/field.mjs).  Disjoint bounding spheres can be ordered by distance,
// which is what FieldWorld.frames() already does, and order_field.mjs
// measures the result at 0.015/255 against an exact global sort.
//
// So objects may be walked in any order as long as their OUTPUTS are
// concatenated in the frames() order.  Within one object the walk is still
// strictly serial and still exact.  That is the whole trick: shard by object,
// never inside one.
//
// WHAT IS SUBTLE
//
// The stream is written BACKWARDS.  AdaptiveStream.begin() sets head = max and
// every emit does --head, so the valid run is [head, max) and the object added
// FIRST ends up at the highest addresses -- drawn last, in front.  add() says
// so in its own comment: "Call these NEAREST FIRST."
//
// Shard that naively and the picture inverts.  Shard 0 holds the nearest
// objects, so shard 0's splats have to land at the END of the final buffer,
// and the shards are uploaded in reverse: the last shard at offset 0, the
// first shard last.  offsets() below is four lines and is the only place that
// knows this; getting it wrong draws the whole field inside out.
//
// None of this needs a GPU, a browser or a worker to test, which is why it is
// a module of its own: tests/shard.mjs drives it on real worker_threads and
// compares the result to the serial walk splat for splat.

/**
 * Split `n` objects, in nearest-first order, into at most `k` CONTIGUOUS runs
 * of roughly equal cost.
 *
 * Contiguous because the concatenation has to come out in frames() order, and
 * a contiguous run of that order is the only split that can be reassembled by
 * offsets alone.
 *
 * Equal COST, not equal count.  One object near the camera can be most of the
 * frame -- cost here is the caller's proxy for how much walking an object will
 * take, and FieldWorld.frames() already computes a serviceable one in
 * `Rw / (d + Rw)`, the apparent size, which is what the LOD threshold is
 * compared against in the first place.
 *
 * The partition minimises the largest run, found by binary search on that
 * bound with a greedy feasibility test -- exact, and at the tens of objects a
 * view holds, immeasurably cheap.  A run may be empty if there are fewer
 * objects than threads.
 */
export function split(costs, k) {
  const n = costs.length;
  const K = Math.max(1, Math.min(k | 0, Math.max(n, 1)));
  if (n === 0) return [];
  if (K === 1) return [{ i0: 0, i1: n, cost: costs.reduce((a, b) => a + b, 0) }];

  let total = 0, worst = 0;
  for (const c of costs) { total += c; if (c > worst) worst = c; }
  // How many runs does a bound of `cap` need, taking objects greedily in
  // order?  An object bigger than cap gets a run to itself, which is why the
  // search starts at the largest single cost.
  const runsFor = (cap) => {
    let runs = 1, acc = 0;
    for (const c of costs) {
      if (acc + c > cap && acc > 0) { runs++; acc = c; } else acc += c;
    }
    return runs;
  };
  let lo = worst, hi = total;
  for (let it = 0; it < 60 && hi - lo > 1e-12 * Math.max(hi, 1); it++) {
    const mid = 0.5 * (lo + hi);
    if (runsFor(mid) <= K) hi = mid; else lo = mid;
  }
  // Lay the runs out at the bound that was found, and produce exactly
  // min(k, n) of them.
  //
  // Minimising the maximum alone will happily return FEWER: if one object is
  // most of the frame, nothing else exceeds the bound and all the rest pack
  // into a single run.  That is optimal for the bottleneck and wrong anyway --
  // the cost proxy is an estimate, idle threads cost nothing, and a run that
  // turns out heavier than predicted has no one to share it with.  So a run is
  // also closed whenever there are exactly as many objects left as runs still
  // to open, which fills every thread without ever exceeding the bound.
  const R = Math.min(K, n);
  const out = [];
  let i0 = 0, acc = 0;
  for (let i = 0; i < n; i++) {
    const c = costs[i];
    const canOpenMore = out.length < R - 1;
    // Non-empty means "holds an object", not "has a positive cost".  On the
    // first frame every measured cost is zero, and testing acc > 0 there
    // closes no run at all and returns one giant shard.
    const nonEmpty = i > i0;
    // joining this object to the current run would leave too few objects for
    // the runs that still have to be opened
    const mustOpen = nonEmpty && canOpenMore && n - i < R - out.length;
    const overBound = nonEmpty && canOpenMore && acc + c > hi;
    if (mustOpen || overBound) {
      out.push({ i0, i1: i, cost: acc });
      i0 = i; acc = c;
    } else acc += c;
  }
  out.push({ i0, i1: n, cost: acc });
  return out;
}

/**
 * Where each shard's splats go in the final buffer.
 *
 * `counts` is per shard in NEAREST-FIRST order, the same order split()
 * returned.  The result is the destination offset for each, in splats.
 *
 * Shard 0 is nearest, and nearest is drawn LAST, so it lands at the top:
 * offset[s] = sum of the counts of every shard AFTER it.  Read the file
 * header before changing this.
 */
export function offsets(counts) {
  const k = counts.length, off = new Array(k).fill(0);
  let acc = 0;
  for (let s = k - 1; s >= 0; s--) { off[s] = acc; acc += counts[s]; }
  return { off, total: acc };
}

/**
 * How much of the splat buffer each shard may use.
 *
 * Divided evenly rather than by cost, and deliberately: the cost proxy says
 * which shard will take LONGEST, not which will emit MOST, and a shard that
 * runs out of room sets its budgetHit, which costs a frame of detail
 * everywhere.  Even shares with a little slack is the safe division.  The
 * shards' regions are disjoint, so one running short cannot corrupt another.
 */
export function capacities(k, maxSplats) {
  // A multiple of four, so that a shard's base in the shared buffer is
  // 41 * splatOffset exactly with no rounding: 41 * 4 = 164 is a multiple of
  // 4, which is the alignment the Float32Array views need.  Round the share
  // up instead and the regions overlap by a few bytes, which is a corruption
  // that would only show as the odd wrong splat.
  const each = Math.floor(maxSplats / Math.max(k, 1)) & ~3;
  return new Array(k).fill(each);
}

/**
 * One frame's measurements from k shards, as the detail controller wants them.
 *
 * The time is the MAXIMUM, not the sum: the shards run at the same time, so
 * what the frame waited for is the slowest of them.  That is also what makes
 * the walk budget mean something again -- targetMs is a wall-clock figure, and
 * k threads each spending 6 ms of it is still 6 ms of frame.
 *
 * The count is the sum, because the GPU draws all of them, and the splat
 * ceiling is a ceiling on the draw.
 *
 * `imbalance` is reported rather than used: the largest shard over the mean,
 * which is how much of the parallel speedup the cost proxy is giving away.
 * 1.0 is a perfect split; 2.0 means half the threads idled.
 */
export function combine(shards) {
  let walkMs = 0, count = 0, hit = false, sum = 0;
  for (const s of shards) {
    if (s.ms > walkMs) walkMs = s.ms;
    count += s.count; sum += s.ms;
    if (s.hit) hit = true;
  }
  const mean = sum / Math.max(shards.length, 1);
  return { walkMs, count, hit,
           imbalance: mean > 0 ? walkMs / mean : 1,
           serialMs: sum };
}
