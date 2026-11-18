// The number behind the proof panel.
//
// Ordered expansion does not sort.  It permutes each node's five children by
// five dot products and nine compare-swaps, and the claim is that the stream
// this produces is close enough to a true back-to-front sort that you cannot
// see the difference.  "Close enough" needs a number, and the number is: of
// all pairs of splats, what fraction does the stream draw in the wrong order
// -- weighted by how much each pair could matter?
//
// Weighted, because an inversion between two nearly transparent splats changes
// nothing, and one between two opaque ones changes everything.  The weight is
// the product of the two opacities, which is what the over-operator makes the
// pair's contribution proportional to.
//
// O(n log n) by Fenwick tree over depth rank: walk the stream in draw order
// and, for each splat, ask how much weight has already been drawn that was
// actually NEARER.  That weight is in the wrong order with respect to this
// one.  A port of weighted_inversions() in py/stage0.py, and
// tests/inversion_parity.mjs checks the two against each other.

export class InversionCounter {
  constructor(n) { this.grow(n); }

  grow(n) {
    this.n = n;
    this.rank = new Int32Array(n);
    this.tree = new Float64Array(n + 1);
    this.idx = new Int32Array(n);
    this.key = new Float64Array(n);
  }

  /**
   * @param order  the stream, as splat indices, back to front
   * @param truth  the exact order, as splat indices, back to front
   * @param alpha  per-splat opacity, indexed by splat index
   * @param count  how many of `order`/`truth` to use
   * @returns {bad, total, rate}
   */
  measure(order, truth, alpha, count) {
    const n = count;
    if (n > this.n) this.grow(n);
    const rank = this.rank, tree = this.tree;
    // truth is back to front, so position in `truth` IS the depth rank:
    // rank 0 = farthest.  An inversion is a splat drawn after one that is
    // farther than it... which is the normal case.  The wrong order is
    // drawing a NEARER splat first, so we count already-drawn weight whose
    // rank is strictly greater.
    for (let k = 0; k < n; k++) rank[truth[k]] = k;
    tree.fill(0, 0, n + 1);
    let bad = 0, sum = 0, sq = 0;
    for (let k = 0; k < n; k++) {
      const w = alpha[order[k]];
      // prefix sum over ranks <= r, i.e. weight already drawn that is FARTHER
      // or equal; the rest of what has been drawn is nearer, and wrong.
      let i = rank[order[k]] + 1, s = 0;
      while (i > 0) { s += tree[i]; i -= i & (-i); }
      bad += w * (sum - s);
      sum += w; sq += w * w;
      i = rank[order[k]] + 1;
      while (i <= n) { tree[i] += w; i += i & (-i); }
    }
    const total = 0.5 * (sum * sum - sq);
    return { bad, total: Math.max(total, 0), rate: total > 0 ? bad / total : 0 };
  }

  /**
   * The same thing on every `stride`-th splat of the stream.  The rate is a
   * ratio of pair weights, so a subsample estimates it without bias while
   * costing stride times less -- which is what makes it affordable to show
   * live rather than once at load.
   */
  sampled(order, truth, alpha, count, stride = 1) {
    if (stride <= 1) return this.measure(order, truth, alpha, count);
    const keep = new Uint8Array(this.n >= count ? this.n : count);
    let m = 0;
    for (let k = 0; k < count; k += stride) { keep[order[k]] = 1; m++; }
    const o = this.idx.length >= m ? this.idx : (this.idx = new Int32Array(m));
    const t = new Int32Array(m);
    let a = 0, b = 0;
    for (let k = 0; k < count; k++) if (keep[order[k]]) o[a++] = order[k];
    for (let k = 0; k < count; k++) if (keep[truth[k]]) t[b++] = truth[k];
    return this.measure(o.subarray(0, a), t, alpha, Math.min(a, b));
  }
}
