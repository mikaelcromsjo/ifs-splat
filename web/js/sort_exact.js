// An exact per-splat depth sort, for the proof panel to compare against.
//
// This is what every web 3DGS viewer does per frame: quantise view-space z into
// 16-bit buckets and counting-sort.  It is the fair opponent -- O(N) with a
// small constant, not a comparison sort -- so the frame times the proof panel
// shows are honest.

export class ExactSorter {
  constructor(N) {
    this.N = N;
    this.depth = new Int32Array(N);
    this.counts = new Uint32Array(65536);
    this.starts = new Uint32Array(65536);
  }

  /** Fill `out` with splat indices, farthest first. */
  sort(pos, dir, out) {
    const N = this.N, depth = this.depth, counts = this.counts, starts = this.starts;
    const dx = dir[0], dy = dir[1], dz = dir[2];
    let lo = Infinity, hi = -Infinity;
    for (let i = 0, j = 0; i < N; i++, j += 3) {
      const d = pos[j] * dx + pos[j + 1] * dy + pos[j + 2] * dz;
      if (d < lo) lo = d;
      if (d > hi) hi = d;
      depth[i] = d * 4096;                  // provisional, rescaled below
    }
    counts.fill(0);
    const s = hi > lo ? 65535.0 / (hi - lo) : 0.0;
    for (let i = 0, j = 0; i < N; i++, j += 3) {
      const d = pos[j] * dx + pos[j + 1] * dy + pos[j + 2] * dz;
      // bucket 0 = farthest, so the stream comes out back to front
      const b = (hi - d) * s | 0;
      depth[i] = b;
      counts[b]++;
    }
    let acc = 0;
    for (let b = 0; b < 65536; b++) { starts[b] = acc; acc += counts[b]; }
    for (let i = 0; i < N; i++) out[starts[depth[i]]++] = i;
    return out;
  }
}
