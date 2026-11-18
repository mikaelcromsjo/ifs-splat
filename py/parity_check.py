#!/usr/bin/env python3
"""Diff the JavaScript generator against the Python one, splat for splat."""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ifs_core as ifs

js = json.load(open(sys.argv[1] if len(sys.argv) > 1 else '/tmp/js_tree.json'))
d, j, s = js['depth'], js['jitter'], js['seed']

print(f"hashes  js {js['hashes']}")
print(f"        py [{ifs.h32(3)}, {ifs.mix(ifs.h32(3),2)}, {ifs.stream(ifs.h32(3),5)}]")

dirs = np.array(js['dirs'])
r = ifs.expand(d, j, s, dirs=dirs)
N = len(r['mu'])
assert N == js['N'], f"splat count {N} vs {js['N']}"

jpos = np.array(js['pos']).reshape(N, 3)
jcov = np.array(js['cov']).reshape(N, 6)
jrgba = np.array(js['rgba']).reshape(N, 4)

pcov = np.stack([r['cov'][:, 0, 0], r['cov'][:, 0, 1], r['cov'][:, 0, 2],
                 r['cov'][:, 1, 1], r['cov'][:, 1, 2], r['cov'][:, 2, 2]], 1)
prgba = np.concatenate([np.round(r['col'] * 255), np.round(r['alpha'][:, None] * 255)], 1)

scale = np.abs(r['mu']).max()
dp = np.abs(jpos - r['mu']).max()
dc = np.abs(jcov - pcov).max() / max(np.abs(pcov).max(), 1e-30)
dr = np.abs(jrgba - prgba).max()
print(f"\n{N} splats, depth {d}, jitter {j}, seed {s}")
print(f"  position   max abs diff {dp:.3e}  (scene scale {scale:.3f})")
print(f"  covariance max rel diff {dc:.3e}")
print(f"  rgba       max abs diff {dr:.0f} / 255")

def replay(levels, dirv, N):
    """Ordered expansion, run on the key points JavaScript actually used.

    Separates the two things a mismatch could mean: a different slot
    arithmetic (a bug) from a different key value (float32 vs float64 on a
    near-tie, which is harmless)."""
    out = np.zeros(N, np.int64) - 1
    ob = {0: 0}
    for L in levels:
        n, ni, cs = L['n'], L['nItems'], L['childSize']
        kp = np.array(L['keyPts'], np.float32).reshape(n, ni, 3).astype(np.float64)
        cb = np.array(L['cbase'], np.int64)
        keys = kp @ dirv
        nob = {}
        for j in range(n):
            order = np.argsort(-keys[j], kind='stable')
            off = ob[j]
            for it in order:
                if it == 0:
                    out[off] = cb[j]; off += 1
                elif ni == 2:
                    out[off] = cb[j] + 1; off += 1
                else:
                    nob[j * 4 + (it - 1)] = off; off += cs
        if ni == 2:
            break
        ob = nob
    return out


# Draw order.  A mismatch only matters if the two splats it swaps are at
# genuinely different depths: JS keys the sort off float32 world points and
# Python off float64, so exact ties can land either way, and a tie is a pair
# the renderer composites identically whichever order it picks.
bad = tie = 0
worst = 0.0
for k, o in enumerate(js['orders']):
    want = np.argsort(r['slot'][k], kind='stable')
    got = np.array(o)
    diff = np.nonzero(want != got)[0]
    if not len(diff):
        continue
    # does the slot arithmetic agree, given JS's own float32 key points?
    same_logic = np.array_equal(replay(js['levels'], dirs[k], N), got)
    z = r['mu'] @ dirs[k]
    gap = np.abs(z[got[diff]] - z[want[diff]])
    if same_logic:
        tie += len(diff)
        print(f"  order dir {k}: {len(diff)} / {N} slots differ -- same slot "
              f"arithmetic, float32 vs float64 key on a near-tie")
    else:
        bad += len(diff)
        worst = max(worst, gap.max())
        print(f"  order dir {k}: {len(diff)} / {N} slots differ -- SLOT "
              f"ARITHMETIC DISAGREES (max depth gap {gap.max():.2e})")
print(f"  draw order: {bad} logic mismatches, {tie} float-tie slots "
      f"out of {N * len(js['orders'])}")

ok = dp < 1e-5 * max(scale, 1) and dc < 1e-5 and dr <= 1 and bad == 0
print("\nPARITY", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
