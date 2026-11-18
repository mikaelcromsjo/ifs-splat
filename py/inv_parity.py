"""Does the browser's live inversion counter agree with the offline one?

The proof panel shows a number live; stage 0's verdict rests on the same
number computed in numpy.  If they disagree the panel is decoration.  Both
sides run on bit-identical trees (py/parity_check.py), so the rates must match
to float64 noise.
"""
import json
import sys

import numpy as np

from ifs_core import expand
from stage0 import weighted_inversions

js = json.load(open(sys.argv[1] if len(sys.argv) > 1 else '/tmp/js_inv.json'))
dirs = np.array([r['f'] for r in js['rows']], float)
t = expand(js['depth'], 0.16, 3, dirs=dirs)
mu, alpha, slot = t['mu'], t['alpha'], t['slot']
N = len(mu)
assert N == js['N'], f"{N} != {js['N']}"

print(f"{N} splats, depth {js['depth']}\n")
print('  direction                     python      js        difference')
worst = 0.0
for k, row in enumerate(js['rows']):
    d = dirs[k]
    z = mu @ d                                   # larger = farther along view
    pos = np.empty(N, np.int64)
    pos[slot[k]] = np.arange(N)                  # slot -> stream index
    bad, tot = weighted_inversions(slot[k], z, alpha)
    rate = bad / tot
    diff = abs(rate - row['rate'])
    worst = max(worst, diff)
    print(f"  [{' '.join(f'{v:+.2f}' for v in d)}]   {rate*100:7.3f}%  "
          f"{row['rate']*100:7.3f}%   {diff*100:.2e}%")
# Not bit-exact, and it cannot be: the browser's ground truth is a 16-bit
# counting sort over float32 depth keys, numpy's is an exact float64 argsort.
# Splats within a key quantum of each other come out in a different order, and
# each such pair moves the rate by about 1/N^2.  A hundredth of a percentage
# point is far below anything the panel displays or the verdict turns on.
TOL = 1e-4
ok = worst < TOL
print(f"\n  worst difference {worst*100:.3e} percentage points "
      f"(tolerance {TOL*100:.2f}, float32 key ties) -- "
      f"{'PARITY OK' if ok else 'MISMATCH'}")
sys.exit(0 if ok else 1)
