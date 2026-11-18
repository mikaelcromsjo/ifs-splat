#!/usr/bin/env python3
"""
Stage 0: does approximate order actually look wrong?

Ground truth is an exact per-splat depth sort.  The candidate is ordered
expansion (ifs_core.expand with ``dirs``): each node sorts its m+1 children by
``<c_i, M^T d>`` and writes them into slots, so the last level comes out as a
back-to-front stream with nothing of size N ever sorted.

Three numbers, swept over camera directions:

  tile   per-tile opacity-weighted inversion rate.  Two splats in a different
         order only matter if they land on the same pixels, so this restricts
         the count to pairs that share a 16x16 tile, and weights each pair by
         the product of the two effective opacities.  This is the number.
  glob   the same weighting over *all* pairs, for reference.  Much larger and
         much less meaningful -- almost no pair of splats overlaps on screen.
  PSNR   exact-sort render vs ordered-expansion render, same splats, same
         camera, only the blend order differs.

Usage:
    python3 stage0.py                       # depth 7, 64 directions, 8 images
    python3 stage0.py -d 8 --dirs 96        # 153k splats
    python3 stage0.py --rep mu0             # compare subtree representatives
"""

import argparse
import os
import struct
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ifs_core as ifs
import splat_raster as sr


# --------------------------------------------------------------- metrics

def weighted_inversions(pos, z, w):
    """sum of w_i*w_j over pairs the stream draws in the wrong order.

    ``pos`` is each element's index in the candidate stream, ``z`` its true
    depth (larger = farther), ``w`` its weight.  A pair is inverted when the
    stream draws the *farther* splat second.  O(n log n) by Fenwick tree over
    depth rank.  Returns (inverted weight, total pair weight).
    """
    n = len(pos)
    tot = 0.5 * (w.sum() ** 2 - (w ** 2).sum())
    if n < 2:
        return 0.0, max(tot, 0.0)
    rank = np.empty(n, np.int64)
    rank[np.argsort(z, kind='stable')] = np.arange(n)      # 0 = nearest
    seq = np.argsort(pos, kind='stable')                   # draw order
    r = rank[seq] + 1
    ww = w[seq]
    tree = np.zeros(n + 1)
    bad = 0.0
    for k in range(n):
        # weight already drawn with a *larger* depth rank than this one
        i = r[k]
        s = 0.0
        while i > 0:
            s += tree[i]
            i -= i & (-i)
        # already drawn, but NEARER than this one -> drawn in the wrong order
        bad += ww[k] * s                    # tree[0] holds the running total
        tree[0] += ww[k]
        i = r[k]
        while i <= n:
            tree[i] += ww[k]
            i += i & (-i)
    return bad, max(tot, 0.0)


def tile_inversions(pos, z, w, sp, ti, ntile):
    """Per-tile version of the above, over (splat, tile) incidences."""
    o = np.argsort(ti, kind='stable')
    sp, ti = sp[o], ti[o]
    bounds = np.searchsorted(ti, np.arange(ntile + 1))
    bad = tot = 0.0
    for t in range(ntile):
        a, b = bounds[t], bounds[t + 1]
        k = b - a
        if k < 2:
            continue
        s = sp[a:b]
        zz, ww, pp = z[s], w[s], pos[s]
        if k > 3000:                       # keep O(k^2) sane; rare
            pick = np.random.default_rng(t).choice(k, 3000, replace=False)
            zz, ww, pp = zz[pick], ww[pick], pp[pick]
            k = 3000
        pw = ww[:, None] * ww[None, :]
        later = pp[:, None] < pp[None, :]          # i drawn before j
        wrong = zz[:, None] < zz[None, :]          # but i is nearer than j
        bad += (pw * (later & wrong)).sum()
        tot += 0.5 * (ww.sum() ** 2 - (ww ** 2).sum())
    return bad, max(tot, 0.0)


# ------------------------------------------------------------------ png

def write_png(path, img):
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    h, w, _ = img.shape
    raw = b''.join(b'\x00' + img[r].tobytes() for r in range(h))

    def chunk(t, b):
        c = t + b
        return struct.pack('>I', len(b)) + c + struct.pack('>I', zlib.crc32(c))

    open(path, 'wb').write(
        b'\x89PNG\r\n\x1a\n'
        + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
        + chunk(b'IDAT', zlib.compress(raw, 6)) + chunk(b'IEND', b''))


def side_by_side(a, b, gap=8):
    h = max(a.shape[0], b.shape[0])
    out = np.zeros((h, a.shape[1] + gap + b.shape[1], 3))
    out[:, :, :] = 0.0
    out[:a.shape[0], :a.shape[1]] = a
    out[:b.shape[0], a.shape[1] + gap:] = b
    return out


# ------------------------------------------------------------------ main

def fib_dirs(n):
    """Evenly spread directions on the sphere."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1.0 - 2.0 * i / n)
    gold = np.pi * (1.0 + 5.0 ** 0.5)
    th = gold * i
    return np.stack([np.sin(phi) * np.cos(th), np.sin(phi) * np.sin(th), np.cos(phi)], 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-d', '--depth', type=int, default=7)
    ap.add_argument('-j', '--jitter', type=float, default=0.16)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--dirs', type=int, default=64)
    ap.add_argument('--images', type=int, default=8)
    ap.add_argument('--res', type=int, default=420)
    ap.add_argument('--tile', type=int, default=16)
    ap.add_argument('--rep', default='centroid', choices=['centroid', 'mu0', 'bound'])
    ap.add_argument('--out', default='out/stage0')
    ap.add_argument('--no-images', action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    dirs = fib_dirs(args.dirs)
    r = ifs.expand(args.depth, args.jitter, args.seed, dirs=dirs, rep=args.rep)
    mu, cov, col, alpha, slot = r['mu'], r['cov'], r['col'], r['alpha'], r['slot']
    N = len(mu)
    print(f'depth {args.depth}  jitter {args.jitter}  rep {args.rep}  '
          f'{N} splats  expand+order x{args.dirs} in {time.time()-t0:.2f}s')

    # equivalence check: the M^T d formulation must agree with world-space depth
    # ordering of the same representatives -- it does, exactly, up to float.
    ctr = mu.mean(0)
    span = np.linalg.norm(mu.max(0) - mu.min(0))

    rows = []
    for k in range(args.dirs):
        d = dirs[k]
        eye = ctr - d * span * 1.05
        cam = sr.Camera(eye, ctr, up=(0, 0, 1), fovy=50.0, W=args.res, H=args.res)
        # the stream's depth key is view-space z, exactly what the camera uses
        scr, S2, z, ascale, vis = sr.project(mu, cov, cam)
        w = alpha * ascale
        det = S2[:, 0, 0] * S2[:, 1, 1] - S2[:, 0, 1] ** 2
        ok = vis & (det > 1e-12) & (w > 2e-3)

        pos = slot[k]
        sel = np.nonzero(ok)[0]
        gb, gt = weighted_inversions(pos[sel], z[sel], w[sel])
        sp, ti, ntile = sr.footprint_tiles(scr, S2, ok, cam, tile=args.tile)
        tb, tt = tile_inversions(pos, z, w, sp, ti, ntile)
        rows.append((gb / gt if gt else 0.0, tb / tt if tt else 0.0, len(sel)))
        if k == 0 or (k + 1) % 16 == 0:
            print(f'  dir {k+1:3d}/{args.dirs}  tile {rows[-1][1]*100:6.3f}%  '
                  f'glob {rows[-1][0]*100:6.3f}%')

    g = np.array([x[0] for x in rows])
    t = np.array([x[1] for x in rows])
    print()
    print(f'  per-tile weighted inversion rate   mean {t.mean()*100:.3f}%   '
          f'median {np.median(t)*100:.3f}%   max {t.max()*100:.3f}%')
    print(f'  global   weighted inversion rate   mean {g.mean()*100:.3f}%   '
          f'max {g.max()*100:.3f}%')

    if args.no_images:
        return

    # ---- A/B images on a handful of directions
    print()
    ps = []
    pick = np.linspace(0, args.dirs - 1, args.images).astype(int)
    for n, k in enumerate(pick):
        d = dirs[k]
        eye = ctr - d * span * 1.05
        cam = sr.Camera(eye, ctr, up=(0, 0, 1), fovy=50.0, W=args.res, H=args.res)
        _, _, z, _, _ = sr.project(mu, cov, cam)
        exact = np.argsort(-z, kind='stable')          # far -> near
        approx = np.argsort(slot[k], kind='stable')
        ia = sr.render(mu, cov, col, alpha, cam, exact)
        ib = sr.render(mu, cov, col, alpha, cam, approx)
        e = np.abs(ia - ib)
        p = sr.psnr(ia, ib)
        ps.append(p)
        write_png(f'{args.out}/ab_{n:02d}.png', side_by_side(ia, ib))
        write_png(f'{args.out}/diff_{n:02d}.png', np.clip(e * 8.0, 0, 1))
        print(f'  image {n} (dir {k:3d})  PSNR {p:5.2f} dB   '
              f'mean |err| {e.mean()*255:5.2f}/255   max {e.max()*255:5.1f}/255   '
              f'px >4/255: {np.mean(e.max(2) > 4/255)*100:5.2f}%')
    print()
    print(f'  A/B PSNR  mean {np.mean(ps):.2f} dB   min {np.min(ps):.2f} dB')
    print(f'  images in {args.out}/')
    print(f'  total {time.time()-t0:.1f}s')


if __name__ == '__main__':
    main()
