#!/usr/bin/env python3
"""
Stage 0, the diagnostics that say whether 15% inversions matter.

Three things the raw inversion rate cannot tell you:

1.  **What a swap actually costs.**  For two splats over a background, swapping
    their order changes the pixel by exactly

        a_i * a_j * |c_i - c_j|

    (expand both orderings of the over-operator; everything else cancels).  So
    inversions between splats of the same colour are free, and in a tree most
    co-located pairs are two leaves of the same green or two rods of the same
    brown.  ``contrast`` below weights by this, which makes it a prediction of
    the image error in units of 1/255 -- checkable against the measured PSNR.

2.  **Where in the hierarchy they come from.**  An inversion between two
    splats whose addresses diverge at level 1 means two whole limbs were
    ordered wrongly; at level 7 it means two twigs of the same branchlet
    swapped, which is invisible and would be unstable under an exact sort too.

3.  **Temporal churn.**  3DGS popping comes from order *changing* between
    frames, not from order being wrong.  ``churn`` counts how many co-located
    pairs swap places from one frame to the next, for the exact sort and for
    ordered expansion.  IDEAS.md claims hierarchical order is much more stable;
    this is that claim as a number.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ifs_core as ifs
import splat_raster as sr
from stage0 import fib_dirs, write_png, side_by_side


def tile_pairs(scr, S2, ok, cam, tile, cap):
    """Iterate (indices,) of co-located splat groups, one per screen tile."""
    sp, ti, ntile = sr.footprint_tiles(scr, S2, ok, cam, tile=tile)
    o = np.argsort(ti, kind='stable')
    sp, ti = sp[o], ti[o]
    bounds = np.searchsorted(ti, np.arange(ntile + 1))
    rng = np.random.default_rng(12345)
    for t in range(ntile):
        a, b = bounds[t], bounds[t + 1]
        if b - a < 2:
            continue
        s = sp[a:b]
        if len(s) > cap:
            s = s[rng.choice(len(s), cap, replace=False)]
        yield s


def analyse(mu, cov, col, alpha, addr, level, cam, pos, tile=16, cap=600, vis_w=None):
    scr, S2, z, ascale, vis = sr.project(mu, cov, cam)
    w = alpha * ascale
    det = S2[:, 0, 0] * S2[:, 1, 1] - S2[:, 0, 1] ** 2
    ok = vis & (det > 1e-12) & (w > 2e-3)
    sig = np.sqrt(np.maximum(S2[:, 0, 0] + S2[:, 1, 1], 1e-9))      # screen extent

    nlev = addr.shape[1] + 1
    op_bad = op_tot = 0.0
    ct_bad = ct_tot = 0.0
    vs_bad = vs_tot = 0.0
    by_lev = np.zeros(nlev)
    gap_hist = np.zeros(4)      # |dz| vs the pair's own depth extent: <.25,<1,<4,>=4

    for s in tile_pairs(scr, S2, ok, cam, tile, cap):
        ww, zz, pp, cc = w[s], z[s], pos[s], col[s]
        pw = ww[:, None] * ww[None, :]
        dc = np.abs(cc[:, None, :] - cc[None, :, :]).mean(2)
        cw = pw * dc
        wrong = (pp[:, None] < pp[None, :]) & (zz[:, None] < zz[None, :])
        tri = np.triu(np.ones_like(wrong), 1) > 0
        allp = tri | tri.T
        op_bad += pw[wrong].sum(); op_tot += pw[allp].sum() * 0.5
        ct_bad += cw[wrong].sum(); ct_tot += cw[allp].sum() * 0.5
        if vis_w is not None:
            # attenuate by the transmittance standing in front of the nearer
            # of the two: a swap behind a saturated canopy changes no pixel
            v = vis_w[s]
            vw = cw * np.minimum(v[:, None], v[None, :])
            vs_bad += vw[wrong].sum(); vs_tot += vw[allp].sum() * 0.5

        # which level do the two addresses diverge at?
        A = addr[s]
        eq = (A[:, None, :] == A[None, :, :]) & (A[:, None, :] >= 0)
        div = np.argmin(np.concatenate([eq, np.zeros_like(eq[:, :, :1])], 2), 2)
        bl = np.bincount(div[wrong].ravel(), weights=pw[wrong].ravel(), minlength=nlev)
        by_lev[:len(bl)] += bl[:nlev]

        # how far apart in depth, relative to the splats' own size?
        # a splat of screen extent sig at depth z spans sig*z/f in world depth
        dz = np.abs(zz[:, None] - zz[None, :])
        wext = sig[s] * zz / cam.f
        ext = wext[:, None] + wext[None, :]
        rel = dz / np.maximum(ext, 1e-12)
        for k, (lo, hi) in enumerate([(0, .25), (.25, 1), (1, 4), (4, np.inf)]):
            gap_hist[k] += pw[wrong & (rel >= lo) & (rel < hi)].sum()

    return dict(op=op_bad / max(op_tot, 1e-30), ct=ct_bad / max(ct_tot, 1e-30),
                vs=vs_bad / max(vs_tot, 1e-30),
                ct_abs=ct_bad, by_lev=by_lev / max(by_lev.sum(), 1e-30),
                gap=gap_hist / max(gap_hist.sum(), 1e-30))


def churn(mu, cov, alpha, cam_a, cam_b, slot_a, slot_b, tile=16, cap=600):
    """Co-located pairs that swap drawing order between two frames, for the
    exact sort and for ordered expansion.  Lower is less popping."""
    out = {}
    scr, S2, za, asc, vis = sr.project(mu, cov, cam_a)
    w = alpha * asc
    det = S2[:, 0, 0] * S2[:, 1, 1] - S2[:, 0, 1] ** 2
    ok = vis & (det > 1e-12) & (w > 2e-3)
    _, _, zb, _, visb = sr.project(mu, cov, cam_b)
    ok &= visb

    ea = np.empty(len(mu), np.int64); ea[np.argsort(-za, kind='stable')] = np.arange(len(mu))
    eb = np.empty(len(mu), np.int64); eb[np.argsort(-zb, kind='stable')] = np.arange(len(mu))

    for name, pa, pb in (('exact', ea, eb), ('ordered', slot_a, slot_b)):
        bad = tot = 0.0
        for s in tile_pairs(scr, S2, ok, cam_a, tile, cap):
            ww = w[s]
            pw = ww[:, None] * ww[None, :]
            tri = np.triu(np.ones((len(s), len(s)), bool), 1)
            fa = pa[s][:, None] < pa[s][None, :]
            fb = pb[s][:, None] < pb[s][None, :]
            bad += pw[tri & (fa != fb)].sum()
            tot += pw[tri].sum()
        out[name] = bad / max(tot, 1e-30)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--depth', type=int, default=7)
    ap.add_argument('-j', '--jitter', type=float, default=0.16)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--dirs', type=int, default=12)
    ap.add_argument('--res', type=int, default=420)
    ap.add_argument('--orbit', type=int, default=24, help='frames in the churn orbit')
    ap.add_argument('--fps-deg', type=float, default=0.5,
                    help='camera rotation per frame, degrees (60fps, 30 deg/s)')
    ap.add_argument('--out', default='out/stage0')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dirs = fib_dirs(args.dirs)
    r = ifs.expand(args.depth, args.jitter, args.seed, dirs=dirs)
    mu, cov, col, alpha = r['mu'], r['cov'], r['col'], r['alpha']
    ctr = mu.mean(0)
    span = np.linalg.norm(mu.max(0) - mu.min(0))
    print(f'{len(mu)} splats, depth {args.depth}\n')

    print('  dir   opacity-wtd   contrast-wtd   visible-wtd   PSNR     '
          'where the two addresses diverge')
    accs = []
    for k in range(args.dirs):
        cam = sr.Camera(ctr - dirs[k] * span * 1.05, ctr, up=(0, 0, 1),
                        fovy=50.0, W=args.res, H=args.res)
        exact = np.argsort(-sr.project(mu, cov, cam)[2], kind='stable')
        ia, vw = sr.render(mu, cov, col, alpha, cam, exact, visibility=True)
        ib = sr.render(mu, cov, col, alpha, cam,
                       np.argsort(r['slot'][k], kind='stable'))
        a = analyse(mu, cov, col, alpha, r['addr'], r['level'], cam, r['slot'][k],
                    vis_w=vw)
        a['psnr'] = sr.psnr(ia, ib)
        accs.append(a)
        lev = a['by_lev']
        top = ' '.join(f'L{i}:{lev[i]*100:.0f}%' for i in np.argsort(-lev)[:3])
        print(f'  {k:3d}   {a["op"]*100:8.2f}%   {a["ct"]*100:9.2f}%   '
              f'{a["vs"]*100:8.2f}%   {a["psnr"]:5.1f}dB   {top}')

    op = np.array([a['op'] for a in accs]); ct = np.array([a['ct'] for a in accs])
    lev = np.mean([a['by_lev'] for a in accs], 0)
    gap = np.mean([a['gap'] for a in accs], 0)
    print(f'\n  opacity-weighted  mean {op.mean()*100:.2f}%  max {op.max()*100:.2f}%')
    vs = np.array([a['vs'] for a in accs]); ps = np.array([a['psnr'] for a in accs])
    print(f'  contrast-weighted mean {ct.mean()*100:.2f}%  max {ct.max()*100:.2f}%')
    print(f'  visible-weighted  mean {vs.mean()*100:.2f}%  max {vs.max()*100:.2f}%'
          '   <- what reaches the image')
    print(f'  A/B PSNR          mean {ps.mean():.1f} dB  min {ps.min():.1f} dB')
    print('  inversions by address-divergence level: '
          + ' '.join(f'L{i} {lev[i]*100:.1f}%' for i in range(len(lev)) if lev[i] > 5e-3))
    print('  depth gap of inverted pairs, in units of the pair\'s own extent: '
          f'<0.25x {gap[0]*100:.1f}%  <1x {gap[1]*100:.1f}%  '
          f'<4x {gap[2]*100:.1f}%  >=4x {gap[3]*100:.1f}%')

    # ---- temporal churn over a slow orbit
    print(f'\n  churn over {args.orbit} frames at {args.fps_deg} deg/frame '
          '(co-located pairs that swap order between consecutive frames)')
    th = np.radians(np.arange(args.orbit + 1) * args.fps_deg)
    el = np.radians(18.0)
    od = np.stack([np.cos(th) * np.cos(el), np.sin(th) * np.cos(el),
                   np.full_like(th, -np.sin(el))], 1)
    ro = ifs.expand(args.depth, args.jitter, args.seed, dirs=od)
    ex, ordr = [], []
    for i in range(args.orbit):
        ca = sr.Camera(ctr - od[i] * span * 1.05, ctr, up=(0, 0, 1), fovy=50.0,
                       W=args.res, H=args.res)
        cb = sr.Camera(ctr - od[i + 1] * span * 1.05, ctr, up=(0, 0, 1), fovy=50.0,
                       W=args.res, H=args.res)
        c = churn(mu, cov, alpha, ca, cb, ro['slot'][i], ro['slot'][i + 1])
        ex.append(c['exact']); ordr.append(c['ordered'])
    ex, ordr = np.array(ex), np.array(ordr)
    print(f'    exact per-splat sort   {ex.mean()*100:.3f}%  (max {ex.max()*100:.3f}%)')
    print(f'    ordered expansion      {ordr.mean()*100:.3f}%  (max {ordr.max()*100:.3f}%)')
    print(f'    ordered expansion is {ex.mean()/max(ordr.mean(),1e-12):.1f}x more stable')


if __name__ == '__main__':
    main()
