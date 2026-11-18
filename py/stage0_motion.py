#!/usr/bin/env python3
"""
Stage 0, the test that actually decides it: does the approximate order look
wrong *in motion*?

A still A/B is not enough.  A constant bias between the two renders is
invisible -- nobody sees the exact-sort image to compare against.  What a
viewer sees is *change*: if the ordering error stays put as the camera moves
it vanishes into the picture, and if it flickers it pops.

So, over an orbit, with  E_t = I_exact,t - I_ordered,t :

    bias     mean |E_t|              how wrong a frame is (invisible on its own)
    flicker  mean |E_t - E_{t-1}|    how much that wrongness moves (what pops)

and for scale, the same quantity for a *correct* renderer:

    motion   mean |I_exact,t - I_exact,t-1|   the frame-to-frame change the
                                              camera itself produces

If flicker is far below motion, the ordering error is hidden inside the
camera's own movement and nobody can see it.  That is the number this stage
exists to produce.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ifs_core as ifs
import splat_raster as sr
from stage0 import write_png, side_by_side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--depth', type=int, default=7)
    ap.add_argument('-j', '--jitter', type=float, default=0.16)
    ap.add_argument('--seed', type=int, default=3)
    ap.add_argument('--frames', type=int, default=24)
    ap.add_argument('--deg', type=float, default=0.5,
                    help='camera rotation per frame (0.5 deg at 60fps = 30 deg/s)')
    ap.add_argument('--res', type=int, default=360)
    ap.add_argument('--out', default='out/stage0')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    F = args.frames
    th = np.radians(np.arange(F) * args.deg)
    el = np.radians(18.0)
    dirs = np.stack([np.cos(th) * np.cos(el), np.sin(th) * np.cos(el),
                     np.full_like(th, -np.sin(el))], 1)

    r = ifs.expand(args.depth, args.jitter, args.seed, dirs=dirs)
    mu, cov, col, alpha = r['mu'], r['cov'], r['col'], r['alpha']
    ctr = mu.mean(0)
    span = np.linalg.norm(mu.max(0) - mu.min(0))
    print(f'{len(mu)} splats, {F} frames at {args.deg} deg/frame, {args.res}px\n')

    Ia, Ib = [], []
    for k in range(F):
        cam = sr.Camera(ctr - dirs[k] * span * 1.05, ctr, up=(0, 0, 1),
                        fovy=50.0, W=args.res, H=args.res)
        z = sr.project(mu, cov, cam)[2]
        Ia.append(sr.render(mu, cov, col, alpha, cam, np.argsort(-z, kind='stable')))
        Ib.append(sr.render(mu, cov, col, alpha, cam,
                            np.argsort(r['slot'][k], kind='stable')))
    Ia = np.array(Ia); Ib = np.array(Ib)
    E = Ia - Ib

    bias = np.abs(E).mean(axis=(1, 2, 3)) * 255
    flick = np.abs(np.diff(E, axis=0)).mean(axis=(1, 2, 3)) * 255
    motion = np.abs(np.diff(Ia, axis=0)).mean(axis=(1, 2, 3)) * 255
    fl_o = np.abs(np.diff(Ib, axis=0)).mean(axis=(1, 2, 3)) * 255

    # worst-pixel versions: a pop is local and rare, so both the spatial mean
    # and the average over frames can hide it.  Take the worst frame's worst
    # pixels, not the average frame's.
    dE = np.abs(np.diff(E, axis=0))
    dI = np.abs(np.diff(Ia, axis=0))
    flick_p99 = np.array([np.percentile(x, 99.9) for x in dE]) * 255
    motion_p99 = np.array([np.percentile(x, 99.9) for x in dI]) * 255
    worst = int(np.argmax(flick_p99))

    print(f'  bias     mean |I_exact - I_ordered|          '
          f'{bias.mean():6.3f}/255   (max {bias.max():.3f})')
    print(f'  flicker  mean |E_t - E_(t-1)|                '
          f'{flick.mean():6.3f}/255   (max {flick.max():.3f})')
    print(f'  motion   mean |I_exact,t - I_exact,t-1|      '
          f'{motion.mean():6.3f}/255')
    print(f'  ratio    flicker / motion                    '
          f'{flick.mean()/max(motion.mean(),1e-9):6.3f}')
    print()
    print(f'  frame-to-frame change, exact order    {motion.mean():6.3f}/255')
    print(f'  frame-to-frame change, ordered expn   {fl_o.mean():6.3f}/255'
          f'   ({fl_o.mean()/max(motion.mean(),1e-9):.3f}x)')
    print()
    print(f'  worst 0.1% of pixels, averaged over frames: '
          f'flicker {flick_p99.mean():5.2f}/255   camera motion {motion_p99.mean():5.2f}/255')
    print(f'  worst 0.1% of pixels, in the WORST frame ({worst}): '
          f'flicker {flick_p99[worst]:5.2f}/255   camera motion {motion_p99[worst]:5.2f}/255')
    print(f'  single worst pixel change from ordering: '
          f'{dE.max()*255:5.1f}/255   from the camera: {dI.max()*255:5.1f}/255')

    # a filmstrip of the error over time, amplified, so a pop is visible
    k = min(6, F - 1)
    sel = [max(0, min(F - 2, worst - 2 + i)) for i in range(k)]
    strip = np.concatenate([np.clip(np.abs(E[i]) * 8.0, 0, 1) for i in sel], 1)
    write_png(f'{args.out}/motion_err.png', strip)
    strip2 = np.concatenate([np.clip(dE[i] * 24.0, 0, 1) for i in sel], 1)
    write_png(f'{args.out}/motion_flicker.png', strip2)
    print(f'\n  {args.out}/motion_err.png      error per frame, x8')
    print(f'  {args.out}/motion_flicker.png  change in the error, x24')


if __name__ == '__main__':
    main()
