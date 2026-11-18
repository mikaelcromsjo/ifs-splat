#!/usr/bin/env python3
"""
Perspective Gaussian-splat rasteriser that blends in a *given* order.

This is the reference implementation of the projection the WebGL vertex
shader has to reproduce, so it is written to be transcribed:

  view        v  = Rv (mu - eye),  with Rv's rows = (right, up, forward)
  depth       z  = v.z                       -- the sort key, linear in mu
  3D dilation S3 = S3 + (kappa * z / f)^2 I  -- Mip-Splatting's 3D filter, with
                                                the pixel footprint at this z
  projection  S2 = J Rv S3 Rv^T J^T          -- J = d(screen)/d(view)
  2D low-pass S2 = S2 + 0.3 I,  alpha *= sqrt(det S2 / det S2')

Sorting by ``z`` (not by distance from the eye) is exactly what web 3DGS
viewers do, and z is an affine function of mu -- which is why the hierarchical
ordering in ifs_core is comparing the same quantity the ground truth does.
"""

import numpy as np

LOWPASS = 0.3       # EWA 2D low-pass variance, in pixels^2
KAPPA = 0.35        # 3D dilation, in pixels of footprint
CUT = 3.0           # Gaussian support, in sigmas


def look_at(eye, target, up=(0.0, 0.0, 1.0)):
    eye = np.asarray(eye, float)
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    up = np.asarray(up, float)
    r = np.cross(f, up)
    nr = np.linalg.norm(r)
    if nr < 1e-9:                                   # looking along up
        r = np.cross(f, np.array([1.0, 0.0, 0.0]))
        nr = np.linalg.norm(r)
    r /= nr
    u = np.cross(r, f)
    return eye, np.stack([r, u, f])                 # (3,3), rows


class Camera:
    def __init__(self, eye, target, up=(0, 0, 1), fovy=50.0, W=512, H=512, znear=1e-4):
        self.eye, self.Rv = look_at(eye, target, up)
        self.fovy, self.W, self.H, self.znear = fovy, W, H, znear
        self.f = (H / 2.0) / np.tan(np.radians(fovy) / 2.0)

    @property
    def fwd(self):
        return self.Rv[2]


def project(mu, cov, cam, dilate=True, lowpass=True):
    """-> scr (N,2), S2 (N,2,2), z (N,), alpha_scale (N,), vis (N,) bool"""
    v = (mu - cam.eye) @ cam.Rv.T
    z = v[:, 2]
    vis = z > cam.znear
    zs = np.where(vis, z, 1.0)

    S3 = cov
    if dilate:
        # one pixel subtends zs/f in world units at this depth
        d2 = (KAPPA * zs / cam.f) ** 2
        S3 = S3 + d2[:, None, None] * np.eye(3)

    f = cam.f
    J = np.zeros((len(mu), 2, 3))
    J[:, 0, 0] = f / zs
    J[:, 0, 2] = -f * v[:, 0] / zs ** 2
    J[:, 1, 1] = -f / zs
    J[:, 1, 2] = f * v[:, 1] / zs ** 2
    W = J @ cam.Rv                                  # (N,2,3)
    S2 = np.einsum('nij,njk,nlk->nil', W, S3, W)

    a = np.ones(len(mu))
    if lowpass:
        det0 = S2[:, 0, 0] * S2[:, 1, 1] - S2[:, 0, 1] ** 2
        S2 = S2 + LOWPASS * np.eye(2)
        det1 = S2[:, 0, 0] * S2[:, 1, 1] - S2[:, 0, 1] ** 2
        a = np.sqrt(np.clip(np.maximum(det0, 0.0) / np.maximum(det1, 1e-20), 0.0, 1.0))

    scr = np.stack([f * v[:, 0] / zs + cam.W / 2.0,
                    cam.H / 2.0 - f * v[:, 1] / zs], 1)
    return scr, S2, z, a, vis


def render(mu, cov, col, alpha, cam, order, bg=(0.045, 0.05, 0.065),
           gamma=1.1, max_radius=400, visibility=False, **kw):
    """Blend ``order`` (an array of splat indices, back to front) over ``bg``.

    With ``visibility=True`` also returns, per splat, the mean transmittance
    standing between it and the camera, weighted by its own kernel: how much of
    this splat the viewer actually sees.  Deep inside a saturated canopy that is
    near zero, which is why order errors there cost nothing.
    """
    scr, S2, z, ascale, vis = project(mu, cov, cam, **kw)
    al = alpha * ascale
    det = S2[:, 0, 0] * S2[:, 1, 1] - S2[:, 0, 1] ** 2
    ok = vis & (det > 1e-12) & (al > 2e-3)
    inv00 = np.where(ok, S2[:, 1, 1] / np.where(det > 0, det, 1), 0.0)
    inv11 = np.where(ok, S2[:, 0, 0] / np.where(det > 0, det, 1), 0.0)
    inv01 = np.where(ok, -S2[:, 0, 1] / np.where(det > 0, det, 1), 0.0)
    rad = np.ceil(CUT * np.sqrt(np.maximum(np.stack([S2[:, 0, 0], S2[:, 1, 1]], 1), 0.0)))
    ok &= rad.max(1) <= max_radius
    ok &= (scr[:, 0] + rad[:, 0] >= 0) & (scr[:, 0] - rad[:, 0] < cam.W)
    ok &= (scr[:, 1] + rad[:, 1] >= 0) & (scr[:, 1] - rad[:, 1] < cam.H)

    H, W = cam.H, cam.W
    buf = np.empty((H, W, 3))
    buf[:] = np.asarray(bg, float)
    Tr = np.ones((H, W)) if visibility else None
    vis_w = np.zeros(len(mu)) if visibility else None
    radi = rad.astype(np.int32)
    sx, sy = scr[:, 0], scr[:, 1]

    for i in order:
        if not ok[i]:
            continue
        x0 = max(int(np.floor(sx[i])) - radi[i, 0], 0)
        x1 = min(int(np.floor(sx[i])) + radi[i, 0] + 1, W)
        y0 = max(int(np.floor(sy[i])) - radi[i, 1], 0)
        y1 = min(int(np.floor(sy[i])) + radi[i, 1] + 1, H)
        if x1 <= x0 or y1 <= y0:
            continue
        dx = np.arange(x0, x1) + 0.5 - sx[i]
        dy = np.arange(y0, y1) + 0.5 - sy[i]
        DX = dx[None, :]
        DY = dy[:, None]
        e = inv00[i] * DX * DX + 2.0 * inv01[i] * DX * DY + inv11[i] * DY * DY
        a2 = al[i] * np.exp(-0.5 * e)
        if visibility:
            t = Tr[y0:y1, x0:x1]
            sw = a2.sum()
            vis_w[i] = float((a2 * t).sum() / sw) if sw > 1e-12 else 0.0
            Tr[y0:y1, x0:x1] = t * (1.0 - a2)
        a = a2[..., None]
        reg = buf[y0:y1, x0:x1]
        buf[y0:y1, x0:x1] = reg * (1.0 - a) + col[i] * a

    img = np.clip(buf, 0.0, 1.0) ** (1.0 / gamma)
    return (img, vis_w) if visibility else img


def footprint_tiles(scr, S2, ok, cam, tile=16):
    """Which tiles each splat's 2-sigma footprint touches.  -> (splat, tile) pairs."""
    nx = (cam.W + tile - 1) // tile
    ny = (cam.H + tile - 1) // tile
    rad = 2.0 * np.sqrt(np.maximum(np.stack([S2[:, 0, 0], S2[:, 1, 1]], 1), 0.0))
    tx0 = np.clip(np.floor((scr[:, 0] - rad[:, 0]) / tile), 0, nx - 1).astype(np.int32)
    tx1 = np.clip(np.floor((scr[:, 0] + rad[:, 0]) / tile), 0, nx - 1).astype(np.int32)
    ty0 = np.clip(np.floor((scr[:, 1] - rad[:, 1]) / tile), 0, ny - 1).astype(np.int32)
    ty1 = np.clip(np.floor((scr[:, 1] + rad[:, 1]) / tile), 0, ny - 1).astype(np.int32)
    inb = ok & (scr[:, 0] + rad[:, 0] >= 0) & (scr[:, 0] - rad[:, 0] < cam.W) \
             & (scr[:, 1] + rad[:, 1] >= 0) & (scr[:, 1] - rad[:, 1] < cam.H)

    sp, ti = [], []
    idx = np.nonzero(inb)[0]
    w = (tx1 - tx0 + 1)[idx]
    h = (ty1 - ty0 + 1)[idx]
    # group by footprint size so the expansion stays vectorised
    for (ww, hh) in set(zip(w.tolist(), h.tolist())):
        sel = idx[(w == ww) & (h == hh)]
        if not len(sel):
            continue
        ox = np.arange(ww)
        oy = np.arange(hh)
        TX = tx0[sel][:, None, None] + ox[None, None, :]
        TY = ty0[sel][:, None, None] + oy[None, :, None]
        T = (TY * nx + TX).reshape(len(sel), -1)
        sp.append(np.repeat(sel, T.shape[1]))
        ti.append(T.ravel())
    if not sp:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), nx * ny
    return np.concatenate(sp), np.concatenate(ti), nx * ny


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse <= 0 else 10.0 * np.log10(1.0 / mse)
