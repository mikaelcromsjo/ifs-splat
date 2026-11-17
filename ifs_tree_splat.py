#!/usr/bin/env python3
"""
IFS tree -> 3D Gaussian splats, with no skeleton.

An affine map applied to a Gaussian is a Gaussian, in closed form:

    w(x) = A x + b      =>      N(mu, S)  ->  N(A mu + b,  A S A^T)

so the IFS acts directly on splats instead of on sampled points.  We
recurse on the branch maps; at every node we emit one "wood" splat =
the trunk map applied to that node's Gaussian.  The trunk map squashes
a whole tree copy into a single thin rod, so it is terminal: recursing
into it would only re-tile the same rod.  That one rod per node is the
branch segment -- the skeleton's geometry falls out of the maps.

Foliage is the recursion's leaf Gaussians, reshaped into discs.

Output: standard INRIA-format 3DGS .ply (spherical harmonics degree 0),
readable by SuperSplat, antimatter15/splat, Nerfstudio, Postshot, etc.
"""

import argparse
import math
import struct
import numpy as np

C0 = 0.28209479177387814   # SH degree-0 basis constant

# ---------------------------------------------------------------- the IFS

S_BR = 0.60    # branch contraction
H_ATT = 1.0    # height at which children attach, in the parent's frame


def rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def branch(tilt, roll, s, twist):
    """Uniform contraction + tilt + roll, attached at z = H_ATT.

    Uniform s means S -> s^2 R S R^T: the initial Gaussian's elongation
    is rotated onto each branch's own axis, so anisotropy propagates for
    free and no map is rank-deficient.
    """
    A = s * (rz(math.radians(roll)) @ ry(math.radians(tilt)) @ rz(math.radians(twist)))
    return A, np.array([0.0, 0.0, H_ATT])


BRANCH = [
    branch(38.0, 0.0, S_BR, 25.0),
    branch(38.0, 120.0, S_BR, 25.0),
    branch(38.0, 240.0, S_BR, 25.0),
    branch(9.0, 60.0, 0.52, 40.0),      # leader shoot: keeps it tree-like, not a 3-fold star
]

# Trunk map: squash the whole tree into a vertical rod.  The z factor is
# 1 - S_BR exactly, which puts the rod's top at the children's attach
# height, so branches emerge from the end of their parent segment.
TRUNK_A = np.diag([0.075, 0.075, 1.0 - S_BR])
TRUNK_B = np.zeros(3)

HEIGHT = H_ATT / (1.0 - S_BR)                      # attractor height, = 2.5
MU0 = np.array([0.0, 0.0, HEIGHT / 2.0])
SIG0 = np.diag([0.45 ** 2, 0.45 ** 2, (HEIGHT / 4.0) ** 2])

# --------------------------------------------------------------- machinery


def gauss(A, b):
    """The Gaussian a node's accumulated affine carries: w(N(MU0, SIG0))."""
    return (np.einsum('nij,j->ni', A, MU0) + b,
            np.einsum('nij,jk,nlk->nil', A, SIG0, A))


def compose(A, b, Ai, bi):
    """w_path . w_i  -- the new map is applied INSIDE the node's own frame."""
    return A @ Ai, np.einsum('nij,j->ni', A, bi) + b


def rand_rots(n, amt, rng):
    """Per-splat small random rotations (Rodrigues), or None if amt == 0."""
    if amt <= 0.0:
        return None
    ax = rng.normal(size=(n, 3))
    ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    ang = rng.normal(scale=amt, size=n)
    K = np.zeros((n, 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -ax[:, 2], ax[:, 1]
    K[:, 1, 0], K[:, 1, 2] = ax[:, 2], -ax[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -ax[:, 1], ax[:, 0]
    I = np.broadcast_to(np.eye(3), (n, 3, 3))
    return I + np.sin(ang)[:, None, None] * K + (1 - np.cos(ang))[:, None, None] * (K @ K)


def decompose(S, floor):
    """Covariance -> (std devs ascending, proper rotation matrix)."""
    S = 0.5 * (S + np.transpose(S, (0, 2, 1)))
    w, V = np.linalg.eigh(S)
    scales = np.sqrt(np.maximum(w, floor ** 2))
    flip = np.linalg.det(V) < 0.0
    V[flip, :, 0] *= -1.0                 # make it a rotation, not a reflection
    return scales, V


def quats(R):
    """Batched rotation matrix -> quaternion (w, x, y, z), Shepperd's method."""
    m = lambda i, j: R[:, i, j]
    t = m(0, 0) + m(1, 1) + m(2, 2)
    cand = []
    r = np.sqrt(np.maximum(1.0 + t, 1e-12)); s = 0.5 / r
    cand.append(np.stack([0.5 * r, (m(2, 1) - m(1, 2)) * s,
                          (m(0, 2) - m(2, 0)) * s, (m(1, 0) - m(0, 1)) * s], 1))
    r = np.sqrt(np.maximum(1.0 + m(0, 0) - m(1, 1) - m(2, 2), 1e-12)); s = 0.5 / r
    cand.append(np.stack([(m(2, 1) - m(1, 2)) * s, 0.5 * r,
                          (m(0, 1) + m(1, 0)) * s, (m(0, 2) + m(2, 0)) * s], 1))
    r = np.sqrt(np.maximum(1.0 - m(0, 0) + m(1, 1) - m(2, 2), 1e-12)); s = 0.5 / r
    cand.append(np.stack([(m(0, 2) - m(2, 0)) * s, (m(0, 1) + m(1, 0)) * s,
                          0.5 * r, (m(1, 2) + m(2, 1)) * s], 1))
    r = np.sqrt(np.maximum(1.0 - m(0, 0) - m(1, 1) + m(2, 2), 1e-12)); s = 0.5 / r
    cand.append(np.stack([(m(1, 0) - m(0, 1)) * s, (m(0, 2) + m(2, 0)) * s,
                          (m(1, 2) + m(2, 1)) * s, 0.5 * r], 1))
    which = np.where(t > 0.0, 0, 1 + np.argmax(
        np.stack([m(0, 0), m(1, 1), m(2, 2)], 1), axis=1))
    q = np.take_along_axis(np.stack(cand, 1), which[:, None, None], 1)[:, 0, :]
    return q / np.linalg.norm(q, axis=1, keepdims=True)


UP = {'z': np.eye(3),
      'y': np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]),
      '-y': np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])}

# -------------------------------------------------------------------- build


def build(depth, jitter, seed):
    rng = np.random.default_rng(seed)
    n = 1
    A = np.eye(3)[None, :, :]
    b = np.zeros((1, 3))
    wood = []

    for d in range(depth + 1):
        wood.append((d,) + gauss(*compose(A, b, TRUNK_A, TRUNK_B)))
        if d == depth:
            break
        nA, nb = [], []
        for Ai, bi in BRANCH:
            R = rand_rots(len(A), jitter, rng)
            A2, b2 = compose(A, b, Ai if R is None else R @ Ai, bi)
            nA.append(A2); nb.append(b2)
        A, b = np.concatenate(nA), np.concatenate(nb)

    lmu, lS = gauss(A, b)
    return wood, lmu, lS


def assemble(wood, leaf_mu, leaf_S, depth, floor, rng):
    mus, scales, rots, cols, alphas = [], [], [], [], []

    # ---- wood: one rod per node, browner at the base, greyer at the twigs
    for d, wmu, wS in wood:
        s, V = decompose(wS, floor)
        f = d / max(depth, 1)
        c = (1 - f) * np.array([0.26, 0.17, 0.10]) + f * np.array([0.47, 0.38, 0.28])
        n = len(wmu)
        mus.append(wmu); scales.append(s); rots.append(V)
        cols.append(np.broadcast_to(c, (n, 3)) *
                    (1.0 + rng.normal(scale=0.06, size=(n, 1))))
        alphas.append(np.full(n, 0.99))

    # ---- foliage: leaf Gaussians reshaped into discs that fan out
    s, V = decompose(leaf_S, floor)
    g = np.prod(s, axis=1) ** (1.0 / 3.0)
    disc = np.stack([1.9 * g, 1.9 * g, 0.30 * g], 1)   # flat along the branch axis
    n = len(leaf_mu)
    base = np.array([0.17, 0.41, 0.13])
    c = base * (1.0 + rng.normal(scale=0.22, size=(n, 1)))
    c[:, 1] *= 1.0 + rng.normal(scale=0.10, size=n)
    mus.append(leaf_mu); scales.append(disc); rots.append(V)
    cols.append(c); alphas.append(np.full(n, 0.55))

    return (np.concatenate(mus), np.concatenate(scales), np.concatenate(rots),
            np.clip(np.concatenate(cols), 0.0, 1.0), np.concatenate(alphas))


def write_ply(path, mu, scale, R, col, alpha):
    n = len(mu)
    q = quats(R)
    data = np.zeros((n, 17), dtype=np.float32)
    data[:, 0:3] = mu
    data[:, 3:6] = 0.0                                  # unused normals
    data[:, 6:9] = (col - 0.5) / C0                     # SH DC
    a = np.clip(alpha, 1e-4, 1 - 1e-4)
    data[:, 9] = np.log(a / (1 - a))                    # opacity is stored as a logit
    data[:, 10:13] = np.log(scale)                      # scales as log std-dev
    data[:, 13:17] = q                                  # (w, x, y, z)

    names = (['x', 'y', 'z', 'nx', 'ny', 'nz'] + [f'f_dc_{i}' for i in range(3)] +
             ['opacity'] + [f'scale_{i}' for i in range(3)] + [f'rot_{i}' for i in range(4)])
    header = ('ply\nformat binary_little_endian 1.0\n'
              f'element vertex {n}\n' +
              ''.join(f'property float {k}\n' for k in names) + 'end_header\n')
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(np.ascontiguousarray(data).tobytes())
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('-o', '--out', default='ifs_tree.ply')
    ap.add_argument('-d', '--depth', type=int, default=7,
                    help='branch recursion depth; splats ~ 5/3 * 4^depth (default 7)')
    ap.add_argument('-j', '--jitter', type=float, default=0.0,
                    help='radians of random rotation per map application. 0 = the exact '
                         'self-similar attractor; ~0.15 looks organic (default 0)')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--up', choices=sorted(UP), default='-y',
                    help="viewer up-axis; try another if the tree loads sideways (default -y)")
    ap.add_argument('--scale', type=float, default=1.0, help='overall size multiplier')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed + 1)
    floor = 2e-4 * HEIGHT                      # sub-pixel splats would vanish or alias
    wood, leaf_mu, leaf_S = build(args.depth, args.jitter, args.seed)
    mu, scale, R, col, alpha = assemble(wood, leaf_mu, leaf_S, args.depth, floor, rng)

    # recentre in the model's own z-up frame, then rotate into the viewer's
    mu[:, 0] -= mu[:, 0].mean()
    mu[:, 1] -= mu[:, 1].mean()
    mu[:, 2] -= mu[:, 2].min()
    U = UP[args.up]
    mu = (mu @ U.T) * args.scale
    R = U[None, :, :] @ R
    scale = scale * args.scale

    n = write_ply(args.out, mu, scale, R, col, alpha)
    lo, hi = mu.min(0), mu.max(0)
    print(f'{n} splats -> {args.out}')
    print(f'  wood {sum(len(w[1]) for w in wood)}   foliage {len(leaf_mu)}')
    print(f'  bbox  min {np.round(lo, 3)}  max {np.round(hi, 3)}')
    print(f'  scale range {scale.min():.2e} .. {scale.max():.2e}')


if __name__ == '__main__':
    main()
