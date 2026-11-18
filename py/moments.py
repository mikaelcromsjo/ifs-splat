#!/usr/bin/env python3
"""
Moment-matched parents, in closed form.

Under adaptive LOD a node that is too small on screen is drawn as one Gaussian
standing for its entire infinite subtree.  For that blob to be a *correct*
rendering rather than a guess, it has to match the subtree's mass, mean and
covariance -- and because the subtree is the attractor of the same maps, those
moments are the solution of a fixed-point equation in the maps, not something
to be fitted.

Give each node's own rod a share ``omega`` of its subtree's mass and split the
rest equally between the m children.  Then, in the node's own frame:

    C = w*mu_w + ((1-w)/m) * SUM_i (A_i C + b_i)                      (mean)
    Q = w*(S_w + (mu_w - C)(mu_w - C)^T)
      + ((1-w)/m) * SUM_i [ A_i Q A_i^T + d_i d_i^T ],  d_i = A_i C + b_i - C

The first is a 3x3 linear solve.  The second is a contraction (its factor is
((1-w)/m) * SUM ||A_i||^2 ~ 0.26 here), so a few dozen iterations reach float
precision.  Both are computed once per map set, never per node: a node's blob
is just ``M C + b`` and ``M Q M^T``.

The same recursion gives a bounding sphere, which the LOD cut and the frustum
test both need, and which costs nothing because every branch map here is a
uniform contraction -- so a node's affine is always (scalar * rotation) and its
world radius is one multiply.
"""

import numpy as np

import ifs_core as ifs


def rod_gauss():
    """The wood rod a node emits, in the node's own frame."""
    Aw = ifs.TRUNK_A
    return Aw @ ifs.MU0, Aw @ ifs.SIG0 @ Aw.T


def invariant_moments(omega=0.28, iters=200, maps=None):
    """-> dict(C, Q, R, omega): mean, covariance and bounding radius of a
    node's whole subtree, in the node's own frame."""
    maps = maps or list(zip(ifs.BR_A, ifs.BR_B))
    m = len(maps)
    p = (1.0 - omega) / m
    mu_w, S_w = rod_gauss()

    # ---- mean: one linear solve
    Asum = sum(A for A, _ in maps)
    bsum = sum(b for _, b in maps)
    C = np.linalg.solve(np.eye(3) - p * Asum, omega * mu_w + p * bsum)

    # ---- covariance: iterate the contraction
    d = [A @ C + b - C for A, b in maps]
    base = omega * (S_w + np.outer(mu_w - C, mu_w - C)) + p * sum(np.outer(x, x) for x in d)
    Q = base.copy()
    for _ in range(iters):
        Qn = base + p * sum(A @ Q @ A.T for A, _ in maps)
        if np.abs(Qn - Q).max() < 1e-15 * max(np.abs(Qn).max(), 1e-30):
            Q = Qn
            break
        Q = Qn

    # ---- bounding sphere about C, covering every splat's 3-sigma ellipsoid
    rod_r = 3.0 * np.sqrt(np.linalg.eigvalsh(S_w).max()) + np.linalg.norm(mu_w - C)
    s = [np.linalg.norm(A, 2) for A, _ in maps]
    R = rod_r
    for _ in range(400):
        Rn = max(rod_r, max(np.linalg.norm(x) + si * R for x, si in zip(d, s)))
        if abs(Rn - R) < 1e-14 * max(Rn, 1e-30):
            R = Rn
            break
        R = Rn

    return dict(C=C, Q=Q, R=R, omega=omega, contraction=max(s))


def uniform_scale(A, tol=1e-9):
    """Every branch map here is s*R with R a rotation, so a node's accumulated
    affine is too.  Returns s, or None if the map is not a similarity -- which
    would make the one-multiply LOD metric invalid and is worth knowing."""
    G = A.T @ A
    s2 = np.trace(G) / 3.0
    if np.abs(G - s2 * np.eye(3)).max() > tol * max(s2, 1e-30):
        return None
    return float(np.sqrt(s2))


def subtree_colour(omega, depth_period, wood_col, iters=400):
    """Average colour of a whole subtree, as a function of the node's depth.

    With a palette that cycles every ``depth_period`` levels the answer is a
    fixed point in P unknowns rather than one:  a node's subtree colour is its
    own rod's colour, weighted omega, blended with the subtree colour one level
    down, weighted (1-omega).
    """
    P = depth_period
    col = np.stack([wood_col(k) for k in range(P)])
    out = col.copy()
    for _ in range(iters):
        nxt = omega * col + (1.0 - omega) * np.roll(out, -1, axis=0)
        if np.abs(nxt - out).max() < 1e-15:
            out = nxt
            break
        out = nxt
    return out


# ------------------------------------------------------------------- check

def monte_carlo(depth=9, omega=0.28, seed=3, jitter=0.0):
    """The same moments, measured off an actually-generated subtree."""
    r = ifs.expand(depth, jitter, seed)
    lv = r['level'].astype(np.float64)
    p = (1.0 - omega) / ifs.M_BR
    # a node at level d holds mass p^d; its rod takes omega of that, and at the
    # truncation depth the leftover mass sits on the leaf disc
    w = np.where(r['kind'] == ifs.KIND_WOOD, omega * p ** lv, 0.0)
    leaf = r['kind'] == ifs.KIND_LEAF
    w[leaf] = p ** lv[leaf] * (1.0 - omega * 1.0)      # what the rod did not take
    w[leaf & (lv < depth)] = 0.0
    w /= w.sum()
    C = (w[:, None] * r['mu']).sum(0)
    dmu = r['mu'] - C
    Q = np.einsum('n,nij->ij', w, r['cov']) + np.einsum('n,ni,nj->ij', w, dmu, dmu)
    return C, Q, w.sum()


if __name__ == '__main__':
    import sys
    om = float(sys.argv[1]) if len(sys.argv) > 1 else 0.28
    M = invariant_moments(om)
    print(f'omega {om}   contraction {M["contraction"]:.4f}')
    print('  C (fixed point) ', np.round(M['C'], 6))
    print('  R (bound)       ', round(M['R'], 5))
    print('  Q (fixed point) \n', np.round(M['Q'], 6))
    for d in (6, 8, 10):
        C, Q, _ = monte_carlo(d, om)
        print(f'  depth {d:2d}: |dC| {np.abs(C - M["C"]).max():.2e}   '
              f'|dQ| {np.abs(Q - M["Q"]).max():.2e}   '
              f'(truncated mass {(1-om)**d:.2e})')
    print('\n  branch maps are similarities (s, or None):',
          [None if uniform_scale(A) is None else round(uniform_scale(A), 4)
           for A in ifs.BR_A])
    print('  trunk map is a similarity:', uniform_scale(ifs.TRUNK_A) is not None)
