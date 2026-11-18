#!/usr/bin/env python3
"""
IFS tree splats: the shared core.

Three things live here, and nothing else should duplicate them:

1.  The IFS itself (the same BRANCH / TRUNK block as ``ifs_tree_splat.py``).
2.  **Address-hashed randomness.**  Every per-node random quantity -- the
    jitter rotation, the colour noise -- is a pure function of the node's
    address, not of how many nodes were generated before it.  That is what
    makes a subtree reproducible on its own, which adaptive LOD (stage 2)
    and order-independent expansion (stage 0) both require.
3.  **Ordered expansion.**  One pass that emits every splat together with
    the slot it would occupy in a back-to-front stream, for one or many
    view directions at once, without ever sorting N things.

The hash is 32-bit and written to be reproducible character-for-character
in JavaScript (``Math.imul`` + ``>>> 0``), so the browser demo generates
the identical tree.  ``js/ifs_core.js`` is the port; ``tests/`` checks they
agree splat for splat.
"""

import math
import numpy as np

C0 = 0.28209479177387814      # SH degree-0 basis constant

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
    A = s * (rz(math.radians(roll)) @ ry(math.radians(tilt)) @ rz(math.radians(twist)))
    return A, np.array([0.0, 0.0, H_ATT])


BRANCH = [
    branch(38.0, 0.0, S_BR, 25.0),
    branch(38.0, 120.0, S_BR, 25.0),
    branch(38.0, 240.0, S_BR, 25.0),
    branch(9.0, 60.0, 0.52, 40.0),      # leader shoot
]
M_BR = len(BRANCH)

BR_A = np.stack([a for a, _ in BRANCH])          # (4,3,3)
BR_B = np.stack([b for _, b in BRANCH])          # (4,3)

TRUNK_A = np.diag([0.075, 0.075, 1.0 - S_BR])
TRUNK_B = np.zeros(3)

HEIGHT = H_ATT / (1.0 - S_BR)                    # 2.5
MU0 = np.array([0.0, 0.0, HEIGHT / 2.0])
SIG0 = np.diag([0.45 ** 2, 0.45 ** 2, (HEIGHT / 4.0) ** 2])

WOOD_MU_LOCAL = TRUNK_A @ MU0                    # wood splat centre, node frame
SCALE_FLOOR = 2e-4 * HEIGHT

KIND_WOOD, KIND_LEAF = 0, 1

# ------------------------------------------------------- address hashing

_U32 = np.uint32
_M1 = _U32(0x85EBCA6B)
_M2 = _U32(0xC2B2AE35)
_GOLD = _U32(0x9E3779B1)


def _imul(a, b):
    """32-bit wrapping multiply, matching JS Math.imul on uint32 inputs."""
    return (np.asarray(a, dtype=_U32) * np.asarray(b, dtype=_U32)).astype(_U32)


def h32(x):
    """murmur3 finalizer. Pure function, uint32 -> uint32."""
    x = np.asarray(x, dtype=_U32).copy()
    x ^= x >> _U32(16)
    x = _imul(x, _M1)
    x ^= x >> _U32(13)
    x = _imul(x, _M2)
    x ^= x >> _U32(16)
    return x


def mix(h, v):
    """Fold one address digit (or any small int) into a hash."""
    return h32(np.asarray(h, dtype=_U32) + _imul(np.asarray(v, dtype=_U32) + _U32(1), _GOLD))


def stream(h, k):
    """k-th independent uniform in [0,1) derived from a node hash."""
    return h32(np.asarray(h, dtype=_U32) ^ _imul(_U32(k) + _U32(1), _U32(0x85EBCA77))
               ).astype(np.float64) * (1.0 / 4294967296.0)


def root_hash(seed):
    return h32(_U32(np.uint32(seed)))


def _normal(h, k):
    """One standard normal from streams k, k+1 (Box-Muller)."""
    u1 = np.maximum(stream(h, k), 1e-12)
    u2 = stream(h, k + 1)
    return np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)


def jitter_rot(h, amt):
    """Per-node random rotation (Rodrigues) from the node's address hash."""
    n = len(h)
    if amt <= 0.0:
        return np.broadcast_to(np.eye(3), (n, 3, 3))
    z = 2.0 * stream(h, 0) - 1.0
    phi = 2.0 * np.pi * stream(h, 1)
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    ax = np.stack([r * np.cos(phi), r * np.sin(phi), z], 1)
    ang = amt * _normal(h, 2)
    K = np.zeros((n, 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -ax[:, 2], ax[:, 1]
    K[:, 1, 0], K[:, 1, 2] = ax[:, 2], -ax[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -ax[:, 1], ax[:, 0]
    I = np.broadcast_to(np.eye(3), (n, 3, 3))
    return I + np.sin(ang)[:, None, None] * K + (1 - np.cos(ang))[:, None, None] * (K @ K)


# -------------------------------------------------- subtree size + centroid

def subtree_sizes(depth):
    """T[r] = splats in a subtree with r branch levels left.

    T[0] = 2 (one wood rod + one foliage disc), T[r] = 1 + m*T[r-1].
    """
    T = np.zeros(depth + 1, dtype=np.int64)
    T[0] = 2
    for r in range(1, depth + 1):
        T[r] = 1 + M_BR * T[r - 1]
    return T


def subtree_centroids(depth):
    """C[r] = centroid of a depth-r subtree's splats, in the node's own frame.

    The exact mean, by the same recursion that builds the tree.  Used as each
    subtree's depth representative: a far better proxy than MU0, because it is
    where the subtree's mass actually is.  Jitter is ignored here (it is small
    and zero-mean), so this is m+1 fixed vectors, not per-node work.
    """
    T = subtree_sizes(depth)
    C = np.zeros((depth + 1, 3))
    C[0] = 0.5 * (WOOD_MU_LOCAL + MU0)
    for r in range(1, depth + 1):
        acc = WOOD_MU_LOCAL.copy()
        for i in range(M_BR):
            acc = acc + T[r - 1] * (BR_A[i] @ C[r - 1] + BR_B[i])
        C[r] = acc / T[r]
    return C


# ----------------------------------------------------------- the expansion

def expand(depth, jitter=0.0, seed=7, dirs=None, rep='centroid'):
    """Generate every splat, in canonical (address-lexicographic) order.

    Returns a dict with, per splat: ``mu`` (N,3), ``cov`` (N,3,3), ``col``
    (N,3), ``alpha`` (N,), ``kind``, ``level``, ``hash``.

    If ``dirs`` is an (D,3) array of view directions (pointing away from the
    camera, into the scene), also returns ``slot`` (D,N): the index this splat
    lands on in the back-to-front stream produced by ordered expansion along
    that direction.  Nothing of size N is ever sorted -- each node sorts
    m+1 = 5 keys, which is 9 compare-swaps in the GPU version.

    ``rep`` picks the local point that stands for a whole subtree when its
    siblings are ordered:
      'centroid' -- the subtree's exact mean (default; best, and free)
      'mu0'      -- the child Gaussian's own centre, as IDEAS.md first proposed
      'bound'    -- the centre of the subtree's exact Hutchinson bound
    """
    T = subtree_sizes(depth)
    C = subtree_centroids(depth)
    N = int(T[depth])

    dirs = None if dirs is None else np.atleast_2d(np.asarray(dirs, dtype=np.float64))
    D = 0 if dirs is None else len(dirs)

    mu = np.zeros((N, 3))
    cov = np.zeros((N, 3, 3))
    kind = np.zeros(N, dtype=np.int8)
    level = np.zeros(N, dtype=np.int8)
    nhash = np.zeros(N, dtype=_U32)
    addr = np.full((N, max(depth, 1)), -1, dtype=np.int8)
    slot = np.zeros((max(D, 1), N), dtype=np.int64)

    if rep not in ('centroid', 'mu0', 'bound'):
        raise ValueError("rep must be 'centroid', 'mu0' or 'bound'")
    bc = hutchinson_bound(depth)[1] if rep == 'bound' else None

    A = np.eye(3)[None, :, :]
    b = np.zeros((1, 3))
    h = np.array([root_hash(seed)], dtype=_U32)
    ad = np.full((1, max(depth, 1)), -1, dtype=np.int8)
    cbase = np.zeros(1, dtype=np.int64)                 # canonical slot base
    obase = np.zeros((max(D, 1), 1), dtype=np.int64)    # ordered slot base

    for d in range(depth + 1):
        r = depth - d
        n = len(A)
        dloc = (np.einsum('nji,dj->dni', A, dirs) if D else None)   # M^T d, per node

        # ---- this node's wood rod: the trunk map, composed in the node's frame
        Aw = A @ TRUNK_A
        idx = cbase                                      # wood is canonical item 0
        mu[idx] = np.einsum('nij,j->ni', Aw, MU0) + b
        cov[idx] = np.einsum('nij,jk,nlk->nil', Aw, SIG0, Aw)
        kind[idx] = KIND_WOOD
        level[idx] = d
        nhash[idx] = h
        addr[idx] = ad

        if r == 0:
            # ---- leaf node: the wood rod plus one foliage disc; no recursion
            lmu = np.einsum('nij,j->ni', A, MU0) + b
            lS = np.einsum('nij,jk,nlk->nil', A, SIG0, A)
            sc, V = decompose(lS, SCALE_FLOOR)
            g = np.prod(sc, axis=1) ** (1.0 / 3.0)
            disc = np.stack([1.9 * g, 1.9 * g, 0.30 * g], 1)   # flat along the branch
            lidx = cbase + 1
            mu[lidx] = lmu
            cov[lidx] = np.einsum('nij,nj,nkj->nik', V, disc ** 2, V)
            kind[lidx] = KIND_LEAF
            level[lidx] = d
            nhash[lidx] = h
            addr[lidx] = ad

            if D:
                keys = np.stack([np.einsum('j,dnj->dn', WOOD_MU_LOCAL, dloc),
                                 np.einsum('j,dnj->dn', MU0, dloc)], 2)   # (D,n,2)
                starts = _slots(keys, np.ones((n, 2), dtype=np.int64))
                slot[:, idx] = obase + starts[:, :, 0]
                slot[:, lidx] = obase + starts[:, :, 1]
            break

        # ---- the m children.  Effective linear part is R(address) @ A_i, so a
        # child's geometry depends only on its own address.
        ch_h = np.stack([mix(h, i) for i in range(M_BR)])        # (m,n)
        ch_Aeff = np.stack([jitter_rot(ch_h[i], jitter) @ BR_A[i] for i in range(M_BR)])
        ch_A = np.einsum('nij,mnjk->mnik', A, ch_Aeff)            # (m,n,3,3)
        ch_b = np.einsum('nij,mj->mni', A, BR_B) + b             # (m,n,3)

        if D:
            # item 0 = this node's wood rod (1 splat); items 1..m = the subtrees
            sizes = np.empty((n, M_BR + 1), dtype=np.int64)
            sizes[:, 0] = 1
            sizes[:, 1:] = T[r - 1]

            q = {'mu0': MU0, 'centroid': C[r - 1], 'bound': bc}[rep]
            loc = np.einsum('mnij,j->mni', ch_Aeff, q) + BR_B[:, None, :]   # (m,n,3)

            keys = np.empty((D, n, M_BR + 1))
            keys[:, :, 0] = np.einsum('j,dnj->dn', WOOD_MU_LOCAL, dloc)
            keys[:, :, 1:] = np.einsum('mni,dni->dnm', loc, dloc)

            starts = _slots(keys, sizes)
            slot[:, idx] = obase + starts[:, :, 0]
            ch_obase = np.stack([obase + starts[:, :, 1 + i] for i in range(M_BR)], 1)

        ch_cbase = np.stack([cbase + 1 + i * T[r - 1] for i in range(M_BR)])  # (m,n)
        ch_ad = np.repeat(ad[None], M_BR, 0).copy()
        for i in range(M_BR):
            ch_ad[i, :, d] = i

        A = ch_A.reshape(M_BR * n, 3, 3)
        b = ch_b.reshape(M_BR * n, 3)
        h = ch_h.reshape(M_BR * n)
        cbase = ch_cbase.reshape(M_BR * n)
        ad = ch_ad.reshape(M_BR * n, max(depth, 1))
        obase = (ch_obase.reshape(max(D, 1), M_BR * n) if D else
                 np.zeros((1, M_BR * n), dtype=np.int64))

    col, alpha = _shade(kind, level, depth, nhash)
    out = dict(mu=mu, cov=cov, col=col, alpha=alpha, kind=kind, level=level,
               hash=nhash, addr=addr)
    if D:
        out['slot'] = slot
    return out


def _slots(keys, sizes):
    """Ordered expansion's slot arithmetic.

    keys (D,n,k), sizes (n,k) -> starts (D,n,k): the offset, within the node's
    own span, of each item once the k items are placed back-to-front.  This is
    the whole algorithm: sort k numbers, prefix-sum k sizes.
    """
    Dn, n, k = keys.shape
    order = np.argsort(-keys, axis=2, kind='stable')             # far -> near
    sz = np.broadcast_to(sizes, (Dn, n, k))
    so = np.take_along_axis(sz, order, axis=2)
    st = np.cumsum(so, axis=2) - so
    starts = np.empty((Dn, n, k), dtype=np.int64)
    np.put_along_axis(starts, order, st, axis=2)
    return starts


def _shade(kind, level, depth, nhash):
    """Colour and opacity.  Address-hashed, so a splat's look never depends on
    when it was generated."""
    n = len(kind)
    col = np.zeros((n, 3))
    alpha = np.zeros(n)

    w = kind == KIND_WOOD
    f = level[w] / max(depth, 1)
    base = ((1 - f)[:, None] * np.array([0.26, 0.17, 0.10])
            + f[:, None] * np.array([0.47, 0.38, 0.28]))
    col[w] = base * (1.0 + 0.06 * _normal(nhash[w], 10)[:, None])
    alpha[w] = 0.99

    l = kind == KIND_LEAF
    c = np.array([0.17, 0.41, 0.13]) * (1.0 + 0.22 * _normal(nhash[l], 12)[:, None])
    c[:, 1] *= 1.0 + 0.10 * _normal(nhash[l], 14)
    col[l] = c
    alpha[l] = 0.55

    return np.clip(col, 0.0, 1.0), alpha


# ------------------------------------------------------------- utilities

def decompose(S, floor):
    S = 0.5 * (S + np.transpose(S, (0, 2, 1)))
    w, V = np.linalg.eigh(S)
    scales = np.sqrt(np.maximum(w, floor ** 2))
    flip = np.linalg.det(V) < 0.0
    V[flip, :, 0] *= -1.0
    return scales, V


def hutchinson_bound(depth=None, iters=80):
    """An ellipsoid that contains the attractor, found by iterating the
    Hutchinson operator on a ball until it stops shrinking.

    Returns (Q, c): the subtree of a node with affine (M, b) lies inside the
    ellipsoid {x : (x - M c - b)^T (M Q M^T)^-1 (x - M c - b) <= 1}.  Exact, no
    fitting -- this is the free bounding-volume hierarchy IDEAS.md describes.
    """
    Q = np.eye(3) * (3.0 * HEIGHT) ** 2
    c = MU0.copy()
    for _ in range(iters):
        pts = []
        for i in range(M_BR):
            pts.append((BR_A[i] @ c + BR_B[i], BR_A[i] @ Q @ BR_A[i].T))
        pts.append((TRUNK_A @ c + TRUNK_B, TRUNK_A @ Q @ TRUNK_A.T))
        c2 = np.mean([p[0] for p in pts], 0)
        Q2 = np.zeros((3, 3))
        for m, S in pts:
            dd = (m - c2)[:, None]
            Q2 = np.maximum(Q2, S + dd @ dd.T)
        c, Q = c2, Q2 * 1.02
    return Q, c


def canonical_frame(mu, cov, up='-y', scale=1.0):
    """Recentre in the model's z-up frame, then rotate into a viewer up-axis."""
    U = {'z': np.eye(3),
         'y': np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]]),
         '-y': np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])}[up]
    m = mu.copy()
    m[:, 0] -= m[:, 0].mean()
    m[:, 1] -= m[:, 1].mean()
    m[:, 2] -= m[:, 2].min()
    m = (m @ U.T) * scale
    S = np.einsum('ij,njk,lk->nil', U, cov, U) * scale ** 2
    return m, S
