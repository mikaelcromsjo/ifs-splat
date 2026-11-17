# A geometric visual world

Notes from the design conversation of 2026-11-17. The generator in `ifs_tree_splat.py` works; this
is about what to build on it. Nothing here is implemented yet.

## Direction

**Build a geometric world, not a forest.**

The method's weakness as naturalism is its strength as style. IFS vegetation looks fractal, not
natural — true of every fractal-plant demo since the 1980s, which is long enough to call it a
property of the method rather than a problem to solve with effort. Real nature reads as natural
because it is non-stationary and historical: competition for light, broken limbs, disease, wind,
age. Self-similarity is the opposite of that. Each step taken toward naturalism (noise-modulated
map parameters, per-species variation, damage models) moves away from the method, until the IFS is
only a geometry compression scheme and the real work is conventional procedural vegetation.

So: a crystalline, lawful, obviously-constructed nature. Recursive and alien, flown through rather
than walked, carried by motion and sound. Weeks of work rather than years, and the ceiling is high
because nothing else looks like it.

The two things that cap the realistic version are worth writing down, because they will tempt
someone later:

- **Lighting.** What makes a real forest beautiful is dappled light and subsurface scatter in
  leaves. Splats with baked spherical harmonics cannot take a moving sun; forward-shaded splats
  lose the appearance that makes 3DGS impressive in the first place. A 3D ellipsoid also has no
  well-defined normal (a flat disc does, which is why foliage discs behave better than wood rods).
- **Scope.** "Open world" adds collision, streaming, terrain, water and gameplay — years of work in
  exactly the areas where the fractal contributes nothing. The interesting part is about 5% of it.

A geometric world sidesteps both: stylised shading is a choice rather than a compromise, and a
flythrough needs no collision or gameplay.

## What the method gives for free

Generation on the fly is unusually well suited here, because generation and level of detail are the
same mechanism rather than two systems fighting each other.

- **Addresses are random-access.** A node's affine is a matrix product along its path, so any
  address can be expanded without materialising its ancestors' other subtrees. A frustum walk
  touches only what is visible.
- **A node's Gaussian is a correct rendering of its whole subtree.** Stop recursing when projected
  size falls under a few pixels; the blob drawn instead of 4^n descendants has the right shape,
  position and anisotropy, because the Jacobian produced it.
- **Memory is O(visible), not O(world).** Store the map set and a seed field; nothing is persisted
  or streamed from disk.
- **Deterministic.** Same address, same splat, every machine. Caching and multiplayer consistency
  are free.
- **Moment-matched LOD in closed form.** The invariant measure's moments satisfy a linear
  fixed-point equation in the maps, so a parent Gaussian matching its children's mixture in mass,
  mean and covariance can be computed rather than fitted. Hierarchical 3DGS (Kerbl et al. 2024) and
  Octree-GS build that tree by fitting to captured data; an IFS hands it over directly.
  Note: this weakens once map parameters vary with world position, since subtree statistics then
  differ by location — fall back to sampling a few representatives per region.

## The cost model is backwards, and that is the real constraint

Splats are cheap to produce and expensive to composite. Correct alpha blending needs per-frame
back-to-front ordering, and canopy overdraw is dozens of translucent layers deep. Procedural
generation makes production nearly free — but production was never the bottleneck. A GPU radix sort
of 2–5M splats is roughly 2–5 ms, a large part of a 16 ms frame, spent before anything is shaded.

Which is what makes the next section the most valuable idea in this file.

## Ordered generation: emit already sorted

**Sorted order is lexicographic order on addresses, under a per-node permutation of the digits.**

Bound the attractor once by iterating the Hutchinson operator on a ball until it stops shrinking,
giving `B`. A node with accumulated affine `(M, b)` then has its entire subtree inside `M·B + b` —
an exact ellipsoid bound, no fitting. That is a bounding volume hierarchy with branching factor m,
zero build cost and zero storage.

Child `i`'s centre in the node's local frame is `c_i = A_i·MU0 + b_i`, fixed and known at build
time. Their order along view direction `d` is the order of `<c_i, M^T d>` — it depends only on the
direction `M^T d`, not on where the node sits in the world. So ordering means sorting m numbers,
not N.

### Ordered expansion

    level k is an array of nodes, already in back-to-front order
    for each node (in parallel):
        d' = M^T d                        # view direction in the node's frame
        rank = order of <c_i, d'>         # m+1 elements incl. the wood splat;
                                          # a 5-element sorting network is 9 compare-swaps
        child j writes to slot = parent_slot * m + rank[j]

Each level is sorted by construction, so the last level is the instance buffer in composite order.
One kernel per depth level, no sort pass, no prefix sum. Cost is about N/3 nodes x 9 compares,
shared by every descendant, against a global radix sort over N keys every frame.

Submission order suffices: within a draw call both GL and Vulkan guarantee fragments blend in
primitive order. Emit, draw with depth writes off, done.

### Closed form at uniform depth

A leaf's index in the sorted stream is

    index = SUM_k  rank_{p_k}(i_k) * m^(D-k)

a mixed-radix number whose digits are the address, each permuted by its node's ordering. A thread
holding an address already computes `M` incrementally along its path, so it gets its own output
slot in O(D) with no communication at all. Pure scatter.

With adaptive depth (wanted, for LOD) subtree sizes vary, so slots come from one scan over ordered
children per level. Still far cheaper than sorting.

### Front-to-back early-out — the bigger win

Once the stream is ordered, run it front-to-back and track per-tile transmittance. When a tile
saturates, stop expanding every node whose footprint lands only in saturated tiles, skipping whole
subtrees. That attacks overdraw, the actual bottleneck, not just the sort. A global sort cannot do
this: it must produce all N before anything is drawn. This is also where the traversal would go
per-tile rather than global, which is closer to per-ray-correct than a global sort ever is.

### Where the order is only approximate

Sibling order by bounding volume is exact only where siblings do not overlap along `d`. In
interpenetrating canopy they do. Three reasons that is acceptable:

- It is exactly as approximate as what web 3DGS viewers already ship, which sort globally by splat
  centre — also not per-ray correct.
- It is far more temporally stable. A node's permutation changes only when `M^T d` crosses one of
  the m(m-1)/2 great circles separating order cells: rare, local, and swapping adjacent elements.
  3DGS popping comes from order *changing*, not from order being wrong, so hierarchical ordering
  tends to look better in motion than an exact per-frame sort. StopThePop (2024) is essentially
  this observation applied to trained scenes.
- Where tighter order is needed, descend further: if two children's `[dmin, dmax]` intervals
  overlap, expand both one more level before emitting. A lazy k-way merge — pay only on ambiguous
  nodes.

The honest limit: none of this transfers to captured 3DGS scenes. It works because the geometry
*is* the hierarchy, with bounds for free. That is why the papers radix-sort instead.

## Other things that will be needed

- **Aliasing is not optional polish.** Small splats at distance shimmer violently under motion;
  Mip-Splatting-style covariance dilation by the pixel footprint is required. It changes how
  everything looks, so build it in from the start rather than retrofitting.
- **LOD popping.** Moment matching fixes the geometry; colour and opacity still jump at a split,
  needing a cross-fade band where parent and children both render — more splats and more ordering
  exactly while the camera moves.
- **Collision, if ever needed.** The wood rods already are capsules: position, axis, radius, from
  the same recursion. Ground and water are not.
- **Variety.** Pure self-similarity reads as synthetic. Map parameters as a function of world
  position via a low-frequency noise field keeps determinism and random access.
- **IFS covers branching only.** Trees, ferns, lichen, river networks, some cloud structure: good.
  Eroded terrain, grass, rock, water: not an IFS.

## What to measure first, before building anything

1. **Does the approximate order matter?** Emit with ordered expansion, compare against a
   ground-truth per-frame sort, count inversions weighted by opacity. A number instead of an
   argument. Cheap — it can be done in the Python generator, no GPU.
2. **What does the frame actually cost?** A few hundred thousand splats from a handful of trees on
   screen, per-frame ordering, camera in motion. Frame time and shimmer decide the project.

Neither needs a world. Both are a few days.

## Decided against

An open-world *game*. The fractal gives nothing to what makes open worlds good, and the work lands
almost entirely in areas where the method has no advantage. Memory footprint is the only axis where
it clearly beats conventional approaches.
