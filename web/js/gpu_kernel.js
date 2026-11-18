// The expansion kernel, written twice.
//
// WGSL below for the GPU; the JS transcription in gpu_ref.js, line for line,
// so that tests/gpu_transcribe.mjs can check the algorithm against
// ifs_core.js on the CPU.  No WebGPU here can be executed outside a browser,
// so the transcription IS the test -- the same arrangement stage 1's vertex
// shader is under, and the same caveat: it proves the arithmetic, not the API
// calls around it.
//
// What the kernel does, per thread, for splat index k:
//
//   walk from the root to the splat, carrying the accumulated affine, the
//   address hash, and the stream offset of the current subtree.  At each node,
//   order its five items (its own rod, its four subtrees) by depth exactly as
//   ordered expansion does on the CPU, prefix-sum their sizes to get each
//   item's offset, and either emit -- if k lands on the rod -- or descend.
//
// This is O(N log N) work where the serial version is O(N), because every
// thread re-walks its own path from the root.  It wins anyway: the walk is
// about four hundred flops a level with no communication between threads, no
// atomics, no prefix scan across the dispatch, and nothing to compact.  And
// because the order falls out of the same walk, there is no index buffer to
// build and upload -- the per-frame CPU cost of the object is zero.
//
// The adaptive expansion of stage 2 does NOT fit this mould, and that is the
// honest limit of this port: under a level-of-detail cut you do not know a
// subtree's size until you have walked it, so the slot arithmetic has nothing
// to prefix-sum. Doing it on the GPU needs three passes -- expand-and-count,
// a bottom-up size scan, then a top-down slot assignment -- which is written
// up in IDEAS.md and not built here.

export const WORKGROUP = 64;

/** The WGSL source, with the map set and tree depth baked in as constants. */
export function wgslSource({ depth, jitter, seed }) {
  return `
// ---- constants baked in at pipeline build
const DEPTH    : u32 = ${depth}u;
const JITTER   : f32 = ${jitter};
const ROOT_HASH: u32 = ${hash32(seed >>> 0)}u;
const M        : u32 = 4u;
const S_BR     : f32 = 0.6;
const H_ATT    : f32 = 1.0;
const HEIGHT   : f32 = 2.5;
const T0       : f32 = 0.075;       // trunk map, diagonal
const T2       : f32 = 0.4;
const MZ       : f32 = 1.25;        // MU0.z
const SIG0     : vec3<f32> = vec3<f32>(0.2025, 0.2025, 0.390625);
const FLOOR2   : f32 = ${(2e-4 * 2.5) ** 2};
const A_DISC   : f32 = 3.61;
const B_DISC   : f32 = 0.09;

struct Uniforms {
  dir   : vec3<f32>,      // points away from the camera
  count : u32,
};
@group(0) @binding(0) var<uniform> U : Uniforms;
@group(0) @binding(1) var<storage, read>       branchA : array<mat3x3<f32>>;
@group(0) @binding(2) var<storage, read>       sizes   : array<f32>;   // T[r]
@group(0) @binding(3) var<storage, read>       cents   : array<vec3<f32>>; // C[r]
@group(0) @binding(4) var<storage, read_write> outPos  : array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> outCovA : array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> outCovB : array<vec4<f32>>;
@group(0) @binding(7) var<storage, read_write> outCol  : array<u32>;
@group(0) @binding(8) var<storage, read_write> outOrder: array<u32>;

fn h32(x0 : u32) -> u32 {
  var x = x0;
  x = x ^ (x >> 16u);
  x = x * 0x85ebca6bu;
  x = x ^ (x >> 13u);
  x = x * 0xc2b2ae35u;
  x = x ^ (x >> 16u);
  return x;
}
fn mixh(h : u32, v : u32) -> u32 { return h32(h + (v + 1u) * 0x9e3779b1u); }
fn strm(h : u32, k : u32) -> f32 {
  return f32(h32(h ^ ((k + 1u) * 0x85ebca77u))) / 4294967296.0;
}
fn nrm(h : u32, k : u32) -> f32 {
  let u1 = max(strm(h, k), 1e-12);
  let u2 = strm(h, k + 1u);
  return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

// Rodrigues, from the address hash.  A node's perturbation is a pure function
// of WHERE it is in the tree, never of when it was generated -- which is what
// lets every thread here produce the same world every other thread assumes.
fn jitterRot(h : u32) -> mat3x3<f32> {
  if (JITTER <= 0.0) { return mat3x3<f32>(1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0); }
  let z   = 2.0 * strm(h, 0u) - 1.0;
  let phi = 6.283185307179586 * strm(h, 1u);
  let r   = sqrt(max(1.0 - z * z, 0.0));
  let ax  = vec3<f32>(r * cos(phi), r * sin(phi), z);
  let ang = JITTER * nrm(h, 2u);
  let s = sin(ang);
  let c = 1.0 - cos(ang);
  // columns of I + s*K + c*K*K, K the cross-product matrix of ax
  let K  = mat3x3<f32>( 0.0, ax.z, -ax.y,  -ax.z, 0.0, ax.x,   ax.y, -ax.x, 0.0);
  let K2 = K * K;
  return mat3x3<f32>(1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0)
       + mat3x3<f32>(s*K[0], s*K[1], s*K[2])
       + mat3x3<f32>(c*K2[0], c*K2[1], c*K2[2]);
}

fn packCol(c : vec3<f32>, a : f32) -> u32 {
  let q = clamp(round(vec4<f32>(c, a) * 255.0), vec4<f32>(0.0), vec4<f32>(255.0));
  return u32(q.x) | (u32(q.y) << 8u) | (u32(q.z) << 16u) | (u32(q.w) << 24u);
}
fn shadeWood(d : u32, h : u32) -> u32 {
  let f = f32(d) / f32(max(DEPTH, 1u));
  let n = 1.0 + 0.06 * nrm(h, 10u);
  return packCol(vec3<f32>(mix(0.26, 0.47, f), mix(0.17, 0.38, f),
                           mix(0.10, 0.28, f)) * n, 0.99);
}
fn shadeLeaf(h : u32) -> u32 {
  let n = 1.0 + 0.22 * nrm(h, 12u);
  let g = 1.0 + 0.10 * nrm(h, 14u);
  return packCol(vec3<f32>(0.17 * n, 0.41 * n * g, 0.13 * n), 0.55);
}

// A diag(s) A^T, as the six upper-triangular entries.
fn congruence(A : mat3x3<f32>, s : vec3<f32>) -> array<f32, 6> {
  // A is column-major in WGSL: A[c] is column c.
  let r0 = vec3<f32>(A[0].x, A[1].x, A[2].x);
  let r1 = vec3<f32>(A[0].y, A[1].y, A[2].y);
  let r2 = vec3<f32>(A[0].z, A[1].z, A[2].z);
  return array<f32, 6>(dot(r0 * s, r0), dot(r0 * s, r1), dot(r0 * s, r2),
                       dot(r1 * s, r1), dot(r1 * s, r2), dot(r2 * s, r2));
}

// The foliage disc: flat across the branch axis.
//
// The CPU does this with a Jacobi eigensolve, because in general you need the
// covariance's principal axis.  Here you do not: every branch map is a
// similarity and so is every jitter rotation, so the accumulated A is s*R
// exactly, its columns all have length s, and the principal axis is simply
// its third column normalised.  Eigenvalues are s^2 * SIG0.  A dozen flops
// instead of a solver, and exact rather than iterated.
fn discCov(A : mat3x3<f32>) -> array<f32, 6> {
  let c3 = vec3<f32>(A[2].x, A[2].y, A[2].z);
  let s2 = dot(c3, c3);
  let v  = c3 * inverseSqrt(max(s2, 1e-30));
  let ev = max(s2 * SIG0, vec3<f32>(FLOOR2));
  let g  = pow(sqrt(ev.x) * sqrt(ev.y) * sqrt(ev.z), 0.3333333333333333);
  let a2 = A_DISC * g * g;
  let k  = B_DISC * g * g - a2;
  return array<f32, 6>(a2 + k*v.x*v.x, k*v.x*v.y, k*v.x*v.z,
                       a2 + k*v.y*v.y, k*v.y*v.z, a2 + k*v.z*v.z);
}

fn emit(slot : u32, idx : u32, p : vec3<f32>, c : array<f32, 6>, col : u32, lvl : u32) {
  outPos[idx]  = vec4<f32>(p, f32(lvl));
  outCovA[idx] = vec4<f32>(c[0], c[1], c[2], c[3]);
  outCovB[idx] = vec4<f32>(c[4], c[5], 0.0, 0.0);
  outCol[idx]  = col;
  outOrder[slot] = idx;            // a permutation, so no atomics are needed
}

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let tid = gid.x;
  if (tid >= U.count) { return; }

  var A    = mat3x3<f32>(1.0,0.0,0.0, 0.0,1.0,0.0, 0.0,0.0,1.0);
  var b    = vec3<f32>(0.0);
  var h    = ROOT_HASH;
  var base = 0u;                   // stream offset of this subtree
  var idx  = 0u;                   // splat index of this subtree's first splat
  var k    = tid;                  // index within this subtree, generation order
  var d    = 0u;

  loop {
    let r = DEPTH - d;

    // the node's own rod: TRUNK_A is diagonal, so it just scales A's columns
    let AW = mat3x3<f32>(A[0] * T0, A[1] * T0, A[2] * T2);
    let w  = vec3<f32>(A[2].x, A[2].y, A[2].z) * (T2 * MZ) + b;

    var key  : array<f32, 5>;
    var size : array<f32, 5>;
    var n    = 5u;
    key[0]  = dot(w, U.dir);
    size[0] = 1.0;

    // the leaf level: the rod plus one foliage disc, and no children
    if (r == 0u) {
      n = 2u;
      let lp = vec3<f32>(A[2].x, A[2].y, A[2].z) * MZ + b;
      key[1]  = dot(lp, U.dir);
      size[1] = 1.0;
      // order the two, then emit whichever this thread owns
      let rodFirst = key[0] >= key[1];
      if (k == 0u) {
        let slot = base + select(1u, 0u, rodFirst);
        emit(slot, idx, w, congruence(AW, SIG0), shadeWood(d, h), d);
      } else {
        let slot = base + select(0u, 1u, rodFirst);
        emit(slot, idx + 1u, lp, discCov(A), shadeLeaf(h), d);
      }
      return;
    }

    // the four children.  Each child's affine depends only on its address.
    var cA : array<mat3x3<f32>, 4>;
    let cb = vec3<f32>(A[2].x, A[2].y, A[2].z) * H_ATT + b;
    let cen = cents[r - 1u];
    let csz = sizes[r - 1u];
    for (var i = 0u; i < M; i = i + 1u) {
      let chh = mixh(h, i);
      let AE  = jitterRot(chh) * branchA[i];
      cA[i]   = A * AE;
      key[1u + i]  = dot(cA[i] * cen + cb, U.dir);
      size[1u + i] = csz;
    }

    // ---- the whole ordering claim, in nine compare-swaps
    var o = array<u32, 5>(0u, 1u, 2u, 3u, 4u);
    cswap(&key, &o, 0u, 1u); cswap(&key, &o, 3u, 4u);
    cswap(&key, &o, 2u, 4u); cswap(&key, &o, 2u, 3u);
    cswap(&key, &o, 0u, 3u); cswap(&key, &o, 0u, 2u);
    cswap(&key, &o, 1u, 4u); cswap(&key, &o, 1u, 3u);
    cswap(&key, &o, 1u, 2u);

    // prefix-sum the sizes in sorted order -> where each item starts
    var off : array<u32, 5>;
    var acc = 0u;
    for (var j = 0u; j < 5u; j = j + 1u) {
      off[o[j]] = acc;
      acc = acc + u32(size[o[j]]);
    }

    if (k == 0u) {                 // this thread is the node's own rod
      emit(base + off[0], idx, w, congruence(AW, SIG0), shadeWood(d, h), d);
      return;
    }
    let ci = (k - 1u) / u32(csz);  // which subtree it is in
    k    = (k - 1u) - ci * u32(csz);
    base = base + off[1u + ci];
    idx  = idx + 1u + ci * u32(csz);
    A    = cA[ci];
    b    = cb;
    h    = mixh(h, ci);
    d    = d + 1u;
  }
}

fn cswap(key : ptr<function, array<f32,5>>, o : ptr<function, array<u32,5>>,
         i : u32, j : u32) {
  // descending: larger key = farther = drawn first
  if ((*key)[i] < (*key)[j]) {
    let tk = (*key)[i]; (*key)[i] = (*key)[j]; (*key)[j] = tk;
    let to = (*o)[i];   (*o)[i]   = (*o)[j];   (*o)[j]   = to;
  }
}
`;
}

function hash32(x) {
  x = x >>> 0;
  x ^= x >>> 16; x = Math.imul(x, 0x85ebca6b);
  x ^= x >>> 13; x = Math.imul(x, 0xc2b2ae35);
  x ^= x >>> 16;
  return x >>> 0;
}
