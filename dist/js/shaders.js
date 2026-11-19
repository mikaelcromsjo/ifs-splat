// The splat shader.  Its projection is a transcription of py/splat_raster.py,
// which is the reference the stage-0 numbers were measured against -- so if
// these two ever disagree, the measurements stop meaning anything.
//
// Anti-aliasing is in here from the first version, not bolted on later:
// without it distant splats shimmer violently under motion and every later
// judgement about how the thing looks is made against a broken image.

const VERT_SRC = `#version 300 es
precision highp float;
precision highp int;

// The near band, in world units: see where nearFade is built below.
const float ZN = 0.02, ZF = 0.12;

uniform vec3  uEye;
uniform mat3  uRv;          // world -> view; rows are right, up, forward
uniform vec2  uFocal;       // pixels per unit at z = 1
uniform vec2  uViewport;    // pixels
uniform float uKappa;       // Mip-Splatting 3D dilation, in pixels of footprint
uniform float uLowpass;     // EWA 2D low-pass variance, in pixels^2
uniform float uCut;         // Gaussian support, in sigmas
uniform int   uTexW;
uniform float uFogNear, uFogFar;

// ---- look ----------------------------------------------------------------
//
// Per-effect bits, ANDed with each splat's own. uFxMask is the UI's master
// switch and an object's aFx is what that object would like; a global off
// always wins, so the UI can never be contradicted by an object's own taste.
//   bit 0  iridescence   bit 1  rim light
uniform int   uFxMask;
uniform float uIridAmt, uIridScale, uIridPhase;
uniform float uRimAmt, uRimPow;
uniform vec3  uRimColor;

//FETCH_DECL

in vec2 aCorner;

out vec4 vColor;
out vec3 vConic;
out vec2 vDelta;

void main() {
  vColor = vec4(0.0); vConic = vec3(1.0, 0.0, 1.0); vDelta = vec2(0.0);
//FETCH_BODY
  vec3 v = uRv * (sPos - uEye);

  // The near band: ZN is where a splat is dropped, ZF is where it is drawn in
  // full, and in between it fades.
  //
  // This used to be a single "if (v.z < 0.02) return", which is a cull on the
  // CENTRE -- so a rod whose body still surrounded the camera was discarded
  // whole, in one frame, the instant its centre crossed behind the eye. The
  // thing you had just flown into blinked out.
  //
  // It cannot be fixed by projecting it properly, because there is nothing to
  // project: J is the affine approximation of the projection AT the centre,
  // and at z <= 0 it does not exist. (Even at z = 0.5 the ellipse it gives is
  // not the splat's real footprint, which is why the honest answer in that
  // regime is a ray-space evaluation and a different alpha model -- a bigger
  // change than this, and one that moves every raster baseline with it.)
  //
  // So what is fixed here is the DISCONTINUITY, not the projection. A splat
  // whose centre reaches the eye still ends up invisible; it now gets there
  // over the band instead of in one frame, so nothing pops and no edge ever
  // appears. The band is absolute, and wide enough to be a ramp rather than a
  // flicker: measured on the real autopilot over the field, at its own default
  // speed (0.0062) a static splat takes 89 frames at the median step and 71 at
  // the p99 step to cross it -- call it a second and a half, against the one
  // frame a pop takes by definition.
  //
  // What it costs, measured rather than assumed: the parity and fidelity
  // scenes never put a splat inside the band, and both are bit-identical to
  // what they were before it existed. The flight does put splats inside it --
  // tests/legible.mjs at 120 280 210 1, same seed, same 3600 frames, runs
  // 28.8% washed out with the band taken out and 23.5% with it in. So the
  // band is worth 5.3 points of fog, all of it on the dive: the climb sits at
  // 32.9% either way, because what washes the climb is splats farther out
  // than ZF, which no near band can reach.
  if (v.z < ZN) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  float nearFade = smoothstep(ZN, ZF, v.z);

  mat3 S = mat3(sCovA.x, sCovA.y, sCovA.z,
                sCovA.y, sCovB.x, sCovB.y,
                sCovA.z, sCovB.y, sCovB.z);

  // Mip-Splatting's 3D filter: dilate by the world size of one pixel at this
  // depth, so nothing is ever asked to be smaller than the sampling grid.
  float px = uKappa * v.z / uFocal.x;
  S[0][0] += px * px; S[1][1] += px * px; S[2][2] += px * px;

  float zi = 1.0 / v.z;
  mat3x2 J = mat3x2( uFocal.x * zi,                 0.0,
                     0.0,                          -uFocal.y * zi,
                    -uFocal.x * v.x * zi * zi,      uFocal.y * v.y * zi * zi);
  mat3x2 W = J * uRv;
  mat2 S2 = W * S * transpose(W);

  // EWA 2D low-pass, with the opacity compensation that keeps a splat's total
  // energy right when it is blurred up to a pixel.
  float det0 = determinant(S2);
  S2[0][0] += uLowpass; S2[1][1] += uLowpass;
  float det1 = determinant(S2);
  if (!(det1 > 0.0)) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  float comp = sqrt(clamp(max(det0, 0.0) / det1, 0.0, 1.0));

  vConic = vec3(S2[1][1], -S2[0][1], S2[0][0]) / det1;

  // quad in the ellipse's own axes, so elongated rods do not drag a huge
  // axis-aligned box of dead fragments behind them
  float mid = 0.5 * (S2[0][0] + S2[1][1]);
  float rr  = sqrt(max(mid * mid - det1, 0.0));
  float l1 = mid + rr, l2 = max(mid - rr, 0.02);
  vec2 ax1 = abs(S2[0][1]) < 1e-9
           ? (S2[0][0] >= S2[1][1] ? vec2(1.0, 0.0) : vec2(0.0, 1.0))
           : normalize(vec2(S2[0][1], l1 - S2[0][0]));
  vec2 ax2 = vec2(ax1.y, -ax1.x);
  float m1 = uCut * sqrt(l1), m2 = uCut * sqrt(l2);
  vec2 maj = m1 * ax1, mnr = m2 * ax2;

  vec2 centre = vec2(uFocal.x * v.x * zi + uViewport.x * 0.5,
                     uViewport.y * 0.5 - uFocal.y * v.y * zi);

  // No cap on how far that quad may reach -- a fix, not an omission.
  //
  // The extent used to be min(uCut * sqrt(l1), 2048.0). The fragment shader
  // only evaluates the kernel INSIDE this quad, so that cap cut the Gaussian
  // dead straight as soon as a splat grew past 2048 px -- not at its 3-sigma
  // tail where nothing is left to see, but at a fraction of a sigma, with
  // most of its alpha still standing. A sigma-2 blob at z = 2 was cut at
  // 1.48 sigma with 33% of peak alpha at the cut; at z = 0.5, 0.37 sigma and
  // 93%. Four quad edges, four straight sides: fly into one large splat and
  // it has the silhouette of a polygon. Worse, once the centre drifted off
  // axis the capped quad stopped covering the viewport at all, so the cut
  // walked across the middle of the view instead of staying off-screen.
  //
  // Big on screen is not an edge case here. A node large on screen does get
  // expanded (sigma > hi gives t = 1, so no blob), but what it expands INTO
  // is rods, and the rod you are flying into is one splat a few metres away:
  // that is the large splat, every time. Blobs join it when the walk stops
  // descending -- "level >= this.maxLevel" forces t = 0 at any size, and
  // budget pressure raises lodScale and with it the size at which a subtree
  // is still drawn as one blob. A splat larger than the window is a routine
  // frame, not a mistake.
  //
  // Why no cap is needed at all: the quad has only to COVER the part of the
  // kernel that is on screen. Coverage is decided per fragment from vDelta,
  // which is affine in the vertex with w = 1, so its interpolation is exact
  // for ANY quad containing ellipse-intersect-viewport. Past the screen
  // diagonal the oriented quad has nothing left to save -- it exists so that
  // elongated rods do not drag a box of dead fragments, and no box larger
  // than the window is cheaper than the window -- so beyond that size draw
  // the kernel's bounding box clipped to the viewport. That cannot cut
  // anything a viewer could see, and it costs the fragments it covers, which
  // is the honest price of being inside a splat.
  //
  // The reference this file transcribes (py/splat_raster.py) never had a cap:
  // its radius is uncapped and clipped to the screen. The cap was this
  // shader's alone, which is why no headless measurement ever showed it --
  // cpu_raster.js bounds a screen-clipped box and so cannot reproduce it.
  vec2 p;
  if (m1 <= length(uViewport)) {
    vDelta = aCorner.x * maj + aCorner.y * mnr;
    p = centre + vDelta;
  } else {
    vec2 r  = abs(maj) + abs(mnr);
    vec2 lo = max(centre - r, vec2(0.0));
    vec2 hi = min(centre + r, uViewport);
    if (lo.x >= hi.x || lo.y >= hi.y) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
    p = mix(lo, hi, 0.5 + 0.5 * aCorner);
    vDelta = p - centre;
  }
  gl_Position = vec4(p.x / uViewport.x * 2.0 - 1.0,
                     1.0 - p.y / uViewport.y * 2.0, 0.0, 1.0);

  vec4 c = sColor;

  // How obliquely is this splat being seen?
  //
  // The ratio of the projected ellipse's two axes, which l1 and l2 above
  // already hold. A splat whose flat face is toward the camera projects round
  // and gives face ~ 1; the same splat edge-on projects to a sliver and gives
  // face ~ 0. That makes this a genuine function of 3D orientation against
  // the view direction, and it costs one divide and one square root because
  // the eigenvalues were needed for the quad anyway.
  //
  // A normal recovered in the fragment shader from the splat's own local
  // coordinate would have been cheaper still and WRONG for this: it is a
  // screen-space quantity, identical from every direction, so nothing would
  // change as the camera moved -- which is the entire effect being asked for.
  float face = sqrt(clamp(l2 / max(l1, 1e-9), 0.0, 1.0));
  int fx = uFxMask & sFx;

  // Iridescence: a thin-film sheen indexed by viewing angle. Three cosines
  // offset by a third of a turn each, which is the cheapest way to sweep hue
  // without converting colour spaces. Multiplied rather than replaced, so an
  // object keeps its palette identity and gains a sheen over it.
  if ((fx & 1) != 0) {
    vec3 sheen = 0.5 + 0.5 * cos(6.28318531 *
      (vec3(0.0, 0.33333, 0.66667) + uIridPhase + face * uIridScale));
    c.rgb = mix(c.rgb, c.rgb * sheen * 2.0, uIridAmt);
  }

  // Rim light: brighten what is seen edge-on, so silhouettes glow. Added
  // before the fog multiply, so a rim fades into the distance with everything
  // else instead of hanging in the void after its object has gone.
  if ((fx & 2) != 0) {
    c.rgb += uRimColor * (uRimAmt * pow(1.0 - face, uRimPow));
  }

  float fog = 1.0 - clamp((v.z - uFogNear) / max(uFogFar - uFogNear, 1e-6), 0.0, 1.0);
  vColor = vec4(c.rgb, c.a * comp * fog * nearFade);
}
`;

export const FRAG = `#version 300 es
precision highp float;

in vec4 vColor;
in vec3 vConic;
in vec2 vDelta;
out vec4 fragColor;

void main() {
  float e = vConic.x * vDelta.x * vDelta.x
          + 2.0 * vConic.y * vDelta.x * vDelta.y
          + vConic.z * vDelta.y * vDelta.y;
  if (e < 0.0) discard;
  float a = vColor.a * exp(-0.5 * e);
  if (a < 0.0039) discard;
  fragColor = vec4(vColor.rgb * a, a);     // premultiplied
}
`;

// Two ways in, one body.
//
//   'static' -- the splats never move, so they live in textures and only a
//               4-byte index per splat is re-uploaded each frame (stage 1).
//   'stream' -- the splats are regenerated every frame by the adaptive walk,
//               so they arrive as instanced attributes already in draw order
//               and there is no index at all (stage 2).
const FETCH = {
  static: {
    decl: `uniform sampler2D uPos;
uniform sampler2D uCovA;
uniform sampler2D uCovB;
uniform sampler2D uCol;
in uint aSplat;`,
    // The static path predates per-object effects and has no fx channel;
    // -1 is every bit set, so it inherits whatever the UI asks for.
    body: `  int sFx = -1;
  int id = int(aSplat);
  ivec2 tc = ivec2(id % uTexW, id / uTexW);
  vec3 sPos  = texelFetch(uPos, tc, 0).xyz;
  vec4 ca    = texelFetch(uCovA, tc, 0);
  vec2 cbb   = texelFetch(uCovB, tc, 0).xy;
  vec3 sCovA = ca.xyz;
  vec3 sCovB = vec3(ca.w, cbb.x, cbb.y);
  vec4 sColor = texelFetch(uCol, tc, 0);`,
  },
  stream: {
    decl: `in vec3 aPos;
in vec3 aCovA;
in vec3 aCovB;
in vec4 aColor;
in float aFx;`,
    body: `  vec3 sPos = aPos;
  vec3 sCovA = aCovA;
  vec3 sCovB = aCovB;
  vec4 sColor = aColor;
  int sFx = int(aFx + 0.5);`,
  },
};

export function vertexSource(kind) {
  const f = FETCH[kind];
  return VERT_SRC.replace('//FETCH_DECL', f.decl).replace('//FETCH_BODY', f.body);
}

export const VERT = vertexSource('static');

// ---------------------------------------------------------------------------
// The nebula background.
//
// Fog currently fades a splat's ALPHA to nothing, so distance resolves to
// whatever the framebuffer was cleared to -- one flat colour, which is why
// the void reads as empty rather than as space. This draws behind everything
// instead, so the same fog now fades into structure.
//
// Deliberately NOT noise. A hash- or texture-based nebula sampled per pixel
// shimmers under motion: the camera here moves by tiny fractions of a world
// unit while scales change by factors of ten, and anything with detail finer
// than a pixel crawls. These are smooth exponential lobes around fixed
// directions plus one broad band, so every spatial frequency present is one
// the screen can actually hold still -- band-limited by construction, which
// is the same reason the splat shader has had anti-aliasing from day one.
export const BG_VERT = `#version 300 es
precision highp float;
in vec2 aCorner;
out vec2 vNdc;
void main() {
  vNdc = aCorner;
  gl_Position = vec4(aCorner, 0.0, 1.0);
}
`;

export const BG_FRAG = `#version 300 es
precision highp float;

uniform mat3  uRvT;        // view -> world (the splat shader's uRv, transposed)
uniform vec2  uTan;        // tan of the half field of view, x and y
uniform vec3  uSky, uDeep; // the two ends of the base gradient
uniform vec3  uLobeCol[3];
uniform vec3  uLobeDir[3];
uniform vec3  uLobeK;      // tightness per lobe
uniform vec3  uBandCol;
uniform vec3  uBandAxis;
uniform float uBandK, uGain;
// Stars.  uStarPx is the angular size of ONE PIXEL, handed in by the caller,
// and it is what keeps this from crawling: see below.
uniform float uStarAmt, uStarPx, uStarDens, uStarFill, uStarWarm;

in vec2 vNdc;
out vec4 fragColor;

vec3 hash3(vec3 p) {
  p = fract(p * vec3(443.8975, 397.2973, 491.1871));
  p += dot(p, p.yzx + 19.19);
  return fract(vec3(p.x * p.y, p.y * p.z, p.z * p.x) * 37.0);
}

/**
 * A star field, on the cube's six faces, band-limited the same way the splats
 * are.
 *
 * The lobes above are smooth on purpose -- a noise nebula crawls, because the
 * camera's scale changes by factors of ten while it moves by fractions of a
 * unit.  Stars are points, which is exactly the thing that was ruled out, and
 * they are still here because the fix is the same one the splat shader has had
 * from the first version: never ask for something smaller than the sampling
 * grid.  Each star's angular sigma is floored at a bit over one pixel
 * (uStarPx), so no star is ever a sub-pixel spike to be sampled, and its
 * profile is a Gaussian rather than a disc, so it has no edge to alias
 * either.  They are fixed in direction, so they do not swim; they sweep as
 * the camera turns, which is what makes a turn read as a turn.
 *
 * Nine cells, not one: a star lives anywhere inside its cell, so its
 * footprint crosses into the neighbours and all nine have to be asked.
 */
vec3 stars(vec3 d) {
  vec3 ad = abs(d);
  float mx = max(ad.x, max(ad.y, ad.z));
  vec2 uv; float axis, sg;
  if (mx == ad.x)      { uv = d.yz / d.x; axis = 0.0; sg = d.x > 0.0 ? 1.0 : -1.0; }
  else if (mx == ad.y) { uv = d.zx / d.y; axis = 1.0; sg = d.y > 0.0 ? 1.0 : -1.0; }
  else                 { uv = d.xy / d.z; axis = 2.0; sg = d.z > 0.0 ? 1.0 : -1.0; }
  float fid = axis * 2.0 + (sg > 0.0 ? 0.0 : 1.0);
  vec2 g = uv * uStarDens, gi = floor(g);
  float sig = max(uStarPx * 1.3, 4.0e-4);
  vec3 acc = vec3(0.0);
  for (int j = -1; j <= 1; j++) {
    for (int i = -1; i <= 1; i++) {
      vec2 c = gi + vec2(float(i), float(j));
      vec3 r = hash3(vec3(c, fid));
      if (r.z > uStarFill) continue;               // most cells hold nothing
      // Inside its own cell, both axes: a star placed outside the cell that
      // owns it would be looked for in the wrong nine, and the ones that fell
      // two cells away would be clipped off at the edge of the search.
      vec2 q = (c + vec2(0.15) + 0.7 * r.xy) / uStarDens;
      vec3 sd = axis < 0.5 ? vec3(1.0, q.x, q.y)
              : axis < 1.5 ? vec3(q.y, 1.0, q.x)
                           : vec3(q.x, q.y, 1.0);
      sd = normalize(sg * sd);
      // 2*(1 - cos) is the angle squared to second order, and it needs no
      // acos -- which would be the one expensive call in this shader.
      float a2 = 2.0 * max(0.0, 1.0 - dot(d, sd));
      // Brightness from the same cell hash, cubed so that most stars are
      // faint and a few are not: a uniform field reads as noise.
      float br = r.z / max(uStarFill, 1e-6);
      br = br * br * br;
      vec3 col = mix(vec3(0.72, 0.84, 1.0), vec3(1.0, 0.86, 0.66),
                     uStarWarm * r.y);
      acc += col * br * exp(-0.5 * a2 / (sig * sig));
    }
  }
  return acc * uStarAmt;
}

void main() {
  // The ray through this pixel, in world space.
  vec3 d = normalize(uRvT * vec3(vNdc.x * uTan.x, vNdc.y * uTan.y, 1.0));

  vec3 c = mix(uDeep, uSky, 0.5 + 0.5 * d.y);

  for (int i = 0; i < 3; i++) {
    // exp(-k * (1 - cos)) -- a smooth cap around uLobeDir[i], brightest at
    // its centre and with no edge anywhere.
    float t = 1.0 - dot(d, normalize(uLobeDir[i]));
    c += uLobeCol[i] * exp(-uLobeK[i] * t);
  }

  // One broad band, for the look of a galactic plane: brightest where the ray
  // is perpendicular to the axis, so it girdles the sky.
  float b = 1.0 - abs(dot(d, normalize(uBandAxis)));
  float band = pow(clamp(b, 0.0, 1.0), uBandK);
  c += uBandCol * band;

  // Stars, dimmed where the band is brightest -- the galactic plane is the
  // one place a real sky has more dust than stars, and it keeps the brightest
  // part of the background from also being the busiest.
  if (uStarAmt > 0.0) c += stars(d) * (1.0 - 0.45 * band);

  fragColor = vec4(c * uGain, 1.0);
}
`;
