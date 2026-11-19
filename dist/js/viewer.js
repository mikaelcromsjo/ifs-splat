// The viewer: one GL context, two ways of feeding it.
//
//   object mode (stage 1)  The splats are expanded once at load and never
//     move, so they live in textures.  Per frame only the ORDER changes:
//     ordered expansion writes an index buffer, ~N/3 nodes at five dot
//     products and nine compare-swaps each.  Press P to put the exact
//     per-splat sort beside it and see that they agree.
//
//   flight mode (stage 2)  There is no stored tree at all.  Every frame an
//     adaptive walk regenerates exactly the splats that are big enough to see,
//     already in draw order, in a frame re-rooted on whichever node the camera
//     is inside.  That is what lets the zoom go on for ever.

import { expand, orderFrame, makeScratch } from './ifs_core.js';
import { ExactSorter } from './sort_exact.js';
import { InversionCounter } from './inversions.js';
import { vertexSource, FRAG, BG_VERT, BG_FRAG } from './shaders.js';
import { AdaptiveStream, PALETTES } from './adaptive.js';
import { World, FieldWorld, VoidCamera, Autopilot, ringLayout } from './world.js';
import { SET_NAMES } from './maps.js';

const TEX_W = 2048;

// ------------------------------------------------------------------ GL help

function compile(gl, type, src, name) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(s);
    const lines = src.split('\n').map((l, i) => `${i + 1}: ${l}`).join('\n');
    console.error(lines);
    throw new Error(`${name} shader: ${log}`);
  }
  return s;
}

function program(gl, vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, vs, 'vertex'));
  gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, fs, 'fragment'));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS))
    throw new Error(`link: ${gl.getProgramInfoLog(p)}`);
  return p;
}

function dataTexture(gl, unit, internal, format, type, w, h, data) {
  const t = gl.createTexture();
  gl.activeTexture(gl.TEXTURE0 + unit);
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texImage2D(gl.TEXTURE_2D, 0, internal, w, h, 0, format, type, data);
  return t;
}

const norm = (v) => {
  const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l];
};
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                         a[0] * b[1] - a[1] * b[0]];

// ------------------------------------------------------------------ camera

export class Orbit {
  constructor(target, radius) {
    this.target = target.slice();
    this.radius = radius;
    this.az = 0.9;
    this.el = 0.22;
  }
  eye() {
    const ce = Math.cos(this.el), se = Math.sin(this.el);
    return [this.target[0] + this.radius * ce * Math.cos(this.az),
            this.target[1] + this.radius * ce * Math.sin(this.az),
            this.target[2] + this.radius * se];
  }
  basis() {
    const e = this.eye(), t = this.target;
    const f = norm([t[0] - e[0], t[1] - e[1], t[2] - e[2]]);
    let r = cross(f, [0, 0, 1]);
    if (Math.hypot(r[0], r[1], r[2]) < 1e-7) r = [1, 0, 0];
    r = norm(r);
    return { eye: e, r, u: cross(r, f), f };
  }
}

// ------------------------------------------------------------------ viewer

/**
 * The detail ladder, coarsest first.
 *
 * What a complexity dial has to drive here is the BUDGET, not the threshold.
 * The LOD holds a time budget: AdaptiveStream.end() nudges `lodScale` every
 * frame so the expansion costs `targetMs`, which is what lets one page give a
 * phone and a desktop each as much detail as it can afford. Set the threshold
 * directly and the controller has undone it by the next frame; raise the
 * budget and the controller spends it.
 *
 * Doubling steps, because the threshold is a length on screen and the splat
 * count goes about as its square -- a finer ladder would be rungs nobody can
 * tell apart.
 */
export const DETAIL = [
  { name: 'lowest',  ms: 1.5,  splats: 24000 },
  { name: 'low',     ms: 3.0,  splats: 48000 },
  { name: 'normal',  ms: 6.0,  splats: 96000 },
  { name: 'high',    ms: 12.0, splats: 192000 },
  { name: 'highest', ms: 24.0, splats: 340000 },
];

export class Viewer {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.opts = Object.assign({
      depth: 8, jitter: 0.16, seed: 3, palette: 'tree',
      kappa: 0.35, lowpass: 0.3, cut: 3.0, fovy: 55,
      maxSplats: 400000, targetMs: 6.0,
      bg: [0.030, 0.035, 0.047],
      // The look. Every effect is switchable, and `mask` is the master: an
      // object's own flags are ANDed with it, so turning one off in the UI
      // turns it off everywhere regardless of what any object asked for.
      look: {
        mask: 3,                   // bit 0 iridescence, bit 1 rim
        iridAmt: 0.38, iridScale: 1.1, iridPhase: 0.0,
        // A brighter, cooler, NARROWER rim. The rim light is the only thing
        // in the shader that draws a SILHOUETTE, and the silhouette is what
        // the dive now flies along, so it is worth more than it was -- but
        // only at the edges. Widening it as well (rimPow 2.5) lit most of a
        // dense canopy and the picture went milky, which is the fault this
        // whole stage is trying to remove. Narrower than it started (2.8,
        // was 3.0) and brighter (0.62, was 0.55): more edge, no haze.
        rimAmt: 0.62, rimPow: 2.8, rimColor: [0.60, 0.74, 1.0],
        nebula: true,
        // Broader, more saturated lobes (lobeK down, colours up): a deep sky
        // with colour in it rather than a dark grey one, which is what makes
        // fog read as distance instead of as fading out.
        sky: [0.040, 0.043, 0.068], deep: [0.006, 0.008, 0.018],
        lobeCol: [[0.090, 0.024, 0.100], [0.016, 0.062, 0.086], [0.082, 0.044, 0.018]],
        lobeDir: [[0.6, 0.5, -0.6], [-0.7, 0.2, 0.68], [0.2, -0.8, 0.55]],
        lobeK: [3.6, 6.0, 9.0],
        bandCol: [0.046, 0.040, 0.066], bandAxis: [0.25, 0.93, 0.27],
        bandK: 5.5, nebGain: 1.0,
        // Stars. `starDens` is cells per cube-face unit and `starFill` the
        // fraction of cells that hold one, so the two together set the count;
        // `starAmt` is brightness and 0 turns the whole thing off.
        starAmt: 0.85, starDens: 40.0, starFill: 0.17, starWarm: 0.55,
      },
    }, opts);
    // A live, mutable copy: the UI writes straight into this, so a toggle
    // takes effect on the next frame with nothing to rebuild or reupload.
    this.look = Object.assign({}, this.opts.look, opts.look || {});

    const gl = canvas.getContext('webgl2', {
      antialias: false, premultipliedAlpha: true, alpha: false,
      powerPreference: 'high-performance', desynchronized: true,
    });
    if (!gl) throw new Error('WebGL2 is not available in this browser.');
    this.gl = gl;

    this.quad = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    gl.bufferData(gl.ARRAY_BUFFER,
                  new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);

    // The background is the one program that may fail without taking the
    // demo with it.
    //
    // Nothing in this repository has ever compiled a shader -- there is no
    // GL here -- so the star field's loop and its five new uniforms are
    // going out to real drivers unexercised. The splat programs below are
    // unchanged from what is already live and stay fatal; a background that
    // will not compile falls back to the flat clear colour, which is what
    // the demo looked like before it existed. A sky is not worth a black
    // page.
    try {
      this.BG = this.makeBgProgram();
    } catch (e) {
      this.BG = null;
      this.look.nebula = false;
      this.bgError = String(e && e.message || e);
      if (typeof console !== 'undefined')
        console.warn('nebula background disabled:', this.bgError);
    }
    this.P = this.makeProgram('static');
    this.S = this.makeProgram('stream');

    this.mode = 'ordered';        // 'ordered' | 'exact'   (object mode)
    // Open on the field, not the single object.  The static object is stage 1
    // and still reachable with F (it is what the proof panel measures), but
    // the thing worth showing first is the world that generates itself.
    this.scene = this.opts.scene || 'void';   // 'object' (stage 1) | 'void' (the field)
    this.proof = false;
    this.dilate = true;
    this.autopilot = true;
    this.stats = { order: 0, exact: 0, expand: 0, morph: 0, frame: 0, fps: 0,
                   splats: 0, depth: 0, scale: 1, here: '', inversion: null,
                   // How often the drawn shape changes, and how often it is
                   // rebuilt: the same slot-frames tests/breath.mjs counts,
                   // measured live. With interpolation on the first should
                   // sit on the frame rate and the second far below it --
                   // which is the claim the panel is now making in public.
                   shapeHz: 0, rebuildHz: 0 };
    this._frames = 0; this._t0 = performance.now(); this._last = this._t0;
    this.keys = new Set();

    // Which rung the ladder starts on: whichever one the configured budget
    // asks for, so opts.targetMs stays the single default and this does not
    // become a second one.
    // The near-side lump fade (AdaptiveStream.rodFade). On by default; the
    // switch is here because the only way to judge it is to fly into
    // something and watch the fog go.
    this.dissolve = this.opts.dissolve ?? true;

    this.detail = 0;
    for (let i = 1; i < DETAIL.length; i++) {
      if (Math.abs(DETAIL[i].ms - this.opts.targetMs)
          < Math.abs(DETAIL[this.detail].ms - this.opts.targetMs)) this.detail = i;
    }

    this.buildObject(this.opts.depth);
    this.buildFlight();
    this.bindEvents();
  }

  /** The nebula background: one fullscreen quad, drawn before anything else. */
  makeBgProgram() {
    const gl = this.gl;
    const p = program(gl, BG_VERT, BG_FRAG);
    const u = {};
    for (const k of ['uRvT', 'uTan', 'uSky', 'uDeep', 'uLobeCol', 'uLobeDir',
                     'uLobeK', 'uBandCol', 'uBandAxis', 'uBandK', 'uGain',
                     'uStarAmt', 'uStarPx', 'uStarDens', 'uStarFill', 'uStarWarm'])
      u[k] = gl.getUniformLocation(p, k);
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    const corner = gl.getAttribLocation(p, 'aCorner');
    gl.enableVertexAttribArray(corner);
    gl.vertexAttribPointer(corner, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
    return { p, u, vao };
  }

  drawNebula(b, W, H) {
    const gl = this.gl, L = this.look, G = this.BG;
    const tanY = Math.tan((this.opts.fovy * Math.PI) / 360);
    gl.useProgram(G.p);
    gl.bindVertexArray(G.vao);
    // view -> world: the splat shader's uRv transposed, so its columns are
    // the camera's right, up and forward.
    gl.uniformMatrix3fv(G.u.uRvT, false, [
      b.r[0], b.r[1], b.r[2], b.u[0], b.u[1], b.u[2], b.f[0], b.f[1], b.f[2]]);
    gl.uniform2f(G.u.uTan, tanY * (W / H), tanY);
    gl.uniform3fv(G.u.uSky, L.sky);
    gl.uniform3fv(G.u.uDeep, L.deep);
    gl.uniform3fv(G.u.uLobeCol, L.lobeCol.flat());
    gl.uniform3fv(G.u.uLobeDir, L.lobeDir.flat());
    gl.uniform3fv(G.u.uLobeK, L.lobeK);
    gl.uniform3fv(G.u.uBandCol, L.bandCol);
    gl.uniform3fv(G.u.uBandAxis, L.bandAxis);
    gl.uniform1f(G.u.uBandK, L.bandK);
    gl.uniform1f(G.u.uGain, L.nebGain);
    gl.uniform1f(G.u.uStarAmt, L.starAmt ?? 0);
    gl.uniform1f(G.u.uStarDens, L.starDens ?? 40);
    gl.uniform1f(G.u.uStarFill, L.starFill ?? 0.17);
    gl.uniform1f(G.u.uStarWarm, L.starWarm ?? 0.55);
    // The angular size of one pixel: two half-fields of view across H pixels.
    // This is what floors each star's width, so it has to come from the real
    // viewport and not from a guess -- a star sized for a 1080-tall window is
    // a sub-pixel spike on a 4K one, and sub-pixel spikes crawl.
    gl.uniform1f(G.u.uStarPx, (2 * tanY) / Math.max(H, 1));
    // Opaque, and it must not blend with the clear it is replacing.
    const wasBlend = gl.isEnabled(gl.BLEND);
    gl.disable(gl.BLEND);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    if (wasBlend) gl.enable(gl.BLEND);
    gl.bindVertexArray(null);
  }

  makeProgram(kind) {
    const gl = this.gl;
    const p = program(gl, vertexSource(kind), FRAG);
    const u = {};
    for (const k of ['uEye', 'uRv', 'uFocal', 'uViewport', 'uKappa', 'uLowpass',
                     'uCut', 'uTexW', 'uPos', 'uCovA', 'uCovB', 'uCol',
                     'uFogNear', 'uFogFar',
                     'uFxMask', 'uIridAmt', 'uIridScale', 'uIridPhase',
                     'uRimAmt', 'uRimPow', 'uRimColor'])
      u[k] = gl.getUniformLocation(p, k);
    const vao = gl.createVertexArray();
    gl.bindVertexArray(vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quad);
    const a = { corner: gl.getAttribLocation(p, 'aCorner') };
    gl.enableVertexAttribArray(a.corner);
    gl.vertexAttribPointer(a.corner, 2, gl.FLOAT, false, 0, 0);
    if (kind === 'static') a.splat = gl.getAttribLocation(p, 'aSplat');
    else {
      a.pos = gl.getAttribLocation(p, 'aPos');
      a.covA = gl.getAttribLocation(p, 'aCovA');
      a.covB = gl.getAttribLocation(p, 'aCovB');
      a.color = gl.getAttribLocation(p, 'aColor');
      a.fx = gl.getAttribLocation(p, 'aFx');
    }
    gl.bindVertexArray(null);
    return { p, u, vao, a };
  }

  // --------------------------------------------- stage 1: the static object

  buildObject(depth) {
    const gl = this.gl;
    const t0 = performance.now();
    const tree = expand(depth, this.opts.jitter, this.opts.seed);
    this.tree = tree;
    this.genMs = performance.now() - t0;

    const N = tree.N, p = tree.pos;
    let mx = 0, my = 0, mz = Infinity;
    for (let i = 0, j = 0; i < N; i++, j += 3) {
      mx += p[j]; my += p[j + 1];
      if (p[j + 2] < mz) mz = p[j + 2];
    }
    mx /= N; my /= N;
    let lo = [1e30, 1e30, 1e30], hi = [-1e30, -1e30, -1e30];
    for (let i = 0, j = 0; i < N; i++, j += 3) {
      p[j] -= mx; p[j + 1] -= my; p[j + 2] -= mz;
      for (let k = 0; k < 3; k++) {
        if (p[j + k] < lo[k]) lo[k] = p[j + k];
        if (p[j + k] > hi[k]) hi[k] = p[j + k];
      }
    }

    const texH = Math.ceil(N / TEX_W), n4 = TEX_W * texH * 4;
    const posD = new Float32Array(n4), cvA = new Float32Array(n4);
    const cvB = new Float32Array(n4), colD = new Uint8Array(n4);
    for (let i = 0; i < N; i++) {
      posD[4 * i] = p[3 * i]; posD[4 * i + 1] = p[3 * i + 1];
      posD[4 * i + 2] = p[3 * i + 2]; posD[4 * i + 3] = tree.level[i];
      for (let k = 0; k < 4; k++) cvA[4 * i + k] = tree.cov[6 * i + k];
      cvB[4 * i] = tree.cov[6 * i + 4]; cvB[4 * i + 1] = tree.cov[6 * i + 5];
      for (let k = 0; k < 4; k++) colD[4 * i + k] = tree.rgba[4 * i + k];
    }
    for (const t of this.tex || []) gl.deleteTexture(t);
    this.tex = [
      dataTexture(gl, 0, gl.RGBA32F, gl.RGBA, gl.FLOAT, TEX_W, texH, posD),
      dataTexture(gl, 1, gl.RGBA32F, gl.RGBA, gl.FLOAT, TEX_W, texH, cvA),
      dataTexture(gl, 2, gl.RGBA32F, gl.RGBA, gl.FLOAT, TEX_W, texH, cvB),
      dataTexture(gl, 3, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, TEX_W, texH, colD),
    ];

    this.scratch = makeScratch(tree);
    this.inv = new InversionCounter(N);
    this.alphaOf = new Float64Array(N);
    for (let i = 0; i < N; i++) this.alphaOf[i] = tree.rgba[4 * i + 3] / 255;
    this.invAt = 0;
    this.orderArr = new Int32Array(N);
    this.exactArr = new Int32Array(N);
    this.sorter = new ExactSorter(N);
    for (const b of [this.idxOrdered, this.idxExact]) if (b) gl.deleteBuffer(b);
    this.idxOrdered = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.idxOrdered);
    gl.bufferData(gl.ARRAY_BUFFER, N * 4, gl.DYNAMIC_DRAW);
    this.idxExact = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.idxExact);
    gl.bufferData(gl.ARRAY_BUFFER, N * 4, gl.DYNAMIC_DRAW);

    const span = Math.hypot(hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
    this.cam = new Orbit([0, 0, (lo[2] + hi[2]) / 2], span * 1.05);
    this.span = span;
  }

  // --------------------------------------------- stage 2: the endless flight

  buildFlight() {
    const gl = this.gl;
    const wopts = { morphing: this.opts.morphing ?? true,
                    morphSeconds: this.opts.morphSeconds ?? 14,
                    holdSeconds: this.opts.holdSeconds ?? 10 };
    this.world = (this.opts.field ?? true)
      ? new FieldWorld({ ...wopts, ...(this.opts.fieldOpts || {}) })
      : new World(ringLayout(SET_NAMES, this.opts.ring), wopts);
    this.stream = new AdaptiveStream(this.world.get(this.world.first()).att, this.opts.maxSplats);
    this.setDetail(this.detail);
    this.stream.rodFade = this.dissolve;
    this.resetFlight();

    const N = this.stream.max;
    this.bPos = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bPos);
    gl.bufferData(gl.ARRAY_BUFFER, N * 12, gl.DYNAMIC_DRAW);
    this.bCov = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bCov);
    gl.bufferData(gl.ARRAY_BUFFER, N * 24, gl.DYNAMIC_DRAW);
    this.bCol = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bCol);
    gl.bufferData(gl.ARRAY_BUFFER, N * 4, gl.DYNAMIC_DRAW);
    this.bFx = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bFx);
    gl.bufferData(gl.ARRAY_BUFFER, N, gl.DYNAMIC_DRAW);

    const S = this.S;
    gl.bindVertexArray(S.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bPos);
    gl.enableVertexAttribArray(S.a.pos);
    gl.vertexAttribPointer(S.a.pos, 3, gl.FLOAT, false, 12, 0);
    gl.vertexAttribDivisor(S.a.pos, 1);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bCov);
    gl.enableVertexAttribArray(S.a.covA);
    gl.vertexAttribPointer(S.a.covA, 3, gl.FLOAT, false, 24, 0);
    gl.vertexAttribDivisor(S.a.covA, 1);
    gl.enableVertexAttribArray(S.a.covB);
    gl.vertexAttribPointer(S.a.covB, 3, gl.FLOAT, false, 24, 12);
    gl.vertexAttribDivisor(S.a.covB, 1);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bCol);
    gl.enableVertexAttribArray(S.a.color);
    gl.vertexAttribPointer(S.a.color, 4, gl.UNSIGNED_BYTE, true, 4, 0);
    gl.vertexAttribDivisor(S.a.color, 1);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bFx);
    gl.enableVertexAttribArray(S.a.fx);
    // NOT normalised: the shader wants the byte's value as a bit field, and
    // normalising would hand it 1/255ths to round back.
    gl.vertexAttribPointer(S.a.fx, 1, gl.UNSIGNED_BYTE, false, 1, 0);
    gl.vertexAttribDivisor(S.a.fx, 1);
    gl.bindVertexArray(null);
  }

  // The two morph switches the panel drives. Accessors rather than fields,
  // for the same reason the keyboard flags are: there is one piece of state
  // and it lives in the thing that does the work, so a box and a reload and
  // a test all read the same answer. The field world forwards both to its
  // pool; the ring world (opts.field false) has no pool and no
  // interpolation, which is what `canInterpolate` is for.
  get morphing() { return !this.world || this.world.morphing !== false; }
  set morphing(on) { if (this.world) this.world.morphing = !!on; }
  get canInterpolate() { return !!(this.world && this.world.pool); }
  get interpolate() { return this.canInterpolate && this.world.interpolate !== false; }
  set interpolate(on) { if (this.canInterpolate) this.world.interpolate = !!on; }

  resetFlight() {
    const W = this.world;
    const id = W.first();
    this.fly = new VoidCamera(W, id);
    const h = W.get(id), c = h.worldCentre(), r = h.worldRadius();
    // the ring is a fixed arrangement to stand back from; the field has no
    // outside, so there the camera starts just clear of whichever cell it
    // woke up in
    const back = W.layout
      ? [c[0] - r * 2.6, c[1] - r * 1.1, c[2] + r * 1.7]
      : [c[0] * 1.9 - 4, c[1] * 1.9 - 4, c[2] + r * 2.4];
    this.fly.pos = [(back[0] - h.t[0]) / h.s, (back[1] - h.t[1]) / h.s,
                    (back[2] - h.t[2]) / h.s];
    this.fly.f = norm([c[0] - back[0], c[1] - back[1], c[2] - back[2]]);
    this.fly.u = [0, 0, 1];
    this.fly.ortho();
    this.pilot = new Autopilot(W, this.fly);
    // The autopilot's own speed, not 0.02.
    //
    // frameFlight passes `flySpeed` into pilot.step(), which overrides
    // Autopilot.speed -- so the browser flew at 0.02 while every headless
    // measurement (tour, legible, flight) used the Autopilot default. They
    // were measuring a flight nobody ever saw, nearly twice as slow as the
    // one on screen. The wheel still scales this; it just starts where the
    // tests are.
    this.flySpeed = this.pilot.speed;
    if (this.stream) this.stream.lodScale = 4.0;
  }


  /** Fade a node's own lump out as it grows too big to be honest, leaving
   *  the finer lumps of its children -- which the walk already draws -- to
   *  carry the picture. */
  setDissolve(on) {
    this.dissolve = !!on;
    if (this.stream) this.stream.rodFade = this.dissolve;
    return this.dissolve;
  }

  /**
   * Move the detail ladder.  Clamped at both ends, and it does NOT reset
   * `lodScale`: the controller is already holding a threshold that worked,
   * and letting it walk to the new budget from there is what keeps the change
   * inside the cross-fade band instead of flipping a frame's worth of nodes
   * at once.  It converges in a few frames, which is the same path a cold
   * start takes.
   */
  setDetail(i) {
    this.detail = Math.max(0, Math.min(DETAIL.length - 1, i | 0));
    const d = DETAIL[this.detail];
    if (this.stream) {
      // Both, and the stream honours whichever binds first -- so a rung
      // always costs fewer splats as well as less time, which is the half a
      // clock alone could not promise. The top rung stays under maxSplats so
      // the ceiling is this dial and not the buffer running out.
      this.stream.budgetBy = 'both';
      this.stream.targetMs = d.ms;
      this.stream.target = Math.min(d.splats, Math.round(this.opts.maxSplats * 0.85));
    }
    return this.detail;
  }

  // ------------------------------------------------------------ interaction

  bindEvents() {
    const c = this.canvas;
    let drag = null;
    c.addEventListener('contextmenu', (e) => e.preventDefault());
    c.addEventListener('pointerdown', (e) => {
      drag = { x: e.clientX, y: e.clientY, b: e.button };
      c.setPointerCapture(e.pointerId);
    });
    const end = (e) => {
      drag = null;
      try { c.releasePointerCapture(e.pointerId); } catch (_) {}
    };
    c.addEventListener('pointerup', end);
    c.addEventListener('pointercancel', end);
    c.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      drag.x = e.clientX; drag.y = e.clientY;
      if (this.scene === 'void') {
        this.autopilot = false;
        this.fly.orient(-dx * 0.004, -dy * 0.004);
      } else if (drag.b === 0) {
        this.cam.az -= dx * 0.006;
        this.cam.el = Math.max(-1.45, Math.min(1.45, this.cam.el + dy * 0.006));
      } else {
        const s = this.cam.radius * 0.0015, b = this.cam.basis();
        for (let k = 0; k < 3; k++)
          this.cam.target[k] += -b.r[k] * dx * s + b.u[k] * dy * s;
      }
    });
    c.addEventListener('wheel', (e) => {
      e.preventDefault();
      if (this.scene === 'void') {
        this.flySpeed = Math.max(0.0015, Math.min(0.2,
          (this.flySpeed ?? this.pilot.speed) * Math.exp(-e.deltaY * 0.0012)));
      } else {
        this.cam.radius = Math.max(1e-3, Math.min(1e4,
          this.cam.radius * Math.exp(e.deltaY * 0.0012)));
      }
    }, { passive: false });

    addEventListener('keydown', (e) => {
      const k = e.key.toLowerCase();
      if (k === 'f') { this.scene = this.scene === 'void' ? 'object' : 'void'; }
      else if (k === 'o') this.mode = this.mode === 'ordered' ? 'exact' : 'ordered';
      else if (k === 'p') this.proof = !this.proof;
      else if (k === 'm') this.dilate = !this.dilate;
      else if (k === ' ') this.autopilot = !this.autopilot;
      else if (k === 'r') this.resetFlight();
      // Detail, on the keys that every other viewer puts it on. '=' as well as
      // '+', because '+' is a shifted key and nobody holds shift for it.
      else if (k === '-' || k === '_') this.setDetail(this.detail - 1);
      else if (k === '=' || k === '+') this.setDetail(this.detail + 1);
      else if ('wasdqe'.includes(k)) { this.keys.add(k); this.autopilot = false; return; }
      else return;
      e.preventDefault();
      if (this.onstate) this.onstate();
    });
    addEventListener('keyup', (e) => this.keys.delete(e.key.toLowerCase()));
    addEventListener('blur', () => this.keys.clear());
  }

  // ------------------------------------------------------------------ frame

  resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w; this.canvas.height = h;
    }
  }

  setCommonUniforms(G, b, W, H, f) {
    const gl = this.gl;
    gl.uniform3fv(G.u.uEye, b.eye);
    gl.uniformMatrix3fv(G.u.uRv, false, [
      b.r[0], b.u[0], b.f[0], b.r[1], b.u[1], b.f[1], b.r[2], b.u[2], b.f[2]]);
    gl.uniform2f(G.u.uFocal, f, f);
    gl.uniform2f(G.u.uViewport, W, H);
    gl.uniform1f(G.u.uKappa, this.dilate ? this.opts.kappa : 0.0);
    gl.uniform1f(G.u.uLowpass, this.dilate ? this.opts.lowpass : 0.0);
    gl.uniform1f(G.u.uCut, this.opts.cut);
    gl.uniform1i(G.u.uTexW, TEX_W);
    gl.uniform1f(G.u.uFogNear, this.fogNear ?? 1e9);
    gl.uniform1f(G.u.uFogFar, this.fogFar ?? 2e9);
    const L = this.look;
    gl.uniform1i(G.u.uFxMask, L.mask);
    gl.uniform1f(G.u.uIridAmt, L.iridAmt);
    gl.uniform1f(G.u.uIridScale, L.iridScale);
    gl.uniform1f(G.u.uIridPhase, L.iridPhase);
    gl.uniform1f(G.u.uRimAmt, L.rimAmt);
    gl.uniform1f(G.u.uRimPow, L.rimPow);
    gl.uniform3fv(G.u.uRimColor, L.rimColor);
  }

  frame() {
    const gl = this.gl, t0 = performance.now();
    const dt = Math.min((t0 - this._last) / 16.67, 4);     // in 60fps ticks
    this._last = t0;
    this.resize();
    const W = this.canvas.width, H = this.canvas.height;
    const f = (H / 2) / Math.tan((this.opts.fovy * Math.PI) / 360);

    gl.viewport(0, 0, W, H);
    const bg = this.opts.bg;
    gl.clearColor(bg[0], bg[1], bg[2], 1);
    gl.disable(gl.SCISSOR_TEST);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.ONE, gl.ONE_MINUS_SRC_ALPHA, gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    if (this.scene === 'void') this.frameFlight(dt, W, H, f);
    else this.frameObject(W, H, f);

    this.stats.frame = performance.now() - t0;
    this._frames++;
    if (t0 - this._t0 > 500) {
      this.stats.fps = (this._frames * 1000) / (t0 - this._t0);
      this._frames = 0; this._t0 = t0;
    }
  }

  frameObject(W, H, f) {
    const gl = this.gl, b = this.cam.basis(), N = this.tree.N;
    this.fogNear = 1e9; this.fogFar = 2e9;              // no fog on the object
    const wantOrdered = this.proof || this.mode === 'ordered';
    const wantExact = this.proof || this.mode === 'exact';
    if (wantOrdered) {
      const t = performance.now();
      orderFrame(this.tree, b.f, this.orderArr, this.scratch);
      this.stats.order = performance.now() - t;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.idxOrdered);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.orderArr);
    }
    if (wantExact) {
      const t = performance.now();
      this.sorter.sort(this.tree.pos, b.f, this.exactArr);
      this.stats.exact = performance.now() - t;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.idxExact);
      gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.exactArr);
    }
    this.stats.splats = N;
    this.stats.expand = 0;

    // The proof panel's number.  Both orders are in hand only while the panel
    // is open, and a full count is O(N log N) -- around fifteen milliseconds
    // at this size, which would halve the frame rate it is meant to be
    // reporting.  So: every sixteenth splat, which estimates the same ratio
    // without bias, and only every tenth frame.
    if (this.proof && --this.invAt <= 0) {
      this.invAt = 10;
      const t = performance.now();
      const r = this.inv.sampled(this.orderArr, this.exactArr, this.alphaOf, N, 16);
      this.stats.inversion = r.rate;
      this.stats.inversionMs = performance.now() - t;
    }

    const G = this.P;
    gl.useProgram(G.p);
    gl.bindVertexArray(G.vao);
    this.setCommonUniforms(G, b, W, H, f);
    gl.uniform1i(G.u.uPos, 0); gl.uniform1i(G.u.uCovA, 1);
    gl.uniform1i(G.u.uCovB, 2); gl.uniform1i(G.u.uCol, 3);
    for (let i = 0; i < 4; i++) {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, this.tex[i]);
    }
    const draw = (buf) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.enableVertexAttribArray(G.a.splat);
      gl.vertexAttribIPointer(G.a.splat, 1, gl.UNSIGNED_INT, 0, 0);
      gl.vertexAttribDivisor(G.a.splat, 1);
      gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, N);
    };
    if (this.proof) {
      const half = (W / 2) | 0;
      gl.enable(gl.SCISSOR_TEST);
      gl.scissor(0, 0, half, H);
      draw(this.idxExact);                       // left: exact per-splat sort
      gl.scissor(half, 0, W - half, H);
      draw(this.idxOrdered);                     // right: ordered expansion
      gl.disable(gl.SCISSOR_TEST);
    } else {
      draw(this.mode === 'ordered' ? this.idxOrdered : this.idxExact);
    }
  }

  frameFlight(dt, W, H, f) {
    const gl = this.gl, cam = this.fly, st = this.stream;
    // The autopilot's rate and the hand-flown rate are not the same number.
    // `flySpeed` is a geometric time constant for the autopilot -- a fraction
    // of the distance remaining -- while WASD moves an absolute distance per
    // frame, so handing the autopilot's 0.0062 straight to the keys would
    // make manual flight three times slower than it has always been. Scaled
    // back to the 0.02 it was, and still driven by the same wheel.
    const speed = (this.flySpeed ?? this.pilot.speed) * 3.2;
    this.world.update((dt * 16.67) / 1000, this.fly);
    if (this.autopilot) this.pilot.step(dt, this.flySpeed ?? this.pilot.speed);
    else {
      const k = this.keys, s = speed * dt * 1.2;
      const dr = (k.has('d') ? s : 0) - (k.has('a') ? s : 0);
      const du = (k.has('e') ? s : 0) - (k.has('q') ? s : 0);
      const df = (k.has('w') ? s : 0) - (k.has('s') ? s : 0);
      if (dr || du || df) cam.move(dr, du, df);
    }
    cam.reroot();
    cam.retarget();

    // Depth fog, in LOCAL units.  That frame rescales every time the camera
    // descends, so one rule covers both regimes: out in the void it has to
    // reach across the ring or the far attractors are simply erased, and
    // inside one it has to close right in or it does nothing at all, because
    // everything around the camera is then within a couple of local units.
    // Tying it to how far the camera is from what it is looking at does both.
    const span = Math.max(2.5, 2.2 * Math.hypot(
      cam.pos[0] - this.world.get(cam.home).att.C[0],
      cam.pos[1] - this.world.get(cam.home).att.C[1],
      cam.pos[2] - this.world.get(cam.home).att.C[2]));
    let near = span * 0.9, far = span * 7.0;
    if (this.world.layout) {
      // In a field, fog is also the horizon.  The rule above is about the
      // object the camera is in and knows nothing about where the generated
      // region stops, so left alone it reaches well past it and cells switch
      // on in clear view at the generation radius.  Fog far has to sit INSIDE
      // that radius, so what arrives is a cell fading up out of the murk.
      //
      // The conversion is the chain in reverse: world -> home's local is a
      // divide by the home cell's scale, local -> the camera's working frame
      // is rootFrame's kappa, which grows as the camera descends.  That is
      // what makes one rule cover both regimes -- dive in and the rest of the
      // field recedes to millions of working units and fogs out on its own.
      const home = this.world.get(cam.home);
      const edge = 0.75 * this.world.layout.genRadius / home.s * cam.rootFrame({}).kap;
      if (edge < far) { far = edge; near = Math.min(near, far * 0.14); }
    }
    this.fogNear = this.opts.fogNear ?? near;
    this.fogFar = this.opts.fogFar ?? far;

    const tanY = Math.tan((this.opts.fovy * Math.PI) / 360);
    const t = performance.now();
    const n = this.world.expand(st, cam, { focal: f, tanX: tanY * (W / H), tanY,
                                           znear: 1e-5, W, H });
    this.stats.expand = performance.now() - t;
    this.stats.order = 0; this.stats.exact = 0;
    this.stats.splats = n;
    this.stats.depth = cam.depth;
    this.stats.scale = cam.scale;
    this.stats.here = this.world.get(cam.home).name;
    this.stats.morph = this.world.reparamMs;
    this.stats.shapeHz = this.world.shapeHz ?? 0;
    this.stats.rebuildHz = this.world.rebuildHzNow ?? 0;
    this.stats.resident = this.world.length;
    this.stats.visible = this.world.visible;
    this.stats.phase = this.pilot.phase;

    // Before the splats, and before the early return below: an empty frame
    // is exactly the frame that most needs a background that is not flat.
    if (this.look.nebula && this.BG) this.drawNebula({ r: cam.r, u: cam.u, f: cam.f }, W, H);
    if (!n) return;

    const o = st.offset;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bPos);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, st.pos, 3 * o, 3 * n);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bCov);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, st.cov, 6 * o, 6 * n);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bCol);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, st.rgba, 4 * o, 4 * n);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bFx);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, st.fx, o, n);

    const G = this.S;
    gl.useProgram(G.p);
    gl.bindVertexArray(G.vao);
    this.setCommonUniforms(G, { eye: cam.pos, r: cam.r, u: cam.u, f: cam.f }, W, H, f);
    gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 4, n);
  }

  start() {
    const loop = () => { this.frame(); this._raf = requestAnimationFrame(loop); };
    this._raf = requestAnimationFrame(loop);
  }
}
