// A CPU rasteriser that transcribes js/shaders.js operation for operation.
//
// Its first job was to check the GLSL against py/splat_raster.py.  Its second,
// and the reason it lives here rather than in tests/, is that it is the only
// way to LOOK at what the adaptive walk produces without a browser -- which
// makes designing map sets possible at all.

import { deflateSync } from 'node:zlib';

export const KAPPA = 0.35, LOWPASS = 0.3, CUT = 3.0;
// The shader's near band, in world units: below ZN a splat is dropped, above
// ZF it is drawn in full, in between it fades.  A centre crossing the eye is
// a ramp, not a pop -- shaders.js carries the argument.
export const ZN = 0.02, ZF = 0.12;
const smoothstep = (a, b, x) => {
  const t = Math.min(Math.max((x - a) / (b - a), 0), 1);
  return t * t * (3 - 2 * t);
};

const nrm3 = (v) => { const l = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / l, v[1] / l, v[2] / l]; };
const fr = (x) => x - Math.floor(x);

function hash3(x, y, z) {
  let px = fr(x * 443.8975), py = fr(y * 397.2973), pz = fr(z * 491.1871);
  const d = px * (py + 19.19) + py * (pz + 19.19) + pz * (px + 19.19);
  px += d; py += d; pz += d;
  return [fr(px * py * 37), fr(py * pz * 37), fr(pz * px * 37)];
}

/**
 * The nebula background, on the CPU: the same lobes, band and stars the
 * BG_FRAG shader draws.
 *
 * Why it is here at all: the void's background is most of what the flight
 * looks like, and until now every headless frame was flat grey behind the
 * splats -- so the one part of the picture that could not be looked at was
 * the one nobody had ever seen run.
 *
 * What it is NOT: a parity transcription. The splat path in this file is
 * checked operation for operation against the GLSL and the Python reference,
 * because measurements depend on it. This is a preview. The star hash is
 * float32 in the shader and float64 here, so the two will not place stars in
 * the same pixels; the colours, the sizes and the band-limiting are what it
 * is here to show.
 *
 * `px` is the angular size of one pixel, exactly as the shader's uStarPx:
 * it is what floors a star's width so it cannot be a sub-pixel spike.
 */
export function skyColour(d, L, px) {
  const sky = L.sky || [0.034, 0.038, 0.056], deep = L.deep || [0.009, 0.011, 0.019];
  const t = 0.5 + 0.5 * d[1];
  const c = [deep[0] + (sky[0] - deep[0]) * t,
             deep[1] + (sky[1] - deep[1]) * t,
             deep[2] + (sky[2] - deep[2]) * t];
  const lc = L.lobeCol || [], ld = L.lobeDir || [], lk = L.lobeK || [];
  for (let i = 0; i < lc.length; i++) {
    const a = nrm3(ld[i]);
    const e = Math.exp(-lk[i] * (1 - (d[0] * a[0] + d[1] * a[1] + d[2] * a[2])));
    c[0] += lc[i][0] * e; c[1] += lc[i][1] * e; c[2] += lc[i][2] * e;
  }
  let band = 0;
  if (L.bandCol) {
    const a = nrm3(L.bandAxis || [0, 1, 0]);
    const b = 1 - Math.abs(d[0] * a[0] + d[1] * a[1] + d[2] * a[2]);
    band = Math.pow(Math.min(Math.max(b, 0), 1), L.bandK ?? 7);
    c[0] += L.bandCol[0] * band; c[1] += L.bandCol[1] * band;
    c[2] += L.bandCol[2] * band;
  }
  const amt = L.starAmt ?? 0;
  if (amt > 0) {
    const dens = L.starDens ?? 40, fill = L.starFill ?? 0.17, warm = L.starWarm ?? 0.55;
    const ax = Math.abs(d[0]), ay = Math.abs(d[1]), az = Math.abs(d[2]);
    const mx = Math.max(ax, ay, az);
    let u, v, axis, sg;
    if (mx === ax) { u = d[1] / d[0]; v = d[2] / d[0]; axis = 0; sg = d[0] > 0 ? 1 : -1; }
    else if (mx === ay) { u = d[2] / d[1]; v = d[0] / d[1]; axis = 1; sg = d[1] > 0 ? 1 : -1; }
    else { u = d[0] / d[2]; v = d[1] / d[2]; axis = 2; sg = d[2] > 0 ? 1 : -1; }
    const fid = axis * 2 + (sg > 0 ? 0 : 1);
    const gx = Math.floor(u * dens), gy = Math.floor(v * dens);
    const sig = Math.max(px * 1.3, 4e-4);
    for (let j = -1; j <= 1; j++) {
      for (let i = -1; i <= 1; i++) {
        const cx = gx + i, cy = gy + j;
        const r = hash3(cx, cy, fid);
        if (r[2] > fill) continue;
        const qx = (cx + 0.15 + 0.7 * r[0]) / dens, qy = (cy + 0.15 + 0.7 * r[1]) / dens;
        let sd = axis === 0 ? [1, qx, qy] : axis === 1 ? [qy, 1, qx] : [qx, qy, 1];
        sd = nrm3([sg * sd[0], sg * sd[1], sg * sd[2]]);
        const a2 = 2 * Math.max(0, 1 - (d[0] * sd[0] + d[1] * sd[1] + d[2] * sd[2]));
        let br = r[2] / Math.max(fill, 1e-6); br = br * br * br;
        const g = br * Math.exp((-0.5 * a2) / (sig * sig)) * amt * (1 - 0.45 * band);
        if (g < 1e-5) continue;
        const w = warm * r[1];
        c[0] += (0.72 + (1.0 - 0.72) * w) * g;
        c[1] += (0.84 + (0.86 - 0.84) * w) * g;
        c[2] += (1.0 + (0.66 - 1.0) * w) * g;
      }
    }
  }
  const gain = L.nebGain ?? 1;
  return [c[0] * gain, c[1] * gain, c[2] * gain];
}

export function basis(f, up = [0, 0, 1]) {
  const n = (v) => { const l = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / l, v[1] / l, v[2] / l]; };
  f = n(f);
  let r = [f[1] * up[2] - f[2] * up[1], f[2] * up[0] - f[0] * up[2],
           f[0] * up[1] - f[1] * up[0]];
  if (Math.hypot(r[0], r[1], r[2]) < 1e-7) r = [1, 0, 0];
  r = n(r);
  const u = [r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2],
             r[0] * f[1] - r[1] * f[0]];
  return { r, u, f };
}

/**
 * Blend a stream of splats, already in back-to-front order, over `bg`.
 *   pos  Float32Array, 3 per splat, starting at `off`
 *   cov  Float32Array, 6 per splat (c00 c01 c02 c11 c12 c22)
 *   rgba Uint8Array,   4 per splat
 */
export function render({ pos, cov, rgba, off = 0, count, eye, B, W, H, fovy = 55,
                         fx = null, look = null,
                         bg = [0.030, 0.035, 0.047], fogNear = 1e9, fogFar = 2e9,
                         sky = null, dilate = true, gamma = 1.1 }) {
  const F = (H / 2) / Math.tan((fovy * Math.PI) / 360);
  const buf = new Float64Array(W * H * 3);
  const { r: R3, u: U3, f: F3 } = B;
  // `sky` defaults to null, which means the flat clear colour -- the same
  // discipline as `look` below. Every washed-out, blank-frame and fidelity
  // number in this repository was measured against a flat background, and
  // tests/legible.mjs in particular measures SPATIAL DETAIL: a star field
  // behind the splats would raise its gradient score without a single splat
  // changing, and the autopilot would look like it had improved. A test that
  // wants to see the sky asks for it.
  if (sky) {
    const tY = Math.tan((fovy * Math.PI) / 360), tX = tY * (W / H);
    const px = (2 * tY) / H;
    for (let y = 0; y < H; y++) {
      const ny = 1 - (2 * (y + 0.5)) / H;
      for (let x = 0; x < W; x++) {
        const nx = (2 * (x + 0.5)) / W - 1;
        const vx = nx * tX, vy = ny * tY;
        const d = nrm3([R3[0] * vx + U3[0] * vy + F3[0],
                        R3[1] * vx + U3[1] * vy + F3[1],
                        R3[2] * vx + U3[2] * vy + F3[2]]);
        const c = skyColour(d, sky, px);
        const o = 3 * (y * W + x);
        buf[o] = c[0]; buf[o + 1] = c[1]; buf[o + 2] = c[2];
      }
    }
  } else {
    for (let i = 0; i < W * H; i++) {
      buf[3 * i] = bg[0]; buf[3 * i + 1] = bg[1]; buf[3 * i + 2] = bg[2];
    }
  }
  let drawn = 0;

  for (let k = 0; k < count; k++) {
    const id = off + k;
    const dx = pos[3 * id] - eye[0], dy = pos[3 * id + 1] - eye[1],
          dz = pos[3 * id + 2] - eye[2];
    const vx = R3[0] * dx + R3[1] * dy + R3[2] * dz;
    const vy = U3[0] * dx + U3[1] * dy + U3[2] * dz;
    const vz = F3[0] * dx + F3[1] * dy + F3[2] * dz;
    if (!(vz >= ZN)) continue;
    const nearFade = smoothstep(ZN, ZF, vz);

    const c = cov, o6 = 6 * id;
    const p2 = dilate ? (KAPPA * vz / F) ** 2 : 0;
    const S = [c[o6] + p2, c[o6 + 1], c[o6 + 2],
               c[o6 + 1], c[o6 + 3] + p2, c[o6 + 4],
               c[o6 + 2], c[o6 + 4], c[o6 + 5] + p2];

    const zi = 1 / vz;
    const J = [F * zi, 0, -F * vx * zi * zi, 0, -F * zi, F * vy * zi * zi];
    const Rv = [R3[0], R3[1], R3[2], U3[0], U3[1], U3[2], F3[0], F3[1], F3[2]];
    const Wm = new Float64Array(6);
    for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++)
      Wm[i * 3 + j] = J[i * 3] * Rv[j] + J[i * 3 + 1] * Rv[3 + j] + J[i * 3 + 2] * Rv[6 + j];
    const WS = new Float64Array(6);
    for (let i = 0; i < 2; i++) for (let j = 0; j < 3; j++)
      WS[i * 3 + j] = Wm[i * 3] * S[j] + Wm[i * 3 + 1] * S[3 + j] + Wm[i * 3 + 2] * S[6 + j];
    let a = WS[0] * Wm[0] + WS[1] * Wm[1] + WS[2] * Wm[2];
    let b = WS[0] * Wm[3] + WS[1] * Wm[4] + WS[2] * Wm[5];
    let d = WS[3] * Wm[3] + WS[4] * Wm[4] + WS[5] * Wm[5];

    const det0 = a * d - b * b;
    const lp = dilate ? LOWPASS : 0;
    a += lp; d += lp;
    const det1 = a * d - b * b;
    if (!(det1 > 0)) continue;
    const comp = Math.sqrt(Math.min(Math.max(det0, 0) / det1, 1));
    const con = [d / det1, -b / det1, a / det1];

    const mid = 0.5 * (a + d), rr = Math.sqrt(Math.max(mid * mid - det1, 0));
    const l1 = mid + rr, l2 = Math.max(mid - rr, 0.02);
    let ax1;
    if (Math.abs(b) < 1e-9) ax1 = a >= d ? [1, 0] : [0, 1];
    else { const nn = Math.hypot(b, l1 - a); ax1 = [b / nn, (l1 - a) / nn]; }
    const ax2 = [ax1[1], -ax1[0]];
    // Uncapped, as the shader's quad and py/splat_raster.py's radius both are:
    // the box below is clipped to the screen anyway, and a cap on the extent
    // cuts the kernel rather than the work.  (The 4096 that used to be here
    // never bit at test resolutions, which is half of why the shader's own
    // 2048 cap went unmeasured for so long.)
    const m1 = CUT * Math.sqrt(l1), m2 = CUT * Math.sqrt(l2);

    const cx = F * vx * zi + W / 2, cy = H / 2 - F * vy * zi;
    const rx = Math.abs(m1 * ax1[0]) + Math.abs(m2 * ax2[0]);
    const ry = Math.abs(m1 * ax1[1]) + Math.abs(m2 * ax2[1]);
    if (cx + rx < 0 || cx - rx >= W || cy + ry < 0 || cy - ry >= H) continue;
    const x0 = Math.max(0, Math.floor(cx - rx)), x1 = Math.min(W - 1, Math.ceil(cx + rx));
    const y0 = Math.max(0, Math.floor(cy - ry)), y1 = Math.min(H - 1, Math.ceil(cy + ry));

    const fog = 1 - Math.min(Math.max((vz - fogNear) / Math.max(fogFar - fogNear, 1e-6), 0), 1);
    const al = (rgba[4 * id + 3] / 255) * comp * fog * nearFade;
    if (al < 0.0039) continue;
    let cr = rgba[4 * id] / 255, cg = rgba[4 * id + 1] / 255, cb = rgba[4 * id + 2] / 255;

    // The same two look-effects as the vertex shader, kept in step with it by
    // hand.  `look` defaults to null, which means OFF -- deliberately, so
    // that every existing headless measurement (shot, legible, tour, the
    // fidelity A/Bs) keeps comparing exactly the images it compared before.
    // A cosmetic addition must not silently move the baselines that the
    // engineering decisions were made against; a test that wants to see the
    // effects asks for them.
    if (look) {
      const bits = (fx ? fx[id] : 0xff) & (look.mask ?? 3);
      const face = Math.sqrt(Math.min(Math.max(l2 / Math.max(l1, 1e-9), 0), 1));
      if (bits & 1) {
        const ph = (look.iridPhase ?? 0) + face * (look.iridScale ?? 1);
        const amt = look.iridAmt ?? 0.35;
        const s0 = 0.5 + 0.5 * Math.cos(6.28318531 * ph);
        const s1 = 0.5 + 0.5 * Math.cos(6.28318531 * (ph + 0.33333));
        const s2 = 0.5 + 0.5 * Math.cos(6.28318531 * (ph + 0.66667));
        cr += (cr * s0 * 2 - cr) * amt;
        cg += (cg * s1 * 2 - cg) * amt;
        cb += (cb * s2 * 2 - cb) * amt;
      }
      if (bits & 2) {
        const rim = Math.pow(1 - face, look.rimPow ?? 3) * (look.rimAmt ?? 0.5);
        const rc = look.rimColor ?? [0.6, 0.8, 1.0];
        cr += rc[0] * rim; cg += rc[1] * rim; cb += rc[2] * rim;
      }
    }
    drawn++;

    for (let y = y0; y <= y1; y++) {
      const ey = y + 0.5 - cy;
      for (let x = x0; x <= x1; x++) {
        const ex = x + 0.5 - cx;
        const e = con[0] * ex * ex + 2 * con[1] * ex * ey + con[2] * ey * ey;
        if (e < 0 || e > 18) continue;
        const av = al * Math.exp(-0.5 * e);
        if (av < 0.0039) continue;
        const o = 3 * (y * W + x);
        buf[o] += (cr - buf[o]) * av;
        buf[o + 1] += (cg - buf[o + 1]) * av;
        buf[o + 2] += (cb - buf[o + 2]) * av;
      }
    }
  }

  const img = new Uint8Array(W * H * 3);
  for (let i = 0; i < W * H * 3; i++)
    img[i] = Math.round(255 * Math.pow(Math.min(Math.max(buf[i], 0), 1), 1 / gamma));
  return { img, W, H, drawn };
}

// ------------------------------------------------------------------- PNG

function crc32(b) {
  let c, t = crc32.t;
  if (!t) {
    t = crc32.t = new Int32Array(256);
    for (let n = 0; n < 256; n++) {
      c = n;
      for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
      t[n] = c;
    }
  }
  c = -1;
  for (let i = 0; i < b.length; i++) c = t[(c ^ b[i]) & 0xff] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
}

function chunk(type, data) {
  const out = Buffer.alloc(data.length + 12);
  out.writeUInt32BE(data.length, 0);
  out.write(type, 4, 'ascii');
  Buffer.from(data).copy(out, 8);
  out.writeUInt32BE(crc32(out.subarray(4, 8 + data.length)), 8 + data.length);
  return out;
}

export function png({ img, W, H }) {
  const raw = Buffer.alloc(H * (W * 3 + 1));
  for (let y = 0; y < H; y++) {
    raw[y * (W * 3 + 1)] = 0;
    Buffer.from(img.buffer, img.byteOffset + y * W * 3, W * 3)
      .copy(raw, y * (W * 3 + 1) + 1);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(W, 0); ihdr.writeUInt32BE(H, 4);
  ihdr[8] = 8; ihdr[9] = 2;
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw, { level: 6 })),
    chunk('IEND', Buffer.alloc(0))]);
}

export function tile(images, cols, gap = 6) {
  const w = images[0].W, h = images[0].H;
  const rows = Math.ceil(images.length / cols);
  const W = cols * w + (cols - 1) * gap, H = rows * h + (rows - 1) * gap;
  const img = new Uint8Array(W * H * 3).fill(18);
  images.forEach((im, k) => {
    const ox = (k % cols) * (w + gap), oy = ((k / cols) | 0) * (h + gap);
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w * 3; x++)
        img[((oy + y) * W + ox) * 3 + x] = im.img[(y * w) * 3 + x];
  });
  return { img, W, H };
}
