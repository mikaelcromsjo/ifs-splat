#!/usr/bin/env python3
"""Quick orthographic splat rasteriser, just to eyeball a .ply. numpy + zlib only."""
import sys, zlib, struct
import numpy as np

C0 = 0.28209479177387814

def read_ply(path):
    with open(path, 'rb') as f:
        hdr = b''
        while not hdr.endswith(b'end_header\n'):
            hdr += f.read(1)
        lines = hdr.decode().splitlines()
        n = int([l for l in lines if l.startswith('element vertex')][0].split()[-1])
        props = [l.split()[-1] for l in lines if l.startswith('property float')]
        d = np.frombuffer(f.read(n * len(props) * 4), dtype='<f4').reshape(n, len(props))
    col = {k: i for i, k in enumerate(props)}
    g = lambda *k: d[:, [col[x] for x in k]]
    return (g('x','y','z').astype(np.float64),
            np.exp(g('scale_0','scale_1','scale_2')).astype(np.float64),
            g('rot_0','rot_1','rot_2','rot_3').astype(np.float64),
            np.clip(g('f_dc_0','f_dc_1','f_dc_2') * C0 + 0.5, 0, 1).astype(np.float64),
            (1/(1+np.exp(-d[:, col['opacity']]))).astype(np.float64))

def quat_to_R(q):
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w,x,y,z = q.T
    return np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], 1).reshape(-1,3,3)

def write_png(path, img):
    h, w, _ = img.shape
    raw = b''.join(b'\x00' + img[r].tobytes() for r in range(h))
    def chunk(t, b):
        c = t + b
        return struct.pack('>I', len(b)) + c + struct.pack('>I', zlib.crc32(c))
    png = (b'\x89PNG\r\n\x1a\n'
           + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 2, 0, 0, 0))
           + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b''))
    open(path, 'wb').write(png)

def main(src, dst, W=620, H=820, ax=(0, 1), depth_ax=2):
    mu, sc, q, col, al = read_ply(src)
    R = quat_to_R(q)
    S = np.einsum('nij,nj,nkj->nik', R, sc**2, R)          # covariance
    P = mu[:, ax]
    lo, hi = P.min(0), P.max(0)
    pad = 0.06 * (hi - lo).max()
    lo -= pad; hi += pad
    span = (hi - lo).max()
    px = span / min(W, H)
    cx = (lo + hi) / 2
    scr = (P - cx) / px + np.array([W/2, H/2])
    cov = S[:, ax][:, :, ax] / px**2
    order = np.argsort(mu[:, depth_ax])                     # far -> near
    buf = np.ones((H, W, 3)) * np.array([0.09, 0.10, 0.12])
    for i in order:
        c = cov[i] + np.eye(2) * 0.35
        det = c[0,0]*c[1,1] - c[0,1]**2
        if det <= 0: continue
        inv = np.array([[c[1,1], -c[0,1]], [-c[0,1], c[0,0]]]) / det
        rad = np.ceil(2.6*np.sqrt(np.maximum(np.diag(c), 0))).astype(int)
        if rad.max() > 220: continue
        x0, y0 = int(scr[i,0]), int(scr[i,1])
        xs = np.arange(max(x0-rad[0],0), min(x0+rad[0]+1, W))
        ys = np.arange(max(y0-rad[1],0), min(y0+rad[1]+1, H))
        if not len(xs) or not len(ys): continue
        dx = xs - scr[i,0]; dy = ys - scr[i,1]
        DX, DY = np.meshgrid(dx, dy)
        e = inv[0,0]*DX*DX + 2*inv[0,1]*DX*DY + inv[1,1]*DY*DY
        a = (al[i] * np.exp(-0.5*e))[..., None]
        reg = buf[ys[0]:ys[-1]+1, xs[0]:xs[-1]+1]
        buf[ys[0]:ys[-1]+1, xs[0]:xs[-1]+1] = reg*(1-a) + col[i]*a
    write_png(dst, (np.clip(buf, 0, 1)**(1/1.1) * 255).astype(np.uint8))
    print('wrote', dst)

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
