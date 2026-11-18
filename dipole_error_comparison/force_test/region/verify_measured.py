"""Verify TFW classifications against the measured interaction force.

Each of the 54 force-test poses is mapped to the Case Study 1 single layout
(fixed magnet at (20,0) mm, heading 90 deg). The interaction depends only on the
relative pose, so the reading gives the measured interaction force on the moving
magnet at that configuration. Field rows and the coil map stay nominal.
For each configuration we compute
  * pointwise thrust authority with the nominal dipole, calibrated L=5, and measured force offset;
  * whole-box classes (nominal, L=5, L=5+reserve) for a small box around the configuration.
"""
from .region_model import *
from .run_regions import reserve, P
from scipy.optimize import linprog
import json
O = Path(__file__).resolve().parent
A = np.load(O.parent / 'analysis.npz')
Dm = A['D']; F = A['F']; bias = A['L5_coef'][1:]          # SI forces (N), fitted sensor bias (N)
HALF_XY = 0.25e-3; HALF_B = np.deg2rad(0.5)                # verification box half-widths

def to_cs1(x_mm, y_mm, th_deg, fx, fy):
    """Test frame -> CS1 frame. Returns X, Y (m), beta (rad), measured force (world, N)."""
    phi = np.deg2rad(90.0 - th_deg)
    R = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    d = R @ (-np.array([x_mm, y_mm]) * 1e-3)               # moving centre relative to fixed centre
    return 0.02 + d[0], d[1], (np.pi / 2 + phi) % (2 * np.pi), R @ np.array([fx, fy])

def point(v, c_idx):
    q = [v[0], v[0], v[1], v[1], v[2], v[2]]
    G, c1, c5 = maps(q, P)
    g = np.array([[e.center for e in row] for row in G])
    return g, np.array([e.center for e in c1]), np.array([e.center for e in c5])

def authority(G, c):
    """Largest t with [0, 20 mT, 0, +-t] reachable within +-15 A (N)."""
    N = 21; Aeq = []; beq = []
    for k, sz in enumerate((-1, 1)):
        for r in range(4):
            row = np.zeros(N); row[10 * k:10 * k + 10] = G[r]
            rhs = [0, .02, 0, 0][r] - c[r]
            if r == 3: row[-1] = -sz
            Aeq.append(row); beq.append(rhs)
    sc = np.array([.02, .02, .1, .1] * 2)
    res = linprog(np.r_[np.zeros(20), -1], A_eq=np.array(Aeq) / sc[:, None], b_eq=np.array(beq) / sc,
                  bounds=[(-15, 15)] * 20 + [(None, None)], method='highs')
    return -res.fun if res.status == 0 else np.nan

rows = []
for j in range(len(Dm)):
    x, y, th = Dm[j, :3]
    meas = F[j] - bias
    X, Y, b, Fw = to_cs1(x, y, th, *meas)
    s = 1e3 * np.hypot(X - .02, Y)
    G, c1, c5 = point([X, Y, b], None)
    ey = np.array([-np.sin(b), np.cos(b)]); ez = np.array([np.cos(b), np.sin(b)])
    cm = c1.copy(); cm[2:] = [Fw @ ey, Fw @ ez]            # measured force rows, nominal field rows
    # Held-out L=5 prediction: measured force plus its leave-one-pose-out residual (pred - obs).
    _, _, _, Fl = to_cs1(x, y, th, *(meas + A['L5_loo'][j]))
    cl = c1.copy(); cl[2:] = [Fl @ ey, Fl @ ez]
    au = [authority(G, c) for c in (c1, c5, cm, cl)]
    box = [X - HALF_XY, X + HALF_XY, Y - HALF_XY, Y + HALF_XY, b - HALF_B, b + HALF_B]
    smin, smax = separation(box)
    Gb, c1b, c5b = maps(box, P)
    eps_in = reserve(max(smin, 14.5)); eps_out = reserve(min(max(smax, 14.5), 59.83))
    cls = [classify(Gb, c1b), classify(Gb, c5b), classify(Gb, c5b, eps_in, eps_out)]
    rows.append(dict(pose=j + 2, test=[x, y, th], cs1_mm=[X * 1e3, Y * 1e3], beta_deg=np.degrees(b), sep_mm=s,
                     in_domain=bool(abs(X) <= .05 and abs(Y) <= .05),
                     f_meas_mN=(cm[2:] * 1e3).tolist(), f_dip_mN=(c1[2:] * 1e3).tolist(), f_L5_mN=(c5[2:] * 1e3).tolist(),
                     authority_mN=[a * 1e3 for a in au], f_L5loo_mN=(cl[2:] * 1e3).tolist(), box_class=cls, eps_in_mN=eps_in * 1e3))
    print(f"{j+2:2d} sep {s:5.1f} beta {np.degrees(b):6.1f} dom {rows[-1]['in_domain']!s:5s} "
          f"auth dip/L5/meas/L5loo {au[0]*1e3:6.1f} {au[1]*1e3:6.1f} {au[2]*1e3:6.1f} {au[3]*1e3:6.1f}  box nom/L5/res {cls}", flush=True)
(O / 'verification.json').write_text(json.dumps(rows, indent=1))
