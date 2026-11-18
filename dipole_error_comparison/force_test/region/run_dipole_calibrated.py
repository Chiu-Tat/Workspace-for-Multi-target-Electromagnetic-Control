"""Diagnostic: point dipole with a calibrated moment product, on the same partition as boxes.npy.
Force rows of the nominal dipole offset are scaled by p_cal / p_nominal; field rows and coil map stay nominal
(as in all other maps). Unresolved boxes wider than the minimum width are subdivided further.
Columns of the output: x0,x1,y0,y1,b0,b1, class_dipole_own_fit, class_dipole_at_L5_product."""
from .region_model import *
from .run_regions import split, MIN_WIDTH, GUARD_MM
from concurrent.futures import ProcessPoolExecutor
import json, time
O = Path(__file__).resolve().parent
A = np.load(O.parent / 'analysis.npz')
P0 = 0.12 * 0.145
PCAL = {'dip_fit': float(A['dip_coef'][0]), 'L5_p': float(A['L5_coef'][0])}

def leaf(q):
    stack = [list(q[:6])]; out = []
    while stack:
        b = stack.pop(); slo, shi = separation(b)
        if slo < GUARD_MM - 1e-9:
            out.append(b + [2, 2]); continue
        G, c1, _ = maps(b, None, include_multipole=False)
        cls = []
        for p in PCAL.values():
            c = c1.copy(); c[2:] = c1[2:] * (p / P0)
            cls.append(classify(G, c))
        if 0 in cls and b[1] - b[0] > MIN_WIDTH * 1.01:
            stack.extend(split(b)); continue
        out.append(b + cls)
    return out

if __name__ == '__main__':
    import os; TAG = os.environ.get('RUN_TAG', ''); base = np.load(O / f'boxes{TAG}.npy'); t = time.time(); rows = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        for k, r in enumerate(pool.map(leaf, base.tolist(), chunksize=64)):
            rows.extend(r)
    a = np.array(rows); np.save(O / f'boxes_dipole_calibrated{TAG}.npy', a)
    ar = (a[:, 1] - a[:, 0]) * (a[:, 3] - a[:, 2]) * 1e6
    smin = np.array([separation(list(r[:6]))[0] for r in a]); dom = smin >= 14.5 - 1e-9
    res = {'p_nominal': P0, 'p': PCAL, 'seconds': time.time() - t, 'leaves': len(a), 'common_area_mm2': float(ar[dom].sum())}
    for j, name in enumerate(PCAL):
        res[name] = {lab: float(ar[dom & (a[:, 6 + j] == c)].sum()) for c, lab in [(1, 'in'), (-1, 'out'), (0, 'unresolved')]}
    (O / f'metrics_dipole_calibrated{TAG}.json').write_text(json.dumps(res, indent=2)); print(json.dumps(res, indent=2))
