#!/usr/bin/env python3
"""Centered-finite-difference checks of the corrected By,y term."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from emns.workspace.fast import (calculate_B_magnet_selective, calculate_B_selective,
    calculate_derivatives_magnet_selective, calculate_derivatives_selective)

def main():
    h, checks = 1e-6, []
    currents = np.array([1.2,-.8,.4,0,.5,-.3,.2,.1,-.2,.7])
    source = {'X':.02,'Y':-.01,'Z':.082,'m':.145,'alpha':1.1,'beta':.8}
    req_b = {'Bx':False,'By':True,'Bz':False}
    req_d = {'Bx_dx':False,'Bx_dy':False,'Bx_dz':False,'By_dy':True,'By_dz':False}
    for x,y,z in [(-.031,.017,.076),(.043,-.026,.089),(-.012,-.038,.081)]:
        bp=calculate_B_selective(currents,x,y+h,z,req_b)['By']; bm=calculate_B_selective(currents,x,y-h,z,req_b)['By']
        ana=calculate_derivatives_selective(currents,x,y,z,req_d)['By_dy']; fd=(bp-bm)/(2*h)
        checks.append({'kind':'coil','pose':[x,y,z],'analytic':ana,'finite_difference':fd,'relative_error':abs(ana-fd)/max(abs(fd),1e-12)})
        bp=calculate_B_magnet_selective(source,x,y+h,z,req_b)['By']; bm=calculate_B_magnet_selective(source,x,y-h,z,req_b)['By']
        ana=calculate_derivatives_magnet_selective(source,x,y,z,req_d)['By_dy']; fd=(bp-bm)/(2*h)
        checks.append({'kind':'target','pose':[x,y,z],'analytic':ana,'finite_difference':fd,'relative_error':abs(ana-fd)/max(abs(fd),1e-12)})
    maximum=max(c['relative_error'] for c in checks)
    assert maximum < 5e-7
    out=Path(__file__).with_name('results'); out.mkdir(exist_ok=True)
    (out/'derivative_checks.json').write_text(json.dumps({'step_m':h,'checks':checks},indent=2)+'\n')
    print(json.dumps({'max_relative_error':maximum,'checks':len(checks)},indent=2))
if __name__ == '__main__': main()
