#!/usr/bin/env python3
"""Validate real coils-off pair observations against the Case-6 predictors.

Measured components are body-frame [By, Bz, fy, fz] of the moving magnet.
Blank component cells are allowed. Metadata are checked, never ignored.
"""
from __future__ import annotations
import argparse, csv, json, math, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parent))
from run_comparison import finite_interaction, dipole_interaction
from cases.case_6_interaction_obstacle_avoidance.simulation import model
COMP=('By','Bz','fy','fz'); UNIT={'By':'T','Bz':'T','fy':'N','fz':'N'}
def finite_float(row,key,required=True):
    raw=row.get(key,'').strip()
    if not raw:
        if required: raise ValueError(f'{key} is required')
        return None
    val=float(raw)
    if not math.isfinite(val): raise ValueError(f'{key} must be finite')
    return val
def validate(path:Path,config:Path,output:Path|None=None):
    cfg=json.loads(config.read_text()); rows=[]; gm,gf=cfg['moving_cylinder'],cfg['fixed_cylinder']
    exclusion=.5*math.hypot(gm['length_m'],gm['diameter_m'])+.5*math.hypot(gf['length_m'],gf['diameter_m'])
    expected={'z_m':model.Z_PLANE,'moving_alpha_rad':model.ALPHA,'fixed_x_m':model.FIXED_X,'fixed_y_m':model.FIXED_Y,'fixed_z_m':model.Z_PLANE,'fixed_alpha_rad':model.ALPHA,'fixed_beta_rad':model.FIXED_BETA,'moving_moment_Am2':model.FREE_M,'fixed_moment_Am2':model.FIXED_M}
    with path.open(newline='') as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames: raise ValueError('CSV header is missing')
        for lineno,row in enumerate(reader,2):
            ident=row.get('sample_id','').strip() or f'line-{lineno}'
            try:
                split=row.get('split','').strip()
                if split not in {'fit','held_out'}: raise ValueError("split must be 'fit' or 'held_out'")
                x=finite_float(row,'x_m'); y=finite_float(row,'y_m'); beta=finite_float(row,'moving_beta_rad')
                if not(cfg['x_limits_m'][0]<=x<=cfg['x_limits_m'][1] and cfg['y_limits_m'][0]<=y<=cfg['y_limits_m'][1]): raise ValueError('moving position is outside configured x/y domain')
                if math.hypot(x-model.FIXED_X,y-model.FIXED_Y)<exclusion: raise ValueError('pose violates configured non-overlap cutoff')
                for key,want in expected.items():
                    got=finite_float(row,key)
                    if not math.isclose(got,want,rel_tol=1e-9,abs_tol=1e-9): raise ValueError(f'{key}={got} unsupported; Case-6 predictor requires {want}')
                obs={}; std={}
                for c in COMP:
                    v=finite_float(row,f'measured_{c}_{UNIT[c]}',False); s=finite_float(row,f'std_{c}_{UNIT[c]}',False)
                    if s is not None and s<0: raise ValueError(f'std_{c}_{UNIT[c]} must be nonnegative')
                    if v is not None: obs[c]=v
                    if s is not None:
                        if v is None: raise ValueError(f'std supplied without measured {c}')
                        std[c]=s
                if not obs: raise ValueError('at least one measured component is required')
            except (ValueError,TypeError) as e: raise ValueError(f'row {ident}: {e}') from e
            dip=dipole_interaction(x,y,beta); finite=finite_interaction(x,y,beta,cfg); idx={c:i for i,c in enumerate(COMP)}
            rows.append({'sample_id':ident,'split':split,'pose_m_rad':[x,y,model.Z_PLANE,beta],'measured':obs,'measurement_std':std,'dipole':{c:float(dip[idx[c]]) for c in obs},'finite':{c:float(finite[idx[c]]) for c in obs},'residual_dipole':{c:float(obs[c]-dip[idx[c]]) for c in obs},'residual_finite':{c:float(obs[c]-finite[idx[c]]) for c in obs}})
    if not rows: raise ValueError('no measured rows')
    for split in ('fit','held_out'):
        if not any(r['split']==split for r in rows): raise ValueError(f'no rows labelled split={split}')
    held={}
    for c in COMP:
        rs=[r['residual_finite'][c] for r in rows if r['split']=='held_out' and c in r['measured']]
        if rs:
            arr=np.asarray(rs); held[c]={'n':len(rs),'mean':float(arr.mean()),'rmse':float(np.sqrt(np.mean(arr**2))),'max_abs':float(np.max(np.abs(arr)))}
    result={'scope':'measured-data residual report; not itself an uncertainty envelope','coordinates':'moving-magnet body-frame [By,Bz,fy,fz]','geometry_scope':cfg['geometry_scope'],'config_file':str(config.resolve()),'nonoverlap_center_separation_m':exclusion,'measurement_uncertainty':'std columns are preserved but not automatically converted into a confidence bound; select and justify coverage separately','held_out_finite_reference_summary':held,'rows':rows}
    if output: output.write_text(json.dumps(result,indent=2)+'\n')
    return result
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('csv',type=Path); ap.add_argument('--config',type=Path,default=HERE/'config.json'); ap.add_argument('--output',type=Path,default=HERE/'results/measured_validation.json'); a=ap.parse_args(); validate(a.csv,a.config,a.output); print(a.output)
if __name__=='__main__': main()
