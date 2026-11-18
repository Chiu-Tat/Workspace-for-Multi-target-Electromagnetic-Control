#!/usr/bin/env python3
"""Fit fixed axial multipole coefficients to measured, body-frame vector fields.

Required CSV: x_m,y_m,z_m,Bx_T,By_T,Bz_T,sigma_Bx_T,sigma_By_T,sigma_Bz_T,split
split must be train/validation/test. Axis is body +z, origin fixed by metrology.
Only run on real supplied data; synthetic tests are not experimental evidence.
"""
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from multipole_stage_a import derivatives,direction,multiply,evaluate,C
import math

def design(points,L):
    table=derivatives(points,L+1);columns=[]
    for ell in range(1,L+1,2):
        op=direction([0,0,1],ell)
        columns.append(np.column_stack([C/math.factorial(ell)*evaluate(multiply(op,direction(e)),table) for e in np.eye(3)]).ravel())
    for e in np.eye(3):columns.append(np.tile(e,(len(points),1)).ravel())
    return np.column_stack(columns)

def fit(points,field,sigma,split):
    split=np.asarray(split);records={};sols={}
    for name in ('train','validation','test'):
        if np.sum(split==name)<4:raise ValueError(f'Need at least four vector samples in {name}')
    if not np.all(np.isin(split,['train','validation','test'])):raise ValueError('Unknown split label')
    for L in (1,3,5):
        A=design(points,L);rows=np.repeat(split=='train',3)
        W=A[rows]/sigma.ravel()[rows,None];b=field.ravel()[rows]/sigma.ravel()[rows]
        scales=np.linalg.norm(W,axis=0)
        if np.any(scales==0):raise ValueError('Unobservable coefficient')
        W=W/scales
        theta,_,rank,s=np.linalg.lstsq(W,b,rcond=None)
        if rank<W.shape[1]:raise ValueError(f'Rank-deficient L={L} measurement geometry')
        theta/=scales;res=(A@theta).reshape(-1,3)-field
        cov=np.linalg.inv(W.T@W)/scales[:,None]/scales[None,:]
        record={'coefficient_names':[f'Q{ell}' for ell in range(1,L+1,2)]+['background_Bx_T','background_By_T','background_Bz_T'],
          'coefficients':theta.tolist(),'covariance_known_sensor_variances':cov.tolist(),
          'scaled_design_condition':float(s[0]/s[-1]),'rank':int(rank),'splits':{}}
        for name in ('train','validation','test'):
            sel=split==name
            record['splits'][name]={'samples':int(sel.sum()),'component_rmse_T':np.sqrt(np.mean(res[sel]**2,axis=0)).tolist(),
                'weighted_rmse':float(np.sqrt(np.mean((res[sel]/sigma[sel])**2))),'max_absolute_T':np.max(abs(res[sel]),axis=0).tolist()}
        records[str(L)]=record
    chosen=min(records,key=lambda k:records[k]['splits']['validation']['weighted_rmse'])
    return {'scope':'Known-pose axial model fit. Sensor-noise covariance only; excludes pose/systematic/magnetization drift uncertainty. Test data are not used in selection.',
            'selected_order':int(chosen),'models':records}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('csv',type=Path);ap.add_argument('--bounding-radius-m',type=float,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    with a.csv.open() as f:rows=list(csv.DictReader(f))
    points=np.array([[float(r[k]) for k in ('x_m','y_m','z_m')] for r in rows])
    field=np.array([[float(r[k]) for k in ('Bx_T','By_T','Bz_T')] for r in rows])
    sigma=np.array([[float(r[k]) for k in ('sigma_Bx_T','sigma_By_T','sigma_Bz_T')] for r in rows])
    if len(points)==0 or not all(np.all(np.isfinite(v)) for v in (points,field,sigma)):raise ValueError('Empty or non-finite measurements')
    if np.any(sigma<=0):raise ValueError('Positive calibrated component uncertainties required')
    if a.bounding_radius_m<=0 or np.any(np.linalg.norm(points,axis=1)<=a.bounding_radius_m):raise ValueError('Samples must be outside the supplied bounding sphere')
    result=fit(points,field,sigma,[r['split'] for r in rows]);result['input_file']=str(a.csv.resolve());result['bounding_radius_m']=a.bounding_radius_m
    a.output.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
