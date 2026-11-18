"""Matched adaptive region classification; all verdicts concern full 3-D boxes."""
from .region_model import *
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
import json,time
OUT=Path(__file__).resolve().parent
DATA=np.load(OUT.parent/'analysis.npz');P=float(DATA['L5_coef'][0]);SEP=DATA['sep'];ERR=np.linalg.norm(DATA['L5_loo'],axis=1)
CONVERGENCE_MM=1e3*sum(math.hypot(g['length_m']/2,g['diameter_m']/2) for g in [SENS,POS])
GUARD_MM=10.
import os
MIN_WIDTH=float(os.environ.get('MIN_WIDTH_MM','0.125'))*1e-3;TAG=os.environ.get('RUN_TAG','');_BC=float(os.environ.get('BAND_CENTER_DEG','46'));BAND=np.deg2rad([_BC-.5,_BC+.5])
def reserve(s):
    if not SEP.min()-1e-9<=s<=SEP.max()+1e-9:raise ValueError('outside empirical separation domain')
    return float(ERR[SEP>=s-1e-9].max())
def split(b):
    xm=(b[0]+b[1])/2;ym=(b[2]+b[3])/2
    return [[xl,xh,yl,yh,b[4],b[5]] for xl,xh in [(b[0],xm),(xm,b[1])] for yl,yh in [(b[2],ym),(ym,b[3])]]
def root_run(b):
    stack=[b];out=[]
    while stack:
        q=stack.pop();slo,shi=separation(q);can=q[1]-q[0]>MIN_WIDTH*1.01
        # Refine cells crossing any distinct domain boundary before classification.
        if can and any(slo<cut<=shi for cut in [GUARD_MM,CONVERGENCE_MM,14.5]):
            stack.extend(split(q));continue
        if slo<GUARD_MM-1e-9:
            out.append(q+[slo,shi,0,2,2,2]);continue
        cor_allowed=slo>CONVERGENCE_MM
        reserve_allowed=slo>=14.5-1e-9
        G,c1,c5=maps(q,P,include_multipole=cor_allowed)
        nom=classify(G,c1);cor=classify(G,c5) if cor_allowed else 2
        rob=2;eps=0.
        if reserve_allowed:
            eps=reserve(slo);rob=classify(G,c5,eps,reserve(shi))
            if rob==1:cor=1
            if cor==-1:rob=-1
        if can and 0 in [nom,cor,rob]:stack.extend(split(q));continue
        out.append(q+[slo,shi,eps,nom,cor,rob])
    return out

def main():
    start=time.time();edges=np.arange(-28,28.1,4)*1e-3
    roots=[[x+.02,xh+.02,y,yh,*BAND]for x,xh in zip(edges[:-1],edges[1:]) for y,yh in zip(edges[:-1],edges[1:])]
    rows=[]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for k,res in enumerate(pool.map(root_run,roots)):
            rows.extend(res)
            if (k+1)%28==0:print('roots',k+1,'leaves',len(rows),'seconds',round(time.time()-start),flush=True)
    a=np.array(rows);np.save(OUT/f'boxes{TAG}.npy',a)
    areas=(a[:,1]-a[:,0])*(a[:,3]-a[:,2])*1e6
    valid=a[:,11]!=2;result={'domain_mm':[-8,48,-28,28],'heading_band_deg':[float(np.degrees(BAND[0])),float(np.degrees(BAND[1]))],'minimum_xy_width_mm':MIN_WIDTH*1e3,'p_L5':P,'seconds':time.time()-start,'leaves':len(a),'valid_leaves':int(valid.sum()),'excluded_area_mm2':float(areas[~valid].sum()),'evaluated_area_mm2':float(areas[valid].sum()),'modes':{},'full_modes':{},'guard_mm':GUARD_MM,'convergence_mm':CONVERGENCE_MM}
    for k,name in enumerate(['nominal','L5_corrected','L5_empirical_reserve'],9):
        result['modes'][name]={label:{'boxes':int(np.sum(valid&(a[:,k]==cl))),'area_mm2':float(areas[valid&(a[:,k]==cl)].sum())}for cl,label in [(1,'in'),(-1,'out'),(0,'unresolved')]}
        result['full_modes'][name]={label:{'boxes':int(np.sum(a[:,k]==cl)),'area_mm2':float(areas[a[:,k]==cl].sum())}for cl,label in [(1,'in'),(-1,'out'),(0,'unresolved'),(2,'not_evaluated')]}
    result['extra_predictions_below_data_limit']={name:int(np.sum((~valid)&(a[:,k]!=2))) for k,name in [(9,'nominal'),(10,'L5_corrected')]}
    result['transitions']={f'{int(i)}->{int(j)}':{'boxes':int(np.sum(valid&(a[:,9]==i)&(a[:,10]==j))),'area_mm2':float(areas[valid&(a[:,9]==i)&(a[:,10]==j)].sum())}for i in [-1,0,1] for j in [-1,0,1]}
    loss=valid&(a[:,10]==1)&(a[:,11]!=1);gain=valid&(a[:,9]!=1)&(a[:,10]==1)
    result['correction_gains_in_area_mm2']=float(areas[gain].sum());result['reserve_loses_in_area_mm2']=float(areas[loss].sum())
    result['gain_examples']=a[np.where(gain)[0][:5]].tolist();result['reserve_loss_examples']=a[np.where(loss)[0][:5]].tolist()
    result['scope']='Local matched region rerun. Taylor remainder bounds, float64 without directed rounding. Force-only correction, fixed nominal coil map and centre-field rows. Empirical reserve is not a verified physical bound. Nominal domain: full cells at or beyond 10 mm; multipole domain: full cells outside the 11.238 mm exterior convergence threshold. Reserve only on full cells within measured separations. Common-domain areas are reported separately from extended model-only predictions.'
    (OUT/f'metrics{TAG}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
