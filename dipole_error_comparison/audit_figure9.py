"""Independent LP and finite-body checks of closest Figure 9 green samples."""
import json
from itertools import product
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
from run_multipole_stage_a import cfg,kw,model,facets,feasible,params_list
from multipole_stage_a import interaction,coil_map,truncation_bounds,bounding_radius
from reference_gauss import integrate,coil_reference
HERE=Path(__file__).resolve().parent
D=np.load(HERE/'results/multipole_stage_a/heading_90.npz')
pts=D['points'];rr=D['radius'];green=D['L5_reserved_mask']
idx=np.flatnonzero(green); idx=idx[np.argsort(rr[idx])[:8]]
# Also test samples immediately outside the white circle, irrespective of their colors.
idx=np.unique(np.r_[idx,np.argsort(rr)[:4]])
p=pts[idx];beta=np.pi/2
c,_,_,T=interaction(p,beta,5,**kw);c[:,0]=T/kw['free_m']
A,Ta=coil_map(p,beta,5,kw['free_m'],kw['moving_cyl'],params_list,model.Z_PLANE);A[:,0,:]=Ta/kw['free_m']
radii=[bounding_radius(g['length_m'],g['diameter_m']) for g in (kw['fixed_cyl'],kw['moving_cyl'])]
b=truncation_bounds(rr[idx],5,*radii,kw['fixed_m'],kw['free_m'],source_length=kw['fixed_cyl']['length_m'],receiver_length=kw['moving_cyl']['length_m'])
reserve=np.column_stack((b['torque']/kw['free_m'],b['field'],b['force'],b['force']))
for par in params_list:
 dist=np.linalg.norm(np.column_stack((p,np.full(len(p),model.Z_PLANE)))-par[3:],axis=1)
 cb=truncation_bounds(dist,5,0.,radii[1],np.linalg.norm(par[:3]),kw['free_m'])
 reserve+=cfg['current_limit_A']*np.column_stack((cb['torque']/kw['free_m'],cb['field'],cb['force'],cb['force']))
print('cache reserve max difference',np.max(abs(reserve-D['L5_physical_reserve'][idx])),flush=True)
print('finite-body quadrature',flush=True)
ref,raw=integrate(p,beta,**kw,level=(7,28,12));ref[:,0]=raw[:,8]/kw['free_m']
Ar,Tr=coil_reference(p,beta,kw['free_m'],kw['moving_cyl'],params_list,model.Z_PLANE,level=(7,28,12));Ar[:,0,:]=Tr/kw['free_m']
center=np.array([0.,model.B_HOLD,0.,0.]);half=np.array([0.,0.,0.,model.F_MAX]);scale=np.array([.02,.02,.1,.1])
def solve(A,rhs):
 # Minimize peak absolute current with independently scaled equalities.
 return linprog(np.r_[np.zeros(10),1.],A_ub=np.r_[np.c_[np.eye(10),-np.ones(10)],np.c_[-np.eye(10),-np.ones(10)]],b_ub=np.zeros(20),A_eq=np.c_[A/scale[:,None],np.zeros(4)],b_eq=rhs/scale,bounds=[(None,None)]*10+[(0,None)],method='highs')
records=[]
for j,i in enumerate(idx):
 verts=[solve(A[j],center+(half+reserve[j])*sign-c[j]) for sign in np.array(list(product([-1,1],repeat=4)))]
 peak=max(r.fun for r in verts)
 ends=[solve(A[j],center+sign*half-c[j]) for sign in [-1,1]]
 refends=[solve(Ar[j],center+sign*half-ref[j]) for sign in [-1,1]]
 # Evaluate independently integrated physical response for the nominal endpoint currents.
 residual=max(np.max(abs(Ar[j]@r.x[:10]+ref[j]-(center+sign*half))) for sign,r in zip([-1,1],ends))
 item=dict(index=int(i),xy_mm=(p[j]*1000).tolist(),separation_mm=rr[i]*1000,green=bool(green[i]),offset=c[j].tolist(),reserve=reserve[j].tolist(),nominal_peak_A=max(r.fun for r in ends),reserve_peak_A=peak,reference_peak_A=max(r.fun for r in refends),reference_offset=ref[j].tolist(),nominal_current_reference_max_mixed_residual=residual,coil_independent_row_capacity=(15*np.sum(abs(A[j]),axis=1)).tolist(),endpoint_currents=[r.x[:10].tolist() for r in ends])
 assert bool(green[i]) == (peak<=15+1e-7)
 records.append(item)
 print(json.dumps({k:v for k,v in item.items() if k not in ['endpoint_currents']}),flush=True)
report=dict(task_center=center.tolist(),halfwidth=half.tolist(),z_plane_m=model.Z_PLANE,source_moment=kw['fixed_m'],receiver_moment=kw['free_m'],closest_green_mm=float(rr[green].min()*1000),checks=records)
(HERE/'results/multipole_stage_a/figure9_audit.json').write_text(json.dumps(report,indent=2)+'\n')
