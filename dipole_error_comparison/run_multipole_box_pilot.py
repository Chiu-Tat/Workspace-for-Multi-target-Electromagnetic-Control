"""3-D whole-box inner-set pilot for the ideal finite-cylinder model.

Independent conservative Lipschitz enclosures plus analytical multipole tails.
This is NOT the existing affine-hybrid backend or an outward-rounded proof.
No outside certificate: failed boxes remain unresolved or outside model domain.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from multipole_stage_a import interaction,coil_map,truncation_bounds,bounding_radius,C
from run_multipole_stage_a import facets,feasible
from test_multipole_stage_a import kw,cfg
from run_comparison import model
from emns.workspace.fast import params_list

def variation(gap,ms,mr,a,dr,db):
    """Bounds for [Tz/m, Bz, Fy, Fz] over translation and planar rotation."""
    g=np.asarray(gap)
    transl=np.column_stack((6/g**4+24*a/g**5,6/g**4,24*mr/g**5,24*mr/g**5))
    angular=np.column_stack((2/g**3+18*a/g**4+24*a*a/g**5,2/g**3+6*a/g**4,
                            mr*(12/g**4+24*a/g**5),mr*(12/g**4+24*a/g**5)))
    return C*abs(ms)*(transl*dr+angular*db)

def evaluate_boxes(points,beta,dx,db,L):
    a_s=bounding_radius(**dict(length=kw['fixed_cyl']['length_m'],diameter=kw['fixed_cyl']['diameter_m']))
    a_r=bounding_radius(kw['moving_cyl']['length_m'],kw['moving_cyl']['diameter_m'])
    d=np.linalg.norm(points-kw['fixed_xy'],axis=1);dr=np.sqrt(2)*dx/2
    valid=d-dr>a_s+a_r
    for par in params_list:
        dc=np.linalg.norm(np.column_stack((points,np.full(len(points),model.Z_PLANE)))-par[3:],axis=1)
        valid &= dc-dr>a_r
    p=points[valid];d=d[valid]
    if not len(p):return valid,np.zeros(0,bool),np.zeros((0,4))
    offsets,_,_,torque=interaction(p,beta,L,**kw)
    A,T=coil_map(p,beta,L,kw['free_m'],kw['moving_cyl'],params_list,model.Z_PLANE)
    A[:,0,:]=T/kw['free_m'];offsets[:,0]=torque/kw['free_m']
    b=truncation_bounds(d,L,a_s,a_r,kw['fixed_m'],kw['free_m'],source_length=kw['fixed_cyl']['length_m'],receiver_length=kw['moving_cyl']['length_m'])
    E=np.column_stack((b['torque']/kw['free_m'],b['field'],b['force'],b['force']))
    E+=variation(d-dr-a_s-a_r,kw['fixed_m'],kw['free_m'],a_r,dr,db/2)
    for par in params_list:
        dc=np.linalg.norm(np.column_stack((p,np.full(len(p),model.Z_PLANE)))-par[3:],axis=1)
        ms=np.linalg.norm(par[:3]);b=truncation_bounds(dc,L,0,a_r,ms,kw['free_m'])
        ee=np.column_stack((b['torque']/kw['free_m'],b['field'],b['force'],b['force']))
        ee+=variation(dc-dr-a_r,ms,kw['free_m'],a_r,dr,db/2)
        E+=cfg['current_limit_A']*ee
    return valid,feasible(facets(A),offsets,E),E

def main(n,nb,L):
    start=time.perf_counter();dx=.1/n;db=2*np.pi/nb
    xx,yy=np.meshgrid(-.05+(np.arange(n)+.5)*dx,-.05+(np.arange(n)+.5)*dx)
    points=np.column_stack((xx.ravel(),yy.ravel()));inside=[];domain=[]
    for k in range(nb):
        beta=(k+.5)*db;valid,mask,E=evaluate_boxes(points,beta,dx,db,L)
        ids=np.flatnonzero(valid);inside.extend(np.column_stack((points[ids[mask]],np.full(mask.sum(),beta))).tolist())
        domain.append(int(valid.sum()))
        if k%8==0:print(f'heading band {k+1}/{nb}, inside so far {len(inside)}',flush=True)
    report={'scope':'Whole-box inner-set numerical pilot for uniform axial cylinders and fixed calibrated coil dipoles; conservative analytic tails and Lipschitz enclosures. Float64 without directed rounding; not hardware or machine-verified certification.',
      'L':L,'xy_cells':n,'heading_cells':nb,'cell_width_m':dx,'heading_width_deg':360/nb,
      'total_boxes':n*n*nb,'domain_valid_boxes':sum(domain),'inside_boxes':len(inside),
      'unresolved_valid_boxes':sum(domain)-len(inside),'outside_certificates':0,
      'inside_fraction_full_search':len(inside)/(n*n*nb),'inside_fraction_valid':len(inside)/max(1,sum(domain)),
      'elapsed_seconds':time.perf_counter()-start}
    out=Path(__file__).parent/'results'/'multipole_stage_a'
    np.savez_compressed(out/'box_pilot.npz',inside_centers=np.array(inside),dx=dx,db=db,L=L)
    (out/'box_pilot.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--xy',type=int,default=32);ap.add_argument('--heading',type=int,default=32);ap.add_argument('--order',type=int,default=5);a=ap.parse_args();main(a.xy,a.heading,a.order)
