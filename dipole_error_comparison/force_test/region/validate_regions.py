from .region_model import *
import json
from scipy.optimize import linprog
OUT=Path(__file__).resolve().parent
A=np.load(OUT.parent/'analysis.npz');p=float(A['L5_coef'][0]);rng=np.random.default_rng(20261108)
# Strong check: actual point value minus shared linear part must fit remainder,
# not merely the wider scalar range. It supplements the analytic remainder proof.
max_excess=-np.inf;checked=0;worst_point_error=0.
for t in range(24):
    while True:
        x,y=rng.uniform([-.008,-.028],[.048,.028]);w=rng.choice([.0005,.001,.002,.004]);b=np.deg2rad(46)
        q=[x-w/2,x+w/2,y-w/2,y+w/2,b-np.deg2rad(.5),b+np.deg2rad(.5)]
        if separation(q)[0]>11.2375147625:break
    G,c1,c5=maps(q,p);forms=list(G.flat)+list(c1)+list(c5)
    for j in range(12):
        z=rng.uniform(-1,1,3) if j>=8 else np.array(list(itertools.product([-1,1],repeat=3))[j])
        mid=np.array([(q[2*k]+q[2*k+1])/2 for k in range(3)]);rad=np.array([(q[2*k+1]-q[2*k])/2 for k in range(3)]);v=mid+rad*z
        # Zero-width form evaluates the explicit analytic dipole equations.
        gg,aa,bb=maps(np.repeat(v,2).tolist(),p);true=np.array([f.center for f in list(gg.flat)+list(aa)+list(bb)])
        pred=np.array([f.center+f.linear@z for f in forms]);err=np.array([f.error for f in forms]);excess=abs(true-pred)-err
        max_excess=max(max_excess,float(excess.max()));checked+=len(forms)
        assert np.all(excess<1e-10),(q,z,excess.max())
        ref=multipole_interaction(v[0],v[1],v[2],5,free_m=.12,fixed_m=p/.12,fixed_xy=(.02,0),fixed_beta=np.pi/2,moving_cyl=SENS,fixed_cyl=POS)
        diff=max(abs(np.array([f.center for f in bb[2:]])-ref[2:]));worst_point_error=max(worst_point_error,float(diff))
        assert diff<1e-10
result={'random_boxes':24,'points_per_box':12,'scalar_affine_residual_checks':checked,'maximum_remainder_excess':max_excess,'L5_independent_point_max_abs_N':worst_point_error,'note':'Analytic Taylor remainders plus numerical regression checks. This does not constitute directed-rounding verification.'}
if (OUT/'boxes.npy').exists():
    a=np.load(OUT/'boxes.npy');valid=a[:,9]!=2;area=(a[:,1]-a[:,0])*(a[:,3]-a[:,2])*1e6
    assert abs(area.sum()-56**2)<1e-7
    assert np.all(a[a[:,11]==1,10]==1)
    result['partition_area_mm2']=area.sum();result['robust_inside_subset_checked']=True
(OUT/'validation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
