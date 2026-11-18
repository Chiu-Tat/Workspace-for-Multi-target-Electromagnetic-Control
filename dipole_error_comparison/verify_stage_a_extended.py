"""Additional gradient, quadrature-moment, coil, box, and calibration checks."""
import json,math,sys
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from multipole_stage_a import *
from reference_gauss import points as quadrature_points,reference_error_bound,integrate
from test_multipole_stage_a import kw
from run_multipole_box_pilot import evaluate_boxes
from run_comparison import model
from emns.workspace.fast import params_list
from multipole_model import multipole_interaction


def main():
    pts=np.array([[.035,.008],[-.02,.02],[.04,-.02]])
    beta=.73
    for L in (1,3,5):
        out,U,F,T=interaction(pts,beta,L,**kw)
        assert_allclose(np.array([multipole_interaction(x,y,beta,L,**kw) for x,y in pts]),out,rtol=1e-10,atol=1e-13)
        J=field_gradient(pts,beta,L,**kw)
        u=np.array([np.cos(beta),np.sin(beta),0]);v=np.array([-u[1],u[0],0])
        for axis in range(2):
            h=1e-7;delta=np.eye(2)[axis]*h
            plus=interaction(pts+delta,beta,L,**kw)[0];minus=interaction(pts-delta,beta,L,**kw)[0]
            derivative=(plus[:,:2]-minus[:,:2])/(2*h)
            assert_allclose(derivative,np.column_stack((J[:,:,axis]@v,J[:,:,axis]@u)),rtol=1e-7,atol=1e-8)
        A,At=coil_map(pts,beta,L,kw['free_m'],kw['moving_cyl'],params_list,model.Z_PLANE)
        h=1e-5
        Ap,_=coil_map(pts,beta+h,L,kw['free_m'],kw['moving_cyl'],params_list,model.Z_PLANE)
        Am,_=coil_map(pts,beta-h,L,kw['free_m'],kw['moving_cyl'],params_list,model.Z_PLANE)
        assert_allclose(kw['free_m']*(Ap[:,1,:]-Am[:,1,:])/(2*h),At,rtol=1e-7,atol=1e-12)
    q,w=quadrature_points(2.,2.,4,16,7)
    # Exact all polynomial moments through degree 13 in a unit-radius cylinder.
    for a in range(14):
      for b in range(14-a):
       for c in range(14-a-b):
        actual=np.sum(w*q[:,0]**a*q[:,1]**b*q[:,2]**c)
        exact=0. if (a%2 or b%2 or c%2) else 2/(a+b+2)*math.gamma((a+1)/2)*math.gamma((b+1)/2)/(math.pi*math.gamma((a+b+2)/2))/(c+1)
        assert_allclose(actual,exact,atol=2e-15,rtol=1e-12)
    low,_=integrate(pts,beta,**kw,level=(4,16,7));high,_=integrate(pts,beta,**kw,level=(5,20,9))
    sep=np.linalg.norm(pts-kw['fixed_xy'],axis=1)
    b=reference_error_bound(sep,(4,16,7),**kw)
    assert np.all(abs(low-high)<=np.column_stack((b['field'],b['field'],b['force'],b['force'])))
    # Spot-check the full-box test against refined reference at extreme corners.
    archive=np.load(Path(__file__).parent/'results/multipole_stage_a/box_pilot.npz')
    boxes=archive['inside_centers'];dx=float(archive['dx']);db=float(archive['db'])
    from reference_gauss import coil_reference
    from run_multipole_stage_a import facets,feasible
    for idx in np.linspace(0,len(boxes)-1,8,dtype=int):
        x,y,beta=boxes[idx]
        corners=np.array([[x+sx*dx/2,y+sy*dx/2] for sx in (-1,1) for sy in (-1,1)])
        for angle in (beta-db/2,beta+db/2):
            out,raw=integrate(corners,angle,**kw,level=(5,20,9))
            A,T=coil_reference(corners,angle,kw['free_m'],kw['moving_cyl'],params_list,model.Z_PLANE)
            out[:,0]=raw[:,8]/kw['free_m'];A[:,0,:]=T/kw['free_m']
            assert np.all(feasible(facets(A),out))
    sys.path.insert(0,str(Path(__file__).parent/'stage_b'))
    from fit_axial_field import design,fit
    rng=np.random.default_rng(132);p=rng.normal(size=(60,3));p=p/np.linalg.norm(p,axis=1)[:,None]*rng.uniform(.02,.05,(60,1))
    theta=np.array([.12,2e-6,1e-5,-2e-5,3e-5]);B=(design(p,3)@theta).reshape(-1,3)
    r=fit(p,B,np.full_like(B,1e-7),['train']*30+['validation']*15+['test']*15)
    assert_allclose(r['models']['3']['coefficients'],theta,rtol=1e-10,atol=1e-15)
    result={'status':'PASS','checks':['legacy and batch multipole agreement','field-gradient finite differences','coil torque virtual rotation','Gauss moments through degree 13','quadrature majorant spot check','64 extreme corners of 8 accepted 3D boxes against independent finite-body reference','synthetic calibration coefficient recovery (not hardware evidence)']}
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
