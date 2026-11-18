"""Physics regressions: run with the project Python, no test package needed."""
import json,math,time
from pathlib import Path
import numpy as np
from numpy.testing import assert_allclose
from multipole_stage_a import *
from reference_gauss import integrate,points
from run_comparison import model,finite_interaction
from emns.workspace.fast import params_list

cfg=json.loads(Path(__file__).with_name('config.json').read_text())
kw=dict(free_m=model.FREE_M,fixed_m=model.FIXED_M,fixed_xy=np.array([model.FIXED_X,model.FIXED_Y]),fixed_beta=model.FIXED_BETA,moving_cyl=cfg['moving_cylinder'],fixed_cyl=cfg['fixed_cylinder'])

def main():
    # Independent surface-charge Legendre integration verifies Q1,Q3,Q5.
    for geom in (kw['moving_cyl'],kw['fixed_cyl']):
        t,w=np.polynomial.legendre.leggauss(20);rho2=(geom['diameter_m']/2)**2*(t+1)/2
        z=geom['length_m']/2;r=np.sqrt(rho2+z*z)
        for ell in (1,3,5):
            coeff=np.zeros(ell+1);coeff[ell]=1
            Q=2/geom['length_m']*np.sum(w/2*r**ell*np.polynomial.legendre.legval(z/r,coeff))
            assert_allclose(Q,cylinder_Q(1,geom['length_m'],geom['diameter_m'],ell),rtol=2e-13)
    poses=np.array([[.033,.006],[-.01,.02],[.04,-.015]])
    for beta in (0.,.7,np.pi/2):
        dip=interaction(poses,beta,1,**kw)[0]
        baseline=np.array([model.actuation_matrix_body(x,y,beta)[:,-1] for x,y in poses])
        assert_allclose(dip,baseline,rtol=3e-6,atol=1e-9) # legacy adds 1 nm to radius
        maps,_=coil_map(poses,beta,1,kw['free_m'],kw['moving_cyl'],params_list,model.Z_PLANE)
        assert_allclose(maps,np.array([model.actuation_matrix_body(x,y,beta)[:,:10] for x,y in poses]),rtol=2e-12,atol=1e-14)
        for L in (1,3,5):
            out,U,F,T=interaction(poses,beta,L,**kw)
            h=1e-7
            for i in range(2):
                delta=np.eye(2)[i]*h
                fd=-(interaction(poses+delta,beta,L,**kw)[1]-interaction(poses-delta,beta,L,**kw)[1])/(2*h)
                assert_allclose(fd,F[:,i],rtol=2e-7,atol=1e-9)
            h=1e-5
            fd=-(interaction(poses,beta+h,L,**kw)[1]-interaction(poses,beta-h,L,**kw)[1])/(2*h)
            assert_allclose(fd,T,rtol=2e-7,atol=1e-10)
            # Rotation covariance and exchange reciprocity.
            a=.41; R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
            kr=dict(kw,fixed_xy=kw['fixed_xy']@R.T,fixed_beta=kw['fixed_beta']+a)
            oo,uu,ff,tt=interaction(poses@R.T,beta+a,L,**kr)
            assert_allclose(oo,out,rtol=3e-11,atol=1e-13)
            assert_allclose(uu,U,rtol=3e-11,atol=1e-13)
            for p,Ui,Fi,Ti in zip(poses,U,F,T):
                ks=dict(kw,free_m=kw['fixed_m'],fixed_m=kw['free_m'],fixed_xy=p,fixed_beta=beta,moving_cyl=kw['fixed_cyl'],fixed_cyl=kw['moving_cyl'])
                _,Us,Fs,Ts=interaction([kw['fixed_xy']],kw['fixed_beta'],L,**ks)
                assert_allclose(Us,[Ui],rtol=1e-11,atol=1e-13)
                assert_allclose(Fs,[-Fi],rtol=1e-11,atol=1e-12)
                r=p-kw['fixed_xy'];assert_allclose(Ti+Ts[0]+r[0]*Fi[1]-r[1]*Fi[0],0,atol=1e-12)
    # Compiled independent integral matches legacy NumPy integral.
    mid,_=integrate(poses,np.pi/2,**kw,level=(4,16,5),midpoint=True)
    old=np.array([finite_interaction(x,y,np.pi/2,cfg) for x,y in poses])
    assert_allclose(mid,old,rtol=2e-12,atol=1e-13)
    # Shrinking bodies and reference convergence; derivative/torque reference.
    tiny=dict(kw)
    for key in ('moving_cyl','fixed_cyl'):tiny[key]={k:v*.001 for k,v in kw[key].items()}
    assert_allclose(interaction(poses,.7,5,**tiny)[0],interaction(poses,.7,1,**kw)[0],rtol=3e-6,atol=1e-10)
    ref,raw=integrate(poses,.7,**kw,level=(5,20,9))
    lo,_=integrate(poses,.7,**kw,level=(4,16,7))
    L5,U,F,T=interaction(poses,.7,5,**kw)
    sep=np.linalg.norm(poses-kw['fixed_xy'],axis=1)
    radii=[bounding_radius(g['length_m'],g['diameter_m']) for g in (kw['fixed_cyl'],kw['moving_cyl'])]
    bounds=truncation_bounds(sep,5,*radii,kw['fixed_m'],kw['free_m'],source_length=kw['fixed_cyl']['length_m'],receiver_length=kw['moving_cyl']['length_m'])
    assert np.all(np.abs(L5-ref)<=np.column_stack((bounds['field'],bounds['field'],bounds['force'],bounds['force'])))
    assert np.all(np.abs(T-raw[:,8])<=bounds['torque'])
    for p in (2,3):
        t=.36;N=6
        direct=sum(math.prod(range(n+1,n+p+1))*t**n for n in range(N,300,2))
        assert_allclose(even_tail(t,N,p),direct,rtol=1e-14)
    print(json.dumps({'status':'PASS','checks':['Q1/Q3/Q5 surface integral','L1 dipole/coil recovery','spatial and rotational energy derivatives','rotation covariance','force reciprocity','angular momentum balance','C/NumPy reference equivalence','shrinking body','tail rational sum','sampled field/force/torque bound coverage'],'reference_refinement_max':np.max(abs(ref-lo),axis=0).tolist(),'L5_error_max':np.max(abs(ref-L5),axis=0).tolist()},indent=2))

if __name__=='__main__':main()
