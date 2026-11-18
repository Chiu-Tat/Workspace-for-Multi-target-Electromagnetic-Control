"""Independent offline volume quadrature with positive Gauss weights.

Refinement differences are numerical error estimates, NOT rigorous enclosures.
The C loop only accelerates the same distributed dipole integral as the legacy
NumPy reference. All sources and compiler options are retained for reproduction.
"""
import ctypes
import hashlib
from pathlib import Path
import subprocess
import tempfile
from functools import lru_cache
import numpy as np
from run_comparison import rotation_from_z

@lru_cache(None)
def kernel():
    src=Path(__file__).with_name('reference_kernel.c')
    digest=hashlib.sha256(src.read_bytes()).hexdigest()[:16]
    library=Path(tempfile.gettempdir())/('ijrr_reference_'+digest+'.so')
    if not library.exists():
        subprocess.run(['cc','-O3','-shared','-fPIC',str(src),'-o',str(library)],check=True)
    fn=ctypes.CDLL(str(library)).reference
    ptr=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    fn.argtypes=[ctypes.c_int,ptr,ctypes.c_int,ptr,ptr,ctypes.c_int,ptr,ptr,ptr,ptr,ctypes.c_double,ctypes.c_double,ptr]
    fn.restype=None
    return fn

@lru_cache(None)
def points(length,diameter,nr,nphi,nz):
    tr,wr=np.polynomial.legendre.leggauss(nr); tz,wz=np.polynomial.legendre.leggauss(nz)
    rho=diameter/2*np.sqrt((tr+1)/2); zs=length/2*tz
    ph=2*np.pi*(np.arange(nphi)+.5)/nphi
    rs,phis,zz=np.meshgrid(rho,ph,zs,indexing='ij')
    pts=np.column_stack((rs.ravel()*np.cos(phis.ravel()),rs.ravel()*np.sin(phis.ravel()),zz.ravel()))
    weights=np.broadcast_to(wr[:,None,None]*wz[None,None,:]/(4*nphi),(nr,nphi,nz)).ravel().copy()
    return pts,weights

def integrate(centers,beta,*,free_m,fixed_m,fixed_xy,fixed_beta,moving_cyl,fixed_cyl,level=(4,16,7),midpoint=False):
    factory=points
    if midpoint:
        from run_comparison import cylinder_points
        factory=cylinder_points
    pm,wm=factory(moving_cyl['length_m'],moving_cyl['diameter_m'],*level)
    ps,ws=factory(fixed_cyl['length_m'],fixed_cyl['diameter_m'],*level)
    Rm,Rs=rotation_from_z(beta),rotation_from_z(fixed_beta)
    pp=np.asarray(centers)-fixed_xy
    centers=np.ascontiguousarray(np.column_stack((pp,np.zeros(len(pp)))))
    pm=np.ascontiguousarray(pm@Rm.T);ps=np.ascontiguousarray(ps@Rs.T)
    u=np.ascontiguousarray(Rm[:,2]);v=np.ascontiguousarray(Rs[:,2])
    out=np.zeros((len(centers),19))
    kernel()(len(centers),centers,len(pm),pm,wm,len(ps),ps,ws,u,v,free_m,fixed_m,out)
    body=np.column_stack((out[:,:3]@Rm[:,1],out[:,:3]@Rm[:,2],out[:,3:6]@Rm[:,1],out[:,3:6]@Rm[:,2]))
    return body,out

def coil_reference(centers,beta,moment,geom,params,z,level=(5,20,9)):
    pm,wm=points(geom['length_m'],geom['diameter_m'],*level)
    R=rotation_from_z(beta);pm=np.ascontiguousarray(pm@R.T);u=np.ascontiguousarray(R[:,2])
    pp=np.column_stack((centers,np.full(len(centers),z)));maps=[];torques=[]
    for par in params:
        strength=np.linalg.norm(par[:3]);axis=np.ascontiguousarray(np.asarray(par[:3])/strength)
        out=np.zeros((len(pp),19));disp=np.ascontiguousarray(pp-par[3:])
        kernel()(len(pp),disp,len(pm),pm,wm,1,np.zeros((1,3)),np.ones(1),u,axis,moment,strength,out)
        maps.append(np.column_stack((out[:,:3]@R[:,1],out[:,:3]@R[:,2],out[:,3:6]@R[:,1],out[:,3:6]@R[:,2])))
        torques.append(out[:,8])
    return np.stack(maps,axis=-1),np.column_stack(torques)

def reference_error_bound(separation,level,*,free_m,fixed_m,moving_cyl,fixed_cyl,**unused):
    """Analytic quadrature remainder majorant, often extremely conservative.

    Positive symmetric Gauss product rules match all monomials through degree d.
    Both exact and discrete moments are supported in the same bounding spheres.
    Their unmatched even Taylor tails each obey the multipole majorant, hence 2x.
    This bounds ideal integration error, not floating-point evaluation roundoff.
    """
    from multipole_stage_a import bounding_radius,truncation_bounds
    nr,nphi,nz=level
    if nphi%2:raise ValueError('Even azimuthal count required for symmetry')
    d=min(2*nz-1,nphi-1,4*nr-2)
    L=2*(d//2)+1
    radii=[bounding_radius(g['length_m'],g['diameter_m']) for g in (fixed_cyl,moving_cyl)]
    b=truncation_bounds(separation,L,*radii,fixed_m,free_m)
    return {key:2*value for key,value in b.items()}
