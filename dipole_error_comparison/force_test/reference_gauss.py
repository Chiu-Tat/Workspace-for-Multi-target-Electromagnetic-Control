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
def rotation_from_z(beta):
    z = np.array([np.cos(beta), np.sin(beta), 0.])
    y = np.array([-np.sin(beta), np.cos(beta), 0.])
    return np.column_stack((np.cross(y,z),y,z))

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
        raise ValueError("This reference uses Gauss quadrature")
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
