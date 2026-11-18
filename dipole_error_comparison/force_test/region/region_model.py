"""Local first-order affine/Taylor enclosures with explicit nonlinear remainders.
Three shared symbols encode x,y,beta. Other dependencies are conservatively
collapsed into a scalar remainder at each operation. Float64, no directed rounding.
No production/hardware modules are patched.
"""
import math,itertools
from pathlib import Path
from functools import lru_cache
import numpy as np
from emns.workspace.fast import params_list
from ...multipole_stage_a import numerator
from ...multipole_model import cylinder_Q,multipole_interaction
from ..analyze import SENS,POS
from hybrid_region.certificates import affine_constructive_inside_certificate,dual_outside_certificate,box_vertices
_IDS=itertools.count(3)
class Form:
    def __init__(self,center,deviations=None,linear=None,error=0):
        self.center=float(center)
        self.linear=np.zeros(3) if linear is None else np.array(linear,float)
        self.error=float(error)
        if deviations is not None:
            for k,v in deviations.items():
                if k<3:self.linear[k]=v
                else:self.error+=abs(v)
        self.symbol=next(_IDS)
    @property
    def deviations(self):
        d={k:v for k,v in enumerate(self.linear) if v}
        if self.error:d[self.symbol]=self.error
        return d
    @property
    def radius(self):return np.abs(self.linear).sum()+self.error
    def __add__(self,b):
        if not isinstance(b,Form):return Form(self.center+b,linear=self.linear,error=self.error)
        return Form(self.center+b.center,linear=self.linear+b.linear,error=self.error+b.error)
    __radd__=__add__
    def __neg__(self):return Form(-self.center,linear=-self.linear,error=self.error)
    def __sub__(self,b):return self+-b
    def __rsub__(self,b):return -self+b
    def __mul__(self,b):
        if not isinstance(b,Form):return Form(self.center*b,linear=self.linear*b,error=self.error*abs(b))
        err=abs(self.center)*b.error+abs(b.center)*self.error+self.radius*b.radius
        return Form(self.center*b.center,linear=self.center*b.linear+b.center*self.linear,error=err)
    __rmul__=__mul__
    def __truediv__(self,b):
        if isinstance(b,Form):return self*b.power(-1)
        return self*(1/b)
    def __pow__(self,n):
        if n==0:return Form(1)
        if n==1:return self
        if isinstance(n,int) and n>0:
            out=Form(1)
            for _ in range(n):out=out*self
            return out
        return self.power(n)
    def power(self,p):
        c=self.center;r=self.radius;a=c-r;b=c+r
        if a<=0:raise ValueError('nonpositive power domain')
        slope=p*c**(p-1);M=abs(p*(p-1))*max(a**(p-2),b**(p-2))
        return Form(c**p,linear=slope*self.linear,error=abs(slope)*self.error+.5*M*r*r)
    def sin(self):return Form(math.sin(self.center),linear=math.cos(self.center)*self.linear,error=abs(math.cos(self.center))*self.error+.5*self.radius**2)
    def cos(self):return Form(math.cos(self.center),linear=-math.sin(self.center)*self.linear,error=abs(math.sin(self.center))*self.error+.5*self.radius**2)

def variables(box):
    return [Form((box[2*k]+box[2*k+1])/2,linear=np.eye(3)[k]*(box[2*k+1]-box[2*k])/2) for k in range(3)]

def dip_source(x,y,beta,source_xyz,momentvec,mr=.12):
    r=[x-source_xyz[0],y-source_xyz[1],Form(.082-source_xyz[2])]
    ux,uy=beta.cos(),beta.sin();u=[ux,uy,Form(0)]
    rr=sum(a*a for a in r);inv3=rr.power(-1.5);inv5=rr.power(-2.5);inv7=rr.power(-3.5)
    mdotr=sum(a*b for a,b in zip(momentvec,r));udotr=sum(a*b for a,b in zip(u,r));udotm=sum(a*b for a,b in zip(u,momentvec))
    B=[1e-7*(3*r[k]*mdotr*inv5-momentvec[k]*inv3) for k in range(2)]
    F=[3e-7*mr*((udotr*momentvec[k]+mdotr*u[k]+udotm*r[k])*inv5-5*udotr*mdotr*r[k]*inv7) for k in range(2)]
    return np.array([-uy*B[0]+ux*B[1],ux*B[0]+uy*B[1],-uy*F[0]+ux*F[1],ux*F[0]+uy*F[1]],object)

def multipole_force(x,y,beta,p,L=5):
    # Fixed source axis +y. Derivative evaluation in normalized coordinates.
    scale=.02;rx=(x-.02)/scale;ry=y/scale
    rr=rx*rx+ry*ry;ux,uy=beta.cos(),beta.sin()
    xp=[rx**a for a in range(12)];yp=[ry**a for a in range(12)]
    up=[ux**a for a in range(6)];vp=[uy**a for a in range(6)]
    radial={n:rr.power(-n-.5) for n in range(3,12,2)}
    @lru_cache(None)
    def D(a,b):
        n=a+b
        num=sum(c*xp[i]*yp[j] for (i,j),c in numerator((a,b)).items())
        return num*radial[n]/scale**(n+1)
    fx=Form(0);fy=Form(0)
    for l in range(1,L+1,2):
        qs=cylinder_Q(1.,POS['length_m'],POS['diameter_m'],l)
        for k in range(0,L,2):
            qr=cylinder_Q(1.,SENS['length_m'],SENS['diameter_m'],k+1)
            fac=1e-7*p*qs*qr/(math.factorial(l)*math.factorial(k+1))
            for j in range(k+2):
                w=fac*math.comb(k+1,j)*up[j]*vp[k+1-j]
                fx+=w*D(j+1,l+k+1-j);fy+=w*D(j,l+k+2-j)
    return np.array([-uy*fx+ux*fy,ux*fx+uy*fy],object)

def maps(box,p,include_multipole=True):
    x,y,b=variables(box)
    G=np.column_stack([dip_source(x,y,b,par[3:],par[:3]) for par in params_list])
    c1=dip_source(x,y,b,[.02,0,.082],[0,.145,0])
    c5=None
    if include_multipole:
        c5=c1.copy();c5[2:]=multipole_force(x,y,b,p)
    return G,c1,c5

def bounds(a):
    centers=np.array([v.center for v in a.flat]).reshape(a.shape)
    radius=np.array([v.radius for v in a.flat]).reshape(a.shape)
    return centers-radius,centers+radius

def task(eps):return box_vertices(np.array([0,.02,-eps,-.1-eps]),np.array([0,.02,eps,.1+eps]))

def classify(G,c,eps_inside=0.,eps_outside=0.):
    # Dimensionless rows improve LP numerics; exact same physical task.
    scale=np.array([.02,.02,.1,.1]);g=G/scale[:,None];cc=c/scale
    r=affine_constructive_inside_certificate(g,task(eps_inside)/scale,-15*np.ones(10),15*np.ones(10),c_affine=cc)
    if r.certified:return 1
    gl,gu=bounds(g);cl,cu=bounds(cc)
    out=dual_outside_certificate(gl,gu,task(eps_outside)/scale,-15*np.ones(10),15*np.ones(10),c_lower=cl,c_upper=cu)
    return -1 if out.certified else 0

def separation(box):
    nearest=[max(box[0]-.02,0,.02-box[1]),max(box[2],0,-box[3])]
    farthest=[max(abs(box[0]-.02),abs(box[1]-.02)),max(abs(box[2]),abs(box[3]))]
    return np.linalg.norm(nearest)*1e3,np.linalg.norm(farthest)*1e3
