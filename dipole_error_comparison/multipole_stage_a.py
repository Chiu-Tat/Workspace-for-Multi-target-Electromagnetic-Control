"""Analytic, batch-capable axial multipoles, coil coupling and absolute tails.

No numerical integration is used in this operational model. Derivatives of 1/r
are generated as exact polynomial recurrences and evaluated in float64.
The reference integration is confined to reference_gauss.py.
"""
import math
from functools import lru_cache
from itertools import product
import numpy as np
from numpy.polynomial import Polynomial
if __package__:
    from .multipole_model import cylinder_Q, bounding_radius
else:
    from multipole_model import cylinder_Q, bounding_radius

C = 1e-7

@lru_cache(None)
def numerator(alpha):
    """P_alpha such that partial^alpha(1/r)=P_alpha(r)/r^(2|alpha|+1)."""
    n = sum(alpha)
    if not n:
        return {(0,)*len(alpha): 1}
    i = next(i for i,a in enumerate(alpha) if a)
    prev = list(alpha); prev[i] -= 1
    out = {}
    def add(a,c):
        a=tuple(a); out[a]=out.get(a,0)+c
    for a,c in numerator(tuple(prev)).items():
        if a[i]:
            for j in range(len(alpha)):
                b=list(a); b[i]-=1; b[j]+=2
                add(b,c*a[i])
        b=list(a); b[i]+=1; add(b,-(2*n-1)*c)
    return {a:c for a,c in out.items() if c}

def derivatives(points, order):
    points=np.atleast_2d(points); dim=points.shape[1]
    radius=np.linalg.norm(points,axis=1)
    if np.any(radius<=0): raise ValueError('Singular source position')
    unit=points/radius[:,None]
    powers=[unit[:,i,None]**np.arange(order+1) for i in range(dim)]
    table={}
    for alpha in product(range(order+1),repeat=dim):
        n=sum(alpha)
        if n>order: continue
        val=np.zeros(len(points))
        for a,c in numerator(alpha).items():
            term=np.full(len(points),float(c))
            for i,ai in enumerate(a): term*=powers[i][:,ai]
            val+=term
        table[alpha]=val/radius**(n+1)
    return table

def multiply(a,b):
    out={}
    for x,c in a.items():
        for y,d in b.items():
            z=tuple(i+j for i,j in zip(x,y)); out[z]=out.get(z,0)+c*d
    return {x:c for x,c in out.items() if c}

def direction(v,k=1):
    dim=len(v); out={(0,)*dim:1.}
    lin={tuple(int(j==i) for j in range(dim)):float(v[i]) for i in range(dim) if v[i]}
    for _ in range(k): out=multiply(out,lin)
    return out

def evaluate(op,table):
    return sum(c*table[a] for a,c in op.items())

def coeffs(moment, geom, L):
    if L not in (1,3,5): raise ValueError('Implemented orders: 1, 3, 5')
    return {k: cylinder_Q(moment,geom['length_m'],geom['diameter_m'],k+1)/math.factorial(k+1)
            for k in range(0,L,2)}

def average_op(axis, geom, L):
    out={}
    for k,c in coeffs(1.,geom,L).items():
        for a,v in direction(axis,k).items(): out[a]=out.get(a,0)+c*v
    return out

def angular_average_op(axis,tangent,geom,L):
    out={}
    for k,c in coeffs(1.,geom,L).items():
        if not k: continue
        for a,v in multiply(direction(axis,k-1),direction(tangent)).items():
            out[a]=out.get(a,0)+c*k*v
    return out

def interaction(points,beta,L,*,free_m,fixed_m,fixed_xy,fixed_beta,moving_cyl,fixed_cyl):
    """Return body outputs, energy, world planar force, physical world-z torque.

    Outputs are [volume-averaged By,Bz,Fy,Fz]. By is NOT a finite-body
    torque surrogate. Torque includes rotation of the receiver averaging operator.
    """
    u=np.array([math.cos(beta),math.sin(beta)]); v=np.array([-u[1],u[0]])
    w=np.array([math.cos(fixed_beta),math.sin(fixed_beta)])
    r=np.linalg.norm(np.asarray(points)-fixed_xy,axis=1)
    radii=sum(bounding_radius(g['length_m'],g['diameter_m']) for g in (moving_cyl,fixed_cyl))
    if np.any(r<=radii): raise ValueError('Outside two-body exterior-series domain')
    table=derivatives(np.asarray(points)-fixed_xy,2*L+1)
    source=multiply(direction(w),average_op(w,fixed_cyl,L))
    rec=average_op(u,moving_cyl,L)
    base=multiply(source,rec)
    B=np.column_stack([C*fixed_m*evaluate(multiply(base,direction(e)),table) for e in np.eye(2)])
    energy_op=multiply(base,direction(u))
    U=-C*fixed_m*free_m*evaluate(energy_op,table)
    F=np.column_stack([C*fixed_m*free_m*evaluate(multiply(energy_op,direction(e)),table) for e in np.eye(2)])
    rot=multiply(base,direction(v))
    extra=multiply(multiply(source,angular_average_op(u,v,moving_cyl,L)),direction(u))
    torque=C*fixed_m*free_m*(evaluate(rot,table)+evaluate(extra,table))
    outputs=np.column_stack((B@v,B@u,F@v,F@u))
    return outputs,U,F,torque

def coil_map(points,beta,L,moment,geom,params,z):
    """Analytic finite-receiver [By,Bz,Fy,Fz] map and physical z-torque row.

    Each calibrated coil remains its existing dipole-per-ampere source.
    This corrects target extent; it does not recalibrate or improve coil sources.
    """
    u=np.array([math.cos(beta),math.sin(beta),0.]); v=np.array([-u[1],u[0],0.])
    avg=average_op(u,geom,L); avgrot=angular_average_op(u,v,geom,L)
    pp=np.column_stack((np.asarray(points),np.full(len(points),z)))
    maps=[]; torques=[]
    for par in params:
        table=derivatives(pp-np.asarray(par[3:]),L+2)
        base=multiply(direction(par[:3]),avg)
        B=np.column_stack([C*evaluate(multiply(base,direction(e)),table) for e in (v,u)])
        scalar=multiply(base,direction(u))
        F=np.column_stack([C*moment*evaluate(multiply(scalar,direction(e)),table) for e in (v,u)])
        torques.append(C*moment*(evaluate(multiply(base,direction(v)),table)+
            evaluate(multiply(multiply(direction(par[:3]),avgrot),direction(u)),table)))
        maps.append(np.column_stack((B,F)))
    return np.stack(maps,axis=-1),np.column_stack(torques)

def even_tail(t,N,p,rotation=False):
    """Sum_{n=N,N+2,...} (n+p)!/n! t^n [times n+1 for torque].

    Closed rational form avoids subtraction of almost equal full/tail sums.
    """
    t=np.asarray(t)
    if np.any((t<0)|(t>=1)): raise ValueError('Tail requires 0 <= radius/separation < 1')
    poly=Polynomial([1.])
    for j in range(1,p+1): poly*=Polynomial([N+j,2.])
    if rotation: poly*=Polynomial([N+1,2.])
    z=t*t
    sums=[1/(1-z),z/(1-z)**2,z*(1+z)/(1-z)**3,
          z*(1+4*z+z*z)/(1-z)**4,z*(1+11*z+11*z*z+z**3)/(1-z)**5]
    return t**N*sum(c*sums[k] for k,c in enumerate(poly.coef))

def truncation_bounds_total_degree(separation,L,source_radius,receiver_radius,source_m,receiver_m):
    """Superseded total-degree majorant (kept for comparison).

    Sums every combined degree n >= L+1, including terms the rectangular model
    keeps; valid but conservative by one to two orders of magnitude.
    """
    r=np.asarray(separation); t=(source_radius+receiver_radius)/r
    b=C*abs(source_m)/r**3*even_tail(t,L+1,2)
    g=C*abs(source_m)/r**4*even_tail(t,L+1,3)
    torque=C*abs(source_m*receiver_m)/r**3*even_tail(t,L+1,2,True)
    return {'field':b,'gradient':g,'force':abs(receiver_m)*g,'torque':torque}


def _full(x,y,p):
    """Sum over even (ks, kr) of C(ks+kr,ks) (n+1)...(n+p) x^ks y^kr, n = ks+kr."""
    return (even_tail(x+y,0,p)+even_tail(np.abs(y-x),0,p))/2


def _odd_derivative(t):
    """d/dt of sum_{n even} (n+1)(n+2) t^n, an odd function of t."""
    a=np.abs(t)
    with np.errstate(invalid='ignore',divide='ignore'):
        g=np.where(a>1e-8,(even_tail(a,0,3)-3*even_tail(a,0,2))/np.where(a>1e-8,a,1.),24*a)
    return np.sign(t)*g


def _kept(x,y,p,L,rotation=False):
    out=0.
    for ks in range(0,L,2):
        for kr in range(0,L,2):
            n=ks+kr
            c=math.comb(n,ks)*math.prod(range(n+1,n+p+1))*((1+kr) if rotation else 1)
            out=out+c*x**ks*y**kr
    return out


def truncation_bounds_dipole_density(separation,L,source_radius,receiver_radius,source_m,receiver_m):
    """Uniform component bounds for symmetric uniformly magnetized bodies.

    Dipole-density picture: each body is a positive dipole density of total
    moment |m| inside its bounding sphere.  With d the centre offset (|d| = s),
    receiver offset xi and source offset eta, the degree-(ks, kr) part of the
    Taylor expansion of the kernel D^2(1/r) at d + xi - eta is bounded by
    C(ks+kr, ks) (n+1)(n+2) a_s^ks a_r^kr / s^(n+3), because the n-th derivative
    tensor of 1/r has norm n!/s^(n+1) and multilinear and polynomial norms of
    symmetric forms agree on a Hilbert space (Banach).  Inversion symmetry makes
    odd volume moments vanish, so only even ks and kr remain.  The rectangular
    order-L model keeps ks, kr <= L-1; the bound sums every other even part,
    in closed form as the full sum minus the finite kept sum.  One more
    derivative gives the gradient and force; a receiver rotation multiplies a
    part by at most (1 + kr).  With source_radius = 0 (a point-dipole coil) it
    reduces to the receiver tail alone.  Floating-point evaluation itself is
    not an outward-rounded certificate.
    """
    r=np.asarray(separation,dtype=float)
    x=source_radius/r; y=receiver_radius/r
    if np.any(x+y>=1): raise ValueError('Tail requires (a_s + a_r)/s < 1')
    tail_b=np.maximum(_full(x,y,2)-_kept(x,y,2,L),0.)
    tail_g=np.maximum(_full(x,y,3)-_kept(x,y,3,L),0.)
    full_rot=_full(x,y,2)+y*(_odd_derivative(x+y)+_odd_derivative(y-x))/2
    tail_t=np.maximum(full_rot-_kept(x,y,2,L,True),0.)
    b=C*abs(source_m)/r**3*tail_b
    g=C*abs(source_m)/r**4*tail_g
    torque=C*abs(source_m*receiver_m)/r**3*tail_t
    return {'field':b,'gradient':g,'force':abs(receiver_m)*g,'torque':torque}

def _parity_sum(t,parity,p):
    """sum over n of the given parity of (n+1)...(n+p) t^n; odd sums are odd in t."""
    a=np.abs(t)
    v=even_tail(a,parity,p)
    return np.sign(t)*v if parity==1 else v


def _charge_full(x,y,parity,p):
    """sum over odd l (source) of C(n,l) (n+1)...(n+p) x^l y^(n-l), n of given parity."""
    return (_parity_sum(x+y,parity,p)-_parity_sum(y-x,parity,p))/2


def _charge_kept(x,y,p,lmax,kset):
    out=0.
    for l in range(1,lmax+1,2):
        for k in kset:
            n=l+k
            out=out+math.comb(n,l)*math.prod(range(n+1,n+p+1))*x**l*y**k
    return out


def truncation_bounds_charge(separation,L,source_radius,receiver_radius,source_m,receiver_m,
                             source_length,receiver_length):
    """Surface-charge picture for uniformly axially magnetized cylinders.

    Each body is end-face charge of total absolute value 2|m|/length inside its
    bounding sphere.  The field and its gradient are Coulomb kernels of the
    source charge (odd source degree l), averaged over the receiver volume
    (even degree k); the force and torque use the receiver charge (odd degree)
    and, for the torque, the lever arm |xi| <= a_r.  The model keeps l <= L and
    receiver charge degree <= L, i.e. k <= L-1.  Same norm argument as the
    dipole-density bound; the two are independent and both rigorous.
    """
    r=np.asarray(separation,dtype=float)
    x=source_radius/r; y=receiver_radius/r
    if np.any(x+y>=1): raise ValueError('Tail requires (a_s + a_r)/s < 1')
    Qs=2*abs(source_m)/source_length; Qr=2*abs(receiver_m)/receiver_length
    even_k=range(0,L,2); odd_k=range(1,L+1,2)
    tb=np.maximum(_charge_full(x,y,1,1)-_charge_kept(x,y,1,L,even_k),0.)   # n odd
    tg=np.maximum(_charge_full(x,y,1,2)-_charge_kept(x,y,2,L,even_k),0.)
    tf=np.maximum(_charge_full(x,y,0,1)-_charge_kept(x,y,1,L,odd_k),0.)    # n even
    field=C*Qs/r**2*tb
    grad=C*Qs/r**3*tg
    force=C*Qs*Qr/r**2*tf
    torque=C*Qs*Qr*receiver_radius/r**2*tb
    return {'field':field,'gradient':grad,'force':force,'torque':torque}


def truncation_bounds(separation,L,source_radius,receiver_radius,source_m,receiver_m,
                      source_length=None,receiver_length=None):
    """Smaller of the two rigorous truncation bounds, componentwise.

    The dipole-density bound always applies.  When both cylinder lengths are
    given (two finite bodies), the surface-charge bound is also computed and the
    minimum is returned.  A point-dipole coil source has no length; then only the
    dipole-density bound, which reduces to the receiver tail, is used.
    """
    b=truncation_bounds_dipole_density(separation,L,source_radius,receiver_radius,source_m,receiver_m)
    if source_length is None or receiver_length is None or source_radius==0:
        return b
    c=truncation_bounds_charge(separation,L,source_radius,receiver_radius,source_m,receiver_m,
                               source_length,receiver_length)
    return {k:np.minimum(b[k],c[k]) for k in b}


def field_gradient(points,beta,L,*,fixed_xy,fixed_beta,moving_cyl,fixed_cyl,fixed_m,free_m):
    """World 3x3 gradient of the receiver-averaged source field, planar axes."""
    u=np.array([math.cos(beta),math.sin(beta)])
    w=np.array([math.cos(fixed_beta),math.sin(fixed_beta)])
    table=derivatives(np.asarray(points)-fixed_xy,2*L+1)
    base=multiply(multiply(direction(w),average_op(w,fixed_cyl,L)),average_op(u,moving_cyl,L))
    J=np.zeros((len(points),3,3))
    for i,e in enumerate(np.eye(2)):
        for j,f in enumerate(np.eye(2)):
            J[:,i,j]=C*fixed_m*evaluate(multiply(multiply(base,direction(e)),direction(f)),table)
    J[:,2,2]=-J[:,0,0]-J[:,1,1]
    return J
