"""Fit the cylinder force models to the published CSV and check reference convergence."""
from pathlib import Path
import json, hashlib
import numpy as np
from . import reference_gauss as rg
from .. import multipole_model as mm
OUT=Path(__file__).resolve().parent
SOURCE = OUT / 'data_original.csv'
SENS={'length_m':.011,'diameter_m':.004}; POS={'length_m':.010,'diameter_m':.004}

def dipole(x,y,t):
    r=np.array([-x,-y,0.]); s=np.linalg.norm(r); h=r/s
    u=np.array([0.,1.,0.]); v=np.array([np.cos(t),np.sin(t),0.])
    a=u@h; b=v@h; c=u@v
    F=3e-7/s**4*(a*v+b*u+c*h-5*a*b*h)
    B=1e-7/s**3*(3*b*h-v)
    return np.r_[F[:2],np.cross(u,B)[2]]

def finite(x,y,t,level=(6,24,11)):
    _,r=rg.integrate(np.zeros((1,2)),np.pi/2,free_m=1.,fixed_m=1.,fixed_xy=np.array([x,y]),fixed_beta=t,moving_cyl=SENS,fixed_cyl=POS,level=level)
    return r[0,[3,4,8]]

def design(U,ix):
    A=np.zeros((2*len(ix),3)); A[:,0]=U[ix,:2].ravel(); A[::2,1]=1; A[1::2,2]=1
    return A

def fit(U,F,ix):return np.linalg.lstsq(design(U,ix),F[ix].ravel(),rcond=None)[0]

def cv(U,F,groups):
    pred=np.zeros_like(F)
    for g in np.unique(groups):
        test=groups==g; c=fit(U,F,np.where(~test)[0]); pred[test]=c[0]*U[test,:2]+c[1:]
    return pred-F

def metrics(e):
    n=np.linalg.norm(e,axis=1)*1e3
    return {'rms_mN':float(np.sqrt(np.mean(n*n))),'max_mN':float(n.max())}

def main():
    with SOURCE.open() as source:
        headers = source.readline().strip().split(',')
        D = np.loadtxt(source, delimiter=',')
    n = len(D)
    ix = np.arange(n)
    xy=D[:,:2]*1e-3; th=np.deg2rad(D[:,2]); sep=np.linalg.norm(xy,axis=1)*1e3
    F=D[:,3:5]*9.80665e-3; T=D[:,8]*1e-6 # user-specified gf and mN mm
    _,groups=np.unique(D[:,:2],axis=0,return_inverse=True)
    U={}; cache=OUT/'kernels.npz'
    if cache.exists():U=dict(np.load(cache))
    else:
        U['dip']=np.array([dipole(x,y,t) for (x,y),t in zip(xy,th)])
        U['finite']=np.array([finite(x,y,t) for (x,y),t in zip(xy,th)])
        for L in (3,5):
            rows=[]
            for (x,y),t in zip(xy,th):
                a=mm.multipole_interaction(0.,0.,np.pi/2,L,free_m=1.,fixed_m=1.,fixed_xy=(x,y),fixed_beta=t,moving_cyl=SENS,fixed_cyl=POS)
                rows.append([-a[2],a[3]])
            U[f'L{L}']=np.array(rows)
        np.savez(cache,**U)
    result={'source':SOURCE.name,'sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'headers':headers,'n':n,'positions':len(np.unique(groups)),'models':{}}
    residuals={}; coefs={}
    for k in ['dip','L3','L5','finite']:
        c=fit(U[k],F,ix); coefs[k]=c
        e=cv(U[k],F,ix); eg=cv(U[k],F,groups); residuals[k]=e
        sensitivity={}
        for label,omit in [('omit_row54',[52]),('omit_row55',[53]),('omit_both',[52,53])]:
            keep=np.array([j for j in ix if j not in omit]); cc=fit(U[k],F,keep)
            sensitivity[label]={'p':cc[0],**metrics(cv(U[k][keep],F[keep],np.arange(len(keep))))}
        result['models'][k]={'p':c[0],'bias_mN':(c[1:]*1e3).tolist(),'loo':metrics(e),'leave_position_out':metrics(eg),'sensitivity':sensitivity}
    # Nominal with no bias correction (operational baseline) and held-out bias fitting.
    p0=.12*.145; en=U['dip'][:,:2]*p0-F
    eb=np.zeros_like(F)
    for j in ix:
        keep=ix!=j; b=np.mean(F[keep]-p0*U['dip'][keep,:2],axis=0)
        eb[j]=p0*U['dip'][j,:2]+b-F[j]
    result['nominal_raw']=metrics(en); result['nominal_loo_bias']=metrics(eb)
    result['nominal_bias_mN']=(np.mean(F-p0*U['dip'][:,:2],axis=0)*1e3).tolist()
    result['torque']={}
    for k in ['dip','finite']:
        # Independent moment estimate from torque; measured forces account for lever arm.
        A=np.column_stack([U[k][:,2],np.ones(n),F[:,1],-F[:,0]])
        c=np.linalg.lstsq(A,T,rcond=None)[0]; e=A@c-T
        el=np.zeros(n)
        for j in ix:
            keep=ix!=j; cc=np.linalg.lstsq(A[keep],T[keep],rcond=None)[0]; el[j]=A[j]@cc-T[j]
        # Stronger cross-channel test: moment fixed from force, nuisance bias/lever only.
        B=A[:,1:]; cc=np.linalg.lstsq(B,T-coefs[k][0]*U[k][:,2],rcond=None)[0]
        ec=coefs[k][0]*U[k][:,2]+B@cc-T
        result['torque'][k]={'p':c[0],'bias_mNmm':c[1]*1e6,'lever_mm':(c[2:]*1e3).tolist(),'fit_rms_mNmm':np.sqrt(np.mean(e*e))*1e6,'loo_rms_mNmm':np.sqrt(np.mean(el*el))*1e6,'force_p_fixed_rms_mNmm':np.sqrt(np.mean(ec*ec))*1e6}
    result['null_channels']={}
    for name,col,scale in [('Fz_mN',5,9.80665),('Tx_mNmm',6,1),('Ty_mNmm',7,1)]:
        a=D[:,col]*scale; result['null_channels'][name]={'mean':a.mean(),'sample_sd':a.std(ddof=1),'max_abs':np.abs(a).max(),'corr_row':np.corrcoef(ix,a)[0,1]}
    result['bins']=[]
    for lo,hi in [(14.5,16),(16,20),(20,23),(23,35),(35,60)]:
        mask=(sep>=lo)&(sep<hi)
        result['bins'].append({'lo':lo,'hi':hi,'n':int(mask.sum()),**{k:metrics(e[mask]) for k,e in {**residuals,'nominal':en}.items()}})
    result['rho_at_mm']={str(s):float(np.max(np.linalg.norm(residuals['finite'][sep>=s-1e-10],axis=1))*1e3) for s in [14.5,16,17.62,20,23,24.5,30,35,40,50]}
    # Same-parameter ideal-model discrepancy is distinct from data residual.
    p=coefs['finite'][0]
    result['same_p_ideal_discrepancy']={k:metrics(p*(U[k][:,:2]-U['finite'][:,:2])) for k in ['dip','L3','L5']}
    # Refine every measured configuration. This estimates numerical convergence, not a bound.
    fine=np.array([finite(x,y,t,(8,32,15)) for (x,y),t in zip(xy,th)])
    result['quadrature_refinement_force_max_mN']=float(np.linalg.norm((fine[:,:2]-U['finite'][:,:2])*p,axis=1).max()*1e3)
    result['quadrature_refinement_torque_max_mNmm']=float(np.abs((fine[:,2]-U['finite'][:,2])*p).max()*1e6)
    # Independent analytical dipole vs multipole L=1, sign and coordinate check.
    l1=[]
    for (x,y),t in zip(xy,th):
        a=mm.multipole_interaction(0.,0.,np.pi/2,1,free_m=1,fixed_m=1,fixed_xy=(x,y),fixed_beta=t,moving_cyl=SENS,fixed_cyl=POS);l1.append([-a[2],a[3]])
    result['L1_dipole_max_abs_unit_p']=float(np.abs(np.array(l1)-U['dip'][:,:2]).max())
    row=np.column_stack([np.arange(2,n+2),D[:,:3],sep,F*1e3,en*1e3,*[residuals[k]*1e3 for k in residuals]])
    np.savetxt(OUT/'per_pose_errors.csv',row,delimiter=',',header='excel_row,x_mm,y_mm,angle_deg,s_mm,Fx_mN,Fy_mN,nom_dx_mN,nom_dy_mN,'+','.join(k+'_'+a+'_mN' for k in residuals for a in ['dx','dy']),comments='')
    np.savez(OUT/'analysis.npz',D=D,F=F,T=T,sep=sep,p=p,**{k+'_loo':e for k,e in residuals.items()},**{k+'_coef':c for k,c in coefs.items()})
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
