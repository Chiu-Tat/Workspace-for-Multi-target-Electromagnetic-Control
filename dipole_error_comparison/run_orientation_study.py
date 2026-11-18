#!/usr/bin/env python3
"""Bounded multi-heading sensitivity study and manuscript figure generator."""
import json, math, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from cases.case_6_interaction_obstacle_avoidance.simulation import model
from run_comparison import finite_interaction, dipole_interaction, contains_box

HEADINGS_DEG = (0, 45, 90, 135)
N = 31

def main():
    cfg=json.loads((HERE/'config.json').read_text())
    xs=np.linspace(*cfg['x_limits_m'],N); ys=np.linspace(*cfg['y_limits_m'],N)
    gm,gf=cfg['moving_cylinder'],cfg['fixed_cylinder']
    exclusion=.5*math.hypot(gm['length_m'],gm['diameter_m'])+.5*math.hypot(gf['length_m'],gf['diameter_m'])
    center=np.array([0.,model.B_HOLD,0.,0.]); half=np.array([0.,0.,0.,model.F_MAX])
    summaries=[]; arrays={}
    bins=np.linspace(exclusion,.09,10)
    for deg in HEADINGS_DEG:
        beta=math.radians(deg); dip=np.zeros((N,N),bool); finite=np.zeros_like(dip); valid=np.zeros_like(dip)
        disc=np.full((N,N,4),np.nan); sep=np.full((N,N),np.nan)
        for j,y in enumerate(ys):
            for i,x in enumerate(xs):
                rr=math.hypot(x-model.FIXED_X,y-model.FIXED_Y); sep[j,i]=rr
                if rr<exclusion: continue
                valid[j,i]=True
                Ah=model.actuation_matrix_body(x,y,beta); A,c=Ah[:,:10],Ah[:,10]
                cfinite=finite_interaction(x,y,beta,cfg); d=np.abs(cfinite-c); disc[j,i]=d
                dip[j,i]=contains_box(A,center-c,half,cfg['current_limit_A'])
                finite[j,i]=contains_box(A,center-cfinite,half,cfg['current_limit_A'])
        radial=[]
        for lo,hi in zip(bins[:-1],bins[1:]):
            take=valid&(sep>=lo)&(sep<hi)
            if take.any():
                radial.append({'r_mid_m':float((lo+hi)/2),'n':int(take.sum()),
                    'force_error_p95_N':float(np.percentile(np.max(disc[take,2:],axis=1),95)),
                    'force_error_max_N':float(np.max(disc[take,2:]))})
        both=dip&finite&valid; either=(dip|finite)&valid
        summaries.append({'heading_deg':deg,'valid':int(valid.sum()),
            'dipole':int((dip&valid).sum()),'finite_body':int((finite&valid).sum()),
            'dipole_only':int((dip&~finite&valid).sum()),
            'finite_body_only':int((finite&~dip&valid).sum()),
            'jaccard':float(both.sum()/max(1,either.sum())),
            'radial_force_discrepancy':radial})
        arrays[f'dip_{deg}']=dip; arrays[f'finite_{deg}']=finite; arrays[f'valid_{deg}']=valid; arrays[f'disc_{deg}']=disc
    out=HERE/'results'; out.mkdir(exist_ok=True)
    np.savez_compressed(out/'orientation_study.npz',xs=xs,ys=ys,**arrays)
    result={'scope':'sampled multi-heading numerical sensitivity; not continuous/physical envelope','grid_n':N,
        'headings_deg':list(HEADINGS_DEG),'nonoverlap_center_separation_m':exclusion,'summaries':summaries}
    (out/'orientation_metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    make_figure(result)
    print(json.dumps(result,indent=2))

def make_figure(orientation):
    base=np.load(HERE/'results/masks_and_errors.npz'); xs=base['xs']*1e3; ys=base['ys']*1e3
    dip,finite,valid=base['dipole'],base['finite_body'],base['valid']
    dipole_only=dip&~finite&valid; finite_only=finite&~dip&valid
    X,Y=np.meshgrid(xs,ys)
    fig,ax=plt.subplots(2,2,figsize=(10.5,7.0),constrained_layout=True)
    state=np.full(dip.shape,-1.0)
    state[valid&~dip&~finite]=0.0
    state[finite_only]=1.0
    state[dipole_only]=2.0
    state[dip&finite]=3.0
    for a,title in [(ax[0,0],'(a) Full $\\beta=90^\\circ$ slice'),(ax[0,1],'(b) Boundary-detail view')]:
        a.contourf(X,Y,state,levels=[-1.5,-.5,.5,1.5,2.5,3.5],
                   colors=['#d9d9d9','#f2d7d5','#f2c66d','#e66b62','#ccebd8'])
        a.contour(X,Y,dip.astype(float),levels=[.5],colors=['#1976d2'],linewidths=2)
        a.contour(X,Y,finite.astype(float),levels=[.5],colors=['#0b7a3e'],linewidths=2)
        a.scatter(X[dipole_only],Y[dipole_only],s=18,c='#9d0b0b',marker='x',label='dipole only')
        a.scatter(X[finite_only],Y[finite_only],s=18,c='#8a5a00',marker='+',label='finite-body only')
        a.plot(model.FIXED_X*1e3,model.FIXED_Y*1e3,'ko',ms=5)
        a.set_title(title); a.set_xlabel('$x$ (mm)'); a.set_ylabel('$y$ (mm)'); a.set_aspect('equal')
    ax[0,0].plot([],[],color='#1976d2',lw=2,label='point-dipole boundary'); ax[0,0].plot([],[],color='#0b7a3e',lw=2,label='finite-body boundary')
    ax[0,0].legend(fontsize=8,loc='lower left')
    disagreement=dipole_only|finite_only
    if disagreement.any():
        pad=8; ax[0,1].set_xlim(X[disagreement].min()-pad,X[disagreement].max()+pad); ax[0,1].set_ylim(Y[disagreement].min()-pad,Y[disagreement].max()+pad)
    for s in orientation['summaries']:
        rr=np.array([v['r_mid_m'] for v in s['radial_force_discrepancy']])*1e3
        ee=np.array([v['force_error_p95_N'] for v in s['radial_force_discrepancy']])*1e3
        ax[1,0].plot(rr,ee,'o-',label=f"$\\beta={s['heading_deg']}^\\circ$")
    ax[1,0].set_yscale('log'); ax[1,0].set_xlabel('centre separation (mm)'); ax[1,0].set_ylabel('sampled 95th-percentile\nmax force discrepancy (mN)'); ax[1,0].set_title('(c) Separation and orientation sensitivity'); ax[1,0].grid(True,which='both',alpha=.25); ax[1,0].legend(fontsize=8)
    deg=[s['heading_deg'] for s in orientation['summaries']]
    dip_only=[s['dipole_only'] for s in orientation['summaries']]
    finite_only_counts=[s['finite_body_only'] for s in orientation['summaries']]
    xx=np.arange(len(deg)); width=.36
    ax[1,1].bar(xx-width/2,dip_only,width,color='#e66b62',label='dipole only')
    ax[1,1].bar(xx+width/2,finite_only_counts,width,color='#f2c66d',label='finite-body only')
    ax[1,1].set_xticks(xx,[str(v) for v in deg])
    ax[1,1].set_xlabel('moving-magnet heading $\\beta$ (deg)'); ax[1,1].set_ylabel('disagreement samples'); ax[1,1].set_title('(d) Matched pointwise slices')
    ax[1,1].legend(fontsize=8)
    fig.savefig(HERE/'results/dipole_error_tfw_comparison.pdf')
    fig.savefig(HERE/'results/dipole_error_tfw_comparison.png',dpi=240)
    plt.close(fig)

if __name__=='__main__': main()
