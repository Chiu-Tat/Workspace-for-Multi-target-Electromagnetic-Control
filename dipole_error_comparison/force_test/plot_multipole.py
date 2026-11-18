"""Paper figure: experimental validation of multipoles and same-parameter truncation check."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
O=Path(__file__).resolve().parent
A=np.load(O/'analysis.npz');K=np.load(O/'kernels.npz');s=A['sep'];p=float(A['p'])
plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
fig,axes=plt.subplots(1,2,figsize=(9.4,3.8),layout='constrained')
for k,label,color,marker in [('dip',r'$L=1$','#b2182b','o'),('L3',r'$L=3$','#e69f00','^'),('L5',r'$L=5$','#0072b2','s')]:
    measured=np.linalg.norm(A[k+'_loo'],axis=1)*1e3
    ideal=np.linalg.norm(p*(K[k][:,:2]-K['finite'][:,:2]),axis=1)*1e3
    for ax,e in zip(axes,[measured,ideal]):
        ax.scatter(s,e,s=27,marker=marker,facecolors='none',edgecolors=color,linewidths=1.1,label=label,alpha=.8)
for ax in axes:
    ax.set(xlabel='Centre separation (mm)',yscale='log',xlim=(12,62))
    ax.grid(axis='y',which='major',color='.9',lw=.6)
    ax.legend(frameon=False,ncol=3,loc='upper right')
axes[0].set(title='(a) Comparison with measured force',ylabel='Leave-one-pose-out vector error (mN)',ylim=(.05,100))
axes[1].set(title='(b) Multipole truncation check',ylabel='Difference from volume-integral reference (mN)',ylim=(1e-7,100))
fig.savefig(O/'multipole_validation.pdf',bbox_inches='tight')
fig.savefig(O/'multipole_validation.png',dpi=220,bbox_inches='tight')
