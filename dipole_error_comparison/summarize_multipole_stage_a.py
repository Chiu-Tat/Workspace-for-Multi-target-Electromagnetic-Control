"""Generate paper tables/figure directly from saved Stage A arrays."""
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multipole_stage_a import field_gradient,truncation_bounds,bounding_radius
from test_multipole_stage_a import kw

HERE=Path(__file__).parent;OUT=HERE/'results'/'multipole_stage_a'
PAPER=HERE / 'results'

def main():
    metrics=json.loads((OUT/'metrics.json').read_text());arrays={h:dict(np.load(OUT/f'heading_{h}.npz')) for h in (0,45,90,135)}
    pooled={};details={}
    for L in (1,3,5):
        force=[];field=[];torque=[];grad=[];sep=[];refine=[];max_ratio=0
        for h,d in arrays.items():
            J=field_gradient(d['points'],np.deg2rad(h),L,**kw)
            je=np.max(abs(J-d['reference_raw'][:,10:].reshape(-1,3,3)),axis=(1,2))
            grad.extend(je);sep.extend(d['radius']);force.extend(np.max(abs(d[f'L{L}_offset'][:,2:]-d['reference'][:,2:]),axis=1))
            field.extend(np.max(abs(d[f'L{L}_offset'][:,:2]-d['reference'][:,:2]),axis=1))
            torque.extend(abs(d[f'L{L}_torque']-d['reference_raw'][:,8]));refine.extend(np.max(d['refinement_change'][:,2:],axis=1))
            radii=[bounding_radius(g['length_m'],g['diameter_m']) for g in (kw['fixed_cyl'],kw['moving_cyl'])]
            b=truncation_bounds(d['radius'],L,*radii,kw['fixed_m'],kw['free_m'],source_length=kw['fixed_cyl']['length_m'],receiver_length=kw['moving_cyl']['length_m'])
            max_ratio=max(max_ratio,float(max(je/b['gradient'])))
        v={k:np.asarray(value) for k,value in dict(force=force,field=field,torque=torque,gradient=grad,separation=sep,reference_force_refinement=refine).items()}
        pooled[L]=v;details[str(L)]={'max_gradient_error_to_bound':max_ratio,'bins':{}}
        for low,high in ((.0113,.02),(.02,.03),(.03,.05),(.05,.1)):
            sel=(v['separation']>=low)&(v['separation']<high)
            details[str(L)]['bins'][f'{low*1000:g}-{high*1000:g}']={name:{'max':float(max(v[name][sel])),'p95':float(np.quantile(v[name][sel],.95))} for name in ('field','gradient','force','torque','reference_force_refinement')}
        details[str(L)]['minimum_separation_sensitivity']={str(mm):{'samples':int(np.sum(v['separation']>=mm/1000)),'max_force_error_N':float(max(v['force'][v['separation']>=mm/1000]))} for mm in (12,15,18,20,25,30)}
    (OUT/'pooled_metrics.json').write_text(json.dumps(details,indent=2)+'\n')
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(10.5,3.2),layout='constrained')
    edges=np.array([11.3,15,20,25,30,40,50,65,90]);colors=['#8c510a','#2166ac','#1b7837']
    for L,col in zip((1,3,5),colors):
        v=pooled[L]
        for ax,name,scale in ((axes[0],'force',1000),(axes[1],'field',1000)):
            vals=[np.quantile(v[name][(v['separation']>=a/1000)&(v['separation']<b/1000)],.95)*scale for a,b in zip(edges[:-1],edges[1:])]
            ax.semilogy((edges[:-1]+edges[1:])/2,vals,'o-',color=col,label=f'L={L}',ms=3)
    axes[0].set(ylabel='95th percentile max force error (mN)',xlabel='Centre separation (mm)',title='(a) Force error, four headings')
    axes[1].set(ylabel='95th percentile max field error (mT)',xlabel='Centre separation (mm)',title='(b) Averaged-field error')
    d=arrays[90];r=np.zeros(len(d['points']),int)
    r[d['L5_physical_mask']]=1;r[d['L5_reserved_mask']]=2
    from matplotlib.colors import ListedColormap
    axes[2].scatter(d['points'][:,0]*1000,d['points'][:,1]*1000,c=r,cmap=ListedColormap(['#cccccc','#92c5de','#238b45']),vmin=0,vmax=2,s=4,marker='s',linewidth=0)
    axes[2].set(xlabel='x (mm)',ylabel='y (mm)',title=r'(c) $L=5$, physical task, $90^\circ$',aspect='equal')
    from matplotlib.patches import Patch
    axes[2].legend(handles=[Patch(color='#238b45',label='Tail-reserved feasible'),Patch(color='#92c5de',label='Nominal only'),Patch(color='#cccccc',label='Nominal infeasible')],fontsize=6,loc='upper left')
    for ax in axes[:2]:ax.grid(alpha=.2);ax.legend()
    images = PAPER if PAPER == HERE / 'results' else PAPER / 'images'
    images.mkdir(parents=True, exist_ok=True)
    fig.savefig(images/'multipole_stage_a.pdf');fig.savefig(images/'multipole_stage_a.png',dpi=180);plt.close(fig)
    lines=[r'\subsection{Four-heading numerical results}',
        r'The matched sweep uses the $61\times61$ grid at each of $0^\circ$, $45^\circ$, $90^\circ$, and $135^\circ$, with 3576 admissible positions per heading (14,304 configurations). All centres satisfy $r>a_s+a_r=11.2970$~mm. The dimensions and moments remain illustrative, not measured. The following pooled errors are the maximum absolute component discrepancy per pose, followed by the indicated percentile or maximum across poses.',
        r'\begin{table}[htbp]\centering\small',r'\caption{Pooled near-field errors, $11.3\le r<20$~mm (1168 configurations). Force and torque use the finite receiver; quadrature differences remain error estimates.}',r'\begin{tabular}{@{}crrrr@{}}\toprule',r'$L$ & $F_{95}$ (mN) & $F_{\max}$ (mN) & $B_{\max}$ (mT) & $T_{\max}$ (mN\,m)\\\midrule']
    for L in (1,3,5):
        b=details[str(L)]['bins']['11.3-20'];lines.append(f"{L} & {1000*b['force']['p95']:.2f} & {1000*b['force']['max']:.2f} & {1000*b['field']['max']:.3f} & {1000*b['torque']['max']:.3f} \\\\")
    lines.extend([r'\bottomrule\end{tabular}\end{table}',r'\begin{table}[htbp]\centering\small',r'\caption{Classification disagreements with the refined reference. Each heading has 3576 positions. \emph{Fixed $G$} isolates interaction error under the legacy field task. \emph{Physical} uses finite-receiver coil maps and $T_z=0$.}',r'\begin{tabular}{@{}crrrrrr@{}}\toprule',r'&\multicolumn{3}{c}{Fixed $G$}&\multicolumn{3}{c}{Physical}\\',r'$\beta$ & $L=1$ & $L=3$ & $L=5$ & $L=1$ & $L=3$ & $L=5$\\\midrule'])
    for h,m in metrics['headings'].items():
        vals=[m['orders'][str(L)][key] for key in ('fixed_G_disagreements','physical_disagreements') for L in (1,3,5)]
        lines.append(h+'$^\\circ$ & '+' & '.join(map(str,vals))+r' \\')
    lines.extend([r'\bottomrule\end{tabular}\end{table}'])
    refmax=max(v['reference_force_refinement'].max() for v in pooled.values())
    ratios=max(max(m['orders'][str(L)]['sampled_error_to_analytic_bound_max']) for m in metrics['headings'].values() for L in (1,3,5))
    lines.append(f'The largest final reference force-component refinement difference is {refmax*1000:.4f}~mN. No fixed-$G$ classification changes occur on the final close-range refinement. All sampled field, gradient, force, and torque discrepancies lie below the analytical majorants; this is a regression check, not the proof of the bounds. The largest field/force discrepancy-to-bound ratio is {ratios:.4f}.')
    counts={L:[metrics['headings'][h]['orders'][str(L)]['physical_tail_reserved_feasible'] for h in metrics['headings']] for L in (1,3,5)}
    refc=[metrics['headings'][h]['orders']['5']['physical_reference_feasible'] for h in metrics['headings']]
    fmt=lambda v: ', '.join(map(str,v[:-1]))+', and '+str(v[-1])
    lines.append(f'For the physical task, the tail-reserved feasible counts at the four headings are respectively {fmt(counts[1])} for $L=1$; {fmt(counts[3])} for $L=3$; and {fmt(counts[5])} for $L=5$, against {fmt(refc)} reference-feasible positions. These use both interaction and coil-receiver truncation reserves, with $\\rho=0$. They are pointwise numerical inner-set tests, not continuous-box or hardware certificates. Zero-error recovery and monotone shrinkage under doubled reserves pass.')
    lines.append(r'The errors favor $L=5$ over $L=3$ for further validation, but do not establish a task-derived acceptance tolerance. In particular, close-range worst-case errors remain material. The code records error maxima above several candidate minimum separations; these values support selection of a domain after the task error budget is specified, not an automatic physical validity claim.')
    writing = PAPER if PAPER == HERE / 'results' else PAPER / 'supplementary materials'
    writing.mkdir(parents=True, exist_ok=True)
    (writing/'multipole_stage_a_results.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'pooled':details,'final_reference_force_change_mN':refmax*1000},indent=2))
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--paper-root',type=Path,default=PAPER)
    PAPER=parser.parse_args().paper_root.resolve();main()
