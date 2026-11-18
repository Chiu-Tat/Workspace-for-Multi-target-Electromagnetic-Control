"""Reproducible four-heading Stage A: analytic models, reference, TFW, bounds."""
import argparse,json,time
from itertools import combinations
from pathlib import Path
import numpy as np
from multipole_stage_a import interaction,coil_map,truncation_bounds,bounding_radius
from reference_gauss import integrate,coil_reference
from test_multipole_stage_a import cfg,kw
from run_comparison import model,contains_box
from emns.workspace.fast import params_list

HERE=Path(__file__).resolve().parent

def facets(A):
    """All 3-generator normals for a full-row-rank 4-D zonotope."""
    triples=np.array(list(combinations(range(A.shape[-1]),3)))
    blocks=A.transpose(0,2,1)[:,triples,:]
    normals=np.stack([(-1)**j*np.linalg.det(np.delete(blocks,j,axis=-1)) for j in range(4)],axis=-1)
    lengths=np.linalg.norm(normals,axis=-1)
    normals/=np.maximum(lengths[...,None],1e-100)
    if np.any(np.linalg.matrix_rank(A)<4):raise ValueError('Rank-deficient map requires separate handling')
    support=cfg['current_limit_A']*np.sum(np.abs(np.einsum('nki,nij->nkj',normals,A)),axis=-1)
    return normals,support

def feasible(facet,c,reserve=None):
    normals,support=facet
    center=np.array([0.,model.B_HOLD,0.,0.])
    half=np.array([0.,0.,0.,model.F_MAX])
    if reserve is not None:half=half+reserve
    radius=np.einsum('nki,ni->nk',abs(normals),np.broadcast_to(half,c.shape))
    margin=support-abs(np.einsum('nki,ni->nk',normals,center-c))-radius
    return np.min(margin,axis=1)>=-1e-10

def summary_stats(errors):
    return {'max':np.max(errors,axis=0).tolist(),'p95':np.quantile(errors,.95,axis=0).tolist()}

def run(n):
    outdir=HERE/'results'/'multipole_stage_a';outdir.mkdir(exist_ok=True)
    xs=np.linspace(*cfg['x_limits_m'],n);ys=np.linspace(*cfg['y_limits_m'],n)
    xx,yy=np.meshgrid(xs,ys);allpoints=np.column_stack((xx.ravel(),yy.ravel()))
    radii=[bounding_radius(g['length_m'],g['diameter_m']) for g in (kw['fixed_cyl'],kw['moving_cyl'])]
    rr=np.linalg.norm(allpoints-kw['fixed_xy'],axis=1);valid=rr>sum(radii)
    points=allpoints[valid];rr=rr[valid]
    report={'scope':'Numerical Stage A, illustrative uniform axial cylinders. Pointwise masks; no continuous-box/hardware certificate.',
        'grid':n,'valid_per_heading':len(points),'domain_min_m':sum(radii),'reference_levels':[[4,16,7],[5,20,9]],
        'reference_error_scope':'Absolute difference of independent quadrature refinements; estimate only, not an upper bound.',
        'orders':[1,3,5],'headings':{},'config':cfg}
    for degrees in (0,45,90,135):
        start=time.perf_counter();beta=np.deg2rad(degrees)
        print(f'{degrees}: reference 4x16x7',flush=True)
        low,rawlow=integrate(points,beta,**kw,level=(4,16,7))
        print(f'{degrees}: reference 5x20x9',flush=True)
        ref,raw=integrate(points,beta,**kw,level=(5,20,9))
        print(f'{degrees}: models and coil maps',flush=True)
        maps={};offsets={};torques={};tmap={};timings={};bounds={}
        for L in (1,3,5):
            t=time.perf_counter();offsets[L],_,_,torques[L]=interaction(points,beta,L,**kw)
            timings[str(L)]=(time.perf_counter()-t)/len(points)
            maps[L],tmap[L]=coil_map(points,beta,L,kw['free_m'],kw['moving_cyl'],params_list,model.Z_PLANE)
            bounds[L]=truncation_bounds(rr,L,*radii,kw['fixed_m'],kw['free_m'],source_length=kw['fixed_cyl']['length_m'],receiver_length=kw['moving_cyl']['length_m'])
        refmap,reftmap=coil_reference(points,beta,kw['free_m'],kw['moving_cyl'],params_list,model.Z_PLANE)
        fixedfacets=facets(maps[1]);reference_mask=feasible(fixedfacets,ref)
        refphysical=refmap.copy();refphysical[:,0,:]=reftmap/kw['free_m']
        refc=ref.copy();refc[:,0]=raw[:,8]/kw['free_m']
        physical_refmask=feasible(facets(refphysical),refc)
        data={'points':points,'radius':rr,'valid':valid,'reference':ref,'reference_raw':raw,'refinement_change':abs(ref-low),'reference_fixed_G_mask':reference_mask,'reference_physical_mask':physical_refmask}
        metrics={'quadrature_change':summary_stats(abs(ref-low)),'quadrature_torque_change_max_Nm':float(max(abs(raw[:,8]-rawlow[:,8]))),'timing_seconds_per_pose':timings,'orders':{},'bins':{}}
        # Critical refinement: all close poses plus any coarse/fine boundary changes.
        critical=(rr<.02)|(feasible(fixedfacets,low)!=reference_mask)
        high,rawhi=integrate(points[critical],beta,**kw,level=(7,28,12))
        metrics['critical_refinement']={'count':int(critical.sum()),'level':[7,28,12],'change':summary_stats(abs(high-ref[critical])),'fixed_G_classification_changes':int(np.sum(feasible((fixedfacets[0][critical],fixedfacets[1][critical]),high)!=reference_mask[critical]))}
        # Use highest evaluated reference on critical poses in all final comparisons.
        reference_before=ref.copy();ref[critical]=high;raw[critical]=rawhi
        estimate=abs(reference_before-low);estimate[critical]=abs(high-reference_before[critical])
        reference_mask=feasible(fixedfacets,ref)
        refc=ref.copy();refc[:,0]=raw[:,8]/kw['free_m']
        physical_refmask=feasible(facets(refphysical),refc)
        data.update(reference=ref,reference_raw=raw,refinement_change=estimate,reference_fixed_G_mask=reference_mask,reference_physical_mask=physical_refmask)
        for L in (1,3,5):
            fixedmask=feasible(fixedfacets,offsets[L])
            A=maps[L].copy();A[:,0,:]=tmap[L]/kw['free_m']
            c=offsets[L].copy();c[:,0]=torques[L]/kw['free_m']
            phyfac=facets(A);physical_mask=feasible(phyfac,c)
            # Independent orientation-uniform tail bounds, plus coil-target extent tails.
            bb=bounds[L];reserve=np.column_stack((bb['torque']/kw['free_m'],bb['field'],bb['force'],bb['force']))
            for par in params_list:
                d=np.linalg.norm(np.column_stack((points,np.full(len(points),model.Z_PLANE)))-par[3:],axis=1)
                cb=truncation_bounds(d,L,0.,radii[1],np.linalg.norm(par[:3]),kw['free_m'])
                reserve+=cfg['current_limit_A']*np.column_stack((cb['torque']/kw['free_m'],cb['field'],cb['force'],cb['force']))
            robust=feasible(phyfac,c,reserve)
            assert np.all(~robust|physical_mask)
            assert np.array_equal(feasible(phyfac,c,np.zeros_like(c)),physical_mask)
            assert np.all(~feasible(phyfac,c,2*reserve)|robust)
            error=abs(offsets[L]-ref)
            coverage=np.column_stack((bb['field'],bb['field'],bb['force'],bb['force']))
            record={'fixed_G_feasible':int(fixedmask.sum()),'fixed_G_reference_feasible':int(reference_mask.sum()),
              'fixed_G_disagreements':int(np.sum(fixedmask!=reference_mask)),
              'fixed_G_model_only':int(np.sum(fixedmask&~reference_mask)),'fixed_G_reference_only':int(np.sum(~fixedmask&reference_mask)),
              'physical_feasible':int(physical_mask.sum()),'physical_reference_feasible':int(physical_refmask.sum()),
              'physical_disagreements':int(np.sum(physical_mask!=physical_refmask)),
              'physical_tail_reserved_feasible':int(robust.sum()),'errors':summary_stats(error),
              'torque_error_max_Nm':float(max(abs(torques[L]-raw[:,8]))),
              'sampled_error_to_analytic_bound_max':np.max(error/coverage,axis=0).tolist(),
              'sampled_torque_error_to_bound_max':float(max(abs(torques[L]-raw[:,8])/bb['torque']))}
            metrics['orders'][str(L)]=record
            data.update({f'L{L}_offset':offsets[L],f'L{L}_torque':torques[L],f'L{L}_fixed_G_mask':fixedmask,f'L{L}_physical_mask':physical_mask,f'L{L}_reserved_mask':robust,f'L{L}_bounds':coverage,f'L{L}_physical_reserve':reserve})
            # Verify facet result against the existing LP on boundary and spread poses.
            indices=np.unique(np.r_[np.linspace(0,len(points)-1,8,dtype=int),np.flatnonzero(fixedmask!=reference_mask)[:8]])
            for i in indices:
                assert fixedmask[i]==contains_box(maps[1][i],np.array([0.,model.B_HOLD,0.,0.])-offsets[L][i],np.array([0.,0.,0.,model.F_MAX]),cfg['current_limit_A'])
        for lo,hi in ((sum(radii),.02),(.02,.03),(.03,.05),(.05,.1)):
            sel=(rr>=lo)&(rr<hi)
            if not np.any(sel):continue
            metrics['bins'][f'{lo*1000:.2f}-{hi*1000:.0f}mm']={'count':int(sel.sum()),'refinement_change':summary_stats(estimate[sel]),'orders':{str(L):summary_stats(abs(offsets[L][sel]-ref[sel])) for L in (1,3,5)}}
        metrics['elapsed_seconds']=time.perf_counter()-start
        report['headings'][str(degrees)]=metrics
        np.savez_compressed(outdir/f'heading_{degrees}.npz',**data)
        (outdir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        print(f'{degrees}: complete in {metrics["elapsed_seconds"]:.1f}s; disagreements '+str({L:metrics['orders'][str(L)]['fixed_G_disagreements'] for L in (1,3,5)}),flush=True)
    return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--grid',type=int,default=61);args=ap.parse_args();run(args.grid)
