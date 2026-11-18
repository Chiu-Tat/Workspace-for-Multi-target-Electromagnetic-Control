"""Case Study 1 figure (Figure 14), metrics and provenance from the sharp 3-D TFWs.

Run from the repository root (after run_tfw3d_fine and presentation --only plans):
  .venv/bin/python -B -m revised_case_studies.cs1_cylinder_interaction.publication
  .venv/bin/python -B -m revised_case_studies.cs1_cylinder_interaction.publication --render

Every TFW panel is the sharp 3-D (x, y, beta) TFW (corrected derivative,
0.781 mm x 5.625 deg) or a section of it.  The planned path is shared with the
slide deck.  Motion is an explicitly illustrative simulation, stopped before
the conservative non-overlap boundary.
"""
from __future__ import annotations

import os
for _thread_var in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_thread_var] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import shutil
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Polygon, FancyBboxPatch, Rectangle, Ellipse
from matplotlib.transforms import Affine2D
import numpy as np
from scipy.optimize import linprog

from cases.case_6_interaction_obstacle_avoidance import workspace_hybrid as W
from emns.numerics.interval import interval
from emns.workspace import affine as AM, fast as FM
from revised_case_studies.cs1_cylinder_interaction import common as C
from revised_case_studies.cs1_cylinder_interaction.closed_loop import (
    DT_CTRL, DT_PLANT, DRAG_C, TAU_ROT, K_P, K_H, MAX_HEADING_CORR,
    control_command, plant_step)
from revised_case_studies.cs1_cylinder_interaction.partB_paths import retime
from revised_case_studies.shared.allocator import QPAllocator, row_weights
from analysis.sections_67.common import CLASS_COLORS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'revised_case_studies/results/cs1/publication'
INNER, OUTER, UNKNOWN = (CLASS_COLORS[k] for k in ('inside','outside','undetermined'))
EXCLUDED = '#ffffff'
BLUE, ORANGE, INK = '#215f98', '#bd5c31', '#252a30'
PURPLE = '#755a9c'
MOTION_SERIES = [('detour', BLUE, 'Planned'), ('crossing', ORANGE, 'Crossing'),
                 ('passage', PURPLE, 'Passage')]
LIMIT = .05
EXCL = .003  # Numerical singularity cutout, not physical cylinder clearance.
SIMULATION_CLEARANCE = .5 * math.hypot(.009, .005) + .5 * math.hypot(.011, .0055)
SCALES = np.array([.02, .02, .1, .1])
HEADINGS = [0, 45, 90, 135]
STEMS = ['cs1_compact']


def source_hash():
    paths = [Path(W.__file__), Path(AM.__file__), Path(FM.__file__), Path(__file__)]
    return hashlib.sha256(b''.join(p.read_bytes() for p in paths)).hexdigest()


def classifier_hash():
    return hashlib.sha256(b''.join(Path(p.__file__).read_bytes() for p in (W,AM,FM))).hexdigest()


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def matrix(x, y, beta, layout='single'):
    return C.body_matrix(float(x), float(y), float(beta), layout)


def point_test(x, y, beta, layout='single', thrust=None):
    a = matrix(x, y, beta, layout)
    G, c = a[:, :10], a[:, 10]
    fs = [-.1, .1] if thrust is None else [float(thrust)]
    feasible = True
    for f in fs:
        h = np.array([0, .02, 0, f])
        r = linprog(np.zeros(10), A_eq=G / SCALES[:, None],
                    b_eq=(h - c) / SCALES, bounds=[(-15, 15)] * 10, method='highs')
        if r.status == 2:
            feasible = False
        elif r.status != 0:
            return 'unresolved'
    return 'inside' if feasible else 'outside'


def sep_box(box, fixed):
    x0, x1, y0, y1 = box
    lows, highs = [], []
    for x, y, _, _ in fixed:
        lows.append(math.hypot(max(x0-x, 0, x-x1), max(y0-y, 0, y-y1)))
        highs.append(max(math.hypot(px-x, py-y) for px in (x0,x1) for py in (y0,y1)))
    return min(lows), min(highs)


def compute_slice(task):
    layout, heading, stop = task
    beta = math.radians(heading)
    sig = {'source_hash': source_hash(), 'layout': layout, 'heading_deg': heading,
           'stop_m': stop, 'exclusion_m': EXCL, 'classifier_hash':classifier_hash()}
    path = OUT / f'slice_{layout}_{heading:03d}.json'
    if path.exists():
        prior = json.loads(path.read_text())
        ps=prior.get('signature',{})
        if all(ps.get(k)==sig[k] for k in ('layout','heading_deg','stop_m','exclusion_m','classifier_hash')):
            return prior
    t0 = time.monotonic()
    I, b, v, lo, hi = W._task_setup()
    edges = np.linspace(-LIMIT, LIMIT, 9)
    queue = [(x0,x1,y0,y1) for x0,x1 in zip(edges[:-1],edges[1:])
             for y0,y1 in zip(edges[:-1],edges[1:])]
    boxes = {k: [] for k in ('inside','outside','undetermined','excluded')}
    extension=None
    if path.exists():
        ps=prior['signature']
        provenance=(OUT/'PROVENANCE.md').read_text()
        backend_unchanged=ps.get('classifier_hash')==classifier_hash() or all(
            hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest() in provenance for mod in (W,AM,FM))
        if (backend_unchanged and ps['layout']==layout and ps['heading_deg']==heading
            and ps['stop_m']==stop and ps['exclusion_m']>EXCL):
            archive=OUT/'archive_11p3mm'
            archive.mkdir(exist_ok=True)
            if not (archive/path.name).exists(): shutil.copy2(path,archive/path.name)
            queue=[tuple(q) for q in prior['boxes']['excluded']]
            for kind in ('inside','outside','undetermined'):
                boxes[kind]=list(prior['boxes'][kind])
            extension={'previous_exclusion_m':ps['exclusion_m'],
                       'previous_source_hash':ps['source_hash'],
                       'recomputed_boxes':len(queue),
                       'preserved_boxes':{k:len(boxes[k]) for k in ('inside','outside','undetermined')}}
    reasons = {}
    while queue:
        box = queue.pop()
        x0,x1,y0,y1 = box
        near, far = sep_box(box, W.LAYOUTS[layout])
        last = max(x1-x0,y1-y0) <= stop * (1+1e-9)
        if far < EXCL or (near < EXCL and last):
            boxes['excluded'].append(box)
            continue
        if near < EXCL:
            verdict, reason = 'undetermined', 'exclusion_edge'
        else:
            q = [[interval[x0,x1]], [interval[y0,y1]], [interval[beta,beta]]]
            try:
                verdict, reason = W.classify_box_hybrid(q,b,I,v,lo,hi,fixed=W.LAYOUTS[layout])
            except (ValueError,ZeroDivisionError,OverflowError,FloatingPointError):
                verdict, reason = 'undetermined', 'evaluation_error'
        if verdict != 'undetermined' or last:
            boxes[verdict].append(box)
            reasons[reason] = reasons.get(reason,0)+1
        else:
            xm,ym = (x0+x1)/2,(y0+y1)/2
            queue.extend([(x0,xm,y0,ym),(xm,x1,y0,ym),(x0,xm,ym,y1),(xm,x1,ym,y1)])
    area = {k:sum((q[1]-q[0])*(q[3]-q[2]) for q in qs)/(.1*.1) for k,qs in boxes.items()}
    result = {'signature':sig,'boxes':boxes,'area_fraction_full_square':area,
              'elapsed_s':time.monotonic()-t0,'reasons':reasons,
              'incremental_extension':extension,
              'numerical_scope':'Nominal model classifications; floating-point affine backend without outward rounding.'}
    dump(path,result)
    return result


def contains_region(data, x, y):
    if abs(x)>LIMIT or abs(y)>LIMIT:
        return 'domain_exit'
    # Conservatively prioritize excluded/unresolved labels on shared edges.
    for kind in ('excluded','outside','undetermined','inside'):
        bs = np.asarray(data['boxes'][kind])
        if bs.size and np.any((bs[:,0]<=x)&(x<=bs[:,1])&(bs[:,2]<=y)&(y<=bs[:,3])):
            return kind
    return 'undetermined'


def independent_checks(maps):
    rng = np.random.default_rng(41)
    tested = 0
    for d in maps.values():
        layout = d['signature']['layout']
        beta = math.radians(d['signature']['heading_deg'])
        for kind in ('inside','outside'):
            boxes = d['boxes'][kind]
            for idx in rng.choice(len(boxes), min(20,len(boxes)),replace=False):
                x0,x1,y0,y1 = boxes[int(idx)]
                for x,y in [((x0+x1)/2,(y0+y1)/2),(x0,y0),(x1,y1)]:
                    observed=point_test(x,y,beta,layout)
                    if observed != kind:
                        raise RuntimeError(f'Box/point contradiction: {layout}, {beta}, {kind}, {observed}, {(x,y)}')
                    tested += 1
    # Check degenerate affine maps against the corrected point map.
    errs=[]
    for beta in (0,.7,math.pi/2,2.4):
        a=W.calculate_affine_A(interval[-.02,-.02],interval[.02,.02],interval[beta,beta],return_affine=True)
        val=np.array([[e.center for e in row] for row in a])
        errs.append(float(np.max(np.abs((val-matrix(-.02,.02,beta))/SCALES[:,None]))))
    if max(errs)>1e-8:
        raise RuntimeError(f'Affine/point derivative check failed: {errs}')
    return {'point_checks':tested,'max_scaled_degenerate_map_difference':max(errs),
            'degenerate_map_scaled_tolerance':1e-8,
            'meaning':'Implementation consistency checks, not continuous verification or hardware validation.'}


def force_data(x,y,beta,layout='single'):
    fr=C.force_ranges(float(x),float(y),float(beta),layout)
    return {k: list(v) if isinstance(v,tuple) else float(v) for k,v in fr.items()}


def fresh_simulation(traj):
    """Original illustrative controller/plant, with a physical-domain abort.

    Field alignment is modeled by a first-order relaxation. Allocation uses the
    commanded heading; its LP feasibility is logged separately from actual pose.
    No physical contact or post-abort trajectory is fabricated.
    """
    alloc=QPAllocator(4)
    weights=row_weights(SCALES)
    x,y,beta=map(float,traj[0,:3])
    prev=np.zeros(10)
    records=[]
    abort=None
    nsub=int(round(DT_CTRL/DT_PLANT))
    T=float(traj[-1,3])+3
    for k in range(int(T/DT_CTRL)+1):
        t=k*DT_CTRL
        bc,fc,(xr,yr,br)=control_command(traj,t,x,y)
        a=matrix(x,y,bc)
        h=np.array([0,.02,0,fc])
        sol=alloc.solve(a[:,:10],a[:,10],h,weights,prev)
        currents=sol['currents']; prev=currents.copy()
        r={'t':t,'x':x,'y':y,'beta':beta,'xr':xr,'yr':yr,'betar':br,
           'beta_command':bc,'fz_command':fc,'currents':currents.tolist(),
           'residual':sol['slack'].tolist(),'weighted_residual':float(sol['weighted_slack']),
           'solver_status':sol['status'],'sep':math.hypot(x-C.FIXED_XY[0],y-C.FIXED_XY[1]),
           'position_error':math.hypot(x-xr,y-yr),
           'actual_pose_task_class':point_test(x,y,beta),
           'allocation_command_class':point_test(x,y,bc,thrust=fc)}
        records.append(r)
        for _ in range(nsub):
            xx,yy,bb,_,_=plant_step(x,y,beta,currents)
            # Reject steps outside the conservative geometry/model domain.
            if math.hypot(xx-C.FIXED_XY[0],yy-C.FIXED_XY[1]) < SIMULATION_CLEARANCE:
                abort={'time_s':t,'reason':'Conservative non-overlap boundary reached; simulation terminated.'}
                break
            x,y,beta=xx,yy,bb
        if abort:
            break
    active=[r for r in records if r['t']<=traj[-1,3]]
    e=np.array([r['position_error'] for r in active])
    angle=np.array([(r['beta']-r['betar']+math.pi)%(2*math.pi)-math.pi for r in active])
    summary={'active_interval_s':[0,min(records[-1]['t'],float(traj[-1,3]))],
             'reference_duration_s':float(traj[-1,3]),'recorded_duration_s':records[-1]['t'],
             'position_rmse_mm':float(np.sqrt(np.mean(e**2))*1e3),
             'position_mae_mm':float(np.mean(e)*1e3),'position_p95_mm':float(np.percentile(e,95)*1e3),
             'heading_rmse_deg':float(np.sqrt(np.mean(angle**2))*180/math.pi),
             'last_goal_error_mm':float(math.hypot(records[-1]['x']-traj[-1,0],records[-1]['y']-traj[-1,1])*1e3),
             'min_separation_mm':float(min(r['sep'] for r in records)*1e3),
             'peak_current_A':float(max(max(abs(v) for v in r['currents']) for r in records)),
             'active_saturated_fraction':float(np.mean([max(abs(v) for v in r['currents'])>14.999 for r in active])),
             'command_infeasible_samples':sum(r['allocation_command_class']=='outside' for r in records),
             'actual_pose_full_task_infeasible_samples':sum(r['actual_pose_task_class']=='outside' for r in records),
             'abort':abort}
    return {'records':records,'summary':summary,'reference':traj.tolist()}


def science_style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8.5,
                        'axes.labelsize':8,'legend.fontsize':7,'xtick.labelsize':7,
                        'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,
                        'axes.edgecolor':'#777b80','text.color':INK,'axes.labelcolor':INK,
                        'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':250})


def panel(ax,letter,title):
    ax.set_title(f'({letter}) {title}',loc='left',pad=6,fontweight='normal',fontsize=7.8)


def background(ax,data):
    for kind,color in [('undetermined',UNKNOWN),('inside',INNER),('outside',OUTER),('excluded',EXCLUDED)]:
        polys=[[(b[0]*1e3,b[2]*1e3),(b[1]*1e3,b[2]*1e3),
                (b[1]*1e3,b[3]*1e3),(b[0]*1e3,b[3]*1e3)] for b in data['boxes'][kind]]
        if polys:
            ax.add_collection(PolyCollection(polys,facecolors=color,edgecolors='none',rasterized=True))
    for x,y,beta,m in W.LAYOUTS[data['signature']['layout']]:
        ax.add_patch(Circle((x*1e3,y*1e3),EXCL*1e3,facecolor='none',edgecolor='#66707b',ls=':',lw=.4,zorder=4))
        magnet_symbol(ax,x*1e3,y*1e3,beta)
    ax.set(xlim=(-50,50),ylim=(-50,50),xlabel='$x$ (mm)',ylabel='$y$ (mm)')
    ax.set_xticks([-50,-25,0,25,50]); ax.set_yticks([-50,-25,0,25,50])
    ax.set_aspect('equal')


def class_legend(fig, y=.015):
    handles=[Patch(fc=INNER,label='TFW-in'),Patch(fc=OUTER,label='TFW-out'),
             Patch(fc=UNKNOWN,ec='#adb0b4',lw=.5,label='Unresolved'),
             Patch(fc=EXCLUDED,ec='#66707b',lw=.5,label='3 mm cutout')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,y),ncol=4,frameon=False)


def export(fig,stem):
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'{stem}.{ext}',facecolor='white',bbox_inches='tight',pad_inches=.08)
    plt.close(fig)
    print(f'Exported {stem}',flush=True)


def magnet_symbol(ax,x,y,beta,*,moving=False):
    """Smooth vector glyph; schematic and deliberately not to physical scale."""
    length,width=(16,7.2) if moving else (12,5.5)
    transform=Affine2D().rotate(beta).translate(x,y)+ax.transData
    outline=FancyBboxPatch((-length/2,-width/2),length,width,
                          boxstyle=f'round,pad=0,rounding_size={width*.18}',
                          facecolor=BLUE if moving else '#69747e',edgecolor='#29313a',
                          linewidth=.65,transform=transform,zorder=7)
    ax.add_patch(outline)
    pole=Rectangle((0,-width/2),length/2,width,facecolor='#b75c55',
                   edgecolor='none',transform=transform,zorder=8)
    pole.set_clip_path(outline); ax.add_patch(pole)
    ax.plot([-length/2+width*.22,length/2-width*.22],[width*.23,width*.23],
            color='white',lw=.5,alpha=.65,transform=transform,zorder=9)
    cap=Ellipse((length/2-width*.09,0),width*.24,width*.8,facecolor='#d38a81',
                edgecolor='#29313a',lw=.35,transform=transform,zorder=9)
    ax.add_patch(cap)
    if moving:
        u=np.array([math.cos(beta),math.sin(beta)])
        ax.annotate('',xy=np.array([x,y])+24*u,xytext=np.array([x,y])+10*u,
                    arrowprops={'arrowstyle':'-|>','color':INK,'lw':.85,'mutation_scale':8},zorder=10)


def fig_compact(maps,analysis,runs,details):
    """Superseded twelve-panel grid; retained for reference."""
    fig=plt.figure(figsize=(7.5,8.65))
    gs=fig.add_gridspec(4,3,left=.075,right=.985,bottom=.115,top=.97,
                        hspace=.58,wspace=.42,height_ratios=[1,1,.94,.78])
    panels=[('single',0,'a',r'1 fixed, $\beta=0^\circ$'),
            ('single',45,'b',r'1 fixed, $\beta=45^\circ$'),
            ('single',90,'c',r'1 fixed, $\beta=90^\circ$'),
            ('single',135,'d',r'1 fixed, $\beta=135^\circ$'),
            ('gate',90,'e',r'2 fixed, $\beta=90^\circ$'),
            ('triangle',90,'f',r'3 fixed, $\beta=90^\circ$')]
    for idx,(layout,heading,letter,title) in enumerate(panels):
        ax=fig.add_subplot(gs[idx//3,idx%3]); data=maps[(layout,heading)]
        background(ax,data); panel(ax,letter,title)
        if layout=='single': magnet_symbol(ax,-30,-25,math.radians(heading),moving=True)
        if idx%3: ax.set_ylabel('')
        ax.set_xticks([-50,0,50]); ax.set_yticks([-50,0,50])
        frac=data['area_fraction_full_square']
        ax.text(.025,.025,f'In {frac["inside"]*100:.1f}% / Out {frac["outside"]*100:.1f}%',
                transform=ax.transAxes,fontsize=6.1,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':1.3})
        if letter=='c':
            for k,(x,y) in enumerate(details['probes_m']):
                ax.plot(x*1e3,y*1e3,'o',ms=2.3,mfc='white',mec=INK,zorder=10)
                offset=(4,2) if k==0 else (-14,-7) if k==1 else (-14,4)
                ax.annotate(f'P{k+1}',(x*1e3,y*1e3),xytext=offset,
                            textcoords='offset points',fontsize=5.7,zorder=11)
    ax=fig.add_subplot(gs[2,0])
    classes=np.array([[2 if v=='inside' else 0 if v=='outside' else 1
                       for v in row] for row in details['probe_classes']])
    headings=details['probe_headings_deg']
    step=headings[1]-headings[0]
    ax.imshow(classes,cmap=ListedColormap([OUTER,UNKNOWN,INNER]),vmin=0,vmax=2,
              aspect='auto',interpolation='nearest',
              extent=(headings[0]-step/2,headings[-1]+step/2,2.5,-.5))
    for y in [.5,1.5]: ax.axhline(y,color='white',lw=3)
    for k,row in enumerate(details['probe_classes']):
        ax.text(348,k,f'{sum(v=="inside" for v in row)/len(row)*100:.0f}%',
                ha='right',va='center',fontsize=6.5,color=INK,
                bbox={'fc':'white','ec':'none','alpha':.8,'pad':1.3})
    ax.set(xlabel=r'Heading $\beta$ (deg)',xlim=(-3,360),
           xticks=[0,90,180,270,360],yticks=[0,1,2],yticklabels=['P1','P2','P3'])
    ax.set_ylabel('Position probe'); panel(ax,'g','Heading availability')
    ax=fig.add_subplot(gs[2,1]); background(ax,maps[('single',90)])
    for name,color in [('detour',BLUE),('crossing',ORANGE)]:
        reference=np.asarray(runs[name]['reference']); rows=runs[name]['records']
        ax.plot(reference[:,0]*1e3,reference[:,1]*1e3,ls='--',color=color,
                lw=.8,zorder=5)
        ax.plot([r['x']*1e3 for r in rows],[r['y']*1e3 for r in rows],
                color=color,lw=1.15,zorder=6)
        if runs[name]['summary']['abort']:
            ax.plot(rows[-1]['x']*1e3,rows[-1]['y']*1e3,'x',
                    color=ORANGE,ms=5,mew=1,zorder=10)
    ax.plot(*np.asarray(runs['detour']['reference'])[0,:2]*1e3,'o',
            mfc='white',mec=INK,ms=3,zorder=10)
    ax.plot(*np.asarray(runs['detour']['reference'])[-1,:2]*1e3,'s',
            mfc='white',mec=INK,ms=3,zorder=10)
    ax.annotate('Detour',(-11,25),xytext=(-46,30),fontsize=6,color=BLUE,
                arrowprops={'arrowstyle':'-','color':BLUE,'lw':.5},zorder=10)
    ax.text(13,35,'Crossing',fontsize=6,color=ORANGE,zorder=10)
    ax.set_xticks([-50,0,50]); ax.set_yticks([-50,0,50]); ax.set_ylabel('')
    panel(ax,'h','Paths and simulated motion')
    ax=fig.add_subplot(gs[2,2]); approach=analysis['approach']
    distance=np.array([r['separation_mm'] for r in approach])
    interaction=np.array([r['fz_interaction'] for r in approach])
    lower=np.array([r['fz_range'][0] for r in approach])-interaction
    upper=np.array([r['fz_range'][1] for r in approach])-interaction
    ax.fill_between(distance,lower,upper,color=INNER,alpha=.85,label='Available coil force')
    ax.plot(distance,.1-interaction,color=INK,lw=1.1,label=r'Required: $f_z=+0.1$ N')
    ax.plot(distance,-.1-interaction,color=ORANGE,lw=1.1,ls='--',label=r'Required: $f_z=-0.1$ N')
    bad=np.array([r['point_class']=='outside' for r in approach])
    if bad.any(): ax.axvspan(distance[bad].min(),distance[bad].max(),color=OUTER,alpha=.3,zorder=0)
    ax.axhline(0,color='#a3a5a8',lw=.45)
    ax.set(xlabel='Separation (mm)',ylabel='Coil force (N)',
           xlim=(distance.max(),distance.min()),ylim=(-.3,.4))
    ax.set_xticks([35,25,15]); ax.set_yticks([-.2,0,.2,.4])
    panel(ax,'i','Required and available force')
    ax.legend(loc='upper left',fontsize=6.1,frameon=False,handlelength=1.7)
    plots=[('j','Position tracking error','position_error',1e3,'Error (mm)'),
           ('k','Maximum coil current','peak',1,'Current (A)'),
           ('l','Allocation residual','weighted_residual',1,'Scaled residual norm')]
    for col,(letter,title,key,scale,ylabel) in enumerate(plots):
        ax=fig.add_subplot(gs[3,col])
        for name,color in [('detour',BLUE),('crossing',ORANGE)]:
            rows=runs[name]['records']; tt=np.array([r['t'] for r in rows])
            values=np.array([max(abs(v) for v in r['currents']) if key=='peak' else r[key]
                             for r in rows])*scale
            ax.plot(tt,values,color=color,lw=1.1,label=name.capitalize())
            bad=np.array([r['allocation_command_class']=='outside' for r in rows])
            if bad.any(): ax.plot(tt[bad],values[bad],'D',ms=2.5,mfc=INK,mec=INK,zorder=8)
        abort=runs['crossing']['summary']['abort']
        if abort: ax.axvline(abort['time_s'],color=ORANGE,ls=':',lw=.8)
        if key=='peak': ax.axhline(15,color=INK,ls='--',lw=.8)
        ax.set(xlabel='Time (s)',ylabel=ylabel,xlim=(0,27),xticks=[0,10,20])
        panel(ax,letter,title)
        if col==0:
            ax.set_ylim(0,3.4); ax.legend(loc='upper right',fontsize=6.1,frameon=False)
        if key=='peak': ax.set_ylim(0,16.5); ax.set_yticks([0,5,10,15])
        if key=='weighted_residual': ax.set_ylim(-.03,1.4); ax.set_yticks([0,.5,1])
    class_legend(fig,.025)
    fig.text(.5,.006,'SIMULATION PLACEHOLDER: replace motion, tracking, current, and residual traces with hardware data.',
             ha='center',fontsize=6.5,color=ORANGE)
    export(fig,STEMS[0])


def row_header(fig,ax,text,note=None,pad=.045):
    box=ax.get_position()
    fig.text(.012,box.y1+pad,text,fontsize=8.6,fontweight='bold',color=INK,va='bottom')
    if note:
        fig.text(.988,box.y1+pad,note,fontsize=6.6,color=ORANGE,style='italic',
                 va='bottom',ha='right')


def compact_map(ax,data,title,*,ylabel=True,moving=None):
    background(ax,data)
    ax.set_title(title,loc='left',pad=3,fontsize=7.4)
    ax.set_xticks([-40,0,40]); ax.set_yticks([-40,0,40])
    ax.set_xlabel('$x$ (mm)',labelpad=1)
    if ylabel: ax.set_ylabel('$y$ (mm)',labelpad=0)
    else: ax.set_ylabel(''); ax.set_yticklabels([])
    if moving is not None: magnet_symbol(ax,-30,-30,math.radians(moving),moving=True)


def heading_rings(ax,details):
    headings=np.radians(details['probe_headings_deg'])
    step=headings[1]-headings[0]
    shares=[]
    for k,row in enumerate(details['probe_classes']):
        r0=1.25-k*.42
        colors=[INNER if v=='inside' else OUTER if v=='outside' else UNKNOWN for v in row]
        ax.bar(headings,.34,width=step,bottom=r0,color=colors,edgecolor='none',align='center')
        share=sum(v=='inside' for v in row)/len(row)*100
        ax.text(math.radians(180),r0+.17,f'P{k+1}',ha='center',va='center',fontsize=5.6,
                color=INK,bbox={'fc':'white','ec':'none','pad':.5,'alpha':.85})
        shares.append(f'P{k+1} {share:.0f}%')
    for deg in HEADINGS:
        ax.plot([math.radians(deg)]*2,[.35,1.62],color='white',lw=.5)
    ax.set_ylim(0,1.62); ax.set_yticks([])
    ax.set_xticks(np.radians([0,90,180,270]))
    ax.set_xticklabels(['0°','90°','180°','270°'],fontsize=6)
    ax.tick_params(axis='x',pad=-1)
    ax.spines['polar'].set_visible(False); ax.grid(False)
    ax.set_title('(e) Heading tests at P1–P3',loc='center',pad=11,fontsize=7.4)
    ax.text(.5,-.2,'   '.join(shares),transform=ax.transAxes,ha='center',va='top',fontsize=6)


def area_summary(ax,maps):
    rows=[(('single',0),r'1 fixed, $0^\circ$'),(('single',45),r'1 fixed, $45^\circ$'),
          (('single',90),r'1 fixed, $90^\circ$'),(('single',135),r'1 fixed, $135^\circ$'),
          (('gate',90),r'2 fixed, $90^\circ$'),(('triangle',90),r'3 fixed, $90^\circ$')]
    y=np.array([0,1,2,3,4.6,5.6])
    for yy,(key,_) in zip(y,rows):
        frac=maps[key]['area_fraction_full_square']
        inside,unres,out=(frac[k]*100 for k in ('inside','undetermined','outside'))
        left=0
        for width,color in [(inside,INNER),(unres,UNKNOWN),(out,OUTER)]:
            ax.barh(yy,width,left=left,height=.62,color=color,edgecolor='none')
            left+=width
        ax.text(inside-1,yy,f'{inside:.1f}%',ha='right',va='center',fontsize=6,color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r[1] for r in rows],fontsize=6.4)
    ax.invert_yaxis(); ax.tick_params(axis='y',length=0)
    ax.set_xlim(0,100); ax.set_xticks([0,25,50,75,100])
    ax.set_xlabel('Share of the 100 × 100 mm square (%)',labelpad=1)
    ax.axhline(3.8,color='#c4c7ca',lw=.5)
    ax.spines['left'].set_visible(False)
    ax.set_title('(h) Area fractions of the full square',loc='left',pad=3,fontsize=7.4)


def paths_panel(ax,maps,runs):
    background(ax,maps[('single',90)])
    for name,color,label in MOTION_SERIES:
        reference=np.asarray(runs[name]['reference']); rows=runs[name]['records']
        ax.plot(reference[:,0]*1e3,reference[:,1]*1e3,ls='--',color=color,lw=.7,zorder=5)
        ax.plot([r['x']*1e3 for r in rows],[r['y']*1e3 for r in rows],color=color,lw=1.2,zorder=6)
        bad=[r for r in rows if r['actual_pose_task_class']=='outside']
        if bad:
            ax.plot([r['x']*1e3 for r in bad],[r['y']*1e3 for r in bad],
                    '^',ms=3,mfc='white',mec=color,mew=.65,zorder=9)
        if runs[name]['summary']['abort']:
            ax.plot(rows[-1]['x']*1e3,rows[-1]['y']*1e3,'x',color=ORANGE,ms=4.5,mew=1.1,zorder=10)
    ref=np.asarray(runs['detour']['reference'])
    ax.plot(*ref[0,:2]*1e3,'o',mfc='white',mec=INK,ms=3,zorder=10)
    ax.plot(*ref[-1,:2]*1e3,'s',mfc='white',mec=INK,ms=3,zorder=10)
    ax.text(-44,8,'Planned',fontsize=6,color=BLUE,zorder=10)
    ax.annotate('Passage',xy=(8,22),xytext=(-44,28),fontsize=5.7,color=PURPLE,
                arrowprops={'arrowstyle':'-','color':PURPLE,'lw':.45},zorder=10)
    ax.annotate('Crossing',xy=(10,-30),xytext=(15,-43),fontsize=5.7,color=ORANGE,
                arrowprops={'arrowstyle':'-','color':ORANGE,'lw':.45},zorder=10)
    ax.set_title(r'(i) Paths on $90^\circ$ slice',loc='left',pad=3,fontsize=7.4)
    ax.set_xticks([-40,0,40]); ax.set_yticks([-40,0,40])
    ax.set_xlabel('$x$ (mm)',labelpad=1); ax.set_ylabel('$y$ (mm)',labelpad=0)


def force_panel(ax,analysis,*,compact=False):
    approach=analysis['approach']
    distance=np.array([r['separation_mm'] for r in approach])
    interaction=np.array([r['fz_interaction'] for r in approach])
    lower=np.array([r['fz_range'][0] for r in approach])-interaction
    upper=np.array([r['fz_range'][1] for r in approach])-interaction
    bad=np.array([r['point_class']=='outside' for r in approach])
    if bad.any():
        ax.axvspan(distance[bad].max(),distance.min(),color=OUTER,alpha=.55,lw=0,zorder=0)
        ax.text(distance[bad].mean(),-.27,'infeasible',rotation=90,ha='center',va='bottom',
                fontsize=5.6,color='#9b3b3b')
    ax.fill_between(distance,lower,upper,color=INNER,alpha=.9,lw=0)
    ax.plot(distance,.1-interaction,color=INK,lw=1.1)
    ax.plot(distance,-.1-interaction,color=INK,lw=1.1,ls='--')
    ax.axhline(0,color='#a3a5a8',lw=.45)
    ax.text(34,-.25 if compact else -.205,'available coil force',fontsize=5.8,color='#2f6b3e')
    ax.text(34,.125,r'required for $+0.1$ N',fontsize=5.8,color=INK,va='bottom')
    ax.text(34,-.085,r'required for $-0.1$ N',fontsize=5.8,color=INK,va='bottom')
    ax.set(xlim=(distance.max(),distance.min()),ylim=(-.3,.4))
    ax.set_xticks([35,30,25,20,15]); ax.set_yticks([-.2,0,.2,.4])
    ax.set_xlabel('Separation (mm), approaching →',fontsize=6.3,labelpad=1)
    ax.set_ylabel('Coil force (N)',fontsize=6.3,labelpad=1)
    ax.tick_params(labelsize=6)
    ax.set_title(r'(j) Interaction compensation: $x=10$ mm',loc='left',pad=3,fontsize=7)


def passage_window(run):
    """Time window around the full-task-infeasible part of the passage."""
    bad=[r['t'] for r in run['records'] if r['actual_pose_task_class']=='outside']
    if not bad:
        sep=min(run['records'],key=lambda r:r['sep']); bad=[sep['t']]
    return (math.floor(min(bad))-1.0, math.ceil(max(bad))+1.5)


def passage_capacity(run):
    """LP force capacities at the logged allocation pose; never mix headings."""
    rows=[]
    t0,t1=passage_window(run)
    for r in run['records']:
        if not t0 <= r['t'] <= t1:
            continue
        fr=force_data(r['x'],r['y'],r['beta_command'])
        lo,hi=fr['fz_range']
        task_class=point_test(r['x'],r['y'],r['beta_command'])
        command_class=point_test(r['x'],r['y'],r['beta_command'],thrust=r['fz_command'])
        if command_class != r['allocation_command_class']:
            raise RuntimeError('Passage command LP disagrees with the saved simulation.')
        if not (math.isfinite(lo) and math.isfinite(hi) and lo-1e-8 <= r['fz_command'] <= hi+1e-8):
            raise RuntimeError('Passage command lies outside its computed force interval.')
        if (lo <= -.1+1e-8 and hi >= .1-1e-8) != (task_class=='inside'):
            raise RuntimeError('Force interval and full-task LP classification disagree.')
        rows.append({'t':r['t'],'x':r['x'],'y':r['y'],'beta_command':r['beta_command'],
                     'fz_range':[float(lo),float(hi)],'fz_command':r['fz_command'],
                     'fz_interaction':fr['fz_interaction'],
                     'allocation_pose_task_class':task_class,
                     'actual_pose_task_class':r['actual_pose_task_class'],
                     'allocation_command_class':command_class})
    data={'scope':'Total longitudinal force at logged (x,y,beta_command); total field and zero lateral force enforced.',
          'reference':'Auxiliary x=8 mm passage, 5 mm/s; different geometry from primary crossing.',
          'checks':'Command LP matches saved records; commands lie within force intervals; full-task LP matches interval containment.',
          'simulation_sha256':hashlib.sha256((OUT/'auxiliary_weaker_task_crossing.json').read_bytes()).hexdigest(),
          'classifier_hash':classifier_hash(),'window_s':[t0,t1],'rows':rows}
    dump(OUT/'passage_force_capacity.json',data)
    return data


def passage_force_panel(ax,data):
    rows=data['rows']; t=np.array([r['t'] for r in rows])
    lo=np.array([r['fz_range'][0] for r in rows]); hi=np.array([r['fz_range'][1] for r in rows])
    command=np.array([r['fz_command'] for r in rows])
    bad=np.array([r['allocation_pose_task_class']=='outside' for r in rows])
    for tt in t[bad]: ax.axvspan(tt-.05,tt+.05,color=OUTER,alpha=.55,lw=0,zorder=0)
    ax.fill_between(t,lo,hi,color=INNER,alpha=.9,lw=0,zorder=1)
    for value in (-.1,.1): ax.axhline(value,color=INK,ls='--',lw=.65)
    ax.plot(t,command,color=PURPLE,lw=1.1,zorder=4)
    ax.plot(t[bad],command[bad],'^',mfc='white',mec=PURPLE,ms=3,mew=.65,zorder=5)
    t0,t1=data.get('window_s',(7.5,11.5))
    ax.text(t0+.1,.112,'full task: ±0.1 N',fontsize=5.8,va='bottom')
    ax.text(t0+.62*(t1-t0),.035,'motion command',color=PURPLE,fontsize=5.8,va='bottom')
    ax.text(t0+.1,-.33,'available total force',color='#2f6b3e',fontsize=5.8)
    ax.set(xlim=(t0,t1),ylim=(-.48,.2),yticks=[-.4,-.2,0,.1])
    ax.set_xticks(range(int(math.ceil(t0)),int(t1)+1))
    ax.set_xlabel('Passage time (s)',fontsize=6.3,labelpad=1)
    ax.set_ylabel('Total force (N)',fontsize=6.3,labelpad=1)
    ax.tick_params(labelsize=6)
    ax.set_title('(k) Full-task loss with an attainable command',loc='left',pad=3,fontsize=7)


def motion_summary(ax,runs):
    ax.axis('off')
    ax.text(0,1,'LP-infeasible samples',fontsize=6.2,fontweight='bold',va='top')
    ax.text(0,.77,'Actual full task / command',fontsize=5.6,va='top')
    for y,(name,color,label) in zip([.53,.3,.07],MOTION_SERIES):
        s=runs[name]['summary']
        ax.text(0,y,label,fontsize=6,color=color,va='top')
        ax.text(1,y,f'{s["actual_pose_full_task_infeasible_samples"]} / {s["command_infeasible_samples"]}',
                fontsize=6,ha='right',va='top')


def trace_stack(fig,cell,runs):
    sub=cell.subgridspec(3,1,hspace=.18)
    specs=[('position_error',1e3,'Error\n(mm)',(0,3.2),[0,1,2,3]),
           ('peak',1,'Max |I|\n(A)',(0,17),[0,15]),
           ('weighted_residual',1,'Residual\n(scaled)',(-.05,1.35),[0,1])]
    abort=runs['crossing']['summary']['abort']
    axes=[]
    for k,(key,scale,label,ylim,yticks) in enumerate(specs):
        ax=fig.add_subplot(sub[k],sharex=axes[0] if axes else None); axes.append(ax)
        for name,color,series_name in MOTION_SERIES:
            rows=runs[name]['records']; tt=np.array([r['t'] for r in rows])
            values=np.array([max(abs(v) for v in r['currents']) if key=='peak' else r[key]
                             for r in rows])*scale
            ax.plot(tt,values,color=color,lw=1.0)
            bad=np.array([r['allocation_command_class']=='outside' for r in rows])
            if bad.any(): ax.plot(tt[bad],values[bad],'D',ms=2.2,mfc=INK,mec=INK,zorder=8)
            if key=='peak':
                pose_bad=np.array([r['actual_pose_task_class']=='outside' for r in rows])
                if pose_bad.any(): ax.plot(tt[pose_bad],values[pose_bad],'^',ms=3,
                                          mfc='white',mec=color,mew=.65,zorder=7)
        if abort: ax.axvline(abort['time_s'],color=ORANGE,ls=':',lw=.8)
        if key=='peak': ax.axhline(15,color=INK,ls='--',lw=.6)
        ax.set_ylim(*ylim); ax.set_yticks(yticks)
        ax.set_ylabel(label,fontsize=6.2,labelpad=2,linespacing=.95)
        ax.tick_params(labelsize=6)
        if k<2: ax.tick_params(labelbottom=False)
    axes[0].set_xlim(0,27); axes[0].set_xticks([0,5,10,15,20,25])
    axes[-1].set_xlabel('Time (s)',labelpad=1)
    if abort:
        axes[2].text(abort['time_s']+.6,.55,'geometry stop',fontsize=5.6,color=ORANGE)
    axes[0].set_title('(l) Simulated motion diagnostics',loc='left',pad=3,fontsize=7.4)
    for y,(_,color,label) in zip([2.8,2.2,1.6],MOTION_SERIES):
        axes[0].text(26.5,y,label,fontsize=6,color=color,ha='right')
    return axes


def fig_redesign(maps,analysis,runs,details,stem='cs1_redesign'):
    """Three rows aligned with the study questions on a common five-column map grid."""
    runs=dict(runs)
    runs['passage']=json.loads((OUT/'auxiliary_weaker_task_crossing.json').read_text())
    capacity=passage_capacity(runs['passage'])
    fig=plt.figure(figsize=(7.16,7.05))
    outer=fig.add_gridspec(3,5,left=.075,right=.985,bottom=.06,top=.9,
                           hspace=.72,wspace=.2,height_ratios=[1,1,1.65])
    first=None
    for col,deg in enumerate(HEADINGS):
        ax=fig.add_subplot(outer[0,col]); first=first or ax
        compact_map(ax,maps[('single',deg)],rf'({"abcd"[col]}) $\beta={deg}^\circ$',ylabel=col==0,moving=deg)
        if deg==90:
            for k,(x,y) in enumerate(details['probes_m']):
                ax.plot(x*1e3,y*1e3,'o',ms=2.2,mfc='white',mec=INK,zorder=10)
                offset=(3,2) if k==0 else (-11,-6) if k==1 else (-11,3)
                ax.annotate(f'P{k+1}',(x*1e3,y*1e3),xytext=offset,textcoords='offset points',
                            fontsize=5.5,zorder=11)
    ax=fig.add_subplot(outer[0,4],projection='polar'); heading_rings(ax,details)
    row_header(fig,first,'A  Orientation dependence: one fixed magnet')
    ax=fig.add_subplot(outer[1,0]); second=ax
    compact_map(ax,maps[('gate',90)],r'(f) 2 fixed: gate')
    ax=fig.add_subplot(outer[1,1])
    compact_map(ax,maps[('triangle',90)],r'(g) 3 fixed: triangle',ylabel=False)
    inner=outer[1,2:].subgridspec(1,1)
    ax=fig.add_subplot(inner[0]); area_summary(ax,maps)
    box=ax.get_position(); ax.set_position([box.x0+.075,box.y0,box.width-.075,box.height])
    row_header(fig,second,r'B  Fixed-magnet interactions: common heading $\beta=90^\circ$')
    path_cell=outer[2,0].subgridspec(2,1,height_ratios=[1,.55],hspace=.37)
    ax=fig.add_subplot(path_cell[0]); paths_panel(ax,maps,runs); ax.set_anchor('N')
    ax=fig.add_subplot(path_cell[1]); motion_summary(ax,runs)
    box=ax.get_position(); ax.set_position([box.x0,box.y0-.014,box.width,box.height])
    force_cell=outer[2,1:3].subgridspec(2,1,hspace=.65)
    ax=fig.add_subplot(force_cell[0]); third=ax; force_panel(ax,analysis,compact=True)
    box=ax.get_position(); ax.set_position([box.x0+.035,box.y0,box.width-.035,box.height])
    ax=fig.add_subplot(force_cell[1]); passage_force_panel(ax,capacity)
    box=ax.get_position(); ax.set_position([box.x0+.035,box.y0,box.width-.035,box.height])
    axes=trace_stack(fig,outer[2,3:],runs)
    for a in axes:
        b=a.get_position(); a.set_position([b.x0+.06,b.y0,b.width-.06,b.height])
    row_header(fig,third,'C  Task loss and passage',
               note='Simulation placeholders: replace (i), (l) with hardware data')
    handles=[Patch(fc=INNER,label='TFW-in'),Patch(fc=OUTER,label='TFW-out'),
             Patch(fc=UNKNOWN,label='Unresolved'),
             Patch(fc=EXCLUDED,ec='#66707b',lw=.5,ls=':',label='3 mm numerical cutout')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.995),ncol=4,frameon=False,
               fontsize=6.4,handlelength=1.5,columnspacing=1.2)
    fig.text(.5,.008,'Dashed paths: references; solid paths: simulated.  △ full-task-infeasible sample; ◆ unattainable command; × geometry stop.',
             ha='center',fontsize=6,color=INK)
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'{stem}.{ext}',facecolor='white',bbox_inches='tight',pad_inches=.04)
    plt.close(fig)
    print(f'Exported {stem}',flush=True)


TFW3D = ROOT / 'revised_case_studies/results/cs1/tfw3d_fine'  # 0.781 mm x 5.625 deg
LAYOUT_LABEL = {'single': '1 fixed', 'gate': '2 fixed: gate', 'triangle': '3 fixed: triangle'}


def load_tfw3d(layout):
    return json.loads((TFW3D/f'tfw3d_{layout}.json').read_text())


def cut3d(d3,layout,deg):
    """Section of a 3-D (x,y,beta) TFW at one heading, in the 2-D map format."""
    beta=math.radians(deg)%(2*math.pi)+1e-9
    keys={'inside':'in_workspace','outside':'out_of_workspace','undetermined':'undetermined'}
    boxes={k:[b[:4] for b in d3[v] if b[4]<=beta<b[5]] for k,v in keys.items()}
    boxes['excluded']=[]
    area={k:sum((q[1]-q[0])*(q[3]-q[2]) for q in qs)/(.1*.1) for k,qs in boxes.items()}
    return {'boxes':boxes,'signature':{'layout':layout,'heading_deg':deg},
            'area_fraction_full_square':area}


def volume_fractions(d3):
    total=.1*.1*2*math.pi
    return {k:sum((b[1]-b[0])*(b[3]-b[2])*(b[5]-b[4]) for b in d3[v])/total
            for k,v in (('inside','in_workspace'),('outside','out_of_workspace'),
                        ('undetermined','undetermined'))}


def voxel_panel(ax,d3,layout,title):
    """Uncertified part of the 3-D TFW (unresolved or TFW-out), drawn on a
    64 x 64 x 64 display grid sampled at cell centres; the TFW is the rest."""
    nx,nb,ns=128,64,64
    inside=np.zeros((nx,nx,nb),bool)
    for b in d3['in_workspace']:
        i0,i1=(int(round((v+LIMIT)/(2*LIMIT)*nx)) for v in b[0:2])
        j0,j1=(int(round((v+LIMIT)/(2*LIMIT)*nx)) for v in b[2:4])
        k0,k1=(int(round(v/(2*math.pi)*nb)) for v in b[4:6])
        inside[i0:i1,j0:j1,k0:k1]=True
    idx=np.arange(ns)*nx//ns+nx//(2*ns)
    blocked=~inside[idx][:,idx]
    xs=np.linspace(-50,50,ns+1); bs=np.linspace(0,360,nb+1)
    X,Y,B=np.meshgrid(xs,xs,bs,indexing='ij')
    ax.voxels(X,Y,B,blocked,facecolors='#b9bfc6',linewidth=0,shade=True)
    for x,y,_,_ in W.LAYOUTS[layout]:
        ax.plot([x*1e3]*2,[y*1e3]*2,[0,360],color='#3d4550',lw=1.4,zorder=20)
    ax.set(xlim=(-50,50),ylim=(-50,50),zlim=(0,360))
    ax.set_box_aspect((1,1,1.1),zoom=1.06)
    ax.set_xticks([-40,0,40]); ax.set_yticks([-40,0,40]); ax.set_zticks([0,90,180,270,360])
    ax.set_zticklabels(['0°','','180°','','360°'])
    ax.tick_params(labelsize=5.6,pad=-2)
    ax.set_xlabel('$x$ (mm)',fontsize=6,labelpad=-9); ax.set_ylabel('$y$ (mm)',fontsize=6,labelpad=-9)
    # beta label placed by hand: zlabel padding is too coarse to clear the tick labels.
    ax.text2D(1.0,.7,r'$\beta$',transform=ax.transAxes,fontsize=6.5,ha='left',va='center')
    for pane in (ax.xaxis.pane,ax.yaxis.pane,ax.zaxis.pane):
        pane.set_facecolor(INNER); pane.set_alpha(.25); pane.set_edgecolor('#c4c7ca')
    ax.view_init(elev=22,azim=-58)
    ax.set_title(title,loc='left',pad=-2,fontsize=7.4,y=.98)


def volume_summary(ax,data):
    y=np.arange(len(data))
    for yy,(label,frac) in zip(y,data):
        inside,unres,out=(frac[k]*100 for k in ('inside','undetermined','outside'))
        left=0
        for width,color in [(inside,INNER),(unres,UNKNOWN),(out,OUTER)]:
            ax.barh(yy,width,left=left,height=.55,color=color,edgecolor='none'); left+=width
        ax.text(inside+1.5,yy,f'{inside:.1f}%',ha='left',va='center',fontsize=6,color=INK)
    ax.set_yticks(y); ax.set_yticklabels([d[0] for d in data],fontsize=6.4)
    ax.invert_yaxis(); ax.tick_params(axis='y',length=0)
    ax.set_xlim(0,100); ax.set_xticks([0,50,100])
    ax.set_xlabel(r'Share of $(x,y,\beta)$ volume (%)',labelpad=1,fontsize=6.4)
    ax.tick_params(axis='x',labelsize=6)
    ax.spines['left'].set_visible(False)
    ax.set_title('(d) Volume fractions',loc='left',pad=3,fontsize=7.4)


LAYOUT_TITLE = {'single': '1 fixed', 'gate': '2 fixed', 'triangle': '3 fixed'}
FREE_POSE = (-34.0, -36.0, math.radians(45))   # schematic free-magnet pose, mm / rad


def layout_schematic(ax,layout,letter,*,define=False):
    """Top view of the 100 x 100 mm workspace: fixed magnets and the meaning of q."""
    ax.add_patch(Rectangle((-50,-50),100,100,fc='#f7f9fa',ec='#9aa3ab',lw=.6,zorder=0))
    ax.annotate('',xy=(46,0),xytext=(-46,0),arrowprops={'arrowstyle':'-|>','lw':.5,
                'color':'#8a929b','mutation_scale':6},zorder=1)
    ax.annotate('',xy=(0,46),xytext=(0,-46),arrowprops={'arrowstyle':'-|>','lw':.5,
                'color':'#8a929b','mutation_scale':6},zorder=1)
    ax.text(47,-2,'$x$',fontsize=6.4,ha='right',va='top',color='#5c656e')
    ax.text(2,47,'$y$',fontsize=6.4,ha='left',va='top',color='#5c656e')
    for x,y,beta,m in W.LAYOUTS[layout]:
        magnet_symbol(ax,x*1e3,y*1e3,beta)
    fx,fy,fb=FREE_POSE
    magnet_symbol(ax,fx,fy,fb,moving=True)
    if define:
        ax.plot([fx,fx+19],[fy,fy],color=INK,lw=.5,ls=(0,(2,1.5)),zorder=11)
        arc=np.linspace(0,fb,20)
        ax.plot(fx+13*np.cos(arc),fy+13*np.sin(arc),color=INK,lw=.6,zorder=11)
        ax.text(fx+15.5,fy+5.5,r'$\beta$',fontsize=6.6,zorder=12)
        ax.plot([fx,fx],[fy,-50],color='#8a929b',lw=.4,ls=':',zorder=1)
        ax.plot([fx,-50],[fy,fy],color='#8a929b',lw=.4,ls=':',zorder=1)
        ax.plot(fx,fy,'o',ms=1.6,color=INK,zorder=12)
        ax.text(fx+3,fy-5,'$(x,y)$',fontsize=6,ha='left',va='top',zorder=12,
                bbox={'fc':'#f7f9fa','ec':'none','pad':.4})
    ax.set(xlim=(-50,50),ylim=(-50,50)); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_title(f'({letter}) {LAYOUT_TITLE[layout]}',loc='left',pad=3,fontsize=7.4)


def fig_compact3d(analysis,runs,details,capacity,stem='cs1_compact'):
    """Figure 14: every TFW panel is the sharp 3-D (x, y, beta) TFW or one of its sections."""
    tfw={l:load_tfw3d(l) for l in ('single','gate','triangle')}
    cuts={('single',deg):cut3d(tfw['single'],'single',deg) for deg in HEADINGS}
    for layout in ('gate','triangle'):
        cuts[(layout,90)]=cut3d(tfw[layout],layout,90)
    fig=plt.figure(figsize=(7.16,8.3))
    outer=fig.add_gridspec(4,1,left=.07,right=.985,bottom=.062,top=.95,
                           hspace=.56,height_ratios=[1.1,1,1,1.25])
    rowA=outer[0].subgridspec(1,6,wspace=.12,width_ratios=[.68,1.32]*3)
    for col,(layout,letter) in enumerate(zip(('single','gate','triangle'),'abc')):
        ax=fig.add_subplot(rowA[2*col]); layout_schematic(ax,layout,letter,define=col==0)
        box=ax.get_position(); ax.set_position([box.x0,box.y0-.012,box.width,box.height])
        if col==0: firstA=ax
        ax=fig.add_subplot(rowA[2*col+1],projection='3d')
        if col==0: first3d=ax
        vol=volume_fractions(tfw[layout])['inside']*100
        voxel_panel(ax,tfw[layout],layout,'')
        # Keep the beta axis label above the neighbouring layout panel.
        ax.set_zorder(2); ax.patch.set_visible(False)
        ax.text2D(.04,1.03,f'{vol:.1f}% TFW-in',transform=ax.transAxes,fontsize=6.4,
                  color='#2f6b3e',va='top')
    # Close the blank band under row A: its 3-D labels end well above row B.
    for ax in fig.axes:
        box=ax.get_position(); ax.set_position([box.x0,box.y0-.016,box.width,box.height])
    # The 3-D panels are taller than the layouts; keep the row header above them.
    top=max(firstA.get_position().y1+.03,first3d.get_position().y1+.01)
    fig.text(.012,top,r'A  Fixed-magnet layouts and their 3-D TFWs over $\mathbf{q}=(x,y,\beta)$',
             fontsize=8.6,fontweight='bold',color=INK,va='bottom')
    def section(ax,key,title,ylabel,moving):
        compact_map(ax,cuts[key],title,ylabel=ylabel,moving=moving)
        frac=cuts[key]['area_fraction_full_square']['inside']*100
        ax.text(.03,.03,f'In {frac:.0f}%',transform=ax.transAxes,fontsize=5.8,
                bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':1.2},zorder=12)
    rowB=outer[1].subgridspec(1,5,wspace=.2)
    first=None
    for col,deg in enumerate(HEADINGS):
        ax=fig.add_subplot(rowB[col]); first=first or ax
        section(ax,('single',deg),rf'({"defg"[col]}) $\beta={deg}^\circ$',col==0,deg)
        if deg==90:
            for k,(x,y) in enumerate(details['probes_m']):
                ax.plot(x*1e3,y*1e3,'o',ms=2.2,mfc='white',mec=INK,zorder=10)
                offset=(3,2) if k==0 else (-11,-6) if k==1 else (-11,3)
                ax.annotate(f'P{k+1}',(x*1e3,y*1e3),xytext=offset,textcoords='offset points',
                            fontsize=5.5,zorder=11)
    ax=fig.add_subplot(rowB[4],projection='polar'); heading_rings(ax,details)
    ax.set_title('(h) Point tests at P1–P3',loc='center',pad=5,fontsize=7.4)
    row_header(fig,first,'B  Sections of the one-magnet TFW (a) at four headings',pad=.022)
    rowC=outer[2].subgridspec(1,5,wspace=.2,width_ratios=[1,1,1,.75,.25])
    ax=fig.add_subplot(rowC[0]); second=ax
    section(ax,('gate',90),r'(i) 2 fixed: gate, $\beta=90^\circ$',True,None)
    ax=fig.add_subplot(rowC[1])
    section(ax,('triangle',90),r'(j) 3 fixed: triangle, $\beta=90^\circ$',False,None)
    ax=fig.add_subplot(rowC[2]); paths_panel(ax,cuts,runs)
    ax.set_title(r'(k) Paths over section (f)',loc='left',pad=3,fontsize=7.4)
    ax.set_ylabel(''); ax.set_yticklabels([])
    ax=fig.add_subplot(rowC[3]); motion_summary(ax,runs)
    box=ax.get_position(); ax.set_position([box.x0+.02,box.y0+.03,box.width,box.height*.62])
    row_header(fig,second,r'C  Sections of (b), (c) at $\beta=90^\circ$ and paths for the one-magnet layout',pad=.022)
    rowD=outer[3].subgridspec(1,3,wspace=.38,width_ratios=[1,1,1.05])
    ax=fig.add_subplot(rowD[0]); third=ax; force_panel(ax,analysis,compact=True)
    ax.set_title(r'(l) Interaction compensation: $x=10$ mm',loc='left',pad=3,fontsize=7)
    ax=fig.add_subplot(rowD[1]); passage_force_panel(ax,capacity)
    ax.set_title('(m) Full-task loss, attainable command',loc='left',pad=3,fontsize=7)
    axes=trace_stack(fig,rowD[2],runs)
    axes[0].set_title('(n) Simulated motion diagnostics',loc='left',pad=3,fontsize=7.4)
    row_header(fig,third,'D  Task loss and passage',pad=.022,
               note='Simulation placeholders: replace (k), (n) with hardware data')
    handles=[Patch(fc=INNER,label='TFW-in'),Patch(fc=OUTER,label='TFW-out'),
             Patch(fc=UNKNOWN,label='Unresolved'),
             Patch(fc='#b9bfc6',ec='#8a929b',lw=.4,label='3-D: not TFW-in'),
             Line2D([],[],color='#3d4550',lw=1.4,label='Fixed-magnet axis'),
             Patch(fc=BLUE,ec='#29313a',lw=.5,label='Free magnet')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.997),ncol=6,frameon=False,
               fontsize=6.4,handlelength=1.5,columnspacing=1.0)
    fig.text(.5,.006,'Dashed paths: references; solid paths: simulated.  △ full-task-infeasible sample; ◆ unattainable command; × geometry stop.',
             ha='center',fontsize=6,color=INK)
    for ext in ('pdf','png'):
        fig.savefig(OUT/f'{stem}.{ext}',facecolor='white',bbox_inches='tight',pad_inches=.04)
    plt.close(fig)
    print(f'Exported {stem}',flush=True)
    return tfw,cuts


def tfw3d_class(x,y,beta,layout='single'):
    """Membership of one pose in the sharp 3-D TFW (raster of its TFW-in boxes)."""
    t=C.tfw(layout); cell=t.cell_of(x,y)
    if cell is None:
        return 'domain_exit'
    i,j=cell
    return 'inside' if t.stack[t.beta_level(beta),j,i] else 'not_inside'


def analysis_data():
    planned=C.load_trajectory(C.PLANNED)   # planned in the sharp 3-D TFW (presentation.py)
    # The straight crossing shares the planned endpoints; heading 90 deg, 5 mm/s.
    y=np.linspace(planned[0,1],planned[-1,1],201)
    crossing=np.column_stack([np.full_like(y,planned[0,0]),y,np.full_like(y,math.pi/2),
                              (y-y[0])/.005])
    result={}
    for name,traj in [('detour',planned),('crossing',crossing)]:
        rows=[]
        for x,y,b,t in traj:
            admissible=math.hypot(x-.02,y)>=EXCL
            fr=force_data(x,y,b) if admissible else {'fz_range':[None,None],'fy_range':[None,None],
                                                   'fz_interaction':None,'fy_interaction':None}
            rows.append({'x':float(x),'y':float(y),'beta':float(b),'time':float(t),
                         'point_class':point_test(x,y,b) if admissible else 'excluded',
                         'tfw3d_class':tfw3d_class(x,y,b),**fr})
        result[name]=rows
    result['approach']=[]
    for y in np.linspace(-.035,-.006,161):
        x=float(crossing[0,0]); fr=force_data(x,y,math.pi/2)
        result['approach'].append({'x':x,'y':float(y),'separation_mm':math.hypot(x-.02,y)*1e3,
                                   'point_class':point_test(x,y,math.pi/2),**fr})
    return result,planned,crossing


def independent_checks_3d(tfw):
    """Point LPs at corners and centres of sampled 3-D boxes must agree with their class."""
    rng=np.random.default_rng(41)
    tested=0
    for layout,d in tfw.items():
        for kind,key in (('inside','in_workspace'),('outside','out_of_workspace')):
            boxes=d[key]
            for idx in rng.choice(len(boxes),min(30,len(boxes)),replace=False) if boxes else []:
                x0,x1,y0,y1,b0,b1=boxes[int(idx)]
                for x,y,b in [((x0+x1)/2,(y0+y1)/2,(b0+b1)/2),(x0,y0,b0),(x1,y1,b1),(x0,y1,b1)]:
                    observed=point_test(x,y,b,layout)
                    if observed!=kind:
                        raise RuntimeError(f'3-D box/point contradiction: {layout}, {kind}, {observed}, {(x,y,b)}')
                    tested+=1
    errs=[]
    for beta in (0,.7,math.pi/2,2.4):
        a=W.calculate_affine_A(interval[-.02,-.02],interval[.02,.02],interval[beta,beta],return_affine=True)
        val=np.array([[e.center for e in row] for row in a])
        errs.append(float(np.max(np.abs((val-matrix(-.02,.02,beta))/SCALES[:,None]))))
    if max(errs)>1e-8:
        raise RuntimeError(f'Affine/point derivative check failed: {errs}')
    return {'point_checks_3d':tested,'max_scaled_degenerate_map_difference':max(errs),
            'degenerate_map_scaled_tolerance':1e-8,
            'meaning':'Implementation consistency checks, not continuous verification or hardware validation.'}


def paper_exports(tfw,cuts,analysis,runs,auxiliary):
    """Write the manuscript's metric macros and an inspectable provenance note."""
    d,c=runs['detour']['summary'],runs['crossing']['summary']
    eps=tfw['single']['eps']
    macros={'TerminalWidth':f'{eps[0]*1e3:.3f}','HeadingWidth':f'{math.degrees(eps[2]):.3f}',
            'SimulationClearance':f'{SIMULATION_CLEARANCE*1e3:.2f}'}
    probes=json.loads((OUT/'orientation_probes.json').read_text())
    macros['ProbeHeadings']=str(len(probes['probe_headings_deg']))
    for label,row in zip(('One','Two','Three'),probes['probe_classes']):
        macros['Probe'+label+'Feasible']=str(sum(v=='inside' for v in row))
    for deg,label in [(0,'Zero'),(45,'FortyFive'),(90,'Ninety'),(135,'OneThirtyFive')]:
        macros['Area'+label]=f'{cuts[("single",deg)]["area_fraction_full_square"]["inside"]*100:.2f}'
    for layout in ('gate','triangle'):
        macros['Area'+layout.capitalize()]=f'{cuts[(layout,90)]["area_fraction_full_square"]["inside"]*100:.2f}'
    for layout,label in (('single','One'),('gate','Gate'),('triangle','Triangle')):
        vf=volume_fractions(tfw[layout])
        macros['Volume'+label]=f'{vf["inside"]*100:.1f}'
        macros['VolumeOut'+label]=f'{vf["outside"]*100:.1f}'
        macros['VolumeUnresolved'+label]=f'{vf["undetermined"]*100:.1f}'
    for name,summary in [('Detour',d),('Crossing',c)]:
        rows=analysis[name.lower()]
        macros[name+'Infeasible']=str(sum(r['point_class']=='outside' for r in rows))
        macros[name+'Samples']=str(sum(r['point_class']!='excluded' for r in rows))
        macros[name+'InTFW']=str(sum(r['tfw3d_class']=='inside' for r in rows))
        macros[name+'RMSE']=f'{summary["position_rmse_mm"]:.2f}'
        macros[name+'Separation']=f'{summary["min_separation_mm"]:.2f}'
        macros[name+'PeakCurrent']=f'{summary["peak_current_A"]:.2f}'
        macros[name+'GoalError']=f'{summary["last_goal_error_mm"]:.2f}'
        macros[name+'Active']=f'{summary["active_interval_s"][1]:.1f}'
    ref=np.asarray(runs['detour']['reference'])
    macros['StartY']=f'{ref[0,1]*1e3:.0f}'; macros['GoalY']=f'{ref[-1,1]*1e3:.0f}'
    macros['StartX']=f'{ref[0,0]*1e3:.0f}'
    macros['AbortTime']=f'{c["abort"]["time_s"]:.1f}' if c['abort'] else 'not applicable'
    macros['CrossingCommandInfeasible']=str(c['command_infeasible_samples'])
    macros['AuxiliaryPoseInfeasible']=str(auxiliary['summary']['actual_pose_full_task_infeasible_samples'])
    macros['AuxiliaryCommandInfeasible']=str(auxiliary['summary']['command_infeasible_samples'])
    for key,field in [('RMSE','position_rmse_mm'),('GoalError','last_goal_error_mm'),
                      ('Separation','min_separation_mm'),('PeakCurrent','peak_current_A')]:
        macros['Auxiliary'+key]=f'{auxiliary["summary"][field]:.2f}'
    macros['AuxiliaryActive']=f'{auxiliary["summary"]["active_interval_s"][1]:.1f}'
    tex='% Generated by revised_case_studies.cs1_cylinder_interaction.publication\n'
    tex+=''.join('\\providecommand{\\csone'+key+'}{'+value+'}\n' for key,value in macros.items())
    (OUT / 'case_study_1_metrics.tex').write_text(tex.rstrip() + '\n')
    note=f'''# Case Study 1 figure provenance

Updated on 17 October 2026. No new hardware data.

## Reproduction

From the repository root:

```bash
# Sharp 3-D TFWs (about 12.5 h on 14 cores in total):
.venv/bin/python -B -m revised_case_studies.cs1_cylinder_interaction.run_tfw3d_fine --layout single
.venv/bin/python -B -m revised_case_studies.cs1_cylinder_interaction.run_tfw3d_fine --layout gate
.venv/bin/python -B -m revised_case_studies.cs1_cylinder_interaction.run_tfw3d_fine --layout triangle
# Planned and straight references (also used by the slide deck):
.venv/bin/python -B -m revised_case_studies.cs1_cylinder_interaction.presentation --only plans
# Analysis, simulations, macros and figure:
.venv/bin/python -B -m revised_case_studies.cs1_cylinder_interaction.publication
# Redraw from saved records:
.venv/bin/python -B -m revised_case_studies.cs1_cylinder_interaction.publication --render
```

## TFW data

Every TFW panel is the sharp 3-D (x, y, beta) TFW of `results/cs1/tfw3d_fine`, or a
section of it at one heading level: affine-hybrid cascade, corrected dBy/dy,
stopping widths {eps[0]*1e3:.3f} mm in x and y and {math.degrees(eps[2]):.3f} deg in beta.
A TFW-in box holds for every heading in its beta interval. The 3-D runs have
no numerical cutout; boxes that contain a fixed-dipole centre stay unresolved.
Section shares divide by the full 100 x 100 mm square; volume shares by the full
(x, y, beta) box. The 3-D panels draw the complement of TFW-in on a 64 x 64 x 64
display grid sampled at cell centres. Labels are nominal TFW-in/TFW-out/
unresolved from the floating-point backend, without outward rounding.
`manifest.json` records numerical consistency checks on sampled 3-D boxes.

## Paths, force, and simulation

The planned path is `presentation/trajectory_single_planned.json`: hybrid A* in
(x, y, beta) on the sharp TFW, 2 mm margin at each heading level and its
neighbours, reversing penalised. The straight crossing shares its endpoints at
heading 90 deg and 5 mm/s; the auxiliary passage is the crossing shifted to
x = 8 mm. Reference samples are tested by endpoint LPs at their own headings
and against the 3-D TFW. Panel (k) draws the paths over the 90-deg section only.
The force interval jointly enforces holding-field and zero-lateral-force tasks.
`simulation_records.json` and `auxiliary_weaker_task_crossing.json` retain
references, poses, currents, residuals, feasibility tests and abort records.
The simulation stops at {SIMULATION_CLEARANCE*1e3:.2f} mm centre separation for
illustrative 9x5 mm moving and 11x5.5 mm fixed cylinders. No contact or
post-abort motion is fabricated. `passage_force_capacity.json` contains the
force-interval LPs of panel (m), checked against command and full-task LPs.

## Hardware replacement

Replace the simulated paths in (k) and the traces in (n) with synchronized
measurements; update the metrics and the caption together. The archived
`images/Case_6.pdf` remains untouched for the historical supplement.

## Final source hashes

'''
    for pth in [Path(__file__),Path(AM.__file__),Path(FM.__file__),Path(W.__file__)]:
        note+=f'- `{pth.relative_to(ROOT)}`: `{hashlib.sha256(pth.read_bytes()).hexdigest()}`\n'
    (OUT/'PROVENANCE.md').write_text(note)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--render',action='store_true',help='Reuse saved records; redraw the figure and macros.')
    ap.add_argument('--preview',action='store_true',help='Render to cs1_3d_preview only; do not touch the paper.')
    args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    science_style()
    load=lambda name: json.loads((OUT/name).read_text())
    if not (args.render or args.preview):
        tfw={l:load_tfw3d(l) for l in ('single','gate','triangle')}
        validation=independent_checks_3d(tfw)
        print('3-D box checks:',validation,flush=True)
        analysis,planned,crossing=analysis_data()
        for rows in analysis.values():
            for r in rows:
                for k,v in list(r.items()):
                    if isinstance(v,list): r[k]=[z if z is not None and math.isfinite(z) else None for z in v]
        dump(OUT/'path_force_analysis.json',analysis)
        runs={n:fresh_simulation(t) for n,t in [('detour',planned),('crossing',crossing)]}
        dump(OUT/'simulation_records.json',runs)
        passage=crossing.copy(); passage[:,0]-=.002
        auxiliary=fresh_simulation(passage)
        auxiliary['scope']='Auxiliary x=8 mm passage at 5 mm/s; different geometry from the primary x=10 mm crossing.'
        (OUT/'auxiliary_weaker_task_crossing.json').write_text(json.dumps(auxiliary,indent=2))
        passage_capacity(auxiliary)
        probes=[(-.025,.02),(.008,-.006),(.008,0)]
        angles=list(range(0,360,6))
        details={'probes_m':probes,'probe_headings_deg':angles,
                 'probe_classes':[[point_test(x,y,math.radians(b)) for b in angles] for x,y in probes]}
        dump(OUT/'orientation_probes.json',details)
        manifest={'source_hash':source_hash(),'numerical_validation':validation,
                  'figures':STEMS,'task':{'B_hold_T':.02,'F_range_N':[-.1,.1],'I_bounds_A':[-15,15]},
                  'tfw':{l:{'file':str((TFW3D/f'tfw3d_{l}.json').relative_to(ROOT)),'eps':tfw[l]['eps'],
                            'volume_fraction':volume_fractions(tfw[l])} for l in tfw},
                  'derivative':'Corrected dBy/dy coefficient 6*y*m_y/r^5 in point and affine maps.',
                  'planned_path':str(Path(C.PLANNED).relative_to(ROOT)),
                  'simulation_settings':{'speed_m_s':.005,'outer_step_s':DT_CTRL,'plant_step_s':DT_PLANT,
                      'drag_N_s_m':DRAG_C,'alignment_time_s':TAU_ROT,'K_P_N_m':K_P,'K_H_rad_m':K_H,
                      'geometry_stop_radius_mm':SIMULATION_CLEARANCE*1e3,
                      'heading_correction_cap_rad':MAX_HEADING_CORR,'current_slew':'Not imposed',
                      'allocation_geometry':'Commanded heading; actual heading relaxes in the plant.',
                      'repeat_count':1,'hardware_status':'No new hardware experiment; explicit placeholders.'},
                  'simulation_summary':{n:r['summary'] for n,r in runs.items()},
                  'auxiliary_passage':{'scope':auxiliary['scope'],'summary':auxiliary['summary']}}
        dump(OUT/'manifest.json',manifest)
        print('Simulation summaries:',json.dumps(manifest['simulation_summary']),flush=True)
    analysis=load('path_force_analysis.json'); runs=load('simulation_records.json')
    details=load('orientation_probes.json'); capacity=load('passage_force_capacity.json')
    auxiliary=load('auxiliary_weaker_task_crossing.json')
    runs['passage']=auxiliary
    for rows in analysis.values():
        for r in rows:
            for key in ('fz_range','fy_range'):
                if key in r: r[key]=[np.nan if z is None else z for z in r[key]]
    if args.preview:
        fig_compact3d(analysis,runs,details,capacity,stem='cs1_3d_preview')
        return
    tfw,cuts=fig_compact3d(analysis,runs,details,capacity,stem=STEMS[0])
    paper_exports(tfw,cuts,analysis,{k:runs[k] for k in ('detour','crossing')},auxiliary)


if __name__=='__main__':
    main()
