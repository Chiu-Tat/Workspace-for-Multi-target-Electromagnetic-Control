"""Generate the revised-paper growing-rod figure, model records and metrics.

Run: .venv/bin/python -B -m revised_case_studies.cs3_growing_rods.publication
Redraw: same command with --render. No new hardware data are generated.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq,linprog
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle,Rectangle,FancyBboxPatch
from matplotlib.lines import Line2D
from analysis.sections_67.common import CLASS_COLORS
from cases.case_7_vessel_microspheres import magnetics
from emns.workspace.fast import params_list
from . import config as K,model as M

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'revised_case_studies/results/cs3_growing_rods/publication'
ASSETS=Path(__file__).resolve().parent/'assets'
BLUE,PURPLE,ORANGE,INK='#215f98','#755a9c','#bd5c31','#252a30'
GREEN,RED=CLASS_COLORS['inside'],CLASS_COLORS['outside']


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def compute():
    spacing=np.linspace(8,60,53);offset=np.linspace(0,40,41)
    maps={}
    for z in [55.,85.]:
        maps[str(int(z))]=[[M.capacity(M.Placement(s,d,z))['selective_robust_T'] for s in spacing] for d in offset]
        print(f'Placement map at z={z:g} mm complete.',flush=True)
    curves=[M.capacity(M.Placement(s)) for s in spacing]
    boundary=brentq(lambda s:M.capacity(M.Placement(s))['selective_robust_T']-K.B_ROT,8,25)
    points={'P1':M.Placement(40,0,55),'P2':M.Placement(boundary*1.002,0,55),'P3':M.Placement(8,0,55)}
    placements={key:{'q_mm':p.q,'capacity':M.capacity(p)} for key,p in points.items()}
    runs={}
    for method in ['selective','common']:
        runs[method]=M.simulate(points['P1'],method)
        print(method,json.dumps(runs[method]['summary']),flush=True)
    probes={}
    for key,p in points.items():
        probes[key]={}
        for method in ['selective','common']:
            probes[key][method]=M.simulate(p,method,sequence=[(4.,(.5,0.))])
        print(key,'placement probe complete.',flush=True)
    return {'spacing_mm':spacing.tolist(),'offset_mm':offset.tolist(),'height_maps_T':maps,
            'spacing_curves':curves,'placements':placements,'runs':runs,'probes':probes}


def validate(data):
    rng=np.random.default_rng(4);checks=[]
    for key,entry in data['placements'].items():
        p=M.Placement(*entry['q_mm']);N=M.facets(p.A)
        # Check facet support independently against current-box LPs.
        discrepancies=[]
        for n in N[rng.choice(len(N),10,replace=False)]:
            r=linprog(-(n@p.A),bounds=[(-K.I_LIMIT,K.I_LIMIT)]*10,method='highs')
            assert r.success
            discrepancies.append(abs(-r.fun-K.I_LIMIT*sum(abs(n@p.A))))
        contradictions=0
        for task in ['selective','simultaneous']:
            radius=entry['capacity'][task+'_robust_T']*.999
            for _ in range(32):
                phase=rng.uniform(0,2*np.pi,2);phi=rng.uniform(0,2*np.pi,2)
                target=radius*np.column_stack([np.cos(phi),np.sin(phi)])
                if task=='selective':target[rng.integers(0,2)]=0.
                if radius>0 and not M.exact_feasible(p,target.ravel(),phase):contradictions+=1
        assert contradictions==0
        safe,details=p.force_safe_current_limit();force_max=0.
        for _ in range(100):
            current=rng.uniform(-safe,safe,10);phase=rng.uniform(0,2*np.pi,2)
            _,forces=p.fields_forces(current,phase)
            force_max=max(force_max,float(np.max(np.linalg.norm(forces,axis=1))))
        if safe>0:assert force_max<=K.FORCE_LIMIT+1e-10
        command_contradictions=0
        for method,run in data['probes'][key].items():
            for row in run['records'][:-1]:
                exact=M.exact_feasible(p,row['command_fields_T'],row['phase_rad'])
                if exact!=row['exact_command']:command_contradictions+=1
        assert command_contradictions==0
        checks.append({'placement':key,'maximum_facet_LP_discrepancy_T':max(discrepancies),
            'phase_samples_per_task':32,'capacity_LP_contradictions':contradictions,
            'probe_command_LP_checks':400,'probe_command_LP_contradictions':command_contradictions,
            'force_safe_sample_max_N':force_max,'force_safe_current_limit_A':safe,
            'force_safe_box_exists':bool(safe>0)})
    # A nontrivial facet check against the convex hull of every current-box vertex.
    p=M.Placement(40);vertices=np.array([[(1 if bits&(1<<j) else -1)*K.I_LIMIT for j in range(10)] for bits in range(1024)])@p.A.T
    hull=ConvexHull(vertices);eq=hull.equations
    n=eq[:,:4];norm1=np.linalg.norm(n[:,:2],axis=1);norm2=np.linalg.norm(n[:,2:],axis=1)
    NU=n@p.U;u=np.linalg.norm(NU[:,:2],axis=1)+np.linalg.norm(NU[:,2:],axis=1)
    independent=float(np.min((-eq[:,4]-u)/np.maximum(norm1,norm2)))
    expected=M.capacity(p)['selective_robust_T']
    assert abs(independent-expected)<1e-9
    # Analytic overdamped phase-lock criterion at the illustrative drive rate.
    tau=K.FRICTION+K.LOAD*K.LEAD_PITCH/(2*np.pi*K.GEAR_RATIO*K.GEAR_EFFICIENCY*K.SCREW_EFFICIENCY)
    assert tau+K.ROT_DRAG*2*np.pi*K.FREQ < K.MOMENT*K.B_ROT
    for method,run in data['runs'].items():
        assert run['summary']['inexact_allocation_samples']==0
        assert run['summary']['peak_current_A']<=K.I_LIMIT+1e-6
    return {'point_checks':checks,'independent_convex_hull_radius_T':independent,
            'facet_radius_T':expected,'phase_lock_torque_margin_Nm':K.MOMENT*K.B_ROT-tau-K.ROT_DRAG*2*np.pi*K.FREQ,
            'scope':'Pointwise floating-point/model checks. No outward-rounded placement-region or physical certificate.'}


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'axes.labelsize':7,
        'axes.titlesize':7.5,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6,
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#777b80',
        'text.color':INK,'axes.labelcolor':INK,'pdf.fonttype':42,'savefig.dpi':250})


def schematic(ax):
    ax.set(xlim=(0,100),ylim=(0,100));ax.axis('off')
    # AI-generated anatomical context only; the stored RGBA asset is not model geometry.
    # Select its visible bounds for plotting, without altering the saved bitmap or alpha.
    spine=plt.imread(ASSETS/'spine_posterior_illustration.png')
    rows,cols=np.nonzero(spine[:,:,3]>.125)
    spine=spine[rows.min():rows.max()+1,cols.min():cols.max()+1]
    ax.figure.canvas.draw()
    bounds=ax.get_window_extent()
    height=82.;width=height*spine.shape[1]/spine.shape[0]*bounds.height/bounds.width
    ax.imshow(spine,extent=(28-width/2,28+width/2,10,92),aspect='auto',zorder=0)
    for xpos,color,k in [(19,BLUE,1),(37,PURPLE,2)]:
        ax.plot([xpos,xpos],[19,84],color=color,lw=2,zorder=3)
        # Conceptual proximal/distal anchors; dimensions are not calibrated implant geometry.
        anchor=24 if k==1 else 32
        for yy in [26,78]:
            ax.plot([xpos,anchor],[yy,yy],color=color,lw=1.3,zorder=3)
            ax.add_patch(Circle((anchor,yy),.9,fc='white',ec=color,lw=.6,zorder=4))
        ax.add_patch(FancyBboxPatch((xpos-3.5,42),7,19,boxstyle='round,pad=.5',fc='white',ec=color,lw=1))
        ax.add_patch(Circle((xpos,55),2.7,fc=color,ec='white',lw=.5))
        ax.text(xpos,96,f'Rod {k}',ha='center',color=color,fontsize=6)
        ax.annotate('',(xpos,88),(xpos,81),arrowprops={'arrowstyle':'-|>','color':color,'lw':.8})
    ax.annotate('',(19,7),(37,7),arrowprops={'arrowstyle':'<->','color':INK,'lw':.6})
    ax.text(28,0,'Spacing s',ha='center',fontsize=6)
    ax.text(72,90,'Actuator module',fontsize=5.3,ha='center')
    for yy,label in [(70,'Magnetic rotor'),(48,'Gear reduction'),(26,'Lead screw')]:
        ax.add_patch(Rectangle((58,yy-7),29,13,fc='#f1f3f5',ec='#9da5ac',lw=.6))
        ax.text(72.5,yy,label,ha='center',va='center',fontsize=5.4)
    for yy in [61,39]:ax.annotate('',(72.5,yy-6),(72.5,yy+1),arrowprops={'arrowstyle':'-|>','lw':.6,'color':INK})
    ax.plot([63,81],[18,18],color=INK,lw=.7)
    ax.text(72,10,'Telescoping output',ha='center',fontsize=5.4)
    ax.set_title('(a) Dual-rod phantom concept',loc='left',pad=5)


def spacing_panel(ax,data):
    x=np.array(data['spacing_mm']);curves=data['spacing_curves']
    for key,color,label,ls in [('selective_robust_T',BLUE,'Rotate + null hold','-'),
             ('simultaneous_robust_T',PURPLE,'Independent rotation','-'),
             ('force_safe_selective_lower_T',ORANGE,'Force-safe lower bound','--')]:
        ax.plot(x,[r[key]*1e3 for r in curves],color=color,lw=1.2,ls=ls,label=label)
    ax.axhline(K.B_ROT*1e3,color=INK,ls=':',lw=.7)
    ax.text(58,17,'Design threshold',fontsize=5.5,ha='right')
    ax.set(xlabel='Spacing s (mm)',ylabel='Amplitude (mT)',xlim=(8,60),ylim=(0,105),xticks=[10,20,40,60])
    ax.legend(loc='upper left',frameon=False,fontsize=5.5)
    ax.set_title('(b) In-plane capacity; z = 55 mm',loc='left',pad=5)


def sensitivity_panel(ax,data):
    spacings=[20,40,60];vals=[M.capacity(M.Placement(s)) for s in spacings]
    x=np.arange(3);width=.24
    for shift,key,color,label in [(-1,'selective_nominal_T','#bac4cb','No interaction'),
            (0,'selective_robust_T',BLUE,'All-phase offset'),
            (1,'force_safe_selective_lower_T',ORANGE,'+ force-safe box')]:
        ax.bar(x+shift*width,[v[key]*1e3 for v in vals],width=width*.95,color=color,label=label)
    ax.set(xticks=x,xticklabels=spacings,xlabel='Spacing s (mm)',ylabel='Amplitude (mT)',ylim=(0,115))
    ax.legend(loc='upper left',bbox_to_anchor=(0,.90),frameon=False,fontsize=5.4)
    ax.text(.03,.96,'0.15 N bound; uniform current derating',transform=ax.transAxes,
            ha='left',va='top',fontsize=5.2)
    ax.set_title('(c) Offset and force-bound effects',loc='left',pad=5)


def maps_panel(fig,spec,data):
    grid=spec.subgridspec(1,2,wspace=.2)
    cmap=LinearSegmentedColormap.from_list('capacity',['#f0f1f2','#acd9b4','#45834e'])
    axes=[]
    for k,z in enumerate([55,85]):
        ax=fig.add_subplot(grid[k]);axes.append(ax)
        a=np.array(data['height_maps_T'][str(z)])*1e3
        im=ax.pcolormesh(data['spacing_mm'],data['offset_mm'],a,shading='auto',cmap=cmap,vmin=0,vmax=110,rasterized=True)
        ax.contour(data['spacing_mm'],data['offset_mm'],a,levels=[K.B_ROT*1e3],colors=INK,linewidths=.75)
        if z==55:
            for key,item in data['placements'].items():
                s,d,_=item['q_mm'];color={'P1':BLUE,'P2':ORANGE,'P3':RED}[key]
                ax.plot(s,d,'o',ms=4,color=color,mec='white',mew=.5,clip_on=False,zorder=5)
                ax.annotate(key,(s,d),xytext=(0,6 if key!='P2' else 15),textcoords='offset points',
                            ha='center',fontsize=5.5,color='#a64242' if key=='P3' else color)
        ax.set(xlabel='Spacing s (mm)',xlim=(8,60),ylim=(0,40),xticks=[10,30,50],yticks=[0,20,40])
        if k==0:ax.set_ylabel('Offset Δ (mm)')
        else:ax.tick_params(labelleft=False)
        ax.set_title(f'z = {z} mm',fontsize=6.5,pad=4)
    axes[0].text(0,1.24,'(d) Selective placement maps',transform=axes[0].transAxes,fontsize=7.5)
    cb=fig.colorbar(im,ax=axes,location='bottom',fraction=.075,pad=.27,aspect=38)
    cb.set_label('All-phase rotate + null-hold capacity (mT)',fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    axes[1].text(.98,.98,'Black: 15 mT contour',transform=axes[1].transAxes,fontsize=5.2,ha='right',va='top')


def placement_panel(fig,spec,data):
    grid=spec.subgridspec(2,1,hspace=.10,height_ratios=[1,1])
    top=fig.add_subplot(grid[0]);bot=fig.add_subplot(grid[1],sharex=top)
    x=np.arange(3);width=.28
    for shift,method,color,label in [(-.5,'selective',BLUE,'Selective'),(.5,'common',ORANGE,'Common field')]:
        probes=[data['probes'][key][method]['summary'] for key in ['P1','P2','P3']]
        top.bar(x+shift*width,[p['maximum_held_length_change_mm']*1e3 for p in probes],width,color=color,label=label)
        bot.bar(x+shift*width,[p['maximum_field_component_error_T']*1e3 for p in probes],width,color=color)
    top.set_title('(e) Four-second placement probes',loc='left',pad=5)
    top.set_ylabel('Held motion\n(μm)',fontsize=6);top.tick_params(labelbottom=False)
    top.set_ylim(0,28)
    top.legend(frameon=False,loc='upper right',fontsize=5.4)
    bot.set_ylabel('Field error\n(mT)',fontsize=6);bot.set_xticks(x,labels=['P1: inside','P2: near','P3: outside'],fontsize=5.2)
    bot.set_ylim(0,max(1.,bot.get_ylim()[1]))
    bot.text(.03,.94,'Hard null hold at updates',transform=bot.transAxes,
             ha='left',va='top',fontsize=5.0)


def motion_panel(fig,spec,data):
    grid=spec.subgridspec(2,2,wspace=.3,hspace=.13,width_ratios=[1.9,1])
    length=fig.add_subplot(grid[0,0]);phase=fig.add_subplot(grid[1,0],sharex=length)
    current=fig.add_subplot(grid[0,1]);resid=fig.add_subplot(grid[1,1],sharex=current)
    for method,ls in [('selective','-'),('common','--')]:
        rows=data['runs'][method]['records'];t=np.array([r['t'] for r in rows])
        vals=np.array([r['length_mm'] for r in rows]);angle=np.array([r['phase_rad'] for r in rows])/(2*np.pi)
        for k,color in [(0,BLUE),(1,PURPLE)]:
            length.plot(t,vals[:,k],color=color,lw=1.1,ls=ls)
            phase.plot(t,angle[:,k],color=color,lw=1.1,ls=ls)
        current.plot(t,[max(abs(v) for v in r['currents_A']) for r in rows],color=INK if method=='selective' else ORANGE,lw=1,ls=ls)
        ref_angle=np.array([r['reference_phase_rad'] if method=='selective' else [r['common_phase_rad']]*2 for r in rows])
        lag=abs((ref_angle-angle*2*np.pi+np.pi)%(2*np.pi)-np.pi)*180/np.pi
        if method=='selective':
            for j,r in enumerate(rows):
                for k in [0,1]:
                    if K.PHASE_SEQUENCE[r['segment']][1][k]==0:lag[j,k]=np.nan
        for k,color in [(0,BLUE),(1,PURPLE)]:resid.plot(t,lag[:,k],color=color,lw=.8,ls=ls)
    rows=data['runs']['selective']['records'];t=[r['t'] for r in rows];ref=np.array([r['reference_length_mm'] for r in rows])
    for k,color in [(0,BLUE),(1,PURPLE)]:length.plot(t,ref[:,k],color=color,ls=':',lw=.65)
    edges=[0,32,64,96,112];labels=['1 moves; 2 holds','2 moves; 1 holds','Both: 2:1 rates','1 retracts; 2 holds']
    for k,label in enumerate(labels):
        for ax in [length,phase,current,resid]:
            ax.axvspan(edges[k],edges[k+1],color=BLUE if k%2==0 else PURPLE,alpha=.04,lw=0)
        length.text((edges[k]+edges[k+1])/2,1.025,label,transform=length.get_xaxis_transform(),ha='center',fontsize=5.0)
    length.set_title('(f) Simulated selective sequences and common-field baseline at P1',loc='left',pad=19)
    length.set_ylabel('Rod extension\n(mm)',fontsize=6);length.tick_params(labelbottom=False)
    phase.set_ylabel('Rotor turns',fontsize=6);phase.set_xlabel('Time (s)');phase.set_xticks([0,32,64,96,112]);phase.set_xlim(0,112)
    current.axhline(K.I_LIMIT,color=INK,ls=':',lw=.7)
    current.set_ylabel('Max |I| (A)',fontsize=6);current.set_ylim(0,17);current.tick_params(labelbottom=False)
    current.set_title('Current: solid selective; dashed common',fontsize=5.5,pad=5)
    resid.set_ylabel('Active phase lag\n(deg)',fontsize=6);resid.set_xlabel('Time (s)');resid.set_xlim(0,112);resid.set_xticks([0,32,64,96,112])
    resid.set_ylim(0,20)
    length.text(.02,.94,'Blue: rod 1; purple: rod 2\nSolid: selective; dashed: common; dotted: reference',
                transform=length.transAxes,fontsize=5.2,va='top')


FORCE_GAIN=2*np.pi*K.GEAR_RATIO*K.GEAR_EFFICIENCY*K.SCREW_EFFICIENCY/K.LEAD_PITCH
ROTOR_LOSS=K.FRICTION+K.ROT_DRAG*2*np.pi*K.FREQ
CLINICAL_STALL_N=(100.,250.)


def force_from_field(radius_T):
    """Quasi-static lead-screw force of the illustrative mechanism at a guaranteed field radius."""
    return np.maximum(0.,(K.MOMENT*np.asarray(radius_T)-ROTOR_LOSS)*FORCE_GAIN)


def field_from_force(force_N):
    return (np.asarray(force_N)/FORCE_GAIN+ROTOR_LOSS)/K.MOMENT


def design_data():
    path=OUT/'design_records.json'
    if path.exists():return json.loads(path.read_text())
    heights=np.arange(20.,140.1,2.5);spacings=np.arange(8.,61.,2.);offsets=np.arange(0.,61.,2.)
    height_curve=[M.capacity(M.Placement(40.,0.,z)) for z in heights]
    height_map=[[M.capacity(M.Placement(s,0.,z))['selective_robust_T'] for s in spacings] for z in heights]
    offset_curves={str(int(s)):[M.capacity(M.Placement(s,d,55.))['selective_robust_T'] for d in offsets]
                   for s in [20.,40.,60.]}
    data={'heights_mm':heights.tolist(),'spacings_mm':spacings.tolist(),'offsets_mm':offsets.tolist(),
          'height_curve_selective_T':[c['selective_robust_T'] for c in height_curve],
          'height_curve_simultaneous_T':[c['simultaneous_robust_T'] for c in height_curve],
          'height_map_selective_T':height_map,'offset_curves_selective_T':offset_curves,
          'scope':'Pointwise all-phase disc containment at sampled placements; Delta=0 unless stated.'}
    dump(path,data);return data


def concept_panel(ax):
    ax.set(xlim=(0,100),ylim=(0,100));ax.axis('off')
    # The lateral anatomical curve is illustrative; actuator centres/axes remain parallel.
    spine=plt.imread(ASSETS/'spine_posterior_curved.png')
    rows,cols=np.nonzero(spine[:,:,3]>.125)
    spine=spine[rows.min():rows.max()+1,cols.min():cols.max()+1]
    ax.figure.canvas.draw();bounds=ax.get_window_extent()
    height=80.;width=height*spine.shape[1]/spine.shape[0]*bounds.height/bounds.width
    ax.imshow(spine,extent=(25-width/2,25+width/2,9,89),aspect='auto',zorder=0)
    for xpos,color,k in [(15,BLUE,1),(35,PURPLE,2)]:
        ax.plot([xpos,xpos],[16,84],color=color,lw=2.2,solid_capstyle='round',zorder=3)
        for yy in [20,80]:
            ax.plot([xpos,25+(xpos-25)*.35],[yy,yy],color=color,lw=1.2,zorder=3)
        ax.add_patch(FancyBboxPatch((xpos-3.2,45),6.4,17,boxstyle='round,pad=.4',fc='white',ec=color,lw=1,zorder=4))
        ax.add_patch(Circle((xpos,57),2.3,fc=color,ec='white',lw=.4,zorder=5))
        ax.text(xpos,93,f'Rod {k}',ha='center',color=color,fontsize=6.2,fontweight='bold')
    ax.annotate('',(15,4),(35,4),arrowprops={'arrowstyle':'<->','color':INK,'lw':.6})
    ax.text(25,-2.5,'spacing $s$',ha='center',va='top',fontsize=6)
    ax.annotate('',(48,57),(56,57),arrowprops={'arrowstyle':'<-','color':'#8a9097','lw':.6,'ls':(0,(2,1.5))})
    x0,x1=60,72
    ax.add_patch(FancyBboxPatch((x0,26),x1-x0,62,boxstyle='round,pad=.6',fc='#f4f6f8',ec='#7d858d',lw=.7))
    ax.add_patch(matplotlib.patches.Wedge((66,79),4.4,90,270,fc='#c0504d',ec='none',zorder=3))
    ax.add_patch(matplotlib.patches.Wedge((66,79),4.4,-90,90,fc='#3f6fb0',ec='none',zorder=3))
    ax.add_patch(Circle((66,79),4.4,fc='none',ec=INK,lw=.5,zorder=4))
    ax.annotate('',(70.2,85),(61.8,85),arrowprops={'arrowstyle':'->','color':INK,'lw':.5,
                'connectionstyle':'arc3,rad=-.5'},zorder=5)
    for yy in [69,63.5,58]:
        ax.add_patch(Rectangle((61.5,yy-2.2),9,4.4,fc='#c9ced3',ec='#6b737b',lw=.5))
        for tx in np.linspace(62.5,69.5,6):ax.plot([tx,tx],[yy+2.2,yy+3],color='#6b737b',lw=.4)
    zz=np.linspace(31,52,40);ax.plot(66+2.2*np.sign(np.sin(zz*2.4))*np.abs(np.sin(zz*2.4))**.5,zz,color='#5c646c',lw=.6)
    ax.add_patch(Rectangle((64.4,9),3.2,19,fc='#aab2ba',ec='#5c646c',lw=.5))
    ax.annotate('',(66,3),(66,9),arrowprops={'arrowstyle':'-|>','color':INK,'lw':.6})
    for yy,label in [(79,'Magnet rotor'),(63.5,'Gearbox\n64:1'),(41,'Lead\nscrew'),(15,'Telescoping\noutput')]:
        ax.plot([x1+1,76],[yy,yy],color='#8a9097',lw=.4)
        ax.text(77,yy,label,va='center',fontsize=5.5,linespacing=.95)
    ax.set_title('(a) Dual growing rods and actuator',loc='left',pad=4,fontsize=7.4)


def height_panel(ax,design):
    z=np.array(design['heights_mm'])
    ax.axhspan(*CLINICAL_STALL_N,color='#e9e4f2',lw=0,zorder=0)
    ax.text(137,175,'clinical stall\nforce range',fontsize=5.4,color='#6d5a8e',ha='right',va='center',linespacing=.95)
    for key,color,label in [('height_curve_selective_T',BLUE,'one rotates, other held'),
                            ('height_curve_simultaneous_T',PURPLE,'both rotate independently')]:
        ax.plot(z,force_from_field(design[key]),color=color,lw=1.3,label=label)
    ax.axvline(55,color='#9aa1a8',lw=.5,ls=':')
    ax.text(57,8,'P1–P3',fontsize=5.2,color='#6b737b',va='bottom')
    ax.set(xlim=(20,140),ylim=(0,360),xlabel='Height above coil array $z$ (mm)',ylabel='Guaranteed rod force (N)')
    ax.set_xticks([20,60,100,140]);ax.set_yticks([0,100,200,300])
    sec=ax.secondary_yaxis('right',functions=(lambda f:field_from_force(f)*1e3,lambda b:force_from_field(np.asarray(b)*1e-3)))
    sec.set_ylabel('Guaranteed field (mT)',fontsize=6.4);sec.set_yticks([10,30,50,70]);sec.tick_params(labelsize=6)
    ax.legend(loc='upper right',bbox_to_anchor=(1.02,1.02),frameon=False,fontsize=5.6,handlelength=1.6)
    ax.set_title('(b) Capacity falls with height ($s=40$ mm)',loc='left',pad=4,fontsize=7.4)


def placement_map(ax,design,data,fig):
    s=np.array(design['spacings_mm']);z=np.array(design['heights_mm'])
    force=force_from_field(np.array(design['height_map_selective_T']))
    cmap=LinearSegmentedColormap.from_list('force',['#f3f4f5','#bfe3c6','#4f9a5c','#2e6b3a'])
    im=ax.pcolormesh(s,z,force,shading='auto',cmap=cmap,vmin=0,vmax=350,rasterized=True)
    cs=ax.contour(s,z,force,levels=list(CLINICAL_STALL_N),colors=[INK,INK],linewidths=[.7,.7],linestyles=['--','-'])
    ax.clabel(cs,fmt={100.:'100 N',250.:'250 N'},fontsize=5.4,inline_spacing=2)
    ax.plot([40,60],[136,136],color='#6d5a8e',lw=1.8,solid_capstyle='butt',clip_on=False)
    ax.text(59,131,'typical rod spacing',fontsize=5.0,color='#6d5a8e',ha='right',va='top')
    for key,item in data['placements'].items():
        sp,_,h=item['q_mm'];color={'P1':BLUE,'P2':ORANGE,'P3':'#a64242'}[key]
        ax.plot(sp,h,'o',ms=3.6,color=color,mec='white',mew=.5,zorder=5)
        ax.annotate(key,(sp,h),xytext={'P1':(3,3),'P2':(1,4),'P3':(-1,-9)}[key],textcoords='offset points',fontsize=5.5,color=color)
    ax.set(xlim=(8,60),ylim=(20,140),xlabel='Rod spacing $s$ (mm)',ylabel='Height $z$ (mm)')
    ax.set_xticks([10,20,40,60]);ax.set_yticks([20,60,100,140])
    cb=fig.colorbar(im,ax=ax,fraction=.06,pad=.03,aspect=22)
    cb.set_label('Guaranteed rod force (N)',fontsize=5.8)
    cb.ax.tick_params(labelsize=5.6);cb.set_ticks([0,100,200,300])
    ax.set_title('(c) Selective placement map',loc='left',pad=4,fontsize=7.4)


def offset_panel(ax,design):
    d=np.array(design['offsets_mm'])
    for key,color in [('20','#b8860b'),('40',BLUE),('60','#2e6b3a')]:
        ax.plot(d,force_from_field(design['offset_curves_selective_T'][key]),color=color,lw=1.2)
        ax.text(61,force_from_field(design['offset_curves_selective_T'][key][-1]),f'$s={key}$',
                fontsize=5.6,color=color,va='center')
    ax.axhspan(*CLINICAL_STALL_N,color='#e9e4f2',lw=0,zorder=0)
    ax.set(xlim=(0,60),ylim=(0,520),xlabel=r'Longitudinal offset $\Delta$ (mm)',ylabel='Guaranteed rod force (N)')
    ax.text(2,505,'$z=55$ mm, one rotates, other held',fontsize=5.2,va='top',color='#6b737b')
    ax.set_xticks([0,20,40,60]);ax.set_yticks([0,200,400])
    ax.set_title(r'(d) Offset helps only close rods',loc='left',pad=4,fontsize=7.4)


def sequence_panel(fig,spec,data):
    grid=spec.subgridspec(2,1,hspace=.12,height_ratios=[2.4,1])
    ax=fig.add_subplot(grid[0]);cur=fig.add_subplot(grid[1],sharex=ax)
    edges=[0,32,64,96,112];labels=['rod 1 moves\nrod 2 held','rod 2 moves\nrod 1 held','both, 2:1\nrates','rod 1 retracts\nrod 2 held']
    for k in range(4):
        for a in (ax,cur):a.axvspan(edges[k],edges[k+1],color='#eef1f4' if k%2==0 else 'white',lw=0,zorder=0)
        ax.text((edges[k]+edges[k+1])/2,1.03,labels[k],transform=ax.get_xaxis_transform(),ha='center',va='bottom',
                fontsize=5.2,linespacing=.95)
    rows=data['runs']['selective']['records'];t=np.array([r['t'] for r in rows])
    ref=np.array([r['reference_length_mm'] for r in rows])*1e3
    for method,ls,lw in [('common',(0,(3,1.6)),1.0),('selective','-',1.3)]:
        rr=data['runs'][method]['records'];tt=np.array([r['t'] for r in rr]);L=np.array([r['length_mm'] for r in rr])*1e3
        for k,color in [(0,BLUE),(1,PURPLE)]:ax.plot(tt,L[:,k],color=color,lw=lw,ls=ls)
        peak=np.array([max(abs(v) for v in r['currents_A']) for r in rr])
        window=max(1,int(round(1/(tt[1]-tt[0]))))
        envelope=np.array([peak[max(0,j-window):j+1].max() for j in range(len(peak))])
        cur.plot(tt,envelope,color=INK if method=='selective' else ORANGE,lw=1.,ls=ls)
    for k,color in [(0,BLUE),(1,PURPLE)]:ax.plot(t,ref[:,k],color=color,lw=.6,ls=':')
    ax.set(ylabel='Rod extension (µm)',ylim=(0,400),yticks=[0,100,200,300,400]);ax.tick_params(labelbottom=False)
    handles=[Line2D([],[],color=BLUE,lw=1.3,label='rod 1'),Line2D([],[],color=PURPLE,lw=1.3,label='rod 2'),
             Line2D([],[],color='#666',lw=1.3,label='selective'),Line2D([],[],color='#666',lw=1,ls=(0,(3,1.6)),label='common field'),
             Line2D([],[],color='#666',lw=.6,ls=':',label='reference')]
    ax.legend(handles=handles,loc='upper left',frameon=False,fontsize=5.4,ncol=2,handlelength=1.8,columnspacing=.9)
    cur.axhline(K.I_LIMIT,color=INK,ls=':',lw=.6)
    cur.set(ylabel='Peak |I|\n(A)',ylim=(0,16.5),yticks=[0,15],xlim=(0,112),xticks=edges,xlabel='Time (s)')
    cur.yaxis.label.set_fontsize(6)
    ax.set_title('(e) Simulated adjustment at P1',loc='left',pad=22,fontsize=7.4)


def probe_panel(ax,data):
    turns=2;intended=turns*K.LEAD_PITCH/K.GEAR_RATIO*1e6
    keys=['P1','P2','P3'];y=np.arange(3)[::-1]*1.0
    for method,color,off in [('selective',BLUE,.17),('common',ORANGE,-.17)]:
        for yy,key in zip(y+off,keys):
            L=np.array(data['probes'][key][method]['records'][-1]['length_mm'])*1e3
            ax.barh(yy,L[0],height=.3,color=color,alpha=.9 if method=='selective' else .35,lw=0)
            ax.barh(yy,L[1],height=.3,fc='none',ec=color,lw=.9,hatch='/////')
            ax.text(max(L)+.6,yy,f'held {L[1]:.1f}' if L[1]>=.05 else 'held 0',fontsize=5.1,va='center',color=color)
    ax.axvline(intended,color=INK,ls=':',lw=.6)
    ax.set(yticks=y,xlim=(0,25),xlabel='Extension in 4 s (µm)')
    ax.set_yticklabels(['P1 inside','P2 near\nboundary','P3 outside'],fontsize=5.8,linespacing=.95)
    ax.tick_params(axis='y',length=0)
    handles=[Patch(fc=BLUE,label='selective: active rod 1'),Patch(fc=ORANGE,alpha=.35,label='common: active rod 1'),
             Patch(fc='white',ec='#555',hatch='/////',label='held rod 2 (either method)')]
    ax.legend(handles=handles,loc='upper center',bbox_to_anchor=(.45,1.0),frameon=False,fontsize=5.2,handlelength=1.4)
    ax.text(intended+.4,-.55,'2-turn target',fontsize=5.0,ha='left',va='bottom')
    ax.set_ylim(-.6,3.6)
    ax.set_title('(f) Short probes: held-rod motion',loc='left',pad=4,fontsize=7.4)


def render(data):
    from matplotlib.patches import Patch
    globals()['Patch']=Patch
    design=design_data();style()
    fig=plt.figure(figsize=(7.16,5.5))
    outer=fig.add_gridspec(2,1,left=.07,right=.975,bottom=.08,top=.93,hspace=.5,height_ratios=[1,1.02])
    top=outer[0].subgridspec(1,3,wspace=.55,width_ratios=[1.12,1,1.08])
    bottom=outer[1].subgridspec(1,3,wspace=.5,width_ratios=[.82,1.45,.95])
    concept_ax=fig.add_subplot(top[0]);concept_panel(concept_ax)
    height_panel(fig.add_subplot(top[1]),design)
    placement_map(fig.add_subplot(top[2]),design,data,fig)
    offset_panel(fig.add_subplot(bottom[0]),design)
    sequence_panel(fig,bottom[1],data)
    probe_panel(fig.add_subplot(bottom[2]),data)
    fig.text(.975,.478,'(e), (f): simulation placeholders for bench measurements',fontsize=6.4,color=ORANGE,
             style='italic',ha='right',va='center')
    for ext in ['pdf','png']:
        path=OUT/f'cs3_compact.{ext}';fig.savefig(path,bbox_inches='tight',pad_inches=.04,facecolor='white')
    fig.canvas.draw()
    preview_box=concept_ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).padded(.035)
    preview=OUT/'cs3_anatomical_concept.png'
    fig.savefig(preview,bbox_inches=preview_box,facecolor='white',dpi=300)
    plt.close(fig)


def render_v1(data):
    style();fig=plt.figure(figsize=(7.16,7.3))
    gs=fig.add_gridspec(3,3,left=.075,right=.985,bottom=.07,top=.93,wspace=.4,hspace=.69,height_ratios=[1,1,1.25])
    concept_ax=fig.add_subplot(gs[0,0]);schematic(concept_ax)
    spacing_panel(fig.add_subplot(gs[0,1]),data)
    sensitivity_panel(fig.add_subplot(gs[0,2]),data)
    maps_panel(fig,gs[1,:2],data);placement_panel(fig,gs[1,2],data)
    motion_panel(fig,gs[2,:],data)
    fig.text(.5,.978,'SIMULATION PLACEHOLDERS: replace mechanism/current traces with hardware measurements.',
             color=ORANGE,ha='center',fontsize=7,style='italic')
    fig.text(.5,.012,'Scaled bench concept; no hardware or clinical result.  Maps are pointwise model calculations; z is coil-frame height.',
             ha='center',fontsize=6.2)
    for ext in ['pdf','png']:
        path=OUT/f'cs3_compact.{ext}';fig.savefig(path,bbox_inches='tight',pad_inches=.04,facecolor='white')
    # A panel-only preview uses the same plotted objects as the manuscript figure.
    fig.canvas.draw()
    preview_box=concept_ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted()).padded(.035)
    preview=OUT/'cs3_anatomical_concept.png'
    fig.savefig(preview,bbox_inches=preview_box,facecolor='white',dpi=300)
    plt.close(fig)


def export(data,checks):
    vals={}
    for s in [20,40,60]:
        c=M.capacity(M.Placement(s));tag={20:'Twenty',40:'Forty',60:'Sixty'}[s]
        for task,key in [('Selective','selective_robust_T'),('Simultaneous','simultaneous_robust_T'),('ForceSafe','force_safe_selective_lower_T')]:
            vals[tag+task]=f'{c[key]*1e3:.2f}'
    for key,entry in data['placements'].items():
        tag={'P1':'PointOne','P2':'PointTwo','P3':'PointThree'}[key]
        vals[tag+'Spacing']=f'{entry["q_mm"][0]:.2f}'
        vals[tag+'Capacity']=f'{entry["capacity"]["selective_robust_T"]*1e3:.2f}'
        p=data['probes'][key]['selective']['summary']
        vals[tag+'FieldError']=f'{p["maximum_field_component_error_T"]*1e3:.2f}'
        vals[tag+'Inexact']=str(p['inexact_allocation_samples'])
    for method,tag in [('selective','Selective'),('common','Common')]:
        s=data['runs'][method]['summary']
        vals[tag+'HeldMotion']=f'{s["maximum_held_length_change_mm"]*1e3:.3f}' if s['maximum_held_length_change_mm']*1e3>=.001 else '<0.001'
        vals[tag+'Current']=f'{s["peak_current_A"]:.2f}'
        vals[tag+'OneLength']=f'{s["final_lengths_mm"][0]:.4f}'
        vals[tag+'TwoLength']=f'{s["final_lengths_mm"][1]:.4f}'
        vals[tag+'Force']=f'{s["maximum_predicted_force_N"]:.4f}'
    design=design_data();z=np.array(design['heights_mm'])
    sel=np.array(design['height_curve_selective_T']);sim=np.array(design['height_curve_simultaneous_T'])
    for h,tag in [(35,'ThirtyFive'),(55,'FiftyFive'),(85,'EightyFive'),(120,'OneTwenty')]:
        j=int(np.argmin(abs(z-h)))
        vals['Height'+tag+'SelectiveField']=f'{sel[j]*1e3:.1f}'
        vals['Height'+tag+'SelectiveForce']=f'{force_from_field(sel[j]):.0f}'
        vals['Height'+tag+'SimultaneousForce']=f'{force_from_field(sim[j]):.0f}'
    def crossing(curve,level):
        force=force_from_field(curve);peak=int(np.argmax(force))
        if force[peak]<level:return 'none'
        return f'{np.interp(-level,-force[peak:],z[peak:]):.0f}'
    for level,tag in [(250.,'TwoFifty'),(100.,'Hundred')]:
        vals['Height'+tag+'Selective']=crossing(sel,level)
        vals['Height'+tag+'Simultaneous']=crossing(sim,level)
    vals['HeightPeakSimultaneousForce']=f'{force_from_field(sim).max():.0f}'
    vals['HeightPeakSelectiveForce']=f'{force_from_field(sel).max():.0f}'
    vals['HeightPeakSelective']=f'{z[int(np.argmax(sel))]:.1f}'
    vals['FieldForHundred']=f'{field_from_force(100.)*1e3:.1f}'
    vals['FieldForTwoFifty']=f'{field_from_force(250.)*1e3:.1f}'
    vals['ForceAtDesign']=f'{force_from_field(K.B_ROT):.0f}'
    d=np.array(design['offsets_mm'])
    for s_,tag in [('20','Twenty'),('40','Forty'),('60','Sixty')]:
        curve=force_from_field(design['offset_curves_selective_T'][s_])
        vals['Offset'+tag+'Zero']=f'{curve[0]:.0f}'
        vals['Offset'+tag+'Sixty']=f'{curve[int(np.argmin(abs(d-60)))]:.0f}'
    vals['PrimaryInteraction']=f'{data["placements"]["P1"]["capacity"]["interaction_field_bound_T"]*1e3:.3f}'
    vals['HoldField']=f'{data["runs"]["selective"]["summary"]["maximum_between_update_hold_field_T"]*1e3:.3f}'
    tex='% Generated from growing-rod publication records; all results are model-conditioned.\n'
    tex+=''.join('\\providecommand{\\csthree'+k+'}{'+v+'}\n' for k,v in vals.items())
    (OUT / 'case_study_3_metrics.tex').write_text(tex.rstrip() + '\n')
    sources=[Path(__file__),Path(M.__file__),Path(K.__file__),Path(magnetics.__file__),ROOT/'emns/workspace/fast.py',
             ROOT/'revised_case_studies/cs2_vessel_microspheres/model.py',
             ASSETS/'spine_posterior_illustration.png',ASSETS/'spine_prompt.md',
             ASSETS/'spine_posterior_curved.png',ASSETS/'spine_curved_prompt.md']
    manifest={'date':'2026-10-06','case':'Selective magnetic adjustment of scaled dual growing rods',
        'hardware_status':'No new hardware or clinical measurements.',
        'anatomical_illustration':{'asset':str((ASSETS/'spine_posterior_curved.png').relative_to(ROOT)),
            'provenance':'Built-in image_gen illustration, not a patient scan or scanned mesh.',
            'prompt_file':str((ASSETS/'spine_curved_prompt.md').relative_to(ROOT)),
            'shape':'Gentle lateral S curve; does not alter parallel actuator geometry or model calculations.',
            'model_use':'None; illustrative anatomical context only.'},
        'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'coil_parameters':np.asarray(params_list).tolist(),
        'settings':{k:v for k,v in vars(K).items() if k.isupper()},'validation':checks,
        'run_summary':{k:v['summary'] for k,v in data['runs'].items()},
        'probe_summary':{k:{m:r['summary'] for m,r in runs.items()} for k,runs in data['probes'].items()},
        'limitations':['Pointwise floating-point facet containment, no continuous placement boxes or outward rounding.',
        'Plotted zero radius includes placements with no robust zero-field task; it does not certify zero commands.',
        'All-phase moment discs enclose actual phase circles; current recourse assumes measured phases.',
        'Force-safe variant uses a uniformly derated current box: a sufficient lower bound, not optimized hybrid capacity.',
        'Primary motion arms use full 15 A field-only limits; they do not enforce the 0.15 N sensitivity bound.',
        'No thermal, power, slew, rotor inertia, backlash, axial motion of rotor centres, or fixture magnetization model.',
        'Fixed rotor centres and axes; constant bidirectional bench load and dry friction, no spinal tissue model.',
        '50 Hz null-field hold at allocation updates does not guarantee continuous null fields; inter-update values logged.',
        'P3 is an artificial small-spacing stress placement; coil height is not skin-to-implant depth.',
        'Reverse sequence concerns bench reversibility only; no clinical treatment claim.']}
    dump(OUT/'manifest.json',manifest)
    (OUT/'PROVENANCE.md').write_text('''# Case Study 3: growing-rod numerical preparation

Run from the repository root:
```
.venv/bin/python -B -m revised_case_studies.cs3_growing_rods.publication
.venv/bin/python -B -m revised_case_studies.cs3_growing_rods.publication --render
```
This replaces the main-paper helical-robot proposal; existing helical code/results
are preserved. All mechanism traces are simulations, not hardware or clinical data.
Rods, actuator schematic, labels and quantitative plots are generated by Python/Matplotlib.
Panel (a) embeds an AI-generated curved posterior spine illustration created with the built-in
image_gen tool. Its unchanged RGBA asset and full prompt set are stored in
revised_case_studies/cs3_growing_rods/assets/ and hashed in manifest.json.
The spine is illustrative context, not patient imaging, a scanned anatomical mesh,
measured phantom geometry, or an input to the magnetic/mechanical calculations.
The renderer preserves the anatomical illustration's aspect ratio and transparency.
The gentle lateral S curve is anatomical context only; rotor centres and axes remain
parallel in the schematic and numerical model. The previous straight asset is retained.

Four coil field outputs are selected in the planes normal to parallel y-axis rods.
Exact nominal zonotope facets test whole discs, never a sampled direction circle.
Selective amplitude is the smaller of the two rotate-and-exact-null-hold capacities.
The moment-disc offset enclosure covers every independent pair of rotor phases.
It yields a pointwise model-robust radius with phase-dependent current recourse.
All calculations use floating point. There is no interval placement-region proof.
The 15 mT contour is an illustrative design threshold, not measured B_req.
Negative radius margins are displayed as zero; this convention does not imply
that even the robust zero-field task is feasible at those placements.

The force-sensitive curve uses a sufficient uniform current derating based on
coil gradient operator norms and a bilinear mutual-force enclosure. It is a
lower bound on force-constrained capacity, not its exact optimum. The primary
mechanism runs do not impose this force bound. Rotor centres are fixed; force
predictions are reaction loads, not measured distraction force.

Two simulated comparisons use identical amplitude, main placement, load, timing,
and current bounds. Selective allocation tracks two local field commands; a held
rotor has exact null field at updates. If an active command is unattainable, its
field alone has residual while null hold remains hard. Common-field allocation
rotates both fields at the same phase/rate. Its extra motion is intentional under
that baseline and is unintended relative to the selective reference.

Gear ratio, efficiencies, screw pitch, resistance, friction and rotor moment are
illustrative. The screw model cannot backdrive because external load enters only
as a resisting torque. The reverse sequence is a bench check, not a clinical
instruction. Thermal performance and loaded clinical distraction are not modeled.

design_records.json holds the redesigned figure's design space: selective and
simultaneous radii versus height at s=40 mm, a spacing-by-height selective map,
and offset curves at z=55 mm. Panels (b)-(d) convert field radius r to a
quasi-static rod force F=2*pi*G*eta_g*eta_s*(m0*r-tau_f-2*pi*zeta*f_d)/p with the
illustrative mechanism; the 100-250 N band is a clinical reference scale only.
The previous figure, script and text are archived in archive_v1/.

publication_records.json contains map samples, all runs, phase/length references,
currents, commands, achieved fields and predicted forces. manifest.json records
parameters, hashes, solver-independent facet/hull/LP checks and evidence limits.
Replace the mechanism/current traces with repeated bench measurements and
recompute feasibility from measured moments, placements, phases and constraints.
''')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--render',action='store_true')
    args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
    if args.render:data=json.loads((OUT/'publication_records.json').read_text())
    else:
        data=compute();dump(OUT/'publication_records.json',data)
    checks=validate(data);export(data,checks);render(data)
    print('Generated Case Study 3 records, metrics, figure, and validation.',flush=True)


if __name__=='__main__':main()
