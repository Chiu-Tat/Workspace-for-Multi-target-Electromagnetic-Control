"""Case Study 2: matched vessel-pair simulations and a reproducible paper figure.

Run from the repository root:
  .venv/bin/python -B -m revised_case_studies.cs2_vessel_microspheres.publication
  .venv/bin/python -B -m revised_case_studies.cs2_vessel_microspheres.publication --render

Uses saved nominal TFW boxes and corridor plans. The new allocator holds the
map's total field and transverse bounds as hard constraints, with slack only
on the two tangential motion commands. No hardware measurements are generated.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import cvxpy as cp
import numpy as np
from scipy.optimize import linprog
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
from matplotlib.collections import PolyCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
from analysis.sections_67.common import CLASS_COLORS
from revised_case_studies.cs2_vessel_microspheres import config as K, tfw as T
from revised_case_studies.cs2_vessel_microspheres import model as M, vessel_pair as V

ROOT=Path(__file__).resolve().parents[2]
SAVED=ROOT/'revised_case_studies/results/cs2'
OUT=SAVED/'publication'
CAD=Path(__file__).with_name('cad')
VARIANTS=['side_by_side','over_under']
GEOMETRY_VARIANTS={v:f'branched_{v}' for v in VARIANTS}
BLUE,PURPLE,ORANGE,INK='#215f98','#755a9c','#bd5c31','#252a30'
# Vessel networks and their spheres (as in the Case 7 branched-network preview).
ARTERY,VEIN,SPHERE_GOLD='#c25362','#347da5','#ffd24b'
SPHERE_COLORS=(ARTERY,VEIN)            # sphere 1 in the artery, sphere 2 in the vein
GEOM_COLORS={'side_by_side':'#4a5866','over_under':'#9b6fb0'}
INNER,UNKNOWN=CLASS_COLORS['inside'],CLASS_COLORS['undetermined']
DT_CTRL,DT_PLANT,KP=.1,.005,2.0
V_CMD_MAX=.008                      # m/s, bound on the requested feedback velocity
V_FRICTION=2e-4                     # m/s, smoothing of the friction feedforward sign
FRICTION_ITERATIONS=6
WEIGHT=np.asarray(K.WEIGHT_VECTOR)
from matplotlib.colors import LinearSegmentedColormap
MARGIN_CMAP=LinearSegmentedColormap.from_list('margin',['#3b2f4a','#755a9c','#9fb4cf','#d9eadf','#8fd19e','#2f7d4a'])
SIDE_INDEX=[7,8,10,11]
FORCE_INDEX=[6,9]


def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Geometry:
    def __init__(self,variant):
        # The straight constructor does not rewrite or replan the saved mission.
        self.source_variant=GEOMETRY_VARIANTS[variant]
        self.pair=V.VesselPair(self.source_variant,plan='straight')
        self.lines=[T.AffineCenterline(self.pair.vessel,b) for b in ['RA','RV']]
        self.map=load(SAVED/f'tfw_{self.source_variant}.json')
        self.plan=load(SAVED/f'plan_{self.source_variant}.json')
        geometry_hash=sha(V.GEOM_DIR/V.VARIANTS[self.source_variant][0])
        if self.map.get('geometry_sha256') != geometry_hash or self.plan['key'].get('geometry_sha256') != geometry_hash:
            raise ValueError('Vessel geometry changed; regenerate TFW and corridor plan before publication.')
        expected={'B_T_along_y':K.B_HOLD,'F_req_N':T.F_REQ,'F_side_N':T.F_SIDE,
                  'I_limit_A':K.I_LIMIT,'moment_A_m2':K.MOMENT,'viscosity_Pa_s':K.VISCOSITY}
        for key,value in expected.items():
            if not math.isclose(self.map['requirement'][key],value,rel_tol=1e-12):
                raise ValueError(f'Saved TFW task differs from publication setting {key}.')
        self.lengths=np.array([r.length for r in self.pair.routes])
        self.u=np.tile(T.Y_HAT,(2,1))

    def s(self,a):
        return np.array([rt.s_of_a(v) for rt,v in zip(self.pair.routes,a)])

    def pose(self,a):
        ss=self.s(a)
        return np.array([line.point(s) for line,s in zip(self.lines,ss)]),[line.frame(s) for line,s in zip(self.lines,ss)]

    def rows(self,a):
        p,frames=self.pose(a)
        G,c=M.allocation_rows(p,self.u,frames,interaction=True)
        return p,frames,G,c

    def model_in(self,a):
        s1,s2=self.s(a)
        return any(b[0]<=s1<=b[1] and b[2]<=s2<=b[3] for b in self.map['inside'])


def lp(G,c,force):
    idx=list(range(6))+FORCE_INDEX
    scales=np.array([K.B_HOLD]*6+[T.F_REQ]*2)
    desired=np.array([K.B_HOLD,0,0]*2+list(force))
    side=G[SIDE_INDEX]/T.F_SIDE; off=c[SIDE_INDEX]/T.F_SIDE
    r=linprog(np.zeros(10),A_eq=G[idx]/scales[:,None],b_eq=(desired-c[idx])/scales,
              A_ub=np.vstack([side,-side]),b_ub=np.r_[1-off,1+off],
              bounds=[(-K.I_LIMIT,K.I_LIMIT)]*10,method='highs')
    if r.status==0: return 'feasible'
    if r.status==2: return 'infeasible'
    raise RuntimeError(f'Unresolved LP status {r.status}: {r.message}')


def full_task(G,c):
    return 'feasible' if all(lp(G,c,[f1,f2])=='feasible'
                            for f1 in [-T.F_REQ,T.F_REQ] for f2 in [-T.F_REQ,T.F_REQ]) else 'infeasible'


def force_margin(G,c):
    """Largest lambda such that all four (+-lambda F_REQ, +-lambda F_REQ) tangential
    corners are attainable with both total fields exact, all transverse
    components within F_SIDE, and |I| <= I_LIMIT.  lambda < 1 means the full task is
    infeasible at this configuration (same constraints as ``full_task``)."""
    idx=list(range(6))+FORCE_INDEX
    scales=np.array([K.B_HOLD]*6+[T.F_REQ]*2)
    fixed=np.array([K.B_HOLD,0,0]*2+[0,0])
    side=G[SIDE_INDEX]/T.F_SIDE; off=c[SIDE_INDEX]/T.F_SIDE
    A_ub=np.hstack([np.vstack([side,-side]),np.zeros((8,1))]); b_ub=np.r_[1-off,1+off]
    lam=math.inf
    for f1 in (-1.,1.):
        for f2 in (-1.,1.):
            direction=np.r_[np.zeros(6),f1,f2]
            A_eq=np.hstack([G[idx]/scales[:,None],-direction[:,None]])
            r=linprog(np.r_[np.zeros(10),-1.],A_eq=A_eq,b_eq=(fixed-c[idx])/scales,A_ub=A_ub,b_ub=b_ub,
                      bounds=[(-K.I_LIMIT,K.I_LIMIT)]*10+[(0,None)],method='highs')
            if r.status==2: return 0.
            if r.status!=0: raise RuntimeError(f'Margin LP status {r.status}: {r.message}')
            lam=min(lam,-r.fun)
    return float(lam)


_MARGIN_GEOMETRY=None


def _margin_init(variant):
    """Worker setup; also works where processes are spawned rather than forked."""
    global _MARGIN_GEOMETRY
    if _MARGIN_GEOMETRY is None or _MARGIN_GEOMETRY.pair.variant!=GEOMETRY_VARIANTS[variant]:
        _MARGIN_GEOMETRY=Geometry(variant)


def _margin_row(args):
    """Margins along one s1 column, with the same rows as the run diagnostics."""
    s1,grid=args; g=_MARGIN_GEOMETRY
    out=[]
    for s2 in grid:
        a=np.array([rt.a_of_s(v) for rt,v in zip(g.pair.routes,(s1,s2))])
        out.append(force_margin(*g.rows(a)[2:]))
    return out


def margin_data(data,geometries):
    """Force-margin maps on the stored pointwise grid and along every run (cached)."""
    global _MARGIN_GEOMETRY
    path=OUT/'force_margin.json'
    key={v:sha(SAVED/f'tfw_{GEOMETRY_VARIANTS[v]}.json') for v in VARIANTS}
    key['records']=hashlib.sha256(json.dumps({v:{m:data[v][m]['summary'] for m in data[v]} for v in VARIANTS},
                                             sort_keys=True).encode()).hexdigest()
    if path.exists():
        cached=load(path)
        if cached.get('key')==key: return cached
    import multiprocessing as mp
    out={'key':key,'definition':'Largest scale of the +-F_max tangential task with all four corners attainable; '
         'fields exact, transverse bounds and current bounds enforced. lambda<1 iff full-task LP infeasible.',
         'maps':{},'runs':{},'consistency':{}}
    for v in VARIANTS:
        g=geometries[v]; grid=g.map['pointwise']['grid']
        _MARGIN_GEOMETRY=g
        method='fork' if 'fork' in mp.get_all_start_methods() else 'spawn'
        with mp.get_context(method).Pool(min(6,mp.cpu_count()),initializer=_margin_init,initargs=(v,)) as pool:
            lam=pool.map(_margin_row,[(s1,grid) for s1 in grid])   # [s1][s2]
        lam=np.array(lam); feas=np.array(g.map['pointwise']['feasible_s1_s2'],bool)
        out['maps'][v]={'grid':grid,'lambda_s1_s2':np.round(lam,5).tolist()}
        out['consistency'][v]={'grid_points':int(feas.size),
            'disagreements_with_pointwise_LP':int(np.sum((lam>=1-1e-6)!=feas)),
            'minimum':float(lam.min()),'fraction_below_one':float(np.mean(lam<1))}
        out['runs'][v]={}
        for m in ['coordinated','direct']:
            rows=data[v][m]['records']
            out['runs'][v][m]={'t':[r['t'] for r in rows],
                               'lambda':[round(force_margin(*g.rows(np.array(r['a']))[2:]),5) for r in rows]}
        print(v,'margin',out['consistency'][v],flush=True)
    dump(path,out)
    return out


def margin_map_panel(ax,g,runs,margins,variant,letter,title):
    mp_=margins['maps'][variant]; grid=np.array(mp_['grid']); lam=np.array(mp_['lambda_s1_s2'])
    mesh=ax.pcolormesh(grid,grid,lam.T,cmap=MARGIN_CMAP,vmin=0,vmax=3,shading='nearest',rasterized=True)
    ax.contour(grid,grid,lam.T,levels=[1],colors=['#a64242'],linewidths=1.0)
    for name,color in [('coordinated',INK),('direct',ORANGE)]:
        s_=np.array([r['s'] for r in runs[name]['records']])
        ax.plot(s_[:,0],s_[:,1],color=color,lw=1.1,zorder=5)
    mission=np.asarray(runs['coordinated']['reference'])
    for point,marker in [(g.s(mission[0,1:]),'o'),(g.s(mission[-1,1:]),'s')]:
        ax.plot(*point,marker,ms=3.5,mfc='white',mec=INK,zorder=6)
    ax.set(xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1],xlabel='$s_1$',ylabel='$s_2$')
    ax.set_aspect('equal')
    ax.set_title(f'({letter}) {title}',loc='left',pad=4)
    below=np.mean(lam<1)*100
    ax.text(.025,.025,f'$\\lambda<1$ on {below:.1f}%',transform=ax.transAxes,fontsize=6,
            bbox={'fc':'white','ec':'none','alpha':.9,'pad':1})
    return mesh


def margin_time_panel(ax,margins,letter):
    for v,color in GEOM_COLORS.items():
        for m,ls in [('coordinated','-'),('direct','--')]:
            r=margins['runs'][v][m]; ax.plot(r['t'],r['lambda'],color=color,ls=ls,lw=1)
    ax.axhspan(0,1,fc='#a64242',alpha=.08,lw=0)
    ax.axhline(1,color='#a64242',lw=.8)
    ax.text(.98,.04,'full task infeasible ($\\lambda<1$)',transform=ax.transAxes,ha='right',fontsize=5.8,color='#a64242')
    maximum=max(max(run['lambda']) for methods in margins['runs'].values() for run in methods.values())
    upper=max(2,math.ceil(maximum*1.04))
    ax.set(ylim=(0,upper),yticks=range(upper+1),xlabel='Time (s)',ylabel='Force margin $\\lambda$')
    ax.set_xlim(0,max(max(margins['runs'][v]['coordinated']['t']) for v in VARIANTS))
    ax.set_title(f'({letter}) Margin along the runs',loc='left',pad=4)
    ax.text(.02,.99,'Slate: side by side\nPurple: over-under',transform=ax.transAxes,
            fontsize=5.6,va='top',ha='left',bbox={'fc':'white','ec':'none','alpha':.85,'pad':1})


class Allocator:
    """Hard total-field/side bounds; only tangential forces may have residual."""
    def __init__(self):
        self.i=cp.Variable(10)
        self.Gb=cp.Parameter((6,10)); self.rb=cp.Parameter(6)
        self.Gf=cp.Parameter((2,10)); self.rf=cp.Parameter(2)
        self.Gs=cp.Parameter((4,10)); self.cs=cp.Parameter(4)
        objective=cp.sum_squares(self.Gf@self.i-self.rf)+1e-7*cp.sum_squares(self.i)
        self.problem=cp.Problem(cp.Minimize(objective),[
            self.Gb@self.i==self.rb,self.i>=-K.I_LIMIT,self.i<=K.I_LIMIT,
            self.Gs@self.i+self.cs<=1,self.Gs@self.i+self.cs>=-1])

    def solve(self,G,c,force):
        self.Gb.value=G[:6]/K.B_HOLD
        self.rb.value=(np.array([K.B_HOLD,0,0]*2)-c[:6])/K.B_HOLD
        self.Gf.value=G[FORCE_INDEX]/T.F_REQ
        self.rf.value=(force-c[FORCE_INDEX])/T.F_REQ
        self.Gs.value=G[SIDE_INDEX]/T.F_SIDE
        self.cs.value=c[SIDE_INDEX]/T.F_SIDE
        self.problem.solve(solver='CLARABEL',warm_start=True,tol_gap_abs=1e-9,tol_feas=1e-9)
        if self.problem.status not in ['optimal','optimal_inaccurate'] or self.i.value is None:
            return None,self.problem.status
        current=np.clip(np.asarray(self.i.value),-K.I_LIMIT,K.I_LIMIT)
        achieved=G@current+c
        field_err=np.max(np.abs(achieved[:6]-np.array([K.B_HOLD,0,0]*2)))/K.B_HOLD
        if field_err>1e-6 or np.max(np.abs(achieved[SIDE_INDEX]))>T.F_SIDE*(1+1e-6):
            raise RuntimeError('Allocator violates a hard field or side constraint.')
        return current,self.problem.status


def smooth_segment(start,end,duration,t0=0):
    t=np.linspace(0,duration,int(math.ceil(duration/.025))+1)
    u=t/duration
    z=10*u**3-15*u**4+6*u**5
    return np.column_stack([t+t0,np.asarray(start)[None,:]+z[:,None]*(np.asarray(end)-start)])


def missions(g):
    coordinated=np.asarray(g.plan['mission'],float).copy()
    # Quintic straight travel may need longer than the corridor reference.
    # Slow both arms equally so endpoints, duration and limits remain matched.
    span=float(np.max(np.abs(coordinated[-1,1:]-coordinated[0,1:])))
    minimum=max(1.875*span/V.SPEED,math.sqrt((10/math.sqrt(3))*span/V.ACCEL))
    duration=max(float(coordinated[-1,0]),minimum*(1+1e-8))
    coordinated[:,0]*=duration/coordinated[-1,0]
    direct=smooth_segment(coordinated[0,1:],coordinated[-1,1:],duration)
    return {'coordinated':coordinated,'direct':direct}


def independent_mission(g):
    start=np.asarray(g.plan['mission'][0][1:]); a1=start+np.array([.004,0]); a2=a1-np.array([0,.004])
    chunks=[np.column_stack([np.arange(0,1.001,.025),np.tile(start,(41,1))])]
    chunks.append(smooth_segment(start,a1,4,1)[1:])
    chunks.append(np.column_stack([np.arange(5.025,6.001,.025),np.tile(a1,(40,1))]))
    chunks.append(smooth_segment(a1,a2,4,6)[1:])
    chunks.append(np.column_stack([np.arange(10.025,11.001,.025),np.tile(a2,(40,1))]))
    return np.vstack(chunks)


def ref(mission,t):
    a=np.array([np.interp(t,mission[:,0],mission[:,k]) for k in [1,2]])
    # Use a one-sided derivative at the endpoints, zero beyond the mission.
    if t<0 or t>=mission[-1,0]: return a,np.zeros(2)
    lo=max(0,t-.025); hi=min(mission[-1,0],t+.025)
    da=np.array([np.interp(hi,mission[:,0],mission[:,k])-np.interp(lo,mission[:,0],mission[:,k])
                 for k in [1,2]])/(hi-lo)
    return a,da


def validate_mission(g,mission,name):
    time_=mission[:,0]
    v=np.diff(mission[:,1:],axis=0)/np.diff(time_)[:,None]
    mid=(time_[1:]+time_[:-1])/2
    accel=np.diff(v,axis=0)/np.diff(mid)[:,None]
    vm=np.max(np.abs(v),axis=0); am=np.max(np.abs(accel),axis=0)
    if np.max(vm)>V.SPEED*(1+1e-4) or np.max(am)>V.ACCEL*(1+2e-3):
        raise RuntimeError(f'{name} exceeds motion limits: {vm}, {am}')
    labels=[g.model_in(row[1:]) for row in mission]
    if name in ['coordinated','independent'] and not all(labels):
        raise RuntimeError(f'{name} leaves the saved nominal TFW-in boxes.')
    return {'max_reference_speed_m_s':vm.tolist(),'max_reference_acceleration_m_s2':am.tolist(),
            'reference_samples':len(mission),'reference_model_in_fraction':float(np.mean(labels))}


def wall_contact(F,frame):
    """Net tangential force (magnetic + weight) and wall normal load for one sphere."""
    t,n,b=(np.asarray(e) for e in frame)
    total=np.asarray(F)+WEIGHT
    return float(total@t),float(math.hypot(total@n,total@b))


def friction_velocity(f_net,normal):
    """Quasi-static sliding: stick while |f_net| <= mu N, else wall-corrected drag."""
    excess=abs(f_net)-K.MU_WALL*normal
    return 0. if excess<=0 else math.copysign(excess,f_net)/K.DRAG_WALL


def command_forces(alloc,G,c,frames,requested_v,side_guess):
    """Magnetic tangential commands that overcome friction and weight.

    The wall load depends on the allocated side forces, so the friction
    feedforward and the allocation are iterated to a fixed point."""
    side=np.asarray(side_guess,float)
    for _ in range(FRICTION_ITERATIONS):
        force=np.zeros(2)
        for k in [0,1]:
            t,n,b=(np.asarray(e) for e in frames[k])
            normal=math.hypot(side[2*k]+WEIGHT@n,side[2*k+1]+WEIGHT@b)
            force[k]=(K.DRAG_WALL*requested_v[k]+K.MU_WALL*normal*math.tanh(requested_v[k]/V_FRICTION)
                      -WEIGHT@t)
        force=np.clip(force,-T.F_REQ,T.F_REQ)
        current,status=alloc.solve(G,c,force)
        if current is None: return force,None,status
        new_side=(G@current+c)[SIDE_INDEX]
        if np.max(np.abs(new_side-side))<1e-7: break
        side=new_side
    return force,current,status


def simulate(g,mission,name):
    validation=validate_mission(g,mission,name)
    a=mission[0,1:].copy(); alloc=Allocator(); records=[]; abort=None
    end=math.ceil((mission[-1,0]+2)/DT_CTRL)*DT_CTRL
    between={'maximum_field_error_T':0.,'maximum_transverse_component_N':0.,
             'minimum_alignment_cosine':1.,'minimum_separation_m':math.inf,
             'maximum_normal_load_N':0.}
    side=np.zeros(4)
    for t in np.arange(0,end+DT_CTRL*.5,DT_CTRL):
        ar,vr=ref(mission,t)
        p,frames,G,c=g.rows(a)
        requested_v=np.clip(vr+KP*(ar-a),-V_CMD_MAX,V_CMD_MAX)
        force,current,status=command_forces(alloc,G,c,frames,requested_v,side)
        if current is None:
            abort={'time_s':float(t),'reason':'Hard holding-field/transverse-force problem infeasible.','status':status}
            break
        achieved=G@current+c; side=achieved[SIDE_INDEX]
        m,B,F=M.plant(p,current,interaction=True)
        contact=[wall_contact(F[k],frames[k]) for k in [0,1]]
        records.append({'t':float(t),'a':a.tolist(),'s':g.s(a).tolist(),'a_ref':ar.tolist(),
                        'error':(a-ar).tolist(),'positions':p.tolist(),'currents':current.tolist(),
                        'force_command':force.tolist(),'force_allocated':achieved[FORCE_INDEX].tolist(),
                        'force_residual_scaled':float(np.linalg.norm((achieved[FORCE_INDEX]-force)/T.F_REQ)),
                        'allocated_field_error_T':float(np.max(np.abs(achieved[:6]-np.array([K.B_HOLD,0,0]*2)))),
                        'allocated_side_force':achieved[SIDE_INDEX].tolist(),
                        'plant_field':B.tolist(),'plant_force':F.tolist(),
                        'normal_load_N':[n for _,n in contact],
                        'friction_limit_N':[K.MU_WALL*n for _,n in contact],
                        'separation_m':float(np.linalg.norm(p[0]-p[1])),
                        'model_in':g.model_in(a),'full_task_class':full_task(G,c),
                        'command_class':lp(G,c,force),'solver_status':status})
        if t>=end-DT_CTRL*.25: break
        for _ in range(round(DT_CTRL/DT_PLANT)):
            pp,fr=g.pose(a); mm,bb,ff=M.plant(pp,current,interaction=True)
            if np.linalg.norm(pp[0]-pp[1])<K.MIN_SEPARATION:
                abort={'time_s':float(t),'reason':'Particle-separation limit reached.'}; break
            between['maximum_field_error_T']=max(between['maximum_field_error_T'],float(np.max(np.abs(bb-K.B_HOLD*T.Y_HAT))))
            between['maximum_transverse_component_N']=max(between['maximum_transverse_component_N'],
                max(abs(float(ff[k]@direction)) for k in [0,1] for direction in fr[k][1:]))
            between['minimum_alignment_cosine']=min(between['minimum_alignment_cosine'],float(np.min(mm@T.Y_HAT/K.MOMENT)))
            between['minimum_separation_m']=min(between['minimum_separation_m'],float(np.linalg.norm(pp[0]-pp[1])))
            contact=[wall_contact(ff[k],fr[k]) for k in [0,1]]
            between['maximum_normal_load_N']=max(between['maximum_normal_load_N'],max(n for _,n in contact))
            v=np.array([friction_velocity(f,n) for f,n in contact])
            a=np.clip(a+DT_PLANT*v,0,g.lengths)
        if abort: break
        if len(records)%100==0: print(f'{g.pair.variant}/{name}: {t:.1f} s',flush=True)
    if not records: raise RuntimeError(f'{name}: no valid records.')
    active=[r for r in records if r['t']<=mission[-1,0]+1e-9]
    errors=np.array([r['error'] for r in active]); goal=np.asarray(records[-1]['a'])-mission[-1,1:]
    summary={'reference_duration_s':float(mission[-1,0]),'recorded_duration_s':records[-1]['t'],
             'active_interval_s':[0,float(active[-1]['t'])],
             'position_rmse_mm':(np.sqrt(np.mean(errors**2,axis=0))*1e3).tolist(),
             'final_goal_error_mm':(np.abs(goal)*1e3).tolist(),
             'completed':bool(abort is None and np.max(np.abs(goal))<=.0005),
             'model_in_sample_fraction':float(np.mean([r['model_in'] for r in active])),
             'full_task_infeasible_samples':sum(r['full_task_class']=='infeasible' for r in active),
             'command_infeasible_samples':sum(r['command_class']=='infeasible' for r in active),
             'active_samples':len(active),'peak_current_A':float(max(max(abs(i) for i in r['currents']) for r in records)),
             'saturated_sample_fraction':float(np.mean([max(abs(i) for i in r['currents'])>14.999 for r in active])),
             'minimum_separation_mm':float(between['minimum_separation_m']*1e3),
             'maximum_allocated_field_error_T':max(r['allocated_field_error_T'] for r in records),
             'maximum_allocated_transverse_component_N':max(max(abs(f) for f in r['allocated_side_force']) for r in records),
             'maximum_scaled_force_residual':max(r['force_residual_scaled'] for r in records),
             'maximum_normal_load_N':max(max(r['normal_load_N']) for r in records),
             'clipped_command_samples':sum(max(abs(f) for f in r['force_command'])>=T.F_REQ*(1-1e-9) for r in active),
             'abort':abort,'between_updates':between,**validation}
    if name=='independent':
        def hold_error(index,lo,hi):
            rr=[r for r in records if lo<=r['t']<=hi]
            return max(abs(r['error'][index]) for r in rr)*1e3
        summary['sphere2_hold_error_mm']=hold_error(1,1,5)
        summary['sphere1_hold_error_mm']=hold_error(0,6,10)
    return {'reference':mission.tolist(),'records':records,'summary':summary}


def audit(g):
    rng=np.random.default_rng(42); checked=0; disagreement=[]
    for idx in rng.choice(len(g.map['inside']),min(32,len(g.map['inside'])),replace=False):
        x0,x1,y0,y1=g.map['inside'][idx]
        for s1,s2 in [((x0+x1)/2,(y0+y1)/2),(x0,y0),(x1,y1)]:
            if not T.point_feasible(g.lines,s1,s2): disagreement.append([s1,s2])
            checked+=1
    if disagreement: raise RuntimeError(f'Nominal TFW-in/point LP disagreement: {disagreement}')
    matrix_error=0.; offset_error=0.
    for s1,s2 in [(.2,.8),(.5,.5),(.8,.2)]:
        positions=np.array([g.lines[0].point(s1),g.lines[1].point(s2)])
        frames=[g.lines[0].frame(s1),g.lines[1].frame(s2)]
        G,c=M.allocation_rows(positions,g.u,frames)
        full,side=T.affine_maps(g.lines,((s1,s1),(s2,s2)))
        A=np.array([[e.center for e in row] for row in full])
        As=np.array([[e.center for e in row] for row in side])
        idx=[2,0,1,6,5,3,4,9]; signs=np.array([-1,1,-1,1,-1,1,-1,1])
        matrix_error=max(matrix_error,float(np.max(np.abs(A[:,:10]-G[idx]*signs[:,None]*T.SCALE_CTRL[:,None]))))
        offset_error=max(offset_error,float(np.max(np.abs(A[:,10]-c[idx]*signs*T.SCALE_CTRL))))
        matrix_error=max(matrix_error,float(np.max(np.abs(As[:,:10]-G[SIDE_INDEX]*T.SCALE_SIDE[:,None]))))
        offset_error=max(offset_error,float(np.max(np.abs(As[:,10]-c[SIDE_INDEX]*T.SCALE_SIDE))))
    if max(matrix_error,offset_error)>1e-8: raise RuntimeError('Point and affine allocation maps disagree.')
    return {'inside_samples':checked,'inside_point_contradictions':len(disagreement),
            'maximum_scaled_matrix_difference':matrix_error,'maximum_scaled_offset_difference':offset_error,
            'scope':'Numerical consistency, not continuous enclosure verification or hardware validation.'}


def allocator_check(g,run):
    """Check attainable and unattainable commands at one recorded model pose."""
    bad=[r for r in run['records'] if r['full_task_class']=='infeasible']
    r=bad[len(bad)//2] if bad else run['records'][len(run['records'])//2]
    _,_,G,c=g.rows(r['a'])
    # When the prescribed full task is feasible, probe beyond its LP capacity.
    # Record that distinction rather than inventing a failing task endpoint.
    scale=1. if bad else 1.1*force_margin(G,c)
    corner=next(np.array([f1,f2]) for f1 in [-scale*T.F_REQ,scale*T.F_REQ]
                for f2 in [-scale*T.F_REQ,scale*T.F_REQ] if lp(G,c,[f1,f2])=='infeasible')
    witness_name='unattainable_task_corner' if bad else 'unattainable_force_probe_beyond_full_task'
    allocator=Allocator(); results=[]
    for name,target in [('recorded_motion_command',np.array(r['force_command'])),
                        (witness_name,corner)]:
        current,status=allocator.solve(G,c,target)
        if current is None: raise RuntimeError('Allocator validation unexpectedly lost the hard constraints.')
        h=G@current+c
        error=float(np.linalg.norm((h[FORCE_INDEX]-target)/T.F_REQ))
        classification=lp(G,c,target)
        if classification=='feasible' and error>1e-3:
            raise RuntimeError('Reachable validation command has excessive allocation residual.')
        if classification=='infeasible' and error<1e-3:
            raise RuntimeError('Interior infeasible task witness lacks the expected force residual.')
        results.append({'condition':name,'force_command_N':target.tolist(),'command_class':classification,
            'scaled_force_residual':error,'maximum_field_error_T':float(np.max(np.abs(h[:6]-np.array([K.B_HOLD,0,0]*2)))),
            'maximum_transverse_component_N':float(np.max(np.abs(h[SIDE_INDEX]))),'solver_status':status})
    data={'time_s':r['t'],'pose_a':r['a'],'results':results,
          'logged_pose_full_task_class':r['full_task_class'],'probe_force_scale':scale,
          'scope':'Numerical allocator behavior at a logged direct-arm pose; the recorded command may itself be LP-infeasible there. Not hardware validation.'}
    dump(OUT/'allocator_validation.json',data)
    return data


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':7.5,
        'axes.titlesize':7.2,'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.5,
        'axes.spines.top':False,'axes.spines.right':False,'axes.edgecolor':'#777b80',
        'text.color':INK,'axes.labelcolor':INK,'pdf.fonttype':42,'savefig.dpi':250})


def _mix(color,target,f):
    c=np.array(to_rgba(color)[:3]); t=np.array(to_rgba(target)[:3])
    return tuple(c+(t-c)*f)


def _band(pts,normal,radius,lo,hi):
    """Polygon between offsets lo*r and hi*r from the centreline (data units)."""
    return np.vstack([pts+hi*radius[:,None]*normal,(pts+lo*radius[:,None]*normal)[::-1]])


def vessel_mesh(g):
    """Top-view triangles from the validated current hollow STEP models."""
    light=np.array([-.35,.45,.82]); light/=np.linalg.norm(light)
    tris,shade,depth,net=[],[],[],[]
    for network,(xyz,faces,color) in zip(('artery','vein'),cad_vessel_meshes(g)):
        tri=xyz[faces]
        nrm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        nrm/=np.maximum(np.linalg.norm(nrm,axis=1)[:,None],1e-12)
        tris.append(tri[:,:,:2]); shade.append(.58+.42*np.abs(nrm@light))
        depth.append(tri[:,:,2].mean(axis=1)); net+= [network]*len(tri)
    return np.concatenate(tris),np.concatenate(shade),np.concatenate(depth),np.array(net)


def draw_vessels(ax,g,zorder=2):
    """Shaded 3-D lumen meshes seen from above (+z), painted deepest first."""
    tri,shade,depth,net=vessel_mesh(g)
    base={'artery':np.array(to_rgba(ARTERY)[:3]),'vein':np.array(to_rgba(VEIN)[:3])}
    rgb=np.array([base[n] for n in net])*shade[:,None]
    order=np.argsort(depth)
    ax.add_collection(PolyCollection(tri[order],facecolors=np.column_stack([rgb[order],np.ones(len(order))]),
                                     edgecolors='none',antialiased=False,rasterized=True,zorder=zorder))


def sphere_marker(ax,xy,r=K.RADIUS*1e3,zorder=6):
    """Enlarged gold microsphere marker (data units), as in the Case 7 preview."""
    ax.add_patch(Circle(xy,r,fc=SPHERE_GOLD,ec='white',lw=.8,zorder=zorder))
    ax.add_patch(Circle(xy,r,fc='none',ec='#5b4a12',lw=.35,zorder=zorder+.1))


def geometry_panel(ax,g,letter,title):
    draw_vessels(ax,g)
    for k,(line,rt,color) in enumerate(zip(g.lines,g.pair.routes,SPHERE_COLORS)):
        start=np.asarray(g.plan['mission'][0][1:]); ss=g.s(start)
        p=line.point(ss[k])*1e3
        sphere_marker(ax,p[:2],zorder=8)
        tangent=line.frame(ss[k])[0][:2]*(-1 if k==1 else 1)
        ax.annotate('',xy=p[:2]+8*tangent,xytext=p[:2]+2*tangent,zorder=9,
                    arrowprops={'arrowstyle':'-|>','color':INK,'lw':.9,'mutation_scale':8})
        offset=(-28,10) if k==0 else (6,4)
        ax.annotate(f'Sphere {k+1}',p[:2],xytext=offset,textcoords='offset points',fontsize=6,color=color,zorder=9,
                    bbox={'fc':'white','ec':'none','alpha':.85,'pad':.6})
    ax.set(xlim=(-48,48),ylim=(-58,58),xticks=[-40,0,40],yticks=[-50,0,50],xlabel='$x$ (mm)',ylabel='$y$ (mm)')
    ax.set_aspect('equal')
    ax.set_title(f'({letter}) {title}',loc='left',pad=4)
    ax.text(.64,.035,f'Closest trunks\n{g.pair.d_close*1e3:.2f} mm',transform=ax.transAxes,fontsize=5.8,
            bbox={'fc':'white','ec':'none','alpha':.9,'pad':1})
    ax.text(.03,.97,'Artery',transform=ax.transAxes,fontsize=6,color=ARTERY,va='top')
    ax.text(.03,.88,'Vein',transform=ax.transAxes,fontsize=6,color=VEIN,va='top')


def cad_vessel_meshes(g):
    """Validated CAD triangles in mm relative to the coil-array centre."""
    import cadquery as cq
    metadata=load(CAD/'validation.json')['models'][g.pair.variant.removeprefix('branched_')]
    source=V.GEOM_DIR/V.VARIANTS[g.source_variant][0]
    step=CAD/metadata['step_file']
    if sha(source)!=metadata['source_sha256'] or sha(step)!=metadata['step_sha256']:
        raise ValueError('CAD geometry differs from its validated source; regenerate the STEP models.')
    bodies=cq.importers.importStep(str(step)).val().Solids()
    if len(bodies)!=2 or not all(body.isValid() for body in bodies):
        raise ValueError('The vessel STEP assembly must contain two valid hollow pipe solids.')
    # Remove the global z registration for local, readable scene coordinates.
    center=np.asarray(g.pair.center)*1e3
    meshes=[]
    for body,color in zip(bodies,[ARTERY,VEIN]):
        vertices,faces=body.tessellate(.04,.15)
        xyz=np.array([v.toTuple() for v in vertices])-center
        meshes.append((xyz,np.array(faces),color))
    return meshes


def geometry_panel_3d(ax,g,letter,title,footer=True):
    """Orthographic, true-proportion hollow CAD pipe view for publication panels.

    CadQuery is only needed for this renderer; simulation code is unaffected.
    The STEP solids supply joined branches and spherical terminal caps.
    Only surface shading is rasterized within these geometry views.
    """
    meshes=cad_vessel_meshes(g)
    center=np.asarray(g.pair.center)*1e3
    for xyz,faces,color in meshes:
        ax.add_collection3d(Poly3DCollection(xyz[faces],facecolors=color,
            linewidths=0,shade=True,antialiased=True,rasterized=True,
            lightsource=matplotlib.colors.LightSource(azdeg=315,altdeg=45),zorder=2))
    xyz=np.concatenate([mesh[0] for mesh in meshes]); lo=xyz.min(axis=0)-2; hi=xyz.max(axis=0)+2
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),
           xticks=[-40,-20,0,20,40],yticks=[-40,-20,0,20,40],zticks=[-5,0,5])
    ax.set_box_aspect(hi-lo,zoom=1.02)
    ax.set_proj_type('ortho')
    ax.view_init(elev=28,azim=-65)
    # Light millimetre grid on the three back panes.
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.set_pane_color((1,1,1,0))
        axis.pane.set_edgecolor('#c9cdd2')
        axis._axinfo['grid'].update(color='#d5d8dc',linewidth=.4)
        axis.line.set_linewidth(.5); axis.line.set_color('#9a9fa5')
    ax.grid(True)
    ax.tick_params(labelsize=4.8,pad=-3,length=0)
    ax.set_xlabel('$x$ (mm)',fontsize=5.5,labelpad=-9)
    ax.set_ylabel('$y$ (mm)',fontsize=5.5,labelpad=-9)
    # The z span is only about 16 mm; its grid lines stay, unlabelled, so the
    # narrow panel keeps clear of the neighbouring top view.
    ax.set_zticklabels([])
    ax.zaxis._axinfo['tick'].update(inward_factor=0,outward_factor=0)
    ax.set_yticklabels(['$-40$','','0','','40'])
    ax.set_title(f'({letter}) {title}',loc='left',pad=4)
    ax.text2D(.015,.98,'Artery',transform=ax.transAxes,fontsize=6,color=ARTERY,va='top')
    ax.text2D(.015,.88,'Vein',transform=ax.transAxes,fontsize=6,color=VEIN,va='top')
    start=np.asarray(g.plan['mission'][0][1:]); ss=g.s(start)
    projection=ax.get_proj()
    for k,(line,color) in enumerate(zip(g.lines,SPHERE_COLORS)):
        p=line.point(ss[k])*1e3-center
        tangent=line.frame(ss[k])[0]*(-1 if k==1 else 1)
        # Spheres and arrows are schematic overlays so they stay visible inside
        # the opaque lumen-volume solids, like the original enlarged markers.
        xp,yp,_=proj3d.proj_transform(*p,projection)
        ax.add_artist(Line2D([xp],[yp],marker='o',ms=4.1,mfc=SPHERE_GOLD,mec='#5b4a12',mew=.45,
                            ls='',zorder=8,transform=ax.transData))
        x0,y0,_=proj3d.proj_transform(*(p+2*tangent),projection)
        x1,y1,_=proj3d.proj_transform(*(p+8*tangent),projection)
        ax.annotate('',xy=(x1,y1),xytext=(x0,y0),zorder=9,
                    arrowprops={'arrowstyle':'-|>','color':INK,'lw':.8,'mutation_scale':7})
        offset=(-24,-9) if k==0 else (4,5)
        ax.annotate(f'Sphere {k+1}',(xp,yp),xytext=offset,textcoords='offset points',
                    fontsize=5.8,color=color,zorder=9,
                    bbox={'fc':'white','ec':'none','alpha':.9,'pad':.5})
    if footer:
        ax.text2D(.04,-.13,f'Trunk centreline separation: {g.pair.d_close*1e3:.2f} mm',
                  transform=ax.transAxes,fontsize=5.5)


def geometry_panel_top(ax,g):
    """Equal-scale +z projection of the same CAD bodies used in the 3D view."""
    triangles=[]; colors=[]; depths=[]
    light=matplotlib.colors.LightSource(azdeg=315,altdeg=45).direction
    for xyz,faces,color in cad_vessel_meshes(g):
        tri=xyz[faces]
        normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-12)
        shade=.65+.35*(normal@light)
        rgb=np.asarray(to_rgba(color)[:3])*shade[:,None]
        triangles.append(tri[:,:,:2]); colors.append(rgb); depths.append(tri[:,:,2].mean(axis=1))
    order=np.argsort(np.concatenate(depths))
    ax.add_collection(PolyCollection(np.concatenate(triangles)[order],
        facecolors=np.concatenate(colors)[order],edgecolors='none',linewidths=0,
        antialiased=False,rasterized=True,zorder=2))
    center=np.asarray(g.pair.center)*1e3
    start=np.asarray(g.plan['mission'][0][1:]); ss=g.s(start)
    for k,(line,color) in enumerate(zip(g.lines,SPHERE_COLORS)):
        p=line.point(ss[k])*1e3-center
        sphere_marker(ax,p[:2],r=K.RADIUS*1e3,zorder=8)
        t=line.frame(ss[k])[0][:2]*(-1 if k==1 else 1)
        ax.annotate('',xy=p[:2]+7*t,xytext=p[:2]+2*t,zorder=9,
                    arrowprops={'arrowstyle':'-|>','color':INK,'lw':.7,'mutation_scale':6})
        ax.annotate(str(k+1),p[:2],xytext=(-9,-4) if k==0 else (4,3),
                    textcoords='offset points',fontsize=5.5,color=color,zorder=9)
    ax.set(xlim=(-48,48),ylim=(-58,58),xticks=[-40,-20,0,20,40],yticks=[-40,-20,0,20,40])
    ax.set_aspect('equal')
    ax.grid(True,color='#d5d8dc',lw=.4)
    ax.set_axisbelow(True)
    ax.set_title('Top view (+z)',loc='left',pad=4)
    ax.set_xlabel('$x$ (mm)',fontsize=6,labelpad=1)
    ax.set_ylabel('$y$ (mm)',fontsize=6,labelpad=1)
    ax.tick_params(labelsize=5.5,pad=1,length=2,width=.6)
    for spine in ax.spines.values(): spine.set_linewidth(.5)
    ax.plot([-40,-30],[-54,-54],color=INK,lw=1.1,zorder=8)
    ax.text(-35,-52,'10 mm',fontsize=5.5,ha='center',va='bottom')


def workspace_panel(ax,g,runs,letter):
    for cls,color in [('undetermined',UNKNOWN),('inside',INNER),('outside',CLASS_COLORS['outside'])]:
        verts=[[(x0,y0),(x1,y0),(x1,y1),(x0,y1)] for x0,x1,y0,y1 in g.map[cls]]
        if verts: ax.add_collection(PolyCollection(verts,fc=color,ec='none',rasterized=True))
    grid=np.array(g.map['pointwise']['grid']); feasible=np.array(g.map['pointwise']['feasible_s1_s2'])
    # This array is indexed [s1,s2], unlike the map raster [s2,s1].
    ij=np.array([(i,j) for i in range(0,len(grid),3) for j in range(0,len(grid),3) if not feasible[i,j]])
    if len(ij): ax.plot(grid[ij[:,0]],grid[ij[:,1]],'x',color='#a64242',ms=2.2,mew=.5,alpha=.8,zorder=3)
    for name,color in [('coordinated',INK),('direct',ORANGE)]:
        run=runs[name]; ref_s=np.array([g.s(row[1:]) for row in run['reference']])
        actual_s=np.array([r['s'] for r in run['records']])
        ax.plot(ref_s[:,0],ref_s[:,1],ls='--',color=color,lw=.8,zorder=4)
        ax.plot(actual_s[:,0],actual_s[:,1],color=color,lw=1.1,zorder=5)
    mission=np.asarray(runs['coordinated']['reference'])
    for point,marker in [(g.s(mission[0,1:]),'o'),(g.s(mission[-1,1:]),'s')]:
        ax.plot(*point,marker,ms=3.5,mfc='white',mec=INK,zorder=6)
    ax.set(xlim=(0,1),ylim=(0,1),xticks=[0,.5,1],yticks=[0,.5,1],xlabel='$s_1$',ylabel='$s_2$')
    ax.set_aspect('equal')
    ax.set_title(f'({letter}) Joint configuration space',loc='left',pad=4)
    a=g.map['area_fraction']
    ax.text(.025,.025,f'In {a["inside"]*100:.1f}% / out {a["outside"]*100:.1f}%\nunresolved {a["undetermined"]*100:.1f}%',
            transform=ax.transAxes,fontsize=6,bbox={'fc':'white','ec':'none','alpha':.9,'pad':1})


def progress_panel(ax,runs,letter):
    for name,ls in [('coordinated','-'),('direct','--')]:
        rows=runs[name]['records']; t=[r['t'] for r in rows]; a=np.array([r['a'] for r in rows])*1e3
        for k,color in enumerate(SPHERE_COLORS): ax.plot(t,a[:,k],color=color,ls=ls,lw=1)
    run=runs['coordinated']; end=run['summary']['reference_duration_s']
    ax.set(xlim=(0,end+2.1),ylim=(0,110),yticks=[0,50,100],xlabel='Time (s)',ylabel='Route position $a_i$ (mm)')
    ax.set_title(f'({letter}) Coordinated passing',loc='left',pad=4)
    ax.text(.71,.68,'Solid: aware\nDashed: direct',transform=ax.transAxes,fontsize=5.6,va='top')
    ax.text(.97,.1,'Sphere 2',color=VEIN,transform=ax.transAxes,fontsize=6,ha='right')
    ax.text(.97,.82,'Sphere 1',color=ARTERY,transform=ax.transAxes,fontsize=6,ha='right')


def render(data,geometries,margins):
    style(); fig=plt.figure(figsize=(7.16,7.7))
    gs=fig.add_gridspec(3,3,left=.06,right=.985,bottom=.075,top=.89,
                       width_ratios=[2.1,1,1],hspace=.7,wspace=.28)
    row_axes=[]
    for row,variant in enumerate(VARIANTS):
        g=geometries[variant]; runs=data[variant]
        views=gs[row,0].subgridspec(1,2,width_ratios=[1.2,1],wspace=.36)
        a=fig.add_subplot(views[0,0],projection='3d',computed_zorder=False); row_axes.append(gs[row,0])
        geometry_panel_3d(a,g,'ad'[row],'3D view',footer=False)
        a=fig.add_subplot(views[0,1]); geometry_panel_top(a,g)
        pos=gs[row,0].get_position(fig)
        bounds=load(CAD/'validation.json')['models'][variant]['round_trip']['bounds_mm']
        y_extent=bounds['max'][1]-bounds['min'][1]
        fig.text(pos.x0,pos.y0-.040,f'Y extent: {y_extent:.1f} mm; closest trunk centrelines: {g.pair.d_close*1e3:.2f} mm',fontsize=5.5)
        length=[r.length*1e3 for r in g.pair.routes]
        fig.text(pos.x0,pos.y0-.058,f'Trunk arc lengths: artery {length[0]:.1f} mm; vein {length[1]:.1f} mm',fontsize=5.5)
        a=fig.add_subplot(gs[row,1]); workspace_panel(a,g,runs,'be'[row])
        a=fig.add_subplot(gs[row,2]); progress_panel(a,runs,'cf'[row])
    bottom=gs[2,:].subgridspec(1,3,wspace=.48)
    ai=fig.add_subplot(bottom[0,0]); margin_map_panel(ai,geometries['side_by_side'],data['side_by_side'],margins,'side_by_side','g','Force margin, side by side')
    aj=fig.add_subplot(bottom[0,1]); mesh=margin_map_panel(aj,geometries['over_under'],data['over_under'],margins,'over_under','h','Force margin, over-under')
    cax=aj.inset_axes([1.05,.0,.05,1.]); cb=fig.colorbar(mesh,cax=cax,ticks=[0,1,2,3]); cb.ax.tick_params(labelsize=6)
    cb.set_label('$\\lambda$',fontsize=7,labelpad=1,rotation=0); cb.ax.axhline(1,color='#a64242',lw=1)
    ak=fig.add_subplot(bottom[0,2]); margin_time_panel(ak,margins,'i')
    for row,label in [(0,'A  Side-by-side vessels: joint-position dependence'),(1,'B  Over-under vessels: geometry comparison')]:
        pos=row_axes[row].get_position(fig)
        fig.text(.015,pos.y1+.045,label,fontsize=8.5,fontweight='bold')
    fig.text(.015,ai.get_position().y1+.045,'C  Pointwise force margin: how far the full task is from infeasibility',fontsize=8.5,fontweight='bold')
    handles=[Patch(fc=INNER,label='TFW-in'),Patch(fc=CLASS_COLORS['outside'],label='TFW-out'),Patch(fc=UNKNOWN,label='Unresolved'),
             Line2D([],[],marker='x',color='#a64242',ls='',ms=4,label='LP-infeasible sample'),
             Line2D([],[],color=INK,lw=1.1,label='TFW-aware'),Line2D([],[],color=ORANGE,lw=1.1,label='Direct'),
             Line2D([],[],color='#a64242',lw=1,label='$\\lambda=1$')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.995),ncol=7,frameon=False,fontsize=6.4)
    fig.text(.5,.952,'SIMULATION PLACEHOLDERS: replace motion/current traces with hardware data; re-evaluate LPs at measured configurations.',
             color=ORANGE,ha='center',fontsize=6.6,style='italic')
    fig.text(.5,.012,'Engineered pipe geometry; 3 mm N35 spheres (top views to scale).  Maps: nominal model; red crosses do not classify whole unresolved boxes.',
             ha='center',fontsize=6.2)
    for ext in ['pdf','png']:
        f=OUT/f'cs2_compact.{ext}'; fig.savefig(f,facecolor='white',bbox_inches='tight',pad_inches=.04,dpi=450)
    plt.close(fig)


def export_text_metrics(data,geometries,margins):
    macros={}
    for variant,tag in [('side_by_side','Side'),('over_under','Over')]:
        area=geometries[variant].map['area_fraction']
        macros[tag+'Area']=f'{area["inside"]*100:.2f}'
        macros[tag+'Unresolved']=f'{area["undetermined"]*100:.2f}'
        macros[tag+'MarginBelowOne']=f'{margins["consistency"][variant]["fraction_below_one"]*100:.1f}'
        macros[tag+'MapMinMargin']=f'{margins["consistency"][variant]["minimum"]:.2f}'
        macros[tag+'TrunkSeparation']=f'{geometries[variant].pair.d_close*1e3:.2f}'
        specification=load(V.GEOM_DIR/V.VARIANTS[GEOMETRY_VARIANTS[variant]][0])
        for branch,label in [('RA','Artery'),('RV','Vein')]:
            for endpoint in ['start','end']:
                macros[tag+label+endpoint.title()+'Diameter']=f"{2*specification['branches'][branch]['radius_'+endpoint+'_mm']:.2f}"
        for method,mt in [('coordinated','Aware'),('direct','Direct')]:
            s=data[variant][method]['summary']; prefix=tag+mt
            for key,val in [('Duration',s['reference_duration_s']),('Current',s['peak_current_A']),
                ('Separation',s['minimum_separation_mm']),('Inside',s['model_in_sample_fraction']*100),
                ('OneRMSE',s['position_rmse_mm'][0]),('TwoRMSE',s['position_rmse_mm'][1]),
                ('OneGoal',s['final_goal_error_mm'][0]),('TwoGoal',s['final_goal_error_mm'][1])]:
                macros[prefix+key]=f'{val:.3f}' if key in ['OneRMSE','TwoRMSE'] else f'{val:.2f}'
            run_lambda=margins['runs'][variant][method]['lambda']
            macros[prefix+'MinMargin']=f'{min(run_lambda):.2f}'
            for key,field in [('TaskBad','full_task_infeasible_samples'),('CommandBad','command_infeasible_samples'),('Samples','active_samples')]:
                macros[prefix+key]=str(s[field])
    hold=data['independent']['summary']
    macros['OneHold']=f'{hold["sphere1_hold_error_mm"]:.3f}'
    macros['TwoHold']=f'{hold["sphere2_hold_error_mm"]:.3f}' if hold['sphere2_hold_error_mm']>=.001 else '<0.001'
    macros['ForceMilli']=f'{T.F_REQ*1e3:.2f}'
    macros['SideMilli']=f'{T.F_SIDE*1e3:.2f}'
    macros['Moment']=f'{K.MOMENT*1e3:.1f}'
    macros['WeightMilli']=f'{K.WEIGHT*1e3:.2f}'
    macros['WallDrag']=f'{K.WALL_DRAG_FACTOR:.0f}'
    macros['MaxNormalLoad']=f'{max(data[v][m]["summary"]["maximum_normal_load_N"] for v in VARIANTS for m in ["coordinated","direct"])*1e3:.2f}'
    macros['MaximumFieldDeviation']=f'{max(data[v][m]["summary"]["between_updates"]["maximum_field_error_T"] for v in VARIANTS for m in ["coordinated","direct"])*1e3:.3f}'
    text='% Generated from Case Study 2 publication simulation_records.json.\n'
    text+=''.join('\\providecommand{\\cstwo'+key+'}{'+value+'}\n' for key,value in macros.items())
    (OUT / 'case_study_2_metrics.tex').write_text(text.rstrip() + '\n')


def provenance(data,geometries,checks,margins):
    sources=[Path(__file__),Path(M.__file__),Path(T.__file__),Path(V.__file__),Path(K.__file__)]
    sources.extend([CAD/'validation.json',Path(__file__).with_name('export_step.py')])
    for v in VARIANTS:
        source=GEOMETRY_VARIANTS[v]
        sources.extend([SAVED/f'tfw_{source}.json',SAVED/f'plan_{source}.json',V.GEOM_DIR/V.VARIANTS[source][0]])
        sources.append(CAD/f'cs2_branched_{v}.step')
    manifest={'date':'2026-10-20','hardware_status':'No new hardware measurements.',
        'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in sources},
        'geometry_variants':GEOMETRY_VARIANTS,
        'geometry_rendering':{'updated':'2026-10-12','source':'Validated hollow-pipe STEP assemblies',
            'projection':'orthographic','elevation_degrees':28,'azimuth_degrees':-65,
            'depth_exaggeration':1,'coordinates':'mm relative to the coil-array centre',
            'surface_raster_dpi':450,'scale_bar_mm':10,'orientation_triad':'x, y, z',
            'paired_views':'3D and top (+z) views from the same hollow CAD pipe solids',
            'top_view':'Equal x/y scaling, mm axes; no perspective foreshortening',
            'trunk_arc_lengths_mm':{v:[r.length*1e3 for r in geometries[v].pair.routes] for v in VARIANTS},
            'uniform_scaling':{v:load(V.GEOM_DIR/V.VARIANTS[GEOMETRY_VARIANTS[v]][0])['uniform_scaling'] for v in VARIANTS},
            'main_trunk_pipe':{v:load(V.GEOM_DIR/V.VARIANTS[GEOMETRY_VARIANTS[v]][0])['main_trunk_pipe'] for v in VARIANTS},
            'sphere_markers':'Top views to scale (3 mm); 3D-view markers schematic'},
        'map_audit':checks,'force_margin_consistency':margins['consistency'],'allocator_validation':load(OUT/'allocator_validation.json'),
        'settings':{'field_T':K.B_HOLD,'moment_A_m2':K.MOMENT,
        'sphere_radius_m':K.RADIUS,'material':'N35 NdFeB','remanence_T':K.REMANENCE,
        'buoyant_weight_N':K.WEIGHT,'gravity':'-z of the eMNS frame','wall_friction_coefficient':K.MU_WALL,
        'viscosity_Pa_s':K.VISCOSITY,'drag_N_s_m':K.DRAG,'wall_drag_factor':K.WALL_DRAG_FACTOR,'wall_drag_N_s_m':K.DRAG_WALL,
        'flow_m_s':0,'task_force_interval_N':[-T.F_REQ,T.F_REQ],'transverse_component_bound_N':T.F_SIDE,
        'current_bound_A':K.I_LIMIT,'control_period_s':DT_CTRL,'plant_period_s':DT_PLANT,
        'feedback_gain_per_s':KP,'command_velocity_cap_m_s':V_CMD_MAX,'friction_feedforward_smoothing_m_s':V_FRICTION,'qp_current_penalty':1e-7,
        'hard_constraints':'Both total fields equal 5 mT along +y; all four transverse components bounded; currents bounded.',
        'soft_outputs':'Only the two tangential commands; independent LPs diagnose infeasibility.',
        'plant':'Quasi-static centreline sliding: stick while |magnetic + weight tangential force| <= mu N, '
        'otherwise wall-corrected Stokes drag; N from weight and magnetic side force; mutual aligned moments iterated six times; zero-order-held current.',
        'task_basis':'F_max >= mu (W + sqrt(2) F_side) + wall-corrected drag at 3 mm/s.'},
        'summary':{v:{m:d['summary'] for m,d in data[v].items()} for v in VARIANTS},
        'independent_summary':data['independent']['summary'],
        'limitations':['Illustrative material and dynamics; no physical validation.',
         'Stored nominal maps use box-centre force frames with a 5-degree tangent refinement criterion; varying frames are not fully enclosed.',
         'Floating-point certificates lack outward rounding; TFW-in is a nominal planning restriction.',
         'Hard output constraints hold at allocation updates; inter-update deviations are logged separately.',
         'Centreline support reacts transverse loads; lateral motion and rolling are not simulated; friction coefficient and remanence are nominal, not measured.',
         'Area fractions are in Bezier parameter coordinates, not physical-area or arc-length measure.',
         'One deterministic run per arm; equal endpoints, duration, and motion bounds within each geometry.']}
    dump(OUT/'manifest.json',manifest)
    note='''# Case Study 2 figure provenance

Maps, plans and simulation records updated 20 October 2026 for 3 mm N35 spheres.
The vessel networks retain approximately 100 mm y extent. Both main trunks
are manufactured as 8 mm outer / 4 mm inner diameter pipes with 2 mm walls
and open annular ends. Separate side-branch bores are not part of this CAD
request. Networks are moved apart along x or z to keep the outer pipes separate.
Simulation radius fields describe the 4 mm inner lumen; particles stay on
the trunk centrelines. Side branches remain geometric context.
All motion traces are simulations; no new hardware data.

Run from the repository root:

```bash
.venv/bin/python -B -m revised_case_studies.cs2_vessel_microspheres.publication
.venv/bin/python -B -m revised_case_studies.cs2_vessel_microspheres.publication --render
```

The nine-panel figure uses branched_parallel.json and branched_crossing.json,
with the matching branched TFW files and corridor plans. Geometry panels (a)
and (d) pair shaded 3D and top (+z) views of the validated hollow STEP pipe assemblies, including
all 14 segments: red arterial and blue venous networks. CadQuery is required
to render these panels. An orthographic camera and equal physical scaling
preserve the 3D proportions without depth exaggeration. Each view has a
calibrated 10 mm scale bar. The 3D view includes an x/y/z orientation triad;
the top view has equal-scale x/y axes in millimetres, showing projected length
without perspective foreshortening. Full trunk arc lengths are reported
separately beneath each pair and are not inferred from the top projection.
Scene coordinates are relative to the coil-array centre. Surface shading and the
workspace/force maps are rasterized (450 dpi); labels and trajectory curves
remain vector PDF content.
Spheres are 3 mm N35; top-view markers are to scale, 3D markers and arrows are schematic.
Both spheres travel only on the main trunks; side-branch routing is not tested.
The earlier carotid-bifurcation task and unbranched results are not pooled.
Publication rendering reads the recomputed TFW files and corridor plans without rewriting them. They are
nominal floating-point box classifications, with zero TFW-out area. The sampled
LP-infeasible points in unresolved boxes are crosses, never red region fills.
Stored region force frames are frozen at box centres with tangent refinement;
the figure does not claim rigorous varying-frame or outward-rounded certification.
Point and affine row consistency and sampled inside-box checks are in manifest.json.
Row C (panels g-i): pointwise force margin lambda, the largest scale of the +-F_max
tangential task with all four corners attainable (fields exact, transverse and current
bounds enforced), on the stored 65 x 65 grid and along every logged run; cached in
force_margin.json. lambda < 1 agrees with the stored pointwise LP at every grid point.
It is a diagnostic, not a region certificate.
allocator_validation.json compares the recorded motion command with an
unattainable full-task corner at one recorded direct-arm configuration (with
3 mm N35 spheres the recorded command there is itself LP-infeasible). The QP
keeps the hard field/side constraints in both cases and exposes the force residual.

New simulations enforce the map's 5 mT total field and transverse-force-component
bounds as hard QP constraints. Only tangential motion force has residual. There
is no unconstrained fallback. All four task corners and each individual command
are tested by LP at every logged configuration, using exact vessel frames.
The nonlinear aligned-moment plant includes mutual interaction at every step.
Hard field/side constraints are enforced at the 10 Hz allocation updates;
inter-update field, transverse-load, and alignment deviations are logged, not
claimed to be continuously constrained. Parameters and centreline confinement
remain illustrative. The plant slides each 3 mm N35 sphere on the wall with Coulomb
friction (mu = 0.2) under buoyant weight and magnetic side load. Transverse load is a model prediction, not a measurement.

The TFW-aware arm reuses each saved corridor mission. Its physical endpoints
are shared by a direct, quintic-retimed reference with the same total duration.
Both obey speed <= 3 mm/s and acceleration <= 1.5 mm/s^2; force commands are
bounded by the full task's 0.5 mN tangential limit (friction and weight feedforward included). No waypoint is silently
moved between comparison arms. Per-arm durations, endpoint errors, sample counts,
LP infeasibility, currents, loads, and reference-limit checks are saved in
simulation_records.json. The separate advance/hold demonstration uses the
side-by-side geometry near its starting configuration, not near the close pass.

Replace simulated actual paths/progress/diagnostics in panels (b),(c),(e)-(h)
with synchronized repeated hardware trials. Recompute feasibility at measured
poses. Keep numerical workspace classes, pointwise tests, and measured motion
separate. The 3D sphere markers are schematic. The geometry is engineered,
not patient-specific, and the parameter-area fractions are not anatomical areas.

Source hashes and all settings: manifest.json.
'''
    (OUT/'PROVENANCE.md').write_text(note)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--render',action='store_true',help='Redraw from saved publication time series.')
    args=ap.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    geometries={v:Geometry(v) for v in VARIANTS}
    if args.render:
        data=load(OUT/'simulation_records.json'); checks=load(OUT/'manifest.json')['map_audit']
    else:
        data={}; checks={}
        for v,g in geometries.items():
            checks[v]=audit(g); data[v]={}
            for name,mission in missions(g).items():
                start=time.monotonic(); data[v][name]=simulate(g,mission,name)
                print(v,name,json.dumps(data[v][name]['summary']),f'elapsed {time.monotonic()-start:.1f}s',flush=True)
                dump(OUT/'simulation_records.json',data)
        data['independent']=simulate(geometries['side_by_side'],independent_mission(geometries['side_by_side']),'independent')
        dump(OUT/'simulation_records.json',data)
    allocator_check(geometries['side_by_side'],data['side_by_side']['direct'])
    margins=margin_data(data,geometries)
    export_text_metrics(data,geometries,margins); provenance(data,geometries,checks,margins); render(data,geometries,margins)
    print('Exported cs2_compact',flush=True)


if __name__=='__main__': main()
