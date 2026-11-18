"""Pointwise disc containment, phase offsets, and a bench mechanism model.

The four output coordinates are in the planes normal to parallel y-axis rods.
All-phase offsets are enclosed by the image of two independent moment discs.
Containment uses all zonotope facets, not a sampled field-direction test.
Numerical results are floating-point pointwise evaluations, not region certificates.
"""
from itertools import combinations
import numpy as np
import cvxpy as cp
from scipy.optimize import linprog
from cases.case_7_vessel_microspheres.magnetics import coil_basis
from revised_case_studies.cs2_vessel_microspheres.model import dipole_field, dipole_gradient
from . import config as K

# This basis has e1 x e2 = +y, so positive phase gives positive axial torque.
E=np.array([[1.,0.],[0.,0.],[0.,-1.]])
TRIPLES=np.array(list(combinations(range(10),3)))


class Placement:
    def __init__(self,spacing_mm,offset_mm=0.,height_mm=55.):
        self.q=[float(spacing_mm),float(offset_mm),float(height_mm)]
        self.p=np.array([[-spacing_mm/2,-offset_mm/2,height_mm],
                         [spacing_mm/2,offset_mm/2,height_mm]])*1e-3
        self.H=[];self.grad=[]
        for p in self.p:
            H,D=coil_basis(p);self.H.append(H);self.grad.append(D)
        self.A=np.vstack([E.T@H for H in self.H])
        self.D=[]
        for k in [0,1]:
            r=self.p[k]-self.p[1-k];d=np.linalg.norm(r)
            self.D.append(1e-7*(3*np.outer(r,r)/d**5-np.eye(3)/d**3))
        self.U=np.zeros((4,4))
        self.U[:2,2:]=K.MOMENT*E.T@self.D[0]@E
        self.U[2:,:2]=K.MOMENT*E.T@self.D[1]@E

    def offset(self,phase):
        u=np.column_stack([np.cos(phase),np.sin(phase)])
        return self.U@u.ravel()

    def fields_forces(self,current,phase):
        moments=np.array([K.MOMENT*E@np.array([np.cos(a),np.sin(a)]) for a in phase])
        fields=[];forces=[]
        for k in [0,1]:
            j=1-k;r=self.p[k]-self.p[j]
            fields.append(E.T@(self.H[k]@current+self.D[k]@moments[j]))
            G=np.einsum('abk,k->ab',self.grad[k],current)+dipole_gradient(r,moments[j])
            forces.append(G.T@moments[k])
        return np.array(fields),np.array(forces)

    def force_safe_current_limit(self):
        # Triangle/operator-norm bound for arbitrary rotor phases.
        # The bilinear mutual-force tensor uses a spectral bound per component.
        limits=[];details=[]
        for k in [0,1]:
            r=self.p[k]-self.p[1-k]
            tensor=np.empty((3,2,2))
            for i in [0,1]:
                for j in [0,1]:
                    tensor[:,i,j]=dipole_gradient(r,K.MOMENT*E[:,j]).T@(K.MOMENT*E[:,i])
            mutual=float(np.linalg.norm([np.linalg.norm(tensor[a],2) for a in range(3)]))
            weights=K.MOMENT*np.array([np.linalg.norm(self.grad[k][:,:,j].T@E,2) for j in range(10)])
            limit=max(0.,min(K.I_LIMIT,(K.FORCE_LIMIT-mutual)/sum(weights)))
            limits.append(limit);details.append({'mutual_force_bound_N':mutual,'sum_coil_force_weights_N_A':float(sum(weights))})
        return float(min(limits)),details


def facets(A):
    if np.linalg.matrix_rank(A)!=4: raise ValueError('Rank-deficient joint field map.')
    W=np.transpose(A[:,TRIPLES],(1,0,2))/np.max(abs(A))
    N=np.column_stack([(-1)**j*np.linalg.det(np.delete(W,j,axis=1)) for j in range(4)])
    lengths=np.linalg.norm(N,axis=1);N=N[lengths>1e-12]
    return N/np.linalg.norm(N,axis=1)[:,None]


def capacity(p):
    N=facets(p.A);h=K.I_LIMIT*np.sum(abs(N@p.A),axis=1)
    n1=np.linalg.norm(N[:,:2],axis=1);n2=np.linalg.norm(N[:,2:],axis=1)
    NU=N@p.U;u=np.linalg.norm(NU[:,:2],axis=1)+np.linalg.norm(NU[:,2:],axis=1)
    selective=np.maximum(n1,n2);simultaneous=n1+n2
    lower,details=p.force_safe_current_limit()
    def radius(b,den): return max(0.,float(np.min(b/den)))
    return {'selective_nominal_T':radius(h,selective),
            'simultaneous_nominal_T':radius(h,simultaneous),
            'selective_robust_T':radius(h-u,selective),
            'simultaneous_robust_T':radius(h-u,simultaneous),
            'force_safe_selective_lower_T':radius(h*(lower/K.I_LIMIT)-u,selective),
            'force_safe_current_limit_A':float(lower),'force_bound_details':details,
            'interaction_field_bound_T':max(np.linalg.norm(p.U[:2,2:],2),np.linalg.norm(p.U[2:,:2],2)),
            'zero_task_robust_feasible':bool(np.min(h-u)>=0),
            'facet_count':len(N)}


def exact_feasible(p,target,phase):
    r=linprog(np.zeros(10),A_eq=p.A/K.B_ROT,
              b_eq=(np.asarray(target)-p.offset(phase))/K.B_ROT,
              bounds=[(-K.I_LIMIT,K.I_LIMIT)]*10,method='highs')
    if r.status not in [0,2]:raise RuntimeError(r.message)
    return r.status==0


class Allocator:
    """Hard current bounds; exact four-field commands when attainable.

    For an unattainable rotate/hold command, held field is hard-zero and only
    active field may have residual. There is no common-field fallback.
    """
    def __init__(self,p):
        self.p=p;self.i=cp.Variable(10);self.target=cp.Parameter(4)
        self.exact=cp.Problem(cp.Minimize(cp.sum_squares(self.i)),[
            p.A@self.i/K.B_ROT==self.target,self.i<=K.I_LIMIT,self.i>=-K.I_LIMIT])
        self.soft=[]
        for active in [0,1]:
            ai=slice(2*active,2*active+2);hi=slice(2*(1-active),2*(1-active)+2)
            self.soft.append(cp.Problem(cp.Minimize(cp.sum_squares(p.A[ai]@self.i/K.B_ROT-self.target[ai])+
                1e-8*cp.sum_squares(self.i)),[
                p.A[hi]@self.i/K.B_ROT==self.target[hi],self.i<=K.I_LIMIT,self.i>=-K.I_LIMIT]))

    def solve(self,target,phase,active=None):
        self.target.value=(np.asarray(target)-self.p.offset(phase))/K.B_ROT
        self.exact.solve(solver='CLARABEL',warm_start=True,tol_feas=1e-8,tol_gap_abs=1e-8)
        exact=self.exact.status in ['optimal','optimal_inaccurate']
        if not exact:
            if active is None:raise RuntimeError('Joint command infeasible; no permitted fallback.')
            self.soft[active].solve(solver='CLARABEL',warm_start=True,tol_feas=1e-8,tol_gap_abs=1e-8)
            if self.soft[active].status not in ['optimal','optimal_inaccurate']:
                raise RuntimeError('Held field/current constraints infeasible.')
        current=np.asarray(self.i.value).copy()
        if max(abs(current))>K.I_LIMIT+1e-6:raise RuntimeError('Current bound violated.')
        actual=self.p.A@current+self.p.offset(phase)
        if exact and max(abs(actual-target))>1e-7:raise RuntimeError('Exact field residual too large.')
        if not exact and max(abs(actual[2*(1-active):2*(1-active)+2]))>1e-7:
            raise RuntimeError('Hard null hold violated.')
        return current,exact


def simulate(p,method='selective',sequence=K.PHASE_SEQUENCE):
    load_torque=K.LOAD*K.LEAD_PITCH/(2*np.pi*K.GEAR_RATIO*K.GEAR_EFFICIENCY*K.SCREW_EFFICIENCY)
    threshold=K.FRICTION+load_torque
    alloc=Allocator(p);phase=np.zeros(2);ref=np.zeros(2);common=0.;rows=[]
    maximum_hold_field=0.;maximum_force=0.;t=0.
    for segment,(duration,freq) in enumerate(sequence):
        freq=np.array(freq);active=np.flatnonzero(freq)
        active_index=int(active[0]) if len(active)==1 else None
        for _ in range(round(duration/K.DT_CTRL)):
            if method=='selective':
                target=np.concatenate([K.B_ROT*np.array([np.cos(ref[k]),np.sin(ref[k])])
                            if k in active else np.zeros(2) for k in [0,1]])
            else:
                target=np.tile(K.B_ROT*np.array([np.cos(common),np.sin(common)]),2)
            current,exact=alloc.solve(target,phase,active_index if method=='selective' else None)
            fields,forces=p.fields_forces(current,phase)
            rows.append({'t':t,'segment':segment,'phase_rad':phase.tolist(),'reference_phase_rad':ref.tolist(),
                'common_phase_rad':common,'length_mm':(phase*K.LEAD_PITCH/(2*np.pi*K.GEAR_RATIO)*1e3).tolist(),
                'reference_length_mm':(ref*K.LEAD_PITCH/(2*np.pi*K.GEAR_RATIO)*1e3).tolist(),
                'currents_A':current.tolist(),'command_fields_T':target.tolist(),'fields_T':fields.tolist(),'forces_N':forces.tolist(),
                'field_error_T':(fields.ravel()-target).tolist(),'exact_command':exact})
            for _ in range(round(K.DT_CTRL/K.DT_PLANT)):
                fields,forces=p.fields_forces(current,phase)
                torque=K.MOMENT*(np.cos(phase)*fields[:,1]-np.sin(phase)*fields[:,0])
                velocity=np.sign(torque)*np.maximum(abs(torque)-threshold,0.)/K.ROT_DRAG
                phase+=K.DT_PLANT*velocity
                maximum_force=max(maximum_force,float(np.max(np.linalg.norm(forces,axis=1))))
                if method=='selective' and active_index is not None:
                    maximum_hold_field=max(maximum_hold_field,float(np.linalg.norm(fields[1-active_index])))
            ref+=2*np.pi*freq*K.DT_CTRL
            common+=2*np.pi*freq[active[0]]*K.DT_CTRL
            t+=K.DT_CTRL
    fields,forces=p.fields_forces(current,phase)
    rows.append({**rows[-1],'t':t,'phase_rad':phase.tolist(),'reference_phase_rad':ref.tolist(),
                 'length_mm':(phase*K.LEAD_PITCH/(2*np.pi*K.GEAR_RATIO)*1e3).tolist(),
                 'reference_length_mm':(ref*K.LEAD_PITCH/(2*np.pi*K.GEAR_RATIO)*1e3).tolist(),
                 'fields_T':fields.tolist(),'forces_N':forces.tolist(),
                 'field_error_T':(fields.ravel()-target).tolist(),'final_snapshot':True})
    holding=[]
    for seg,(_,freq) in enumerate(sequence):
        if np.count_nonzero(freq)!=1:continue
        held=1-int(np.flatnonzero(freq)[0]);rr=[r for r in rows if r['segment']==seg]
        value=max(abs(r['length_mm'][held]-rr[0]['length_mm'][held]) for r in rr)
        holding.append({'segment':seg,'held_rotor':held,'maximum_length_change_mm':value})
    summary={'placement_mm':p.q,'method':method,'duration_s':t,'holding_intervals':holding,
        'maximum_held_length_change_mm':max(h['maximum_length_change_mm'] for h in holding),
        'final_lengths_mm':rows[-1]['length_mm'],'reference_final_lengths_mm':rows[-1]['reference_length_mm'],
        'inexact_allocation_samples':sum(not r['exact_command'] for r in rows[:-1]),
        'peak_current_A':max(max(abs(v) for v in r['currents_A']) for r in rows),
        'maximum_field_component_error_T':max(max(abs(v) for v in r['field_error_T']) for r in rows[:-1]),
        'maximum_between_update_hold_field_T':maximum_hold_field,'maximum_predicted_force_N':maximum_force,
        'load_torque_Nm':load_torque,'static_torque_threshold_Nm':threshold}
    return {'records':rows,'summary':summary}
