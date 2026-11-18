#!/usr/bin/env python3
"""Quadrature sensitivity and classification stability at critical samples."""
import json, sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE.parent))
from run_comparison import finite_interaction, contains_box
from cases.case_6_interaction_obstacle_avoidance.simulation import model
LEVELS={'coarse':{'radial':3,'azimuthal':12,'axial':4},'reported':{'radial':4,'azimuthal':16,'axial':5},'fine':{'radial':6,'azimuthal':32,'axial':9}}
def main():
    cfg=json.loads((HERE/'config.json').read_text()); base=json.loads((HERE/'results/metrics.json').read_text())
    poses=[(v['x_m'],v['y_m']) for v in base['quadrature_sensitivity_checks']]
    beta=cfg['moving_heading_rad']; center=np.array([0.,model.B_HOLD,0.,0.]); half=np.array([0.,0.,0.,model.F_MAX]); reports=[]
    for x,y in poses:
        Ah=model.actuation_matrix_body(x,y,beta); A,c=Ah[:,:10],Ah[:,10]; levels={}
        for name,q in LEVELS.items():
            cc=json.loads(json.dumps(cfg)); cc['reference_quadrature']=q
            ref=finite_interaction(x,y,beta,cc); disc=np.abs(ref-c)
            feasible=contains_box(A,center-c,half+cfg['envelope_safety_factor']*disc,cfg['current_limit_A'])
            levels[name]={'reference':ref.tolist(),'absolute_discrepancy':disc.tolist(),'reserved_feasible':bool(feasible)}
        fine=np.array(levels['fine']['reference']); reported=np.array(levels['reported']['reference'])
        reports.append({'pose_m':[x,y,beta],'levels':levels,'reported_to_fine_absolute_change':np.abs(fine-reported).tolist(),'classification_stable_all_levels':len({v['reserved_feasible'] for v in levels.values()})==1})
    result={'scope':'finite quadrature sensitivity only; not a rigorous remainder or physical uncertainty bound','levels':LEVELS,'poses':reports,'all_classifications_stable':all(v['classification_stable_all_levels'] for v in reports)}
    (HERE/'results/reference_convergence.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'poses':len(reports),'all_classifications_stable':result['all_classifications_stable'],'max_reported_to_fine_change':np.max([p['reported_to_fine_absolute_change'] for p in reports],axis=0).tolist()},indent=2))
if __name__=='__main__': main()
