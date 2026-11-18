"""Finite regression checks on retained region geometry and source metrics.

These checks detect implementation/export errors; they are not continuous proofs.
"""
from pathlib import Path
import json
import numpy as np
from analysis.sections_67.common import OUTPUT_ROOT, dump_json
from analysis.sections_67 import afw_tools


def validate(root=OUTPUT_ROOT):
    data=root/'data'
    rows=[]
    for filename in ('resolution_study.json','task_requirement_hybrid.json','coil_configuration.json'):
        payload=json.loads((data/filename).read_text())
        metrics=payload.get('metrics',payload.get('tfw',[])+payload.get('current_limits',[]))
        for report in metrics:
            boxes=report['terminal_boxes_m']
            protocol=report['protocol']
            initial=np.array(protocol['initial_box_m'])
            # The library protocol stores per-axis lower/upper pairs.
            total=float(np.prod(initial[:,1]-initial[:,0]))
            measures={k:sum(float(np.prod(np.array(b['upper'])-b['lower'])) for b in group) for k,group in boxes.items()}
            assert np.isclose(sum(measures.values()),total,atol=1e-12)
            for key in measures:
                assert np.isclose(measures[key]/total,report['area_fractions'][key],atol=1e-10)
            bounds=np.array(protocol['task_box_T'])
            vertices=np.array([[x,y] for x in bounds[0] for y in bounds[1]])
            active=protocol['active_coil_indices']
            limit=protocol['current_A'][1]
            checked=0;unknown=0;contradictions=[]
            for label in ('inside','outside'):
                group=boxes[label]
                if not group:continue
                indices=np.unique(np.linspace(0,len(group)-1,min(12,len(group)),dtype=int))
                for index in indices:
                    box=group[index];lo=np.array(box['lower']);hi=np.array(box['upper'])
                    for pose in ((lo+hi)/2,lo,hi):
                        target={'X':pose[0],'Y':pose[1],'Z':protocol['coil_model_z_m'],'m':0.,'alpha':0.,'beta':0.,
                                **{k:None for k in ('Bx','By','Bz','Bx_dx','Bx_dy','Bx_dz','By_dy','By_dz','fx','fy','fz','tx','ty','tz')}}
                        target.update(Bx=True,By=True)
                        g,c=afw_tools.actuation_map([target],active)
                        statuses=[afw_tools.current_feasibility(v,g,c,limit,1e-7)['feasible'] for v in vertices]
                        expected = all(v is True for v in statuses) if label=='inside' else any(v is False for v in statuses)
                        checked+=1
                        if not expected:
                            if any(v is None for v in statuses):unknown+=1
                            else:contradictions.append({'label':label,'pose_m':pose.tolist(),'vertex_status':statuses})
            rows.append({'source':filename,'case':report.get('case',report.get('subset',report.get('current_limit_A',report.get('eps_m')))),
                         'checked_poses':checked,'unknown':unknown,'contradictions':contradictions,'area_check':'passed'})
    output={'interpretation':'Finite regression only; no continuous or hardware certification','reports':rows,
            'total_checked':sum(r['checked_poses'] for r in rows),'total_unknown':sum(r['unknown'] for r in rows),
            'total_contradictions':sum(len(r['contradictions']) for r in rows)}
    dump_json(root/'validation/review_export_checks.json',output)
    print({k:v for k,v in output.items() if k!='reports'})
    if output['total_contradictions'] or output['total_unknown']:raise SystemExit(1)

if __name__=='__main__':
    validate()
