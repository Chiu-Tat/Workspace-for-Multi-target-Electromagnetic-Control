"""Create the two editable branched-network JSON presets from the simple pairs.

Run explicitly only to reset these presets to their designed defaults.
Preview regeneration reads JSON without resetting user edits.
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent


def bezier(p, s):
    return np.array([(1-s)**3, 3*(1-s)**2*s, 3*(1-s)*s*s, s**3]) @ np.array(p)


def branch_point(branch,s):
    wave=np.sin(np.pi*s)**2*np.sin(2*np.pi*s+branch.get('waviness_phase',0))
    return bezier(branch['control_points_mm'],s)+wave*np.array(branch.get('waviness_mm',[0,0,0]))


def make_preset(kind):
    config = json.loads((HERE/'geometries'/('close_'+kind+'.json')).read_text())
    config['title'] = 'Branched paired vessels: '+('side-by-side approach' if kind == 'parallel' else 'over-under crossing')
    config['description'] = 'Synthetic forearm-inspired vessels with uneven branching, smooth taper, winding paths and rounded junctions; not a patient scan.'
    config.pop('designed_lumen_gap_mm', None)
    config['networks'] = {'artery': {'trunk': 'RA', 'label': 'Arterial network', 'color': '#c25362'},
                          'vein': {'trunk': 'RV', 'label': 'Venous network', 'color': '#347da5'}}
    config['mesh_step_mm'] = 0.20
    config['display_smoothing_max_mm'] = 0.06
    for trunk, group, sign in [('RA', 'artery', -1), ('RV', 'vein', 1)]:
        main = config['branches'][trunk]
        main['network'] = group
        main['radius_end_mm'] = .94 if trunk == 'RA' else .66
        main['smooth_taper'] = True
        main['waviness_mm'] = [0,.65,1.3] if kind=='parallel' else [.85,.45,0]
        main['waviness_phase'] = 0 if trunk=='RA' else .8
        if kind == 'parallel':
            for point,x in zip(main['control_points_mm'],[-7.8,.5,.9,-8.1] if trunk=='RA' else [6.2,-.4,-.4,8.3]):
                point[0]=x
            for point,z in zip(main['control_points_mm'], [-1.5, 3., -3., 1.5]):
                point[2] = z
        else:
            for point,z in zip(main['control_points_mm'], [2.5, .7, .7, 2.5]):
                point[2] = sign*z
        # Four asymmetric, tapered side branches. Their first point attaches
        # exactly to the parent centerline; the union gives a patent junction.
        stations = [.145,.325,.61,.805] if trunk=='RA' else [.235,.425,.685,.9]
        for index,s in enumerate(stations,1):
            root = branch_point(main,s)
            reach = ([9,19,14,8] if trunk=='RA' else [13,10,18,8])[index-1]
            dy = ([8,4,13,7] if trunk=='RA' else [-6,11,6,4])[index-1]
            if kind == 'parallel':
                delta = np.array([sign*reach,dy, [-2,3,-3,2][index-1]])
            else:
                outward = -1 if root[0] < 0 else 1
                delta = np.array([outward*reach,dy,sign*[3.5,5,4.5,3][index-1]])
            # Curved take-off, unequal turning lengths and non-collinear handles.
            bend = np.array([0, [3.1,-3.4,2.3,-1.7][index-1],0])
            p = np.stack([root,root+delta*np.array([.09,.32,.18])+bend*.5,
                          root+delta*np.array([.64,.79,.65])-bend,root+delta])
            if index==4:
                # A short terminal offshoot needs a cleaner departure to avoid
                # grazing the parent surface again at the take-off shoulder.
                p[1]=root+delta*np.array([.42,0,.40])
                p[2]=root+delta*np.array([.76,.20,.80])
            name = trunk+'_B'+str(index)
            r0 = ([.57,.69,.61,.46] if trunk=='RA' else [.51,.57,.48,.40])[index-1]
            config['branches'][name] = {
                'label': ('Side branch ' if trunk=='RA' else 'Tributary ')+str(index),
                'network':group,'parent':trunk,'parent_s':s,
                'control_points_mm':p.round(8).tolist(),
                'radius_start_mm':r0,'radius_end_mm':max(.25,r0*[.57,.49,.61,.65][index-1]),
                'smooth_taper':True,
                'waviness_mm':[0,.55,1.1] if kind=='parallel' else [.5,.45,0],
                'waviness_phase':index*.9+(0 if trunk=='RA' else .7),
                'bulb_amplitude_mm':0,'bulb_center_s':.5,'bulb_width_s':.2}
            if index==4:
                config['branches'][name]['waviness_mm']=[0,0,0]
            if index in (2,3):
                child_s = .56 if index==2 else .72
                child_root = branch_point(config['branches'][name],child_s)
                secondary = delta*np.array([.39,-.42,.45])
                child_bend = np.array([0,2.2 if index==2 else -1.9,0])
                child_p = np.stack([child_root,child_root+secondary*.13+child_bend*.45,
                                    child_root+secondary*.62+child_bend,child_root+secondary])
                config['branches'][name+'_twig'] = {
                    'label':'Secondary '+('branch' if trunk=='RA' else 'tributary'),
                    'network':group,'parent':name,'parent_s':child_s,
                    'control_points_mm':child_p.round(8).tolist(),
                    'radius_start_mm':.36 if trunk=='RA' else .31,
                    'radius_end_mm':.24,'smooth_taper':True,
                    'waviness_mm':[0,.25,.5] if kind=='parallel' else [.2,.3,0],
                    'waviness_phase':index*.7,'bulb_amplitude_mm':0,
                    'bulb_center_s':.5,'bulb_width_s':.2}
    config['task_concept'] = {'sphere_1': {'network':'artery','target_branch':'RA_B3_twig'},
                              'sphere_2': {'network':'vein','target_branch':'RV_B2_twig'},
                              'note':'Target labels only; reachability and flow direction are not evaluated.'}
    (HERE/'geometries'/('branched_'+kind+'.json')).write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    for kind in ('parallel','crossing'):
        make_preset(kind)
