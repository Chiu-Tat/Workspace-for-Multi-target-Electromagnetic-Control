"""Build merged branched lumen surfaces and views, without flow/control simulation.

python -m cases.case_7_vessel_microspheres.build_networks
Requires NumPy, Matplotlib and scikit-image. Reads the editable JSON presets.
"""
import csv
import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from .vessel import Vessel
from .build_variants import coordinate_range

HERE = Path(__file__).parent


def network_checks(vessel):
    """Check graph attachments and a continuous separating-slab bound in mm."""
    config = vessel.config
    axis = config['separation_axis']
    intervals = {}
    counts = {}
    for network, metadata in config['networks'].items():
        keys = [b for b in vessel.branches if vessel.branches[b].get('network') == network]
        trunk = metadata['trunk']
        if trunk not in keys:
            raise ValueError('Network trunk is absent')
        lows, highs = [], []
        for key in keys:
            branch = vessel.branches[key]
            low, high = coordinate_range(branch['control_points_mm'], axis)
            wave_bound=abs(branch.get('waviness_mm',[0,0,0])[axis])
            low-=wave_bound
            high+=wave_bound
            rmax = max(branch['radius_start_mm'],branch['radius_end_mm']) + max(branch['bulb_amplitude_mm'],0)
            lows.append(low-rmax)
            highs.append(high+rmax)
            if key != trunk:
                parent = branch['parent']
                if parent not in keys:
                    raise ValueError('Branch has a missing or cross-network parent')
                if not np.allclose(vessel.centerline(key,0),vessel.centerline(parent,branch['parent_s']),rtol=0,atol=1e-9):
                    raise ValueError('Detached branch: '+key)
                visited = {key}
                while parent != trunk:
                    if parent in visited:
                        raise ValueError('Cycle in attachment graph')
                    visited.add(parent)
                    parent = vessel.branches[parent]['parent']
            if min(branch['radius_start_mm'],branch['radius_end_mm']) <= 0 or branch['bulb_amplitude_mm'] < 0:
                raise ValueError('Preset certificate requires positive radii and nonnegative bulb amplitude')
        intervals[network] = [min(lows),max(highs)]
        counts[network] = len(keys)
    groups = list(intervals)
    if len(groups)!=2:
        raise ValueError('Paired network check needs two groups')
    a,b = [intervals[g] for g in groups]
    bound = max(b[0]-a[1],a[0]-b[1])
    if bound <= 0:
        raise ValueError('Unable to certify network separation on the configured axis')
    return {'network_count':2,'segments_per_network':counts,'attachment_graphs':'connected acyclic trees',
            'continuous_gap_lower_bound_mm':float(bound),'separating_axis_local':axis,
            'method':'Scalar cubic extrema expanded by winding-amplitude and radius bounds; ordinary floating-point arithmetic',
            'note':'Conservative model bound, not an exact minimum, interval-arithmetic certificate, or wall-thickness model.'}


def mesh_topology(vertices, faces):
    edges = Counter(tuple(sorted((int(f[i]),int(f[(i+1)%3])))) for f in faces for i in range(3))
    if any(count!=2 for count in edges.values()):
        raise ValueError('Open or nonmanifold mesh edge')
    neighbors = {}
    for a,b in edges:
        neighbors.setdefault(a,set()).add(b)
        neighbors.setdefault(b,set()).add(a)
    seen=set()
    components=0
    for start in neighbors:
        if start in seen:
            continue
        components+=1
        todo=[start]
        while todo:
            current=todo.pop()
            if current in seen:
                continue
            seen.add(current)
            todo.extend(neighbors[current]-seen)
    euler=len(seen)-len(edges)+len(faces)
    if components!=1 or euler!=2:
        raise ValueError('Unexpected disconnected lumen or mesh handle: components={}, Euler={}'.format(components,euler))
    return {'vertices':len(vertices),'triangles':len(faces),'components':components,'closed_edges':True,'euler_characteristic':euler}


def merged_mesh(vessel, network, step_mm):
    """Rasterize sampled-ball union locally, avoiding a full grid x centers array."""
    try:
        from skimage.measure import marching_cubes
    except ImportError:
        from skimage.measure import marching_cubes_lewiner as marching_cubes
    centers,radii=vessel.samples(301,network=network)
    step=step_mm*1e-3
    lower=(centers-radii[:,None]).min(axis=0)-2*step
    upper=(centers+radii[:,None]).max(axis=0)+2*step
    shape=np.ceil((upper-lower)/step).astype(int)+1
    field=np.full(tuple(shape),-2*step,dtype=np.float32)
    for center,radius in zip(centers,radii):
        lo=np.maximum(0,np.floor((center-radius-2*step-lower)/step).astype(int))
        hi=np.minimum(shape,np.ceil((center+radius+2*step-lower)/step).astype(int)+1)
        xyz=[lower[k]+np.arange(lo[k],hi[k])*step-center[k] for k in range(3)]
        distance=np.sqrt(xyz[0][:,None,None]**2+xyz[1][None,:,None]**2+xyz[2][None,None,:]**2)
        sl=tuple(slice(lo[k],hi[k]) for k in range(3))
        np.maximum(field[sl],radius-distance,out=field[sl])
    vertices,faces,_,_=marching_cubes(field,level=0,spacing=(step,)*3)
    vertices+=lower
    return vertices,faces


def fair_display_mesh(vertices,faces,max_displacement_mm):
    """Bounded display-only Laplacian fairing; analytic lumen is unchanged."""
    if max_displacement_mm<=0:
        return vertices
    original=vertices.copy()
    edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=1),axis=0)
    degree=np.bincount(edges.ravel(),minlength=len(vertices))
    limit=max_displacement_mm*1e-3
    for _ in range(5):
        sums=np.zeros_like(vertices)
        np.add.at(sums,edges[:,0],vertices[edges[:,1]])
        np.add.at(sums,edges[:,1],vertices[edges[:,0]])
        candidate=vertices+.32*(sums/np.maximum(degree[:,None],1)-vertices)
        delta=candidate-original
        factor=np.minimum(1,limit/np.maximum(np.linalg.norm(delta,axis=1),1e-15))
        vertices=original+factor[:,None]*delta
    return vertices


def export_preset(name):
    vessel=Vessel(HERE/'geometries'/(name+'.json'))
    checks=network_checks(vessel)
    out=HERE/'preview'/name
    out.mkdir(parents=True,exist_ok=True)
    data={'title':vessel.config['title'],'description':vessel.config['description'],
          'gap':checks['continuous_gap_lower_bound_mm'],'networks':vessel.config['networks'],
          'curves':{},'meshes':[],'particles':[],'targets':[]}
    checks['mesh_step_mm']=vessel.config['mesh_step_mm']
    checks['display_fairing_max_displacement_mm']=vessel.config.get('display_smoothing_max_mm',0)
    checks['meshes']={}
    for group,metadata in vessel.config['networks'].items():
        vertices,faces=merged_mesh(vessel,group,vessel.config['mesh_step_mm'])
        vertices=fair_display_mesh(vertices,faces,vessel.config.get('display_smoothing_max_mm',0))
        try:
            checks['meshes'][group]=mesh_topology(vertices,faces)
        except ValueError as error:
            raise ValueError(name+'/'+group+': '+str(error)) from error
        local=(vertices-vessel.origin)@vessel.rotation
        axis=vessel.config['separation_axis']
        checks['meshes'][group]['display_axis_range_mm']=[float(local[:,axis].min()*1000),float(local[:,axis].max()*1000)]
        with (out/(group+'_network.obj')).open('w') as f:
            f.write('# EMNS metres; merged branch junctions; terminal viewing caps\n')
            for vertex in vertices:
                f.write('v {:.8f} {:.8f} {:.8f}\n'.format(*vertex))
            for face in faces+1:
                f.write('f {} {} {}\n'.format(*face))
        data['meshes'].append({'network':group,'color':metadata['color'],
                              'vertices':((vertices-vessel.origin)*1000).round(4).tolist(),'faces':faces.tolist()})
    ranges=[entry['display_axis_range_mm'] for entry in checks['meshes'].values()]
    mesh_gap=max(ranges[1][0]-ranges[0][1],ranges[0][0]-ranges[1][1])
    if mesh_gap<=0:
        raise ValueError('Display meshes no longer have a separating slab')
    checks['display_mesh_separation_lower_bound_mm']=mesh_gap
    with (out/'centerlines.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['branch','network','parent','parent_s','s','x_m','y_m','z_m','radius_m'])
        for key,branch in vessel.branches.items():
            ss=np.linspace(0,1,151)
            curve=vessel.centerline(key,ss)
            data['curves'][key]={'network':branch['network'],'points':((curve-vessel.origin)*1000).round(4).tolist(),
                                 'radii':(vessel.radius(key,ss)*1000).round(4).tolist(),'parent':branch.get('parent')}
            for s,p,r in zip(ss,curve,vessel.radius(key,ss)):
                writer.writerow([key,branch['network'],branch.get('parent',''),branch.get('parent_s',''),s,*p,r])
    for particle in vessel.config['illustrative_particles']:
        xyz=(vessel.centerline(particle['branch'],particle['s'])-vessel.origin)*1000
        data['particles'].append(dict(particle,xyz=xyz.tolist(),network=vessel.branches[particle['branch']]['network']))
    for sphere,target in vessel.config['task_concept'].items():
        if not isinstance(target,dict):
            continue
        xyz=(vessel.centerline(target['target_branch'],1)-vessel.origin)*1000
        data['targets'].append(dict(target,xyz=xyz.tolist(),name='Target '+('1' if sphere=='sphere_1' else '2')))
    (out/'geometry_checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf-8')
    template=(HERE/'network_viewer_template.html').read_text(encoding='utf-8')
    (out/'viewer.html').write_text(template.replace('__DATA__',json.dumps(data,separators=(',',':'))),encoding='utf-8')
    print('{}: {} segments, two separate merged networks, gap >= {:.3f} mm'.format(name,len(vessel.branches),data['gap']),flush=True)
    return data


def comparison_figure(models):
    fig,axes=plt.subplots(1,2,figsize=(14,8))
    fig.patch.set_facecolor('#f4f7fb')
    for ax,data in zip(axes,models):
        ax.set_facecolor('#f4f7fb')
        # A metric-preserving orthographic view, slightly oblique to show depth.
        yaw=.25 if 'side-by-side' in data['title'] else .72
        pitch=.20
        R=np.array([[np.cos(yaw),0,np.sin(yaw)],
                    [np.sin(pitch)*np.sin(yaw),np.cos(pitch),-np.sin(pitch)*np.cos(yaw)],
                    [-np.cos(pitch)*np.sin(yaw),np.sin(pitch),np.cos(pitch)*np.cos(yaw)]])
        all_triangles=[]
        all_colors=[]
        all_depth=[]
        from matplotlib.colors import to_rgb
        for mesh in data['meshes']:
            xyz=np.array(mesh['vertices'])@R.T
            faces=np.array(mesh['faces'])
            tri=xyz[faces]
            normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
            normal/=np.maximum(np.linalg.norm(normal,axis=1)[:,None],1e-12)
            shade=.65+.35*np.abs(normal@np.array([.3,.4,.866]))
            rgb=np.array(to_rgb(mesh['color']))
            color=np.column_stack([shade[:,None]*rgb,np.full(len(tri),.96)])
            all_triangles.extend(tri[:,:,:2]);all_colors.extend(color);all_depth.extend(tri[:,:,2].mean(axis=1))
        order=np.argsort(all_depth)
        ax.add_collection(PolyCollection(np.array(all_triangles)[order],facecolors=np.array(all_colors)[order],edgecolors='none'))
        for particle in data['particles']:
            p=np.array(particle['xyz'])@R.T
            ax.scatter(p[0],p[1],s=36,c='#ffd24b',edgecolors='white',zorder=4)
        for target in data['targets']:
            p=np.array(target['xyz'])@R.T
            color=data['networks'][target['network']]['color']
            ax.scatter(p[0],p[1],marker='*',s=90,c=color,edgecolors='white',zorder=4)
            ax.annotate(target['name'],p[:2],xytext=(5,5),textcoords='offset points',fontsize=9,color=color)
        ax.autoscale_view()
        ax.margins(.10)
        ax.set_aspect('equal')
        ax.set_title(data['title'].replace('Branched paired vessels: ','').capitalize(),fontsize=15,pad=14)
        reported_gap=np.floor((data['gap']+1e-9)*100)/100
        ax.text(.02,.02,'14 segments / 2 connected networks\nInter-network gap bound: >= {:.2f} mm'.format(reported_gap),
                transform=ax.transAxes,fontsize=9,color='#52687b',bbox={'facecolor':'#f4f7fb','edgecolor':'none','alpha':.9})
        ax.set(xlabel='Projected horizontal coordinate (mm)',ylabel='Projected vertical coordinate (mm)')
        for side in ('top','right'):
            ax.spines[side].set_visible(False)
    fig.suptitle('Branched vessel networks with close approaches',fontsize=21,y=.97)
    fig.text(.5,.02,'Red: artery + side branches    Blue: vein + tributaries    Gold: enlarged sphere markers    Stars: illustrative targets',ha='center',fontsize=10)
    fig.subplots_adjust(top=.89,bottom=.1,wspace=.19)
    fig.savefig(HERE/'preview'/'branched_networks.png',dpi=180)
    plt.close(fig)


def main():
    comparison_figure([export_preset(name) for name in ('branched_parallel','branched_crossing')])


if __name__=='__main__':
    main()
