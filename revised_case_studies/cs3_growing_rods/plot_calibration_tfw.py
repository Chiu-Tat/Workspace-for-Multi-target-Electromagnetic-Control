"""Render Case Study 3 calibration and TFW panels from recorded data and CAD.

Run this script with the project Python environment containing NumPy, SciPy,
Matplotlib and Pillow. The Blender render and its projected landmarks are
produced by render_model.py. No generative image model is used.
"""
import csv, itertools, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import TwoSlopeNorm
from PIL import Image
from scipy.optimize import linprog

from .experiment_field import coil_basis, rotation_plane
from ..shared.paths import out_dir

here = Path(__file__).resolve().parent
data = here / 'data'
assets = here / 'assets'
output = out_dir('cs3_growing_rods') / 'publication'
output.mkdir(parents=True, exist_ok=True)
cfg = json.loads((data / 'experiment.json').read_text())
limit = cfg['current_limit_A'] * cfg['current_margin']
triples=np.array(list(itertools.combinations(range(10),3)))

def capacities(matrix):
    # In four dimensions a facet has at least three linearly independent generators.
    _, sigma, vt=np.linalg.svd(matrix.T[triples],full_matrices=True)
    normals=vt[sigma[:,-1]>1e-12,-1,:]
    if np.linalg.matrix_rank(matrix)!=4:
        return np.array([np.nan,np.nan])
    support=limit*np.abs(normals@matrix).sum(axis=1)
    left=np.linalg.norm(normals[:,:2],axis=1)
    right=np.linalg.norm(normals[:,2:],axis=1)
    return np.array([np.min((support-5*right)[left>1e-10]/left[left>1e-10]),
                     np.min((support-5*left)[right>1e-10]/right[right>1e-10])])

def matrix_at(s,delta,z):
    p=np.array([[-s/2,-24.5-delta/2,z],[s/2,-24.5+delta/2,z]])*.001
    return np.vstack([coil_basis(q,cfg['coil_parameters'])[0][[0,2]]*[ [1000],[-1000] ] for q in p])
actual=matrix_at(56,25,110)
trial_matrix = np.vstack([
    np.vstack(rotation_plane(m['axis'])[1:]) @ coil_basis(m['position_m'], cfg['coil_parameters'])[0]
    for m in cfg['magnets']
])
np.testing.assert_allclose(actual, trial_matrix * 1000)
print('Actual capacity by role',capacities(actual))
print('Capacities at 56 mm, 110 mm', [min(capacities(matrix_at(56,d,110))) for d in (0,25,50)])
for amp in [(20,5),(5,20),(23,5)]:
    for a,b in itertools.product(np.linspace(0,2*np.pi,24,endpoint=False),repeat=2):
        target = np.array([amp[0]*np.cos(a), amp[0]*np.sin(a), amp[1]*np.cos(b), amp[1]*np.sin(b)])
        assert linprog(np.zeros(10), A_eq=actual,b_eq=target,bounds=[(-limit, limit)]*10).success

plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
ink='#25374a'; teal='#16736a'; orange='#ad5b22'
f=plt.figure(figsize=(9.5,6.5))
grid=f.add_gridspec(2,3,left=.07,right=.985,top=.94,bottom=.095,hspace=.40,wspace=.47)
bottom=grid[1,:].subgridspec(1,6,wspace=.18,width_ratios=[1,1,1,.06,.42,1.05])
cad=assets/'cs3_blender_model.png'
ax=f.add_subplot(grid[0,0])
cad_image=Image.open(cad)
ax.imshow(cad_image.crop(cad_image.getbbox()))
ax.set_axis_off()
ax.set_title('(a) Spine and dual rods',loc='left',fontweight='bold')
ax.text(.5,-.065,'Oblique view',ha='center',transform=ax.transAxes)
bounds=cad_image.getbbox()
landmarks=json.loads(Path(__file__).with_name('model_landmarks.json').read_text())
labels=[('spine','Spine',(1.5,.55),'left'),
        ('rod','Telescopic\nrod',(1.5,.88),'left'),
        ('housing','Screw-drive\nhousing',(-.22,.36),'right'),
        ('left_magnet','Magnet\nrotor',(-.22,.19),'right')]
for key,label,position,alignment in labels:
    x,y=landmarks[key]
    ax.annotate(label,xy=(x-bounds[0],y-bounds[1]),xycoords='data',
                xytext=position,textcoords='axes fraction',ha=alignment,va='center',
                fontsize=7,color=ink,annotation_clip=False,
                bbox=dict(facecolor='white',edgecolor='none',pad=.4),
                arrowprops=dict(arrowstyle='-',color=ink,lw=.65,
                                connectionstyle='angle,angleA=0,angleB=90,rad=0'))
# Dimension lines follow the projection of the coil-frame x and y directions.
import matplotlib.patheffects as pe
dimensions=json.loads(Path(__file__).with_name('model_dimensions.json').read_text())
points={key:np.array(value)-np.array(bounds[:2]) for key,value in dimensions.items()}
dimension_color='#007b83'
for first,second in [('left_centre','s_left'),('corner','s_right'),
                     ('corner','offset_lower'),('right_centre','offset_upper')]:
    ax.plot(*np.stack([points[first],points[second]]).T,color=dimension_color,lw=.6,
            ls=(0,(2,2)),clip_on=False,
            path_effects=[pe.Stroke(linewidth=1.6,foreground='white'),pe.Normal()])
for first,second in [('s_left','s_right'),('offset_lower','offset_upper')]:
    arrow=ax.annotate('',xy=points[first],xytext=points[second],annotation_clip=False,
                     arrowprops=dict(arrowstyle='<->',color=dimension_color,lw=.9,
                                     shrinkA=0,shrinkB=0))
    arrow.arrow_patch.set_path_effects([pe.Stroke(linewidth=2,foreground='white'),pe.Normal()])
mid=(points['s_left']+points['s_right'])/2
ax.annotate('Lateral spacing\n$s=56$ mm',xy=mid,xytext=(0,-4),textcoords='offset points',
            ha='center',va='top',fontsize=6.6,color=dimension_color,
            bbox=dict(facecolor='white',alpha=.9,edgecolor='none',pad=.7),annotation_clip=False)
mid=(points['offset_lower']+points['offset_upper'])/2
ax.annotate('Offset\n$\\Delta=25$ mm',xy=mid,xytext=(4,0),textcoords='offset points',
            ha='left',va='center',fontsize=6.6,color=dimension_color,
            bbox=dict(facecolor='white',alpha=.9,edgecolor='none',pad=.7),annotation_clip=False)
# Working-principle inset in the style of the omnidirectional-requirement figure:
# a magnified view of the left rotor (along its axis) with a star of field arrows;
# grey shades show successive field directions of the rotating field B(t).
from matplotlib.patches import Wedge, Circle, FancyArrowPatch
from matplotlib.lines import Line2D as _Line
rotor_ax=ax; rotor_xy=np.array(landmarks['left_magnet'],float)-np.array(bounds[:2])
ins=f.add_axes([.004,.765,.12,.17]); ins.set(xlim=(-1.05,1.05),ylim=(-1.05,1.05),aspect='equal'); ins.set_axis_off()
ins.add_patch(Circle((0,0),1.0,fc='#efefef',ec='#5a5a5a',lw=.8))
# 3-D rotor: a diametrically magnetized cylinder (red N half, blue S half) whose
# axis is vertical, with a star of field arrows in the plane normal to the axis.
from mpl_toolkits.mplot3d import proj3d
class _Arrow3D(FancyArrowPatch):
    def __init__(self,p0,p1,**kw):
        super().__init__((0,0),(0,0),**kw); self._p=(p0,p1)
    def do_3d_projection(self,renderer=None):
        (x0,y0,z0),(x1,y1,z1)=self._p
        xs,ys,zs=proj3d.proj_transform([x0,x1],[y0,y1],[z0,z1],self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1])); return np.min(zs)
ins3=f.add_axes([.004,.752,.12,.17],projection='3d',computed_zorder=False)
ELEV,AZIM=32,-62
ins3.set_proj_type('ortho'); ins3.view_init(ELEV,AZIM)
ins3.set(xlim=(-1,1),ylim=(-1,1),zlim=(-1,1)); ins3.set_box_aspect((1,1,1),zoom=1.12)
ins3.set_axis_off(); ins3.patch.set_alpha(0); ins3.set_facecolor((0,0,0,0))
from matplotlib.colors import to_rgb
now=np.radians(35); rc,hc=.30,.80; red,blue=np.array(to_rgb('#d0453f')),np.array(to_rgb('#3f6fb5'))
view=np.array([np.cos(np.radians(AZIM)),np.sin(np.radians(AZIM))])
def _col(t,shade):
    base=np.where((np.cos(t-now)>=0)[...,None],red,blue)
    return np.clip(base*shade[...,None],0,1)
t=np.linspace(0,2*np.pi,97); tm=(t[:-1]+t[1:])/2
T,Z=np.meshgrid(t,[-hc/2,hc/2])
side=_col(np.tile(tm,(1,1)),.62+.38*np.clip(np.cos(tm-np.radians(AZIM+40)),0,1)[None,:])
fc=np.ones((2,97,3)); fc[0,:-1]=side[0]
ins3.plot_surface(rc*np.cos(T),rc*np.sin(T),Z,facecolors=fc,shade=False,linewidth=0,antialiased=False,zorder=2)
R,T2=np.meshgrid([0,rc],t)
cap=np.ones((97,2,3)); cap[:-1,0]=_col(tm,np.full_like(tm,1.0))
ins3.plot_surface(R*np.cos(T2),R*np.sin(T2),np.full_like(R,hc/2),facecolors=cap,shade=False,linewidth=0,antialiased=False,zorder=2.5)
for k in range(8):                       # oldest (light) to current (black)
    ang=now-np.radians(45)*(7-k); g=0.72-0.72*k/7
    front=np.cos(ang)*view[0]+np.sin(ang)*view[1]>-0.2
    ins3.add_artist(_Arrow3D((.31*np.cos(ang),.31*np.sin(ang),0),(.95*np.cos(ang),.95*np.sin(ang),0),
                    arrowstyle='-|>',mutation_scale=7,color=(g,g,g),lw=.9,shrinkA=0,shrinkB=0,zorder=3 if front else 1))
ins3.text(0,0,.78,r'$\mathbf{B}(t)$',fontsize=6.8,ha='center',va='bottom',color=ink,zorder=4)
f.canvas.draw()
def _circle_px(axes,center,radius_px=None,radius_data=None):
    c=axes.transData.transform(center)
    r=radius_px if radius_px is not None else abs(axes.transData.transform((radius_data,0))[0]-axes.transData.transform((0,0))[0])
    return c,r
c1,r1=_circle_px(rotor_ax,rotor_xy,radius_px=f.dpi*.085)
c2,r2=_circle_px(ins,(0,0),radius_data=1.0)
to_fig=f.transFigure.inverted()
small=Circle(to_fig.transform(c1),r1/f.bbox.width,transform=f.transFigure,fill=False,ec='#5a5a5a',lw=.8)
f.add_artist(small)
# External tangents between the two circles (display coordinates).
d=c2-c1; L=np.linalg.norm(d); u=d/L; v=np.array([-u[1],u[0]]); cosa=(r1-r2)/L; sina=np.sqrt(1-cosa**2)
for sgn in (1,-1):
    n=cosa*u+sgn*sina*v
    p1,p2=c1+r1*n,c2+r2*n
    f.add_artist(_Line(*to_fig.transform(np.array([p1,p2])).T,transform=f.transFigure,color='#5a5a5a',lw=.7))
ax=f.add_subplot(grid[0,1])
photo=assets/'calibration_setup_photo.jpg'
ax.imshow(np.rot90(np.asarray(Image.open(photo).convert('RGB')),3))
ax.set_axis_off()
ax.set_title('(b) Push-force fixture',loc='left',fontweight='bold')

records=[]
for session in ('session_20261103_154603','session_20261103_160426'):
    for row in csv.DictReader((data / f'{session}_force_records.csv').open()):
        if (float(row['axis_z']),int(row['sense']),float(row['frequency_Hz']),row['force_unit'])==(1.,-1,1.,'N'):
            records.append((float(row['amplitude_setpoint_mT']),float(row['force']),
                            session.endswith('154603') and row['record'] in {'11','12'}))
b,force,flag=np.array(records).T
flag=flag.astype(bool)
def fit(x,y):
    m,c=np.polyfit(x,y,1)
    return m,c,1-np.sum((y-m*x-c)**2)/np.sum((y-y.mean())**2)
b,force=b[~flag],force[~flag]
m,c,r=fit(b,force)
print('Filtered fit',len(b),(m,c,r),'20mT',m*20+c)
ax=f.add_subplot(grid[0,2])
xx=np.array([b.min(),b.max()])
ax.plot(xx,m*xx+c,color=ink,label=f'Linear fit ($R^2={r:.3f}$)',lw=1.4)
ax.scatter(b,force,color=teal,s=19,zorder=3,label='15 readings')
ax.set(xlabel='Commanded amplitude (mT)',ylabel='Push force (N)',xticks=[20,30,40,50,60],ylim=(1,11.7))
ax.set_title('(c) Bench calibration',loc='left',fontweight='bold')
ax.legend(loc='upper left',fontsize=7.4,frameon=False)
ax.grid(alpha=.18)

ss=np.arange(16.,101.,2.)
zs=np.arange(70.,151.,2.)
levels=[0,5,10,15,20,25,30,40,50,65]
for index,delta in enumerate((0,25,50)):
    ax=f.add_subplot(bottom[index])
    v=np.array([[min(capacities(matrix_at(s,delta,z))) for s in ss] for z in zs])
    print('offset',delta,'grid range',float(np.min(v)),float(np.max(v)),flush=True)
    cn=ax.contourf(ss,zs,np.maximum(0,v),levels=levels,cmap='BrBG',norm=TwoSlopeNorm(vcenter=20,vmin=0,vmax=65),extend='max')
    ax.contour(ss,zs,v,levels=[20],colors=ink,linewidths=1.5)
    ax.set(xlabel='Lateral spacing $s$ (mm)',xticks=[20,40,60,80,100],yticks=[70,90,110,130,150])
    if index==0: ax.set_ylabel('Rotor height $z$ (mm)')
    else: ax.set_yticklabels([])
    # Bench (s, z) in every slice: filled at the recorded offset, hollow at the others.
    if delta==25:
        ax.plot(56,110,'*',color='#bc3839',mec='white',mew=.6,ms=11)
        ax.annotate('Bench pose',(56,110),xytext=(62,116),fontsize=7.5)
    else:
        ax.plot(56,110,'*',mfc='none',mec='#bc3839',mew=.9,ms=10)
    ax.set_title(f'({chr(100+index)}) Offset $\\Delta={delta}$ mm',loc='left',fontweight='bold')
cb=f.colorbar(cn,cax=f.add_subplot(bottom[3]),ticks=[0,10,20,30,50,65])
cb.set_label('Active capacity (mT)',fontsize=8,labelpad=1)
cb.ax.tick_params(labelsize=7)
# (g) Capacity along the offset at the bench spacing and height.
deltas=np.arange(0.,60.5,1.)
caps=np.array([capacities(matrix_at(56,d,110)) for d in deltas])
ax=f.add_subplot(bottom[5])
ax.plot(deltas,caps[:,0],color=teal,lw=1.1,label='Left rod active')
ax.plot(deltas,caps[:,1],color=orange,lw=1.1,label='Right rod active')
ax.plot(deltas,caps.min(axis=1),color=ink,lw=1.8,label='Smaller of the two')
ax.axhline(20,color=ink,lw=.8,ls=(0,(3,2)))
ax.text(59,20.6,'20 mT task',fontsize=6.6,ha='right',va='bottom',color=ink)
for d,mk in ((0,'none'),(25,'#bc3839'),(50,'none')):
    v=min(capacities(matrix_at(56,d,110)))
    ax.plot(d,v,'*',mfc=mk,mec='#bc3839',mew=.9,ms=10,zorder=5)
    ax.annotate(f'{v:.1f}',(d,v),xytext=(4,-2),textcoords='offset points',fontsize=6.6,va='top')
ax.set(xlabel='Offset $\\Delta$ (mm)',ylabel='Active capacity (mT)',xlim=(0,60),xticks=[0,25,50],ylim=(0,max(45,np.nanmax(caps)*1.05)))
ax.set_title('(g) Bench $s$, $z$',loc='left',fontweight='bold')
ax.legend(loc='upper left',fontsize=6.6,frameon=False,handlelength=1.4)
ax.grid(alpha=.18)
print('capacity vs offset: min at 0/25/50',[round(min(capacities(matrix_at(56,d,110))),2) for d in (0,25,50)])
f.savefig(output/'cs3_calibration_tfw.pdf')

plt.close(f)
