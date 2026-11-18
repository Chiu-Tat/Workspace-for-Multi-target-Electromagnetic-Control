"""Plot the Case Study 3 dual-rod phantom trial from its recorded files.

Inputs (session_20261103_174942): three extracted camera frames,
linearity.csv (camera shape metrics and the five tracked marker centres), telemetry.csv (field targets
and rod commands) and events.csv (video start/stop). Times are relative to the video start.
No generative image model is used.
"""
from pathlib import Path
from ast import literal_eval
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
from matplotlib.patches import Patch
from PIL import Image

from ..shared.paths import out_dir

run = Path(__file__).with_name('data')
output = out_dir('cs3_growing_rods') / 'publication'
output.mkdir(parents=True, exist_ok=True)
ink, teal, orange, grey = '#25374a', '#16736a', '#ad5b22', '#8a929b'
SHADE = '#ccc36f'
WINDOWS = [(19.5, 22.9), (114.8, 116.7)]      # initial and final hold summaries (trial time, s)
FRAMES = [(2, 'Initial hold'), (46, 'Intermediate hold'), (97, 'Final hold')]   # video time, s

plt.rcParams.update({'font.size': 8, 'pdf.fonttype': 42, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.titlesize': 8.5})
rows = lambda name: list(csv.DictReader((run / name).open()))
events = rows('events.csv')
start = float(next(x['t_s'] for x in events if 'video 001 start' in x['event']))
stop = float(next(x['t_s'] for x in events if 'video 001 stop' in x['event']))
lin = [r for r in rows('linearity.csv') if r['valid'] == 'True' and start <= float(r['t_s']) <= stop]
tele = [r for r in rows('telemetry.csv') if start <= float(r['t_s']) <= stop]
times = lambda rs: np.array([float(r['t_s']) - start for r in rs])
values = lambda rs, key: np.array([float(r[key]) if r[key] else np.nan for r in rs])
lt, t = times(lin), times(tele)


def camera_frame(second):
    return Image.open(run / f'frame_{second}.png')


def markers(window):
    """Median marker centres (camera px) over valid frames in a trial-time window."""
    pts = [np.array(literal_eval(r['segments_px'])) for r in lin if window[0] <= float(r['t_s']) <= window[1]
           and r['n_segments'] == '5']
    return np.median(np.stack(pts), axis=0), len(pts)


f = plt.figure(figsize=(9.5, 6.6))
outer = f.add_gridspec(2, 1, left=.06, right=.985, top=.95, bottom=.075, hspace=.22, height_ratios=[1, 1.3])
top = outer[0].subgridspec(1, 4, wspace=.08, width_ratios=[1, 1, 1, 1.25])
for i, (stamp, label) in enumerate(FRAMES):
    ax = f.add_subplot(top[i])
    ax.imshow(camera_frame(stamp).crop((42, 19, 402, 501)))
    ax.set_title(f'({chr(97 + i)}) {label}', loc='left', fontweight='bold')
    ax.text(.04, .04, f'{stamp} s', transform=ax.transAxes, color='white', fontsize=7.5,
            bbox={'facecolor': ink, 'alpha': .83, 'pad': 1.6, 'edgecolor': 'none'})
    ax.set_axis_off()

# (d) Spine shape from the tracked marker centres: initial versus final hold.
ax = f.add_subplot(top[3])
summary = {}
for window, color, name in ((WINDOWS[0], grey, 'Initial hold'), (WINDOWS[1], teal, 'Final hold')):
    p, n = markers(window)
    sel = [r for r in lin if window[0] <= float(r['t_s']) <= window[1]]
    summary[name] = (np.median(values(sel, 'linearity_ratio')), np.median(values(sel, 'max_tilt_deg')), n)
    ax.plot(p[:, 0], p[:, 1], '-o', color=color, lw=1.6, ms=4.5, mec='white', mew=.6, zorder=3,
            label=name)
    ax.plot(p[[0, -1], 0], p[[0, -1], 1], color=color, lw=.7, ls=(0, (3, 2)), zorder=2)
ax.invert_yaxis(); ax.set_aspect('equal')
ax.set(xlabel='Camera $u$ (px)', ylabel='Camera $v$ (px)')
ax.set_xlim(np.mean(ax.get_xlim()) - 60, np.mean(ax.get_xlim()) + 60)
ax.legend(loc='center right', fontsize=6.6, frameon=False, handlelength=1.6)
(l0, a0, _), (l1, a1, _) = summary['Initial hold'], summary['Final hold']
ax.text(.98, .98, f'Linearity {l0:.3f} → {l1:.3f}\nMax. angle {a0:.1f}° → {a1:.1f}°', transform=ax.transAxes,
        ha='right', va='top', fontsize=6.6, color=ink, bbox={'fc': 'white', 'ec': 'none', 'alpha': .9, 'pad': 1})
ax.set_title('(d) Tracked marker centres', loc='left', fontweight='bold')
ax.grid(alpha=.18)

# (e) Time series.
sub = outer[1].subgridspec(3, 1, hspace=.12, height_ratios=[.7, 1, 1])
axs = [f.add_subplot(sub[0])]
axs += [f.add_subplot(sub[k], sharex=axs[0]) for k in range(1, 3)]
ax = axs[0]
handles = [Patch(fc='#5c656e', label='Push'), Patch(fc='#5c656e', alpha=.45, label='Pull')]
ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(1, 1), borderaxespad=0,
          fontsize=6.6, frameon=False, ncol=2, handlelength=1.4, columnspacing=1)
ax.set_title('(e) Camera and controller records', loc='left', fontweight='bold')
L, R = values(tele, 'left_target_mT'), values(tele, 'right_target_mT')
change = np.r_[0, np.flatnonzero((np.diff(L) != 0) | (np.diff(R) != 0)) + 1, len(t) - 1]
for a in change[1:-1]:
    [x.axvline(t[a], color=ink, lw=.5, ls=(0, (1, 2)), zorder=1) for x in axs]
# Rod commands as a timeline.
for lane, (k, color) in enumerate((('left', teal), ('right', orange))):
    c = values(tele, k + '_command')
    for sign, alpha in ((1, 1.0), (-1, .45)):
        on = c == sign
        idx = np.flatnonzero(on)
        for run_ in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(run_):
                ax.broken_barh([(t[run_[0]], t[run_[-1]] - t[run_[0]])], (1 - lane - .35, .7), facecolor=color,
                               alpha=alpha, edgecolor='none')
ax.set(ylim=(-.6, 1.6), yticks=[1, 0], yticklabels=['Left', 'Right'])
ax = axs[1]
ax.plot(lt, values(lin, 'linearity_ratio'), color=ink, lw=.8)
ax.set(ylabel='Linearity', ylim=(.962, .998), yticks=[.97, .98, .99])
ax = axs[2]
ax.plot(lt, values(lin, 'max_tilt_deg'), color=ink, lw=.8)
ax.set(ylabel='Max. marker\nangle (°)', xlim=(0, 100), xlabel='Time from video start (s)')
for x in axs:
    x.grid(alpha=.18)
    for a, b in WINDOWS:
        x.axvspan(a - start, b - start, color=SHADE, alpha=.5, lw=0, zorder=0)
    if x is not axs[-1]:
        x.tick_params(labelbottom=False)
print('summary', summary, 'stage changes (s)', [round(t[c], 1) for c in change[1:-1]])
f.savefig(output / 'cs3_phantom_trial.pdf')
plt.close(f)
