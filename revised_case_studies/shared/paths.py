"""File locations and figure style shared by the case studies."""

from __future__ import annotations

from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
ROOT = PACKAGE.parent
RESULTS_IN = ROOT / 'Results'                 # existing TFW results (read only)
RESULTS_OUT = PACKAGE / 'results'             # outputs of this package


def out_dir(case):
    d = RESULTS_OUT / case
    d.mkdir(parents=True, exist_ok=True)
    return d


# Colours used consistently across all figures.
C_INSIDE = '#4c9a5b'       # certified inside the TFW
C_BOUNDARY = '#e0a526'     # inside, near the boundary
C_OUTSIDE = '#c8553d'      # not certified / pointwise infeasible
C_FIXED = '#5a5a5a'
C_MOVING = '#2f6db3'
C_REF = '#222222'


def apply_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
        'legend.fontsize': 8, 'figure.dpi': 110, 'savefig.dpi': 300,
        'axes.spines.top': False, 'axes.spines.right': False,
    })


def save(fig, case, name):
    d = out_dir(case)
    for ext in ('png', 'pdf'):
        fig.savefig(d / f'{name}.{ext}', bbox_inches='tight')
    print(f'saved {d / name}.png/.pdf')
