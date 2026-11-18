#!/usr/bin/env python3
# Run from the repository root:
#   .venv/bin/python -B -m revised_case_studies.cs1_cylinder_interaction.run_tfw3d_corrected --layout single [--resume]
"""Fresh 3-D (x, y, beta) affine-hybrid TFW for the Case Study 1 layouts.

Runs the unchanged Case 6 runner (cases/case_6_interaction_obstacle_avoidance/
workspace_hybrid.py) with the corrected By/dy derivative now in emns, but writes
to new files so the archived pre-correction results in Results/ stay untouched.
Outputs: results/cs1/tfw3d_corrected/tfw3d_<layout>.json (+ .png, .checkpoint.json).
"""
import multiprocessing as mp
import os
import sys

from cases.case_6_interaction_obstacle_avoidance import workspace_hybrid as W
from revised_case_studies.shared.paths import out_dir

OUT = out_dir('cs1') / 'tfw3d_corrected'


def _paths(layout=W.DEFAULT_LAYOUT):
    stem = str(OUT / f'tfw3d_{layout}')
    return stem + '.json', stem + '.png', stem + '.checkpoint.json'


if __name__ == '__main__':
    mp.freeze_support()
    W.result_paths = _paths
    sys.exit(W.main())
