#!/usr/bin/env python3
"""Synthetic formula/input fixtures; these are not physical data."""
import csv, tempfile
from pathlib import Path
from validate_measurements import validate, HERE
FIELDS='sample_id split x_m y_m z_m moving_alpha_rad moving_beta_rad fixed_x_m fixed_y_m fixed_z_m fixed_alpha_rad fixed_beta_rad moving_moment_Am2 fixed_moment_Am2 measured_By_T measured_Bz_T measured_fy_N measured_fz_N std_By_T std_Bz_T std_fy_N std_fz_N temperature_C notes'.split()
BASE={'x_m':'-0.03','y_m':'-0.03','z_m':'0.082','moving_alpha_rad':'1.5707963267948966','moving_beta_rad':'1.5707963267948966','fixed_x_m':'0.02','fixed_y_m':'0','fixed_z_m':'0.082','fixed_alpha_rad':'1.5707963267948966','fixed_beta_rad':'1.5707963267948966','moving_moment_Am2':'0.12','fixed_moment_Am2':'0.145'}
def write(path,rows):
    with path.open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
def main():
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'fixture.csv'; fit={**BASE,'sample_id':'synthetic-fit','split':'fit','measured_fy_N':'0','std_fy_N':'0.001'}; hold={**BASE,'sample_id':'synthetic-held','split':'held_out','measured_fz_N':'0'}
        write(p,[fit,hold]); result=validate(p,HERE/'config.json'); assert result['held_out_finite_reference_summary']['fz']['n']==1
        bad={**hold,'sample_id':'bad-moment','fixed_moment_Am2':'0.2'}; write(p,[fit,bad])
        try: validate(p,HERE/'config.json'); raise AssertionError('unsupported moment accepted')
        except ValueError as e: assert 'unsupported' in str(e)
    print('synthetic fixtures passed: force-only rows and metadata rejection')
if __name__=='__main__': main()
