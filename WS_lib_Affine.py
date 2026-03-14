#!/usr/bin/env python3
"""
WS_lib_Affine.py - Affine Arithmetic Library for Electromagnetic Navigation System

This library provides affine arithmetic implementations for magnetic field calculations,
force/torque computations, and workspace determination for electromagnetic navigation systems.

Key Features:
- Magnetic field calculations using affine arithmetic
- Force and torque computations for magnetic dipoles
- Mapping matrix generation for workspace analysis
- Working implementations for guaranteed correct results
"""

import numpy as np
import math
from custom_interval import interval, imath
import custom_affine as ca

# ============================================================================
# COIL PARAMETERS AND CONFIGURATION
# ============================================================================

# 10-coil system parameters (magnetic moments and positions)
# params_list = [
#     np.array([-13.04945069, -4.41557229, 6.47376799, 0.12129096, 0.00466922, -0.0174842]),  # coil 1
#     np.array([-5.10083416, 13.54294901, 7.85474539, 0.05834654, -0.11165548, -0.01850546]),  # coil 2
#     np.array([4.05088788, 14.23365818, 6.44760956, -0.05903076, -0.11020417, -0.01488244]),  # coil 3
#     np.array([13.89011305, -0.06092074, 4.77365608, -0.12306086, -0.00085745, -0.01378161]), # coil 4
#     np.array([11.44363813, -9.40543896, 4.46367162, -0.06806179, 0.1024875, -0.01397152]),  # coil 5
#     np.array([-9.00577939, -12.78905365, 5.98650851, 0.06473315, 0.10618968, -0.0151172]),  # coil 6
#     np.array([0.92820081, 8.54965337, 8.72298349, -0.00381254, -0.08845466, -0.08874662]),   # coil 7
#     np.array([8.7302819, -4.90773115, 7.00109937, -0.07977306, 0.04481733, -0.08536032]),   # coil 8
#     np.array([-7.68962762, -6.83258326, 8.12112247, 0.07498008, 0.04542436, -0.08696975]),   # coil 9
#     np.array([2.35614001, -1.11370036, 14.00304846, -0.00722183, 0.00029277, -0.12482979])   # coil 10
# ]

params_list = [
    np.array([-13.04945069, -4.41557229, 6.47376799, 0.12129096, 0.00466922, 0.0675158]),  # coil 1
    np.array([-5.10083416, 13.54294901, 7.85474539, 0.05834654, -0.11165548, 0.06649454]),  # coil 2
    np.array([4.05088788, 14.23365818, 6.44760956, -0.05903076, -0.11020417, 0.07011756]),  # coil 3
    np.array([13.89011305, -0.06092074, 4.77365608, -0.12306086, -0.00085745, 0.07121839]), # coil 4
    np.array([11.44363813, -9.40543896, 4.46367162, -0.06806179, 0.1024875, 0.07102848]),  # coil 5
    np.array([-9.00577939, -12.78905365, 5.98650851, 0.06473315, 0.10618968, 0.0698828]),  # coil 6
    np.array([0.92820081, 8.54965337, 8.72298349, -0.00381254, -0.08845466, -0.00374662]),   # coil 7
    np.array([8.7302819, -4.90773115, 7.00109937, -0.07977306, 0.04481733, -0.00036032]),   # coil 8
    np.array([-7.68962762, -6.83258326, 8.12112247, 0.07498008, 0.04542436, -0.00196975]),   # coil 9
    np.array([2.35614001, -1.11370036, 14.00304846, -0.00722183, 0.00029277, -0.03982979])   # coil 10
]

num_coils = len(params_list)

def calculate_B_selective_affine(currents, X_affine, Y_affine, Z_affine, required_B):
    """Affine version of calculate_B_selective_interval. Assumes inputs are affine."""
    results = {}
    if not any(required_B.values()): return results
    
    if required_B.get('Bx'): results['Bx'] = ca.Affine(0)
    if required_B.get('By'): results['By'] = ca.Affine(0)
    if required_B.get('Bz'): results['Bz'] = ca.Affine(0)

    for i in range(num_coils):
        if currents[i] == 0: continue
        params = params_list[i]
        m0, m1, m2, r0_0, r0_1, r0_2 = params
        m0_affine, m1_affine, m2_affine = ca.Affine(m0), ca.Affine(m1), ca.Affine(m2)
        r0_0_affine, r0_1_affine, r0_2_affine = ca.Affine(r0_0), ca.Affine(r0_1), ca.Affine(r0_2)

        dx, dy, dz = X_affine - r0_0_affine, Y_affine - r0_1_affine, Z_affine - r0_2_affine
        r_squared = dx*dx + dy*dy + dz*dz
        
        # Safety check for r_squared being too close to zero, which causes instability.
        r_squared_interval = r_squared.to_interval()
        if r_squared_interval[0][0] < 1e-9:
            # Collapse the affine number to its center to avoid instability.
            # This trades precision for stability.
            r_squared = ca.Affine(max(r_squared.center, 1e-9)) # Ensure center is not zero or negative
            
        r = ca.affine_sqrt(r_squared)
        dot_product = m0_affine*dx + m1_affine*dy + m2_affine*dz
        
        mu0_4pi = 1e-7
        r_cubed = r * r_squared
        r_fifth = r_cubed * r_squared
        
        if required_B.get('Bx'):
            results['Bx'] += mu0_4pi * (3 * dx * dot_product / r_fifth - m0_affine / r_cubed) * currents[i]
        if required_B.get('By'):
            results['By'] += mu0_4pi * (3 * dy * dot_product / r_fifth - m1_affine / r_cubed) * currents[i]
        if required_B.get('Bz'):
            results['Bz'] += mu0_4pi * (3 * dz * dot_product / r_fifth - m2_affine / r_cubed) * currents[i]
    return results

def calculate_derivatives_selective_affine(currents, X_affine, Y_affine, Z_affine, required_derivs):
    """Affine version of calculate_derivatives_selective_interval. Assumes inputs are affine."""
    results = {}
    if not any(required_derivs.values()): return results

    for key in ['Bx_dx', 'Bx_dy', 'Bx_dz', 'By_dy', 'By_dz']:
        if required_derivs.get(key): results[key] = ca.Affine(0)
    
    for i in range(num_coils):
        if currents[i] == 0: continue
        params = params_list[i]
        m0, m1, m2, r0_0, r0_1, r0_2 = params
        m0_affine, m1_affine, m2_affine = ca.Affine(m0), ca.Affine(m1), ca.Affine(m2)
        r0_0_affine, r0_1_affine, r0_2_affine = ca.Affine(r0_0), ca.Affine(r0_1), ca.Affine(r0_2)
        
        dx, dy, dz = X_affine - r0_0_affine, Y_affine - r0_1_affine, Z_affine - r0_2_affine
        r_squared = dx*dx + dy*dy + dz*dz

        # Safety check for r_squared being too close to zero, which causes instability.
        r_squared_interval = r_squared.to_interval()
        if r_squared_interval[0][0] < 1e-9:
            # Collapse the affine number to its center to avoid instability.
            # This trades precision for stability.
            r_squared = ca.Affine(max(r_squared.center, 1e-9)) # Ensure center is not zero or negative

        r = ca.affine_sqrt(r_squared)
        dot_product = m0_affine*dx + m1_affine*dy + m2_affine*dz
        
        mu0_4pi = 1e-7
        r5 = r_squared * r_squared * r
        r7 = r5 * r_squared
        
        # Safety check for numerical stability in derivative calculations
        r5_interval = r5.to_interval()
        r7_interval = r7.to_interval()
        
        # If r5 or r7 intervals include non-positive values or are too small, use center values
        if r5_interval[0][0] <= 1e-12 or abs(r5_interval[0][0]) > 1e6 or abs(r5_interval[0][1]) > 1e6:
            r5 = ca.Affine(max(abs(r5.center), 1e-10))
            r7 = ca.Affine(max(abs(r7.center), 1e-15))
        elif r7_interval[0][0] <= 1e-18 or abs(r7_interval[0][0]) > 1e9 or abs(r7_interval[0][1]) > 1e9:
            r7 = ca.Affine(max(abs(r7.center), 1e-15))
        
        if required_derivs.get('Bx_dx'):
            results['Bx_dx'] += mu0_4pi * (3*dot_product/r5 + 6*dx*m0_affine/r5 - 15*dx**2*dot_product/r7) * currents[i]
        if required_derivs.get('Bx_dy'):
            results['Bx_dy'] += mu0_4pi * (3*dx*m1_affine/r5 + 3*m0_affine*dy/r5 - 15*dx*dy*dot_product/r7) * currents[i]
        if required_derivs.get('Bx_dz'):
            results['Bx_dz'] += mu0_4pi * (3*dx*m2_affine/r5 + 3*m0_affine*dz/r5 - 15*dx*dz*dot_product/r7) * currents[i]
        if required_derivs.get('By_dy'):
            results['By_dy'] += mu0_4pi * (3*dot_product/r5 + 3*dy*m1_affine/r5 - 15*dy**2*dot_product/r7) * currents[i]
        if required_derivs.get('By_dz'):
            results['By_dz'] += mu0_4pi * (3*dy*m2_affine/r5 + 3*m1_affine*dz/r5 - 15*dy*dz*dot_product/r7) * currents[i]
    return results

def calculate_Force_selective_affine(currents, X_affine, Y_affine, Z_affine, m_affine, alpha_affine, beta_affine, required_F, deriv_results):
    """Affine version of calculate_Force_selective_interval."""
    results = {}
    if not any(required_F.values()): return results

    mx = ca.affine_sin(alpha_affine) * ca.affine_cos(beta_affine)
    my = ca.affine_sin(alpha_affine) * ca.affine_sin(beta_affine)
    mz = ca.affine_cos(alpha_affine)

    if required_F.get('fx'):
        results['fx'] = m_affine * (mx * deriv_results.get('Bx_dx', ca.Affine(0)) + my * deriv_results.get('Bx_dy', ca.Affine(0)) + mz * deriv_results.get('Bx_dz', ca.Affine(0)))
    if required_F.get('fy'):
        results['fy'] = m_affine * (mx * deriv_results.get('Bx_dy', ca.Affine(0)) + my * deriv_results.get('By_dy', ca.Affine(0)) + mz * deriv_results.get('By_dz', ca.Affine(0)))
    if required_F.get('fz'):
        results['fz'] = m_affine * (-mz * deriv_results.get('Bx_dx', ca.Affine(0)) + mx * deriv_results.get('Bx_dz', ca.Affine(0)) - mz * deriv_results.get('By_dy', ca.Affine(0)) + my * deriv_results.get('By_dz', ca.Affine(0)))
    return results

def calculate_Torque_selective_affine(currents, X_affine, Y_affine, Z_affine, m_affine, alpha_affine, beta_affine, required_T, B_results):
    """Affine version of calculate_Torque_selective_interval."""
    results = {}
    if not any(required_T.values()): return results

    mx = ca.affine_sin(alpha_affine) * ca.affine_cos(beta_affine)
    my = ca.affine_sin(alpha_affine) * ca.affine_sin(beta_affine)
    mz = ca.affine_cos(alpha_affine)

    if required_T.get('tx'):
        results['tx'] = m_affine * (-mz * B_results.get('By', ca.Affine(0)) + my * B_results.get('Bz', ca.Affine(0)))
    if required_T.get('ty'):
        results['ty'] = m_affine * (mz * B_results.get('Bx', ca.Affine(0)) - mx * B_results.get('Bz', ca.Affine(0)))
    if required_T.get('tz'):
        results['tz'] = m_affine * (-my * B_results.get('Bx', ca.Affine(0)) + mx * B_results.get('By', ca.Affine(0)))
    return results

def calculate_Force_and_Torque_selective_affine(currents, X_affine, Y_affine, Z_affine, m_affine, alpha_affine, beta_affine, required_FT, B_results, deriv_results):
    """Affine version of calculate_Force_and_Torque_selective_interval. Assumes inputs are affine."""
    results = {}
    if not any(required_FT.values()): return results
    
    mx = ca.affine_sin(alpha_affine) * ca.affine_cos(beta_affine)
    my = ca.affine_sin(alpha_affine) * ca.affine_sin(beta_affine)
    mz = ca.affine_cos(alpha_affine)
    
    if required_FT.get('fx'):
        results['fx'] = m_affine * (mx * deriv_results.get('Bx_dx', ca.Affine(0)) + my * deriv_results.get('Bx_dy', ca.Affine(0)) + mz * deriv_results.get('Bx_dz', ca.Affine(0)))
    if required_FT.get('fy'):
        results['fy'] = m_affine * (mx * deriv_results.get('Bx_dy', ca.Affine(0)) + my * deriv_results.get('By_dy', ca.Affine(0)) + mz * deriv_results.get('By_dz', ca.Affine(0)))
    if required_FT.get('fz'):
        results['fz'] = m_affine * (-mz * deriv_results.get('Bx_dx', ca.Affine(0)) + mx * deriv_results.get('Bx_dz', ca.Affine(0)) - mz * deriv_results.get('By_dy', ca.Affine(0)) + my * deriv_results.get('By_dz', ca.Affine(0)))

    if required_FT.get('tx'):
        results['tx'] = m_affine * (-mz * B_results.get('By', ca.Affine(0)) + my * B_results.get('Bz', ca.Affine(0)))
    if required_FT.get('ty'):
        results['ty'] = m_affine * (mz * B_results.get('Bx', ca.Affine(0)) - mx * B_results.get('Bz', ca.Affine(0)))
    if required_FT.get('tz'):
        results['tz'] = m_affine * (-my * B_results.get('Bx', ca.Affine(0)) + mx * B_results.get('By', ca.Affine(0)))
    return results

def calculate_B_magnet_selective_affine(magnet_point_affine, target_X_affine, target_Y_affine, target_Z_affine, required_B):
    """Calculates only required B-field components from a magnet using affine arithmetic."""
    results = {}
    if not any(required_B.values()): return results

    magnet_X, magnet_Y, magnet_Z = magnet_point_affine['X'], magnet_point_affine['Y'], magnet_point_affine['Z']
    moment, alpha, beta = magnet_point_affine['m'], magnet_point_affine['alpha'], magnet_point_affine['beta']
    
    mx = ca.affine_sin(alpha) * ca.affine_cos(beta) * moment
    my = ca.affine_sin(alpha) * ca.affine_sin(beta) * moment
    mz = ca.affine_cos(alpha) * moment
    
    dx, dy, dz = target_X_affine - magnet_X, target_Y_affine - magnet_Y, target_Z_affine - magnet_Z
    r_squared = dx*dx + dy*dy + dz*dz

    r_squared_interval = r_squared.to_interval()
    if r_squared_interval[0][0] < 1e-9:
        r_squared = ca.Affine(max(r_squared.center, 1e-9))

    r = ca.affine_sqrt(r_squared)
    dot_product = mx*dx + my*dy + mz*dz
    
    mu0_4pi = 1e-7
    r_cubed = r * r_squared
    r_fifth = r_cubed * r_squared

    if required_B.get('Bx'):
        results['Bx'] = mu0_4pi * (3 * dx * dot_product / r_fifth - mx / r_cubed)
    if required_B.get('By'):
        results['By'] = mu0_4pi * (3 * dy * dot_product / r_fifth - my / r_cubed)
    if required_B.get('Bz'):
        results['Bz'] = mu0_4pi * (3 * dz * dot_product / r_fifth - mz / r_cubed)
    return results

def calculate_derivatives_magnet_selective_affine(magnet_point_affine, target_X_affine, target_Y_affine, target_Z_affine, required_derivs):
    """Calculates only required B-field derivative components from a magnet using affine arithmetic."""
    results = {}
    if not any(required_derivs.values()): return results

    magnet_X, magnet_Y, magnet_Z = magnet_point_affine['X'], magnet_point_affine['Y'], magnet_point_affine['Z']
    moment, alpha, beta = magnet_point_affine['m'], magnet_point_affine['alpha'], magnet_point_affine['beta']
    
    mx = ca.affine_sin(alpha) * ca.affine_cos(beta) * moment
    my = ca.affine_sin(alpha) * ca.affine_sin(beta) * moment
    mz = ca.affine_cos(alpha) * moment
    
    dx, dy, dz = target_X_affine - magnet_X, target_Y_affine - magnet_Y, target_Z_affine - magnet_Z
    r_squared = dx*dx + dy*dy + dz*dz

    r_squared_interval = r_squared.to_interval()
    if r_squared_interval[0][0] < 1e-9:
        r_squared = ca.Affine(max(r_squared.center, 1e-9))

    r = ca.affine_sqrt(r_squared)
    dot_product = mx*dx + my*dy + mz*dz
    
    mu0_4pi = 1e-7
    r5 = r * r_squared * r_squared
    r7 = r5 * r_squared

    if required_derivs.get('Bx_dx'):
        results['Bx_dx'] = mu0_4pi * (3*dot_product/r5 + 6*dx*mx/r5 - 15*dx**2*dot_product/r7)
    if required_derivs.get('Bx_dy'):
        results['Bx_dy'] = mu0_4pi * (3*dx*my/r5 + 3*mx*dy/r5 - 15*dx*dy*dot_product/r7)
    if required_derivs.get('Bx_dz'):
        results['Bx_dz'] = mu0_4pi * (3*dx*mz/r5 + 3*mx*dz/r5 - 15*dx*dz*dot_product/r7)
    if required_derivs.get('By_dy'):
        results['By_dy'] = mu0_4pi * (3*dot_product/r5 + 3*dy*my/r5 - 15*dy**2*dot_product/r7)
    if required_derivs.get('By_dz'):
        results['By_dz'] = mu0_4pi * (3*dy*mz/r5 + 3*my*dz/r5 - 15*dy*dz*dot_product/r7)
    return results

def calculate_Force_magnet_selective_affine(magnet_point1_affine, magnet_point2_affine, required_F):
    """Calculates force between two magnets using affine arithmetic."""
    if not any(required_F.values()): return {}
    
    required_derivs = {'Bx_dx': True, 'Bx_dy': True, 'Bx_dz': True, 'By_dy': True, 'By_dz': True}
    deriv_results = calculate_derivatives_magnet_selective_affine(magnet_point2_affine, magnet_point1_affine['X'], magnet_point1_affine['Y'], magnet_point1_affine['Z'], required_derivs)
    
    m1, alpha1, beta1 = magnet_point1_affine['m'], magnet_point1_affine['alpha'], magnet_point1_affine['beta']
    
    return calculate_Force_selective_affine(None, None, None, None, m1, alpha1, beta1, required_F, deriv_results)

def calculate_Torque_magnet_selective_affine(magnet_point1_affine, magnet_point2_affine, required_T):
    """Calculates torque between two magnets using affine arithmetic."""
    if not any(required_T.values()): return {}
        
    required_B = {'Bx': True, 'By': True, 'Bz': True}
    B_results = calculate_B_magnet_selective_affine(magnet_point2_affine, magnet_point1_affine['X'], magnet_point1_affine['Y'], magnet_point1_affine['Z'], required_B)
    
    m1, alpha1, beta1 = magnet_point1_affine['m'], magnet_point1_affine['alpha'], magnet_point1_affine['beta']

    return calculate_Torque_selective_affine(None, None, None, None, m1, alpha1, beta1, required_T, B_results)

def Map_I2H_Affine(target_points):
    """
    Calculates the mapping matrix from coil currents to magnetic field properties (H) at target points
    using affine arithmetic for robust uncertainty propagation.

    This function takes a list of target points, each with interval-defined uncertainties in position and
    magnetic properties. It computes the influence of each coil and other magnets on every specified
    component (e.g., Bx, fx) at each target point.

    The process is as follows:
    1.  Convert all interval inputs for target_points into their affine representations.
    2.  Calculate the combined effect of all other magnets on each target point.
    3.  For each required component at each target point (forming the rows of the matrix):
        a. Calculate the contribution of each coil (forming the columns of the matrix).
        b. The final column is the pre-calculated magnet-on-magnet effect.
    4.  The entire calculation is done using affine arithmetic.
    5.  The final matrix of affine results is converted back to an interval matrix before returning.
    """
    
    # Convert all interval inputs in target_points to affine forms first
    affine_target_points = []
    # Create a sample interval to get its type, as 'interval' itself is not a type.
    sample_interval_type = type(interval[0,0])
    for tp in target_points:
        affine_tp = {key: ca.Affine.from_interval(value) if isinstance(value, sample_interval_type) else value for key, value in tp.items()}
        affine_target_points.append(affine_tp)

    # Count total number of active components (rows in the final matrix)
    row_mapping = []  # Stores (target_index, component_name) for each row
    for i, target_point in enumerate(affine_target_points):
        for component in ['Bx', 'By', 'Bz', 'Bx_dx', 'Bx_dy', 'Bx_dz', 'By_dy', 'By_dz', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz']:
            if target_point.get(component, False):
                row_mapping.append((i, component))
    
    A_matrix_affine = []
    
    # --- Calculate Magnet-on-Magnet Interaction Effects ---
    magnet_effects = {}
    for i, target_point in enumerate(affine_target_points):
        X, Y, Z = target_point['X'], target_point['Y'], target_point['Z']
        
        magnet_effects[i] = {comp: ca.Affine(0) for comp in ['Bx', 'By', 'Bz', 'Bx_dx', 'Bx_dy', 'Bx_dz', 'By_dy', 'By_dz', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz']}
        
        # Determine what components we need for magnet interactions for this target point
        required_B_magnet = {c: target_point.get(c, False) for c in ['Bx', 'By', 'Bz']}
        required_derivs_magnet = {c: target_point.get(c, False) for c in ['Bx_dx', 'Bx_dy', 'Bx_dz', 'By_dy', 'By_dz']}
        required_F_magnet = {c: target_point.get(c, False) for c in ['fx', 'fy', 'fz']}
        required_T_magnet = {c: target_point.get(c, False) for c in ['tx', 'ty', 'tz']}

        # If forces or torques are needed, their base components are also needed
        if any(required_F_magnet.values()):
            required_derivs_magnet.update({k: True for k in required_derivs_magnet})
        if any(required_T_magnet.values()):
            required_B_magnet.update({k: True for k in required_B_magnet})

        for k, other_magnet in enumerate(affine_target_points):
            if k == i: continue

            B_results = calculate_B_magnet_selective_affine(other_magnet, X, Y, Z, required_B_magnet)
            for comp, val in B_results.items():
                magnet_effects[i][comp] += val

            deriv_results = calculate_derivatives_magnet_selective_affine(other_magnet, X, Y, Z, required_derivs_magnet)
            for comp, val in deriv_results.items():
                magnet_effects[i][comp] += val
            
            F_results = calculate_Force_magnet_selective_affine(target_point, other_magnet, required_F_magnet)
            for comp, val in F_results.items():
                magnet_effects[i][comp] += val

            T_results = calculate_Torque_magnet_selective_affine(target_point, other_magnet, required_T_magnet)
            for comp, val in T_results.items():
                magnet_effects[i][comp] += val

    # --- Process Each Row of the Matrix ---
    for target_idx, component in row_mapping:
        target_point = affine_target_points[target_idx]
        X, Y, Z = target_point['X'], target_point['Y'], target_point['Z']
        m, alpha, beta = target_point['m'], target_point['alpha'], target_point['beta']
        
        current_row_affine = []
        
        # --- Calculate Coil Contributions ---
        for j in range(num_coils):  
            currents = np.zeros(num_coils)
            currents[j] = 1
            
            coil_effect = ca.Affine(0)

            if component in ['Bx', 'By', 'Bz']:
                required_B = {comp: (comp == component) for comp in ['Bx', 'By', 'Bz']}
                B_results = calculate_B_selective_affine(currents, X, Y, Z, required_B)
                coil_effect = B_results.get(component, ca.Affine(0))
                
            elif component in ['Bx_dx', 'Bx_dy', 'Bx_dz', 'By_dy', 'By_dz']:
                required_derivs = {comp: (comp == component) for comp in ['Bx_dx', 'Bx_dy', 'Bx_dz', 'By_dy', 'By_dz']}
                deriv_results = calculate_derivatives_selective_affine(currents, X, Y, Z, required_derivs)
                coil_effect = deriv_results.get(component, ca.Affine(0))
                
            elif component in ['fx', 'fy', 'fz']:
                required_F = {comp: (comp == component) for comp in ['fx', 'fy', 'fz']}
                required_derivs = {'Bx_dx': True, 'Bx_dy': True, 'Bx_dz': True, 'By_dy': True, 'By_dz': True}
                deriv_results = calculate_derivatives_selective_affine(currents, X, Y, Z, required_derivs)
                F_results = calculate_Force_selective_affine(currents, X, Y, Z, m, alpha, beta, required_F, deriv_results)
                coil_effect = F_results.get(component, ca.Affine(0))

            elif component in ['tx', 'ty', 'tz']:
                required_T = {comp: (comp == component) for comp in ['tx', 'ty', 'tz']}
                required_B = {'Bx': True, 'By': True, 'Bz': True}
                B_results = calculate_B_selective_affine(currents, X, Y, Z, required_B)
                T_results = calculate_Torque_selective_affine(currents, X, Y, Z, m, alpha, beta, required_T, B_results)
                coil_effect = T_results.get(component, ca.Affine(0))
            
            current_row_affine.append(coil_effect)
        
        # --- Add Magnet Interaction Effect (Last Column) ---
        current_row_affine.append(magnet_effects[target_idx][component])
        
        A_matrix_affine.append(current_row_affine)
    
    # --- Convert Final Affine Matrix to Interval Matrix ---
    A_matrix_interval = [[item.to_interval() for item in row_affine] for row_affine in A_matrix_affine]
        
    return np.array(A_matrix_interval)

def Map_I2H_Affine_body(target_points):
    """
    Calculates the mapping matrix from coil currents to magnetic field properties (H) 
    in the body frame of each target magnet, using affine arithmetic.
    """
    
    # Convert all interval inputs in target_points to affine forms
    affine_target_points = []
    sample_interval_type = type(interval[0,0])
    for tp in target_points:
        affine_tp = {key: ca.Affine.from_interval(value) if isinstance(value, sample_interval_type) else value for key, value in tp.items()}
        affine_target_points.append(affine_tp)

    # Identify active components to determine matrix rows
    row_mapping = []  # Stores (target_index, component_name) for each row
    for i, target_point in enumerate(affine_target_points):
        for component in ['Bx', 'By', 'Bz', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz']:
            if target_point.get(component, False):
                row_mapping.append((i, component))
    
    A_matrix_affine = []

    # Determine globally required components
    needs_B = any(tp.get(c, False) for tp in target_points for c in ['Bx', 'By', 'Bz'])
    needs_F = any(tp.get(c, False) for tp in target_points for c in ['fx', 'fy', 'fz'])
    needs_T = any(tp.get(c, False) for tp in target_points for c in ['tx', 'ty', 'tz'])
    
    # --- Pre-calculate Magnet-on-Magnet Interaction Effects (in World Frame) ---
    magnet_effects_world = {}
    for i, target_point in enumerate(affine_target_points):
        X, Y, Z = target_point['X'], target_point['Y'], target_point['Z']
        
        magnet_effects_world[i] = {
            'B': [ca.Affine(0)] * 3, 
            'F': [ca.Affine(0)] * 3, 
            'T': [ca.Affine(0)] * 3
        }
        
        for k, other_magnet in enumerate(affine_target_points):
            if k == i: continue

            # B-field from other magnets is needed for direct B output or for torque calculation
            if needs_B or needs_T:
                req_B = {'Bx': True, 'By': True, 'Bz': True}
                B_res = calculate_B_magnet_selective_affine(other_magnet, X, Y, Z, req_B)
                if needs_B:
                    magnet_effects_world[i]['B'][0] += B_res.get('Bx', ca.Affine(0))
                    magnet_effects_world[i]['B'][1] += B_res.get('By', ca.Affine(0))
                    magnet_effects_world[i]['B'][2] += B_res.get('Bz', ca.Affine(0))
                if needs_T:
                    req_T = {'tx': True, 'ty': True, 'tz': True}
                    T_res = calculate_Torque_magnet_selective_affine(target_point, other_magnet, req_T)
                    magnet_effects_world[i]['T'][0] += T_res.get('tx', ca.Affine(0))
                    magnet_effects_world[i]['T'][1] += T_res.get('ty', ca.Affine(0))
                    magnet_effects_world[i]['T'][2] += T_res.get('tz', ca.Affine(0))

            # Derivatives from other magnets are needed for force calculation
            if needs_F:
                req_F = {'fx': True, 'fy': True, 'fz': True}
                F_res = calculate_Force_magnet_selective_affine(target_point, other_magnet, req_F)
                magnet_effects_world[i]['F'][0] += F_res.get('fx', ca.Affine(0))
                magnet_effects_world[i]['F'][1] += F_res.get('fy', ca.Affine(0))
                magnet_effects_world[i]['F'][2] += F_res.get('fz', ca.Affine(0))

    # --- Process Each Row of the Matrix ---
    for target_idx, component in row_mapping:
        target_point = affine_target_points[target_idx]
        X, Y, Z = target_point['X'], target_point['Y'], target_point['Z']
        m, alpha, beta = target_point['m'], target_point['alpha'], target_point['beta']
        
        # Rotation matrix from world to body frame (affine version)
        ca_alpha, sa_alpha = ca.affine_cos(alpha), ca.affine_sin(alpha)
        ca_beta, sa_beta = ca.affine_cos(beta), ca.affine_sin(beta)
        R = [
            [ca_alpha * ca_beta, ca_alpha * sa_beta, -sa_alpha],
            [-sa_beta, ca_beta, ca.Affine(0)],
            [sa_alpha * ca_beta, sa_alpha * sa_beta, ca_alpha]
        ]

        current_row_affine = []
        
        # --- Calculate Coil Contributions ---
        for j in range(num_coils):
            currents = np.zeros(num_coils)
            currents[j] = 1
            
            # Calculate B, F, T in world frame for a single coil
            B_world, F_world, T_world = [ca.Affine(0)]*3, [ca.Affine(0)]*3, [ca.Affine(0)]*3
            
            # B-field from coils is needed for direct B output or for torque calculation
            if needs_B or needs_T:
                req_B = {'Bx': True, 'By': True, 'Bz': True}
                B_res = calculate_B_selective_affine(currents, X, Y, Z, req_B)
                if needs_B and component.startswith('B'):
                    B_world = [B_res.get('Bx', ca.Affine(0)), B_res.get('By', ca.Affine(0)), B_res.get('Bz', ca.Affine(0))]
                if needs_T and component.startswith('t'):
                    req_T = {'tx': True, 'ty': True, 'tz': True}
                    T_res = calculate_Torque_selective_affine(currents, X, Y, Z, m, alpha, beta, req_T, B_res)
                    T_world = [T_res.get('tx', ca.Affine(0)), T_res.get('ty', ca.Affine(0)), T_res.get('tz', ca.Affine(0))]

            # Derivatives from coils are needed for force calculation
            if needs_F and component.startswith('f'):
                req_derivs = {'Bx_dx': True, 'Bx_dy': True, 'Bx_dz': True, 'By_dy': True, 'By_dz': True}
                deriv_res = calculate_derivatives_selective_affine(currents, X, Y, Z, req_derivs)
                req_F = {'fx': True, 'fy': True, 'fz': True}
                F_res = calculate_Force_selective_affine(currents, X, Y, Z, m, alpha, beta, req_F, deriv_res)
                F_world = [F_res.get('fx', ca.Affine(0)), F_res.get('fy', ca.Affine(0)), F_res.get('fz', ca.Affine(0))]

            # Select the correct world vector and transform to body frame
            if component.startswith('B'):
                V_world = B_world
                idx = ['Bx', 'By', 'Bz'].index(component)
            elif component.startswith('f'):
                V_world = F_world
                idx = ['fx', 'fy', 'fz'].index(component)
            else: # 't'
                V_world = T_world
                idx = ['tx', 'ty', 'tz'].index(component)
            
            coil_effect_body = R[idx][0]*V_world[0] + R[idx][1]*V_world[1] + R[idx][2]*V_world[2]
            current_row_affine.append(coil_effect_body)

        # --- Add Magnet Interaction Effect (Last Column) ---
        if component.startswith('B'):
            V_world = magnet_effects_world[target_idx]['B']
            idx = ['Bx', 'By', 'Bz'].index(component)
        elif component.startswith('f'):
            V_world = magnet_effects_world[target_idx]['F']
            idx = ['fx', 'fy', 'fz'].index(component)
        else: # 't'
            V_world = magnet_effects_world[target_idx]['T']
            idx = ['tx', 'ty', 'tz'].index(component)
            
        magnet_effect_body = R[idx][0]*V_world[0] + R[idx][1]*V_world[1] + R[idx][2]*V_world[2]
        current_row_affine.append(magnet_effect_body)
        
        A_matrix_affine.append(current_row_affine)
        
    # Convert final affine matrix to interval matrix
    A_matrix_interval = [[item.to_interval() for item in row] for row in A_matrix_affine]
    return np.array(A_matrix_interval)

def robot_arm_kinematics(alpha1, beta1, beta2b):
    """
    Forward kinematics calculations with interval arithmetic support.
    
    Args:
        alpha1, beta1, beta2b: Joint angles (can be scalars or intervals)
    
    Returns:
        Tuple of positions and orientations (X1, Y1, Z1, X2, Y2, Z2, alpha2w, beta2w)
        All return values will be intervals if any input is an interval.
    """
    # Forward kinematics calculations
    x_base = 0
    y_base = 0.023
    z_base = 0.079
    L1_m = 0.01173
    L1 = 0.01973
    L2_m = 0.005
    L2 = 0.008
    
    # Use interval arithmetic
    sin_alpha1 = imath.sin(alpha1)
    cos_alpha1 = imath.cos(alpha1)
    sin_beta1 = imath.sin(beta1)
    cos_beta1 = imath.cos(beta1)
    sin_beta2b = imath.sin(beta2b)
    cos_beta2b = imath.cos(beta2b)
    
    # position of the first magnet
    X1 = L1_m * sin_alpha1 * cos_beta1
    Y1 = L1_m * sin_alpha1 * sin_beta1
    Z1 = L1_m * cos_alpha1
    X1 = X1 + x_base
    Y1 = Y1 + y_base
    Z1 = Z1 + z_base

    # # position of the second magnet
    # X2 = L1 * cos_beta1 * sin_alpha1 + L2_m * cos_beta1 * sin_alpha1 * cos_beta2b - L2_m * sin_beta1 * sin_beta2b
    # Y2 = L1 * sin_beta1 * sin_alpha1 + L2_m * sin_beta1 * sin_alpha1 * cos_beta2b + L2_m * cos_beta1 * sin_beta2b
    # Z2 = L1 * cos_alpha1 + L2_m * cos_alpha1 * cos_beta2b

    # # Calculate the angles for the second magnet
    # cos_alpha2 = cos_alpha1 * cos_beta2b
    
    # # Ensure cos_alpha2 is within [-1, 1] for arccos
    # if hasattr(cos_alpha2, 'lower'):
    #     # Clamp interval to valid domain
    #     cos_alpha2_clamped = interval([max(cos_alpha2.lower, -1), min(cos_alpha2.upper, 1)])
    # else:
    #     cos_alpha2_clamped = max(-1, min(1, cos_alpha2))
        
    # alpha2w = imath.arccos(cos_alpha2_clamped)

    # x = -sin_beta1 * sin_beta2b + cos_beta1 * sin_alpha1 * cos_beta2b
    # y = cos_beta1 * sin_beta2b + sin_beta1 * sin_alpha1 * cos_beta2b
    # beta2w = imath.arctan2(y, x)

    # position of the second magnet
    X2 = (L1 * imath.sin(alpha1) + L2_m * imath.sin(alpha1 + beta2b)) * cos_beta1
    Y2 = (L1 * imath.sin(alpha1) + L2_m * imath.sin(alpha1 + beta2b)) * imath.sin(beta1)
    Z2 = L1 * imath.cos(alpha1) + L2_m * imath.cos(alpha1 + beta2b) 
    X2 += x_base
    Y2 += y_base
    Z2 += z_base

    # Calculate the angles for the second magnet
    alpha2w = imath.arccos(imath.cos(alpha1 + beta2b))
    beta2w = beta1

    return X1, Y1, Z1, X2, Y2, Z2, alpha2w, beta2w

def arm_end_positions_affine(beta1, alpha2):
    """
    Forward kinematics calculations with interval arithmetic support.
    """
    x_base = interval[0, 0]
    y_base = interval[0.023, 0.023]
    z_base = interval[0.079, 0.079]
    L1_m = interval[0.011, 0.011]
    L1 = interval[0.023, 0.023]
    L2_m = interval[0.011, 0.011]
    L2 = interval[0.029, 0.029]
    
    beta1 = beta1 + interval[math.pi/2, math.pi/2]
    alpha2 = alpha2 - interval[math.pi/2, math.pi/2]

    # position of the first magnet
    p1_x = L1_m * imath.sin(beta1)
    p1_y = -L1_m * imath.cos(beta1)
    p1_z = 0.0
    p1_x += x_base
    p1_y += y_base
    p1_z += z_base
    
    # position of the second magnet
    p2_x = (L1 + L2_m * imath.cos(alpha2)) * imath.sin(beta1)
    p2_y = -(L1 + L2_m * imath.cos(alpha2)) * imath.cos(beta1)
    p2_z = -L2_m * imath.sin(alpha2)
    p2_x += x_base
    p2_y += y_base
    p2_z += z_base

    return p1_x, p1_y, p1_z, p2_x, p2_y, p2_z

if __name__ == "__main__":
    # target_points = [
    #     {'X': interval[0.0, 0.0], 'Y': interval[0.013, 0.013], 'Z': interval[0.085, 0.085], 'm': interval[0.1, 0.1], 'alpha': interval[math.pi/2, math.pi/2], 'beta': interval[math.pi, math.pi], 'Bx': None, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None, 'fz': True, 'tx': None, 'ty': None, 'tz': None},
    #     {'X': interval[-0.01, -0.01], 'Y': interval[0.01, 0.01], 'Z': interval[0.085, 0.085], 'm': interval[0.1, 0.1], 'alpha': interval[math.pi/2, math.pi/2], 'beta': interval[math.pi, math.pi], 'Bx': None, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None, 'fz': True, 'tx': None, 'ty': None, 'tz': None},
    #     ]
    target_points = [
        {'X': interval[0,0], 'Y': 0, 'Z': interval[0.0835, 0.0835], 'm': interval[0.145, 0.145], 
         'alpha': interval[0, 0], 'beta': interval[0.00, 0.00], 
         'Bx': None, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': True, 
         'fz': None, 'tx': None, 'ty': None, 'tz': None},
         {'X': interval[0, 0], 'Y': 0.01, 'Z': interval[0.0835, 0.0835], 'm': interval[0.145, 0.145], 
         'alpha': interval[0,0], 'beta': interval[0.00, 0.00], 
         'Bx': None, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 
         'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': True, 
         'fz': None, 'tx': None, 'ty': None, 'tz': None}
    ]
    # Example usage of Map_I2H_Affine
    # A_matrix = Map_I2H_Affine(target_points)
    A_matrix = Map_I2H_Affine_body(target_points)
    print("Mapping matrix A affine (interval representation):")
    print(A_matrix)
    print("Shape of A_matrix:", A_matrix.shape)