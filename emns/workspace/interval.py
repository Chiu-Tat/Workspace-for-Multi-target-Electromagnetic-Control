import numpy as np
# Run from the repository root: python -m emns.workspace.interval
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, QhullError
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import combinations
from scipy.linalg import null_space
from emns.numerics.interval import interval, imath, fpu
import math

# 10 coils data 3D
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

def calculate_B_selective_interval(currents, X, Y, Z, required_B):
    """Calculate only the required magnetic field components with interval inputs"""
    results = {}

    # Skip if no B components are needed
    if not any(required_B[key] for key in ['Bx', 'By', 'Bz']):
        return results

    # Initialize only needed components
    if required_B['Bx']:
        results['Bx'] = interval[0, 0]
    if required_B['By']:
        results['By'] = interval[0, 0]
    if required_B['Bz']:
        results['Bz'] = interval[0, 0]

    # Loop over the coils
    for i in range(num_coils):
        if currents[i] == 0:  # Skip coils with zero current
            continue

        params = params_list[i]
        m0, m1, m2, r0_0, r0_1, r0_2 = params
        m0 = interval[m0, m0]
        m1 = interval[m1, m1]
        m2 = interval[m2, m2]
        r0_0 = interval[r0_0, r0_0]
        r0_1 = interval[r0_1, r0_1]
        r0_2 = interval[r0_2, r0_2]

        # Calculate displacement with intervals
        dx = X - r0_0
        dy = Y - r0_1
        dz = Z - r0_2

        # Calculate distance
        r = imath.sqrt(dx**2 + dy**2 + dz**2)

        # Calculate dot product
        dot_product = m0 * dx + m1 * dy + m2 * dz

        # Magnetic field constants
        mu0_4pi = (4 * math.pi * 1e-7) / (4 * math.pi)

        # Calculate only required magnetic field components
        if required_B['Bx']:
            Bx = mu0_4pi * (3 * dx * dot_product / (r**5) - m0 / (r**3)) * currents[i]
            results['Bx'] = results['Bx'] + Bx

        if required_B['By']:
            By = mu0_4pi * (3 * dy * dot_product / (r**5) - m1 / (r**3)) * currents[i]
            results['By'] = results['By'] + By

        if required_B['Bz']:
            Bz = mu0_4pi * (3 * dz * dot_product / (r**5) - m2 / (r**3)) * currents[i]
            results['Bz'] = results['Bz'] + Bz

    return results

def calculate_derivatives_selective_interval(currents, X, Y, Z, required_derivs):
    """Calculate only the required magnetic field derivatives with interval inputs"""
    results = {}

    # Skip if no derivatives are needed
    if not any(required_derivs[key] for key in ['Bx_dx', 'Bx_dy', 'Bx_dz', 'By_dy', 'By_dz']):
        return results

    # Initialize only needed components
    for key in ['Bx_dx', 'Bx_dy', 'Bx_dz', 'By_dy', 'By_dz']:
        if required_derivs[key]:
            results[key] = interval[0, 0]

    # Loop over the coils
    for i in range(num_coils):
        if currents[i] == 0:  # Skip coils with zero current
            continue

        params = params_list[i]
        m0, m1, m2, r0_0, r0_1, r0_2 = params
        m0 = interval[m0, m0]
        m1 = interval[m1, m1]
        m2 = interval[m2, m2]
        r0_0 = interval[r0_0, r0_0]
        r0_1 = interval[r0_1, r0_1]
        r0_2 = interval[r0_2, r0_2]

        # Calculate displacement with intervals
        dx = X - r0_0
        dy = Y - r0_1
        dz = Z - r0_2

        # Calculate distance
        r = imath.sqrt(dx**2 + dy**2 + dz**2)

        # Calculate dot product
        dot_product = m0 * dx + m1 * dy + m2 * dz

        # Magnetic field constants
        mu0_4pi = (4 * math.pi * 1e-7) / (4 * math.pi)

        # Calculate partial derivatives only if needed
        r5 = r**5
        r7 = r**7

        if required_derivs['Bx_dx']:
            Bx_dx = mu0_4pi * (3 * dot_product / r5 + 6 * dx * m0 / r5 - 15 * dx**2 * dot_product / r7) * currents[i]
            results['Bx_dx'] = results['Bx_dx'] + Bx_dx

        if required_derivs['Bx_dy']:
            Bx_dy = mu0_4pi * (3 * dx * m1 / r5 + 3 * m0 * dy / r5 - 15 * dx * dy * dot_product / r7) * currents[i]
            results['Bx_dy'] = results['Bx_dy'] + Bx_dy

        if required_derivs['Bx_dz']:
            Bx_dz = mu0_4pi * (3 * dx * m2 / r5 + 3 * m0 * dz / r5 - 15 * dx * dz * dot_product / r7) * currents[i]
            results['Bx_dz'] = results['Bx_dz'] + Bx_dz

        if required_derivs['By_dy']:
            By_dy = mu0_4pi * (3 * dot_product / r5 + 6*dy*m1 / r5 - 15 * dy**2 * dot_product / r7) * currents[i]
            results['By_dy'] = results['By_dy'] + By_dy

        if required_derivs['By_dz']:
            By_dz = mu0_4pi * (3 * dy * m2 / r5 + 3 * m1 * dz / r5 - 15 * dy * dz * dot_product / r7) * currents[i]
            results['By_dz'] = results['By_dz'] + By_dz

    return results

def calculate_Force_selective_interval(currents, X, Y, Z, m, alpha, beta, required_F, deriv_results):
    """Calculate only the required force components with interval inputs"""
    results = {}
    if not any(required_F.values()):
        return results

    mx = imath.sin(alpha) * imath.cos(beta)
    my = imath.sin(alpha) * imath.sin(beta)
    mz = imath.cos(alpha)

    if required_F.get('fx', False):
        fx = m * (mx * deriv_results.get('Bx_dx', interval[0, 0]) +
                 my * deriv_results.get('Bx_dy', interval[0, 0]) +
                 mz * deriv_results.get('Bx_dz', interval[0, 0]))
        results['fx'] = fx

    if required_F.get('fy', False):
        fy = m * (mx * deriv_results.get('Bx_dy', interval[0, 0]) +
                 my * deriv_results.get('By_dy', interval[0, 0]) +
                 mz * deriv_results.get('By_dz', interval[0, 0]))
        results['fy'] = fy

    if required_F.get('fz', False):
        fz = m * (-mz * deriv_results.get('Bx_dx', interval[0, 0]) +
                 mx * deriv_results.get('Bx_dz', interval[0, 0]) -
                 mz * deriv_results.get('By_dy', interval[0, 0]) +
                 my * deriv_results.get('By_dz', interval[0, 0]))
        results['fz'] = fz
    return results

def calculate_Torque_selective_interval(currents, X, Y, Z, m, alpha, beta, required_T, B_results):
    """Calculate only the required torque components with interval inputs"""
    results = {}
    if not any(required_T.values()):
        return results

    mx = imath.sin(alpha) * imath.cos(beta)
    my = imath.sin(alpha) * imath.sin(beta)
    mz = imath.cos(alpha)

    if required_T.get('tx', False):
        tx = m * (-mz * B_results.get('By', interval[0, 0]) +
                 my * B_results.get('Bz', interval[0, 0]))
        results['tx'] = tx

    if required_T.get('ty', False):
        ty = m * (mz * B_results.get('Bx', interval[0, 0]) -
                 mx * B_results.get('Bz', interval[0, 0]))
        results['ty'] = ty

    if required_T.get('tz', False):
        tz = m * (-my * B_results.get('Bx', interval[0, 0]) +
                 mx * B_results.get('By', interval[0, 0]))
        results['tz'] = tz
    return results

def calculate_B_magnet_selective_interval(magnet_point, target_X, target_Y, target_Z, required_B):
    """Calculate only required magnetic field components from one magnet to target point with intervals"""
    results = {}
    if not any(required_B.values()):
        return results

    magnet_X, magnet_Y, magnet_Z = magnet_point['X'], magnet_point['Y'], magnet_point['Z']
    moment, alpha, beta = magnet_point['m'], magnet_point['alpha'], magnet_point['beta']

    mx = imath.sin(alpha) * imath.cos(beta) * moment
    my = imath.sin(alpha) * imath.sin(beta) * moment
    mz = imath.cos(alpha) * moment

    dx, dy, dz = target_X - magnet_X, target_Y - magnet_Y, target_Z - magnet_Z
    r = imath.sqrt(dx**2 + dy**2 + dz**2) + interval[1e-9, 1e-9]
    dot_product = mx * dx + my * dy + mz * dz
    mu0_4pi = (4 * math.pi * 1e-7) / (4 * math.pi)

    r3 = r**3
    r5 = r**5

    if required_B.get('Bx', False):
        results['Bx'] = mu0_4pi * (3 * dx * dot_product / r5 - mx / r3)
    if required_B.get('By', False):
        results['By'] = mu0_4pi * (3 * dy * dot_product / r5 - my / r3)
    if required_B.get('Bz', False):
        results['Bz'] = mu0_4pi * (3 * dz * dot_product / r5 - mz / r3)
    return results

def calculate_derivatives_magnet_selective_interval(magnet_point, target_X, target_Y, target_Z, required_derivs):
    """Calculate only required magnetic field derivatives from one magnet to target point with intervals"""
    results = {}
    if not any(required_derivs.values()):
        return results

    magnet_X, magnet_Y, magnet_Z = magnet_point['X'], magnet_point['Y'], magnet_point['Z']
    moment, alpha, beta = magnet_point['m'], magnet_point['alpha'], magnet_point['beta']

    mx = imath.sin(alpha) * imath.cos(beta) * moment
    my = imath.sin(alpha) * imath.sin(beta) * moment
    mz = imath.cos(alpha) * moment

    dx, dy, dz = target_X - magnet_X, target_Y - magnet_Y, target_Z - magnet_Z
    r = imath.sqrt(dx**2 + dy**2 + dz**2) + interval[1e-9, 1e-9]
    dot_product = mx * dx + my * dy + mz * dz
    mu0_4pi = (4 * math.pi * 1e-7) / (4 * math.pi)

    r5 = r**5
    r7 = r**7

    if required_derivs.get('Bx_dx', False):
        results['Bx_dx'] = mu0_4pi * (3 * dot_product / r5 + 6 * dx * mx / r5 - 15 * dx**2 * dot_product / r7)
    if required_derivs.get('Bx_dy', False):
        results['Bx_dy'] = mu0_4pi * (3 * dx * my / r5 + 3 * mx * dy / r5 - 15 * dx * dy * dot_product / r7)
    if required_derivs.get('Bx_dz', False):
        results['Bx_dz'] = mu0_4pi * (3 * dx * mz / r5 + 3 * mx * dz / r5 - 15 * dx * dz * dot_product / r7)
    if required_derivs.get('By_dy', False):
        results['By_dy'] = mu0_4pi * (3 * dot_product / r5 + 6*dy*my / r5 - 15 * dy**2 * dot_product / r7)
    if required_derivs.get('By_dz', False):
        results['By_dz'] = mu0_4pi * (3 * dy * mz / r5 + 3 * my * dz / r5 - 15 * dy * dz * dot_product / r7)
    return results

def calculate_Force_magnet_selective_interval(magnet_point1, magnet_point2, required_F):
    """Calculate only required force between two magnets with intervals"""
    if not any(required_F.values()):
        return {}

    required_derivs = {'Bx_dx': True, 'Bx_dy': True, 'Bx_dz': True, 'By_dy': True, 'By_dz': True}
    deriv_results = calculate_derivatives_magnet_selective_interval(magnet_point2, magnet_point1['X'], magnet_point1['Y'], magnet_point1['Z'], required_derivs)

    m1, alpha1, beta1 = magnet_point1['m'], magnet_point1['alpha'], magnet_point1['beta']

    return calculate_Force_selective_interval(None, None, None, None, m1, alpha1, beta1, required_F, deriv_results)

def calculate_Torque_magnet_selective_interval(magnet_point1, magnet_point2, required_T):
    """Calculate only required torque between two magnets with intervals"""
    if not any(required_T.values()):
        return {}

    required_B = {'Bx': True, 'By': True, 'Bz': True}
    B_results = calculate_B_magnet_selective_interval(magnet_point2, magnet_point1['X'], magnet_point1['Y'], magnet_point1['Z'], required_B)

    m1, alpha1, beta1 = magnet_point1['m'], magnet_point1['alpha'], magnet_point1['beta']

    return calculate_Torque_selective_interval(None, None, None, None, m1, alpha1, beta1, required_T, B_results)

def Map_I2H_Interval(target_points):
    """Create mapping matrix with interval arithmetic - optimized to only calculate needed components"""

    # Count total number of active components across all target points
    total_rows = 0
    row_mapping = []  # Store which (target_index, component) each row represents

    for i, target_point in enumerate(target_points):
        for component in ['Bx', 'By', 'Bz', 'Bx_dx', 'Bx_dy', 'Bx_dz', 'By_dy', 'By_dz', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz']:
            if target_point.get(component, False):
                row_mapping.append((i, component))
                total_rows += 1

    # Initialize matrix with proper dimensions
    A_matrix = []

    # Calculate magnet interaction effects for each target point
    magnet_effects = {}
    for i, target_point in enumerate(target_points):
        X = target_point['X']
        Y = target_point['Y']
        Z = target_point['Z']

        # Initialize effects for this target point
        magnet_effects[i] = {c: interval[0, 0] for c in ['Bx', 'By', 'Bz', 'Bx_dx', 'Bx_dy', 'Bx_dz', 'By_dy', 'By_dz', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz']}

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

        # Calculate interactions with other magnets
        for k, other_target in enumerate(target_points):
            if k != i:  # Exclude self-interaction
                B_results = calculate_B_magnet_selective_interval(other_target, X, Y, Z, required_B_magnet)
                for comp, val in B_results.items():
                    magnet_effects[i][comp] += val

                deriv_results = calculate_derivatives_magnet_selective_interval(other_target, X, Y, Z, required_derivs_magnet)
                for comp, val in deriv_results.items():
                    magnet_effects[i][comp] += val

                F_results = calculate_Force_magnet_selective_interval(target_point, other_target, required_F_magnet)
                for comp, val in F_results.items():
                    magnet_effects[i][comp] += val

                T_results = calculate_Torque_magnet_selective_interval(target_point, other_target, required_T_magnet)
                for comp, val in T_results.items():
                    magnet_effects[i][comp] += val

    # Process each active row
    for target_idx, component in row_mapping:
        target_point = target_points[target_idx]
        X = target_point['X']
        Y = target_point['Y']
        Z = target_point['Z']
        m = target_point['m']
        alpha = target_point['alpha']
        beta = target_point['beta']

        current_row = []

        # Loop over the coils - each coil contributes to this row
        for j in range(num_coils):
            # Calculate the magnetic field and its derivatives for a unit current in the j-th coil
            currents = np.zeros(num_coils)
            currents[j] = 1

            coil_effect = interval[0, 0]
            # Only calculate what we need for this specific component
            if component in ['Bx', 'By', 'Bz']:
                required_B = {comp: (comp == component) for comp in ['Bx', 'By', 'Bz']}
                B_results = calculate_B_selective_interval(currents, X, Y, Z, required_B)
                coil_effect = B_results.get(component, interval[0, 0])

            elif component in ['Bx_dx', 'Bx_dy', 'Bx_dz', 'By_dy', 'By_dz']:
                required_derivs = {comp: (comp == component) for comp in ['Bx_dx', 'Bx_dy', 'Bx_dz', 'By_dy', 'By_dz']}
                deriv_results = calculate_derivatives_selective_interval(currents, X, Y, Z, required_derivs)
                coil_effect = deriv_results.get(component, interval[0, 0])

            elif component in ['fx', 'fy', 'fz']:
                required_F = {comp: (comp == component) for comp in ['fx', 'fy', 'fz']}
                required_derivs = {'Bx_dx': True, 'Bx_dy': True, 'Bx_dz': True, 'By_dy': True, 'By_dz': True}
                deriv_results = calculate_derivatives_selective_interval(currents, X, Y, Z, required_derivs)
                F_results = calculate_Force_selective_interval(currents, X, Y, Z, m, alpha, beta, required_F, deriv_results)
                coil_effect = F_results.get(component, interval[0, 0])

            elif component in ['tx', 'ty', 'tz']:
                required_T = {comp: (comp == component) for comp in ['tx', 'ty', 'tz']}
                required_B = {'Bx': True, 'By': True, 'Bz': True}
                B_results = calculate_B_selective_interval(currents, X, Y, Z, required_B)
                T_results = calculate_Torque_selective_interval(currents, X, Y, Z, m, alpha, beta, required_T, B_results)
                coil_effect = T_results.get(component, interval[0, 0])

            current_row.append(coil_effect)

        # Add magnet interaction effect for this row
        current_row.append(magnet_effects[target_idx][component])

        # Add this row to the matrix
        A_matrix.append(current_row)

    return np.array(A_matrix)

def Map_I2H_Interval_body(target_points):
    """Create mapping matrix with interval arithmetic in the body frame"""

    total_rows = 0
    row_mapping = []  # Store which (target_index, component) each row represents

    for i, target_point in enumerate(target_points):
        for component in ['Bx', 'By', 'Bz', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz']:
            if target_point.get(component, False):
                row_mapping.append((i, component))
                total_rows += 1

    A_matrix = []

    # Determine globally required components
    needs_B = any(tp.get(c, False) for tp in target_points for c in ['Bx', 'By', 'Bz'])
    needs_F = any(tp.get(c, False) for tp in target_points for c in ['fx', 'fy', 'fz'])
    needs_T = any(tp.get(c, False) for tp in target_points for c in ['tx', 'ty', 'tz'])

    # Pre-calculate magnet interaction effects in world frame
    magnet_effects_world = {}
    for i, target_point in enumerate(target_points):
        X = target_point['X']
        Y = target_point['Y']
        Z = target_point['Z']

        magnet_effects_world[i] = {
            'B': [interval[0,0]] * 3,
            'F': [interval[0,0]] * 3,
            'T': [interval[0,0]] * 3
        }

        for k, other_target in enumerate(target_points):
            if k == i: continue

            # B-field from other magnets is needed for direct B output or for torque calculation
            if needs_B or needs_T:
                req_B = {'Bx': True, 'By': True, 'Bz': True}
                B_res = calculate_B_magnet_selective_interval(other_target, X, Y, Z, req_B)
                if needs_B:
                    magnet_effects_world[i]['B'][0] += B_res.get('Bx', interval[0,0])
                    magnet_effects_world[i]['B'][1] += B_res.get('By', interval[0,0])
                    magnet_effects_world[i]['B'][2] += B_res.get('Bz', interval[0,0])
                if needs_T:
                    req_T = {'tx': True, 'ty': True, 'tz': True}
                    # Torque calculation re-uses the B-field calculated above
                    T_res = calculate_Torque_selective_interval(None, None, None, None, target_point['m'], target_point['alpha'], target_point['beta'], req_T, B_res)
                    magnet_effects_world[i]['T'][0] += T_res.get('tx', interval[0,0])
                    magnet_effects_world[i]['T'][1] += T_res.get('ty', interval[0,0])
                    magnet_effects_world[i]['T'][2] += T_res.get('tz', interval[0,0])

            # Derivatives from other magnets are needed for force calculation
            if needs_F:
                req_F = {'fx': True, 'fy': True, 'fz': True}
                F_res = calculate_Force_magnet_selective_interval(target_point, other_target, req_F)
                magnet_effects_world[i]['F'][0] += F_res.get('fx', interval[0,0])
                magnet_effects_world[i]['F'][1] += F_res.get('fy', interval[0,0])
                magnet_effects_world[i]['F'][2] += F_res.get('fz', interval[0,0])

    for target_idx, component in row_mapping:
        target_point = target_points[target_idx]
        X, Y, Z = target_point['X'], target_point['Y'], target_point['Z']
        m, alpha, beta = target_point['m'], target_point['alpha'], target_point['beta']

        ca, sa = imath.cos(alpha), imath.sin(alpha)
        cb, sb = imath.cos(beta), imath.sin(beta)

        R = [[ca*cb, ca*sb, -sa], [-sb, cb, interval[0,0]], [sa*cb, sa*sb, ca]]

        current_row = []

        # Coil contributions
        for j in range(num_coils):
            currents = np.zeros(num_coils)
            currents[j] = 1

            B_world, F_world, T_world = [interval[0,0]]*3, [interval[0,0]]*3, [interval[0,0]]*3

            # B-field from coils is needed for direct B output or for torque calculation
            if needs_B or needs_T:
                req_B = {'Bx': True, 'By': True, 'Bz': True}
                B_res = calculate_B_selective_interval(currents, X, Y, Z, req_B)
                if needs_B and component.startswith('B'):
                    B_world = [B_res.get('Bx', 0), B_res.get('By', 0), B_res.get('Bz', 0)]
                if needs_T and component.startswith('t'):
                    req_T = {'tx': True, 'ty': True, 'tz': True}
                    T_res = calculate_Torque_selective_interval(currents, X, Y, Z, m, alpha, beta, req_T, B_res)
                    T_world = [T_res.get('tx', 0), T_res.get('ty', 0), T_res.get('tz', 0)]

            # Derivatives from coils are needed for force calculation
            if needs_F and component.startswith('f'):
                req_derivs = {'Bx_dx': True, 'Bx_dy': True, 'Bx_dz': True, 'By_dy': True, 'By_dz': True}
                deriv_res = calculate_derivatives_selective_interval(currents, X, Y, Z, req_derivs)
                req_F = {'fx': True, 'fy': True, 'fz': True}
                F_res = calculate_Force_selective_interval(currents, X, Y, Z, m, alpha, beta, req_F, deriv_res)
                F_world = [F_res.get('fx', 0), F_res.get('fy', 0), F_res.get('fz', 0)]

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
            current_row.append(coil_effect_body)

        # Magnet interaction contribution
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
        current_row.append(magnet_effect_body)

        A_matrix.append(current_row)

    return np.array(A_matrix)


if __name__ == "__main__":
    # Interval example usage
    target_points = [
        {'X': interval[0.0, 0.0], 'Y': interval[0.013, 0.013], 'Z': interval[0.085, 0.085], 'm': interval[0.1, 0.1], 'alpha': interval[math.pi/2, math.pi/2], 'beta': interval[math.pi, math.pi], 'Bx': None, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': True, 'fy': None, 'fz': None, 'tx': None, 'ty': None, 'tz': None},
        {'X': interval[-0.01, -0.01], 'Y': interval[0.01, 0.01], 'Z': interval[0.085, 0.085], 'm': interval[0.1, 0.1], 'alpha': interval[math.pi/2, math.pi/2], 'beta': interval[math.pi, math.pi], 'Bx': None, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': True, 'fy': None, 'fz': None, 'tx': None, 'ty': None, 'tz': None},
        ]

    A = Map_I2H_Interval(target_points)
    # A = Map_I2H_Interval_body(target_points)
    print("Optimized mapping matrix with intervals:")
    print(A)
    print("shape of A:", A.shape)
