#!/usr/bin/env python3
# Run from the repository root: python -m analysis.affine.omni_single_point
"""
Affine analysis for the determination of the workspace of an electromagnetic navigation system
"""

# Standard library imports
import os
import sys
import math
import yaml
from itertools import combinations
from itertools import product

# Scientific computing imports
import numpy as np
from numpy.linalg import lstsq
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.linalg import null_space
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
from scipy.optimize import linprog

# Plotting imports
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm
import matplotlib as mpl
from matplotlib.patches import Circle
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
from matplotlib.colors import LightSource

# Interval analysis imports - use our custom module
from emns.numerics.interval import interval, imath, fpu
from emns.workspace.interval_fast import Map_I2H_Interval_Fast

# ============================================================================
# Affine Arithmetic Engine
# ============================================================================

_next_symbol_index = 0

def get_next_symbol_index():
    global _next_symbol_index
    _next_symbol_index += 1
    return _next_symbol_index - 1

class Affine:
    """A simple Affine Arithmetic class."""
    def __init__(self, center, deviations=None):
        self.center = center
        self.deviations = deviations if deviations is not None else {}

    @classmethod
    def from_interval(cls, interval_obj):
        # Handle both interval objects and scalar values
        if isinstance(interval_obj, (int, float)):
            return cls(interval_obj)

        center = (interval_obj[0][0] + interval_obj[0][1]) / 2
        radius = (interval_obj[0][1] - interval_obj[0][0]) / 2
        if radius == 0:
            return cls(center)
        symbol_index = get_next_symbol_index()
        return cls(center, {symbol_index: radius})

    def to_interval(self):
        radius = sum(abs(v) for v in self.deviations.values())
        return interval[self.center - radius, self.center + radius]

    def __add__(self, other):
        if not isinstance(other, Affine):
            return Affine(self.center + other, self.deviations.copy())

        new_center = self.center + other.center
        new_devs = self.deviations.copy()
        for k, v in other.deviations.items():
            new_devs[k] = new_devs.get(k, 0) + v
        return Affine(new_center, new_devs)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, Affine):
            return Affine(self.center - other, self.deviations.copy())

        new_center = self.center - other.center
        new_devs = self.deviations.copy()
        for k, v in other.deviations.items():
            new_devs[k] = new_devs.get(k, 0) - v
        return Affine(new_center, new_devs)

    def __rsub__(self, other):
        res = self.__sub__(other)
        return Affine(-res.center, {k: -v for k, v in res.deviations.items()})

    def __neg__(self):
        return Affine(-self.center, {k: -v for k, v in self.deviations.items()})

    def __mul__(self, other):
        if not isinstance(other, Affine):
            return Affine(self.center * other, {k: v * other for k, v in self.deviations.items()})

        new_center = self.center * other.center
        new_devs = {}

        # Linear terms
        for k, v in self.deviations.items():
            new_devs[k] = new_devs.get(k, 0) + v * other.center
        for k, v in other.deviations.items():
            new_devs[k] = new_devs.get(k, 0) + v * self.center

        # Quadratic error term
        quad_error_radius = sum(abs(v) for v in self.deviations.values()) * \
                            sum(abs(v) for v in other.deviations.values())

        if quad_error_radius > 1e-9: # Avoid adding tiny error terms
            error_symbol = get_next_symbol_index()
            new_devs[error_symbol] = quad_error_radius

        return Affine(new_center, new_devs)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        # Division is complex in AA. For this problem, we can approximate by inverting.
        # 1 / (c + sum(d_i * e_i)) approx 1/c - (sum(d_i * e_i)) / c^2
        if not isinstance(other, Affine):
            return self * (1.0 / other)

        # Check for division by zero interval
        other_interval = other.to_interval()
        if other_interval[0][0] <= 0 <= other_interval[0][1]:
            # If interval contains zero, use safer division
            inv_center = 1.0 / other.center if abs(other.center) > 1e-9 else 1e9
            return Affine(inv_center)

        inv_other = Affine(1.0 / other.center) # Simplified for this example
        return self * inv_other

    def __pow__(self, exponent):
        if exponent == 2:
            return self * self
        elif exponent == 3:
            return self * self * self
        else:
            # For other powers, convert to interval and back (conservative)
            interval_result = self.to_interval() ** exponent
            return Affine.from_interval(interval_result)

    def __repr__(self):
        dev_str = " + ".join([f"{v:.3f}e{k}" for k, v in self.deviations.items()])
        return f"Affine({self.center:.3f}, {dev_str})"

def affine_sin(a: Affine):
    interval_obj = a.to_interval()
    # Convert to standard interval arithmetic for sin calculation
    sin_interval = imath.sin(interval_obj)
    cos_interval = imath.cos(interval_obj)

    # Use the interval form for the linear part's coefficient
    alpha = cos_interval[0][0] if abs(cos_interval[0][0]) > abs(cos_interval[0][1]) else cos_interval[0][1]

    # Linear approximation
    y = a - a.center
    res_lin = Affine(math.sin(a.center)) + y * alpha

    # Error term
    error_radius = 0.5 * max(abs(sin_interval[0][0]), abs(sin_interval[0][1])) * (interval_obj[0][1] - interval_obj[0][0])**2 / 4
    if error_radius > 1e-9:
        error_symbol = get_next_symbol_index()
        res_lin.deviations[error_symbol] = error_radius
    return res_lin

def affine_cos(a: Affine):
    interval_obj = a.to_interval()
    cos_interval = imath.cos(interval_obj)
    sin_interval = imath.sin(interval_obj)

    alpha = -sin_interval[0][0] if abs(sin_interval[0][0]) > abs(sin_interval[0][1]) else -sin_interval[0][1]

    y = a - a.center
    res_lin = Affine(math.cos(a.center)) + y * alpha

    error_radius = 0.5 * max(abs(cos_interval[0][0]), abs(cos_interval[0][1])) * (interval_obj[0][1] - interval_obj[0][0])**2 / 4
    if error_radius > 1e-9:
        error_symbol = get_next_symbol_index()
        res_lin.deviations[error_symbol] = error_radius
    return res_lin

def affine_sqrt(a: Affine):
    interval_obj = a.to_interval()
    if interval_obj[0][0] < 0:
        interval_obj = interval[0, interval_obj[0][1]]

    # Linear approximation
    c0 = a.center
    if c0 <= 0: c0 = 1e-9 # Avoid division by zero

    alpha = 0.5 / math.sqrt(c0)
    y = a - c0
    res_lin = Affine(math.sqrt(c0)) + y * alpha

    # Error term
    width = interval_obj[0][1] - interval_obj[0][0]
    error_radius = 0.125 * (width**2) * (c0**(-1.5)) if c0 > 1e-9 else 0
    if error_radius > 1e-9:
        error_symbol = get_next_symbol_index()
        res_lin.deviations[error_symbol] = error_radius
    return res_lin

# ============================================================================
# Affine version of magnetic field calculations
# ============================================================================

def calculate_B_selective_affine(currents, X, Y, Z, required_B):
    """Affine version of calculate_B_selective_interval"""
    global _next_symbol_index
    _next_symbol_index = 0  # Reset for each calculation

    # Import actual coil parameters from WS_lib_Interval_fast
    from emns.workspace.interval_fast import params_list, num_coils

    results = {}

    # Skip if no B components are needed
    if not any(required_B[key] for key in ['Bx', 'By', 'Bz']):
        return results

    # Convert inputs to affine forms
    X_affine = Affine.from_interval(X)
    Y_affine = Affine.from_interval(Y)
    Z_affine = Affine.from_interval(Z)

    # Initialize only needed components
    if required_B['Bx']:
        results['Bx'] = Affine(0)
    if required_B['By']:
        results['By'] = Affine(0)
    if required_B['Bz']:
        results['Bz'] = Affine(0)

    # Loop over the coils using real parameters
    for i in range(num_coils):
        if currents[i] == 0:  # Skip coils with zero current
            continue

        params = params_list[i]
        m0, m1, m2, r0_0, r0_1, r0_2 = params
        m0_affine = Affine(m0)
        m1_affine = Affine(m1)
        m2_affine = Affine(m2)
        r0_0_affine = Affine(r0_0)
        r0_1_affine = Affine(r0_1)
        r0_2_affine = Affine(r0_2)

        # Calculate displacement with affine arithmetic
        dx = X_affine - r0_0_affine
        dy = Y_affine - r0_1_affine
        dz = Z_affine - r0_2_affine

        # Calculate distance
        r_squared = dx * dx + dy * dy + dz * dz
        r = affine_sqrt(r_squared)

        # Calculate dot product
        dot_product = m0_affine * dx + m1_affine * dy + m2_affine * dz

        # Magnetic field constants
        mu0_4pi = (4 * math.pi * 1e-7) / (4 * math.pi)
        mu0_4pi_affine = Affine(mu0_4pi)
        current_affine = Affine(currents[i])

        # Calculate only required magnetic field components
        r_cubed = r * r * r
        r_fifth = r_cubed * r_squared

        if required_B['Bx']:
            Bx = mu0_4pi_affine * (Affine(3) * dx * dot_product / r_fifth - m0_affine / r_cubed) * current_affine
            results['Bx'] = results['Bx'] + Bx

        if required_B['By']:
            By = mu0_4pi_affine * (Affine(3) * dy * dot_product / r_fifth - m1_affine / r_cubed) * current_affine
            results['By'] = results['By'] + By

        if required_B['Bz']:
            Bz = mu0_4pi_affine * (Affine(3) * dz * dot_product / r_fifth - m2_affine / r_cubed) * current_affine
            results['Bz'] = results['Bz'] + Bz

    # Convert affine results back to intervals
    interval_results = {}
    for key in results:
        interval_results[key] = results[key].to_interval()

    return interval_results

def affine_target_point_calculation(target_point):
    """Affine version of target point field calculation using real coil parameters"""
    global _next_symbol_index
    _next_symbol_index = 0  # Reset for each calculation

    # Import actual parameters
    from emns.workspace.interval_fast import num_coils

    X = target_point['X']
    Y = target_point['Y']
    Z = target_point['Z']

    # Only calculate components that are requested
    required_B = {'Bx': target_point.get('Bx', False),
                  'By': target_point.get('By', False),
                  'Bz': target_point.get('Bz', False)}

    # Count active components
    active_components = [comp for comp in ['Bx', 'By', 'Bz'] if target_point.get(comp, False)]
    total_rows = len(active_components)

    if total_rows == 0:
        return np.array([]).reshape(0, num_coils), []

    # Initialize matrix
    A_matrix = []

    # Process each active component
    for component in active_components:
        current_row = []

        # Loop over the coils - each coil contributes to this row
        for j in range(num_coils):
            # Calculate the magnetic field for a unit current in the j-th coil
            currents = np.zeros(num_coils)
            currents[j] = 1

            # Calculate only the required component
            required_B_single = {comp: (comp == component) for comp in ['Bx', 'By', 'Bz']}
            B_results = calculate_B_selective_affine(currents, X, Y, Z, required_B_single)
            coil_effect = B_results.get(component, interval[0, 0])

            current_row.append(coil_effect)

        # Add this row to the matrix
        A_matrix.append(current_row)

    return np.array(A_matrix), active_components

# ============================================================================
# Utility Functions for Affine Analysis
# ============================================================================

def create_I_intervals(num_coils, default_min=-15, default_max=15, custom_bounds=None):
    """
    Create current interval list with default bounds and optional custom bounds for specific coils

    Args:
        num_coils: Number of coils
        default_min: Default minimum current
        default_max: Default maximum current
        custom_bounds: Dict of {coil_index: (min, max)} for custom bounds

    Returns:
        List of interval objects for each coil

    Example:
        # 10 coils, most with [-15, 15], coil 9 with [-10, 12]
        I_intervals = create_I_intervals(10, custom_bounds={9: (-10, 12)})
    """
    I_intervals = []
    for i in range(num_coils):
        if custom_bounds and i in custom_bounds:
            min_val, max_val = custom_bounds[i]
            I_intervals.append(interval[min_val, max_val])
        else:
            I_intervals.append(interval[default_min, default_max])
    return I_intervals

def create_b_intervals(num_components, default_min=-0.09, default_max=0.09, custom_bounds=None):
    """
    Create magnetic field interval list with default bounds and optional custom bounds for specific components

    Args:
        num_components: Number of B components (e.g., 2 for 2D: Bx, By)
        default_min: Default minimum magnetic field
        default_max: Default maximum magnetic field
        custom_bounds: Dict of {component_index: (min, max)} for custom bounds

    Returns:
        List of interval objects for each B component

    Example:
        # 2D case: Bx with [-0.09, 0.09], By with [-0.08, 0.08]
        b_intervals = create_b_intervals(2, custom_bounds={1: (-0.08, 0.08)})
    """
    b_intervals = []
    for i in range(num_components):
        if custom_bounds and i in custom_bounds:
            min_val, max_val = custom_bounds[i]
            b_intervals.append(interval[min_val, max_val])
        else:
            b_intervals.append(interval[default_min, default_max])
    return b_intervals

def CreateFieldCombinationMatrix(n):
    """Create combination matrix to test combination of field"""
    nums = np.arange(2**n)
    M = ((nums.reshape(-1,1) & (2**np.arange(n))) != 0).astype(int)
    return M

def get_affine_rotation_matrix_2d(beta):
    """Affine version of the 2D rotation matrix"""
    beta_affine = Affine.from_interval(beta)
    c_beta, s_beta = affine_cos(beta_affine), affine_sin(beta_affine)

    # Convert back to intervals for the rotation matrix
    c_beta_interval = c_beta.to_interval()
    s_beta_interval = s_beta.to_interval()

    return np.array([[-s_beta_interval, c_beta_interval],
                     [c_beta_interval, s_beta_interval]])

def calculate_affine_A(xi, yi, zi, beta):
    """Affine version of calculate_interval_A"""
    target_points = [
    {'X': xi, 'Y': yi, 'Z': zi, 'm': interval[0.00, 0.00], 'alpha': interval[0.00, 0.00], 'beta': interval[0.00, 0.00], 'Bx': True, 'By': True, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None, 'fz': None, 'tx': None, 'ty': None, 'tz': None}
    ]

    A, row = affine_target_point_calculation(target_points[0])
    A_rotated = get_affine_rotation_matrix_2d(beta) @ A
    return A_rotated

# Keep the same feasibility checking functions from the original
def Feasible(Ainter, I_intervals, b_intervals):
    """Test if the box is feasible with independent interval constraints for each component"""
    d = np.shape(Ainter)[0]
    n = np.shape(Ainter)[1]

    # Build the vertex matrices
    # Create combination matrix
    M = CreateFieldCombinationMatrix(d)
    Ytemp = 2*M-1
    Y = Ytemp.transpose()

    # Initialize feasability to True
    isFeasible = True

    for k in range(2**d):
        # Initialization of the vertex matrix and vertex vector
        Ay = [[0 for c in range(len(Ainter[0]))] for r in range(d)]
        Amy = [[0 for c in range(len(Ainter[0]))] for r in range(d)]
        by = np.zeros((d, 1))

        # Construction
        for i in range(d):
            for j in range(len(Ainter[0])):
                Aij = Ainter[i][j]
                Ay[i][j] = Aij[0][0] + (Aij[0][1] - Aij[0][0])*(1 - Y[i,k])/2
                Amy[i][j] = Aij[0][0] + (Aij[0][1] - Aij[0][0])*(1 + Y[i,k])/2

            # Use individual B interval bounds for each component
            b_interval = b_intervals[i]
            bmin_i = b_interval[0][0]
            bmax_i = b_interval[0][1]
            by[i] = bmin_i + (bmax_i - bmin_i) * (1 + Y[i,k])/2

        # Check strong solvability at the vertex using simplex method
        c = np.zeros((2*n, 1))
        A = np.concatenate((np.asarray(Ay), -np.asarray(Amy)), axis=1)
        b = by

        # Use individual current interval bounds for each coil
        Imax_v = np.zeros((n, 1))
        Imin_v = np.zeros((n, 1))
        for i in range(n):
            I_interval = I_intervals[i]
            Imax_v[i] = I_interval[0][1]  # Upper bound
            Imin_v[i] = -I_interval[0][0]  # Negative of lower bound

        beq = np.concatenate((Imax_v, Imin_v), axis=0)
        I = np.eye(n)
        Aeq_temp_up = np.concatenate((I, -I), axis=1)
        Aeq_temp_down = np.concatenate((-I, I), axis=1)
        Aeq = np.concatenate((Aeq_temp_up, Aeq_temp_down), axis=0)

        bnds = (0, None)

        try:
            res = linprog(c, Aeq, beq, A, b, bounds=(bnds))
            isVertexFeasible = res.success
        except:
            isVertexFeasible = False

        if not(isVertexFeasible):
            isFeasible = False
            break

    return isFeasible

def Out(Ainter, I_intervals, b_intervals):
    """
    Fallback implementation using interval constraint propagation
    with independent interval constraints for each component

    This method uses iterative constraint propagation to check if the interval
    system A*I = b has a solution within the given bounds, similar to PyIbex
    but implemented using basic interval arithmetic.
    """
    import itertools

    d = np.shape(Ainter)[0]  # dimension (should be 2 for 2D)
    nc = len(Ainter[0])      # number of coils

    try:
        # Create interval matrix from Ainter
        A_interval = []
        for i in range(d):
            row = []
            for j in range(nc):
                Aij = Ainter[i][j]
                # Create interval object with correct bounds
                lower = Aij[0][0]
                upper = Aij[0][1]
                row.append(interval[lower, upper])
            A_interval.append(row)

        # Convert to numpy array for easier handling
        A_matrix = np.array(A_interval, dtype=object)

        # Define field bounds - test corners of the field polytope using individual B intervals
        field_bounds = []
        for i in range(d):
            b_interval = b_intervals[i]
            bmin_i = b_interval[0][0]
            bmax_i = b_interval[0][1]
            field_bounds.append([bmin_i, bmax_i])

        # Generate all combinations of field corner values
        for b_tuple in itertools.product(*field_bounds):
            b_desired = np.array(b_tuple)

            # Initial current intervals for all coils using individual I intervals
            current_I = []
            for k in range(nc):
                I_interval = I_intervals[k]
                Imin_k = I_interval[0][0]
                Imax_k = I_interval[0][1]
                current_I.append(interval[Imin_k, Imax_k])

            # Iterative constraint propagation (max 10 iterations)
            for iteration in range(10):
                changed = False

                # For each coil (variable) k
                for k in range(nc):
                    # For each equation (constraint) i
                    for i in range(d):
                        A_ik = A_matrix[i, k]

                        # Skip if coefficient is zero
                        if hasattr(A_ik, 'width') and A_ik.width() == 0 and A_ik[0][0] == 0:
                            continue

                        # Calculate sum of other terms: sum(A_ij * I_j for j != k)
                        sum_part = interval[0, 0]  # Start with zero interval
                        for j in range(nc):
                            if j != k:
                                A_ij = A_matrix[i, j]
                                sum_part = sum_part + A_ij * current_I[j]

                        # Calculate right-hand side: b_i - sum_part
                        rhs = b_desired[i] - sum_part

                        # Calculate what I_k should be: I_k = rhs / A_ik
                        try:
                            new_I_k = rhs / A_ik
                        except:
                            # Division by zero or other error - skip this constraint
                            continue

                        # Store original width to detect changes
                        if hasattr(current_I[k], 'width'):
                            original_width = current_I[k].width()
                        else:
                            original_width = current_I[k][0][1] - current_I[k][0][0]

                        # Intersect with current interval
                        current_I[k] = current_I[k] & new_I_k

                        # Check if interval became empty
                        if hasattr(current_I[k], 'is_empty'):
                            if current_I[k].is_empty():
                                return True  # System is infeasible - box is outside workspace
                        else:
                            # Check for empty interval in our custom implementation
                            bounds = current_I[k][0]
                            if bounds[0] > bounds[1] or math.isnan(bounds[0]) or math.isnan(bounds[1]):
                                return True  # System is infeasible - box is outside workspace

                        # Check if interval width decreased (convergence detection)
                        if hasattr(current_I[k], 'width'):
                            new_width = current_I[k].width()
                        else:
                            bounds = current_I[k][0]
                            new_width = bounds[1] - bounds[0]

                        if new_width < original_width - 1e-12:  # Small tolerance for numerical precision
                            changed = True

                # If no changes occurred, we've converged
                if not changed:
                    break

        # If we reach here, at least one field combination is feasible
        return False

    except Exception as e:
        # If any error occurs, use conservative approach
        print(f"Warning in constraint propagation: {e}")
        return False

def Create2DBox(xi, yi):
    """Create a 2D box for interval analysis"""
    # Initialize
    Box = [[0 for c in range(1)] for r in range(2)]

    xmin = xi[0][0]
    xmax = xi[0][1]
    ymin = yi[0][0]
    ymax = yi[0][1]
    Box[0][0] = interval[xmin, xmax]
    Box[1][0] = interval[ymin, ymax]

    return Box

def BisectBox(Box):
    """Bisection of a box and return the list of bisected boxes"""
    d = np.shape(Box)[0]

    # Initialize
    Bb = [[0 for c in range(2)] for r in range(d)]
    comb = product(np.arange(2), repeat=d)
    M = np.asarray(list(comb))

    # Divide each 1D-intervals
    for i in range(d):
        inter = Box[i][0]
        xmin = inter[0][0]
        xmax = inter[0][1]
        Bb[i][0] = interval[xmin, (xmin+xmax)/2]
        Bb[i][1] = interval[(xmin+xmax)/2, xmax]

    # Create list of bisected boxes
    Box_list = []
    for i in range(2**d):
        B_temp = [[0 for c in range(1)] for r in range(d)]
        for j in range(d):
            B_temp[j][0] = Bb[j][M[i,j]]
        Box_list.append(B_temp)

    return Box_list

def Ws2DDetermination(InitBox, b_intervals, I_intervals, eps_pos, eps_ori, zi, angle_interval, max_iter=-1):
    """
    Determination of the 2D workspace of a eMNS using affine analysis with independent constraints
    Modified to handle orientation intervals - finds boxes where ALL orientations in Angle_interval satisfy b_intervals
    """
    from collections import deque

    Lin = []
    Lout = []
    Lneg = []
    outer_queue = deque([InitBox])
    iteration = 0
    memo = {}  # Memoization to avoid recomputing same position boxes

    def is_valid_for_all_orientations(pos_box, full_angle_range):
        """Check if position box is valid for ALL orientations in the given range"""
        # Create memoization key
        xi, yi = pos_box[0][0], pos_box[1][0]
        # full_angle_range is an interval object, access bounds directly
        pos_key = (xi[0][0], xi[0][1], yi[0][0], yi[0][1], full_angle_range[0][0], full_angle_range[0][1])
        if pos_key in memo:
            return memo[pos_key]

        inner_queue = deque([full_angle_range])

        while inner_queue:
            current_angle_range = inner_queue.popleft()

            # Calculate affine matrix A for current position and orientation box
            Ainter = calculate_affine_A(xi, yi, zi, current_angle_range)

            # Check if this orientation range is completely outside workspace
            isOut = Out(Ainter, I_intervals, b_intervals)
            if isOut:
                memo[pos_key] = 'OUT'
                return 'OUT'

            # Check if this orientation range is completely inside workspace
            isFeasible = Feasible(Ainter, I_intervals, b_intervals)
            if isFeasible:
                # This angle range is feasible, continue checking other ranges
                continue

            # If orientation range is too small, mark as unknown
            if (current_angle_range[0][1] - current_angle_range[0][0]) < eps_ori:
                memo[pos_key] = 'UNKNOWN'
                return 'UNKNOWN'

            # Bisect the orientation interval
            angle_min = current_angle_range[0][0]
            angle_max = current_angle_range[0][1]
            angle_mid = (angle_min + angle_max) / 2

            # Add bisected orientation intervals to queue
            inner_queue.append(interval[angle_min, angle_mid])
            inner_queue.append(interval[angle_mid, angle_max])

        # If we've checked all orientation subdivisions without finding any outside regions
        memo[pos_key] = 'IN'
        return 'IN'

    # Main outer loop for position boxes
    while outer_queue:
        if iteration >= max_iter and max_iter >= 1:
            print('Max iter reached')
            break

        pos_box = outer_queue.popleft()
        iteration += 1

        # Check if this position box is valid for all orientations in angle_interval
        # angle_interval is a list containing one interval object
        angle_range = angle_interval[0]  # Get the interval object from the list
        status = is_valid_for_all_orientations(pos_box, angle_range)

        if status == 'IN':
            Lin.append(pos_box)
        elif status == 'OUT':
            Lout.append(pos_box)
        else:  # UNKNOWN
            # Extract position intervals
            xi = pos_box[0][0]
            yi = pos_box[1][0]

            # Check if position box is small enough to stop subdivision
            if (xi[0][1] - xi[0][0]) < eps_pos and (yi[0][1] - yi[0][0]) < eps_pos:
                Lneg.append(pos_box)
            else:
                # Bisect position box
                Bisected_box = BisectBox(pos_box)
                outer_queue.extend(Bisected_box)

    return [], Lin, Lout, Lneg, iteration

def Coils2DPatch(coil_length, coil_width, cal_path_mns):
    """Plot 2D coils"""
    with open(cal_path_mns, 'r') as yaml_file:
        parsed_yaml_file = yaml.load(yaml_file, Loader=yaml.FullLoader)

    patches_list = []

    coil_list = parsed_yaml_file.get("Coil_List")
    n_coil = len(coil_list)

    # Iterate on coils
    for i in range(0, n_coil-1):
        # Get actual coil
        coil_i = parsed_yaml_file.get("Coil_"+str(i))
        n_sources = len(coil_i.get("Source_List"))

        # Iterate on sources for the coil
        for j in range(0, n_sources):
            # Get source parameters
            src_j = coil_i.get("Src_"+str(j))
            src_dir = src_j.get("Source_Direction")
            src_pos = src_j.get("Source_Position")

            # Transform position in source frame
            xpos = src_pos[0]
            ypos = src_pos[1]
            zpos = src_pos[2]

            angle = np.arctan2(src_dir[1], src_dir[0])

            # Coil
            coil = patches.Rectangle((0, 0), coil_length, coil_width, color="black", alpha=0.80)
            toff = mpl.transforms.Affine2D().translate(-coil_length/2, -coil_width/2)
            r = mpl.transforms.Affine2D().rotate_deg(angle*180/math.pi)
            t = mpl.transforms.Affine2D().translate(xpos, ypos)
            coil.set_transform(toff + r + t)

            # Center
            center = patches.Circle((xpos, ypos), coil_width/10, color="blue", alpha=1.)

            patches_list.append(coil)
            patches_list.append(center)

    p = PatchCollection(patches_list, alpha=0.4)
    return p

# ============================================================================
# Main Application
# ============================================================================

def main():
    """Main function reproducing the affine analysis for omnidirectional workspace with flexible constraints and orientation intervals"""

    print("Affine analysis for the determination of the workspace of an electromagnetic navigation system")
    print("Modified to handle ALL orientations in the specified Angle_interval")
    print("=" * 80)

    # Load systems calibration
    cal_path_mns = 'models/mns_10coils.yaml'

    # Check if the calibration file exists
    if not os.path.exists(cal_path_mns):
        print(f"Error: Calibration file {cal_path_mns} not found!")
        return

    print(f"Loading system calibration from: {cal_path_mns}")

    # Build initial box
    pmin = -0.05
    pmax = 0.05
    xi = interval[pmin, pmax]
    yi = interval[pmin, pmax]
    zi = interval[0, 0]
    InitBox = Create2DBox(xi, yi)

    # Define interval constraints
    b_intervals = [
        interval[0, 0],
        interval[0, 0.113],  # Bx bounds
    ]

    I_intervals = [interval[-15, 15]] * 10

    Angle_interval = [
        interval[0, 2 * math.pi]
    ]

    # Define precision of bisection and run the determination algorithm
    eps_pos = 0.001
    eps_ori = 0.03
    print(f"Running workspace determination with position precision eps_pos = {eps_pos}")
    print(f"and orientation precision eps_ori = {eps_ori}")
    print("This may take some time...")

    L, Lin, Lout, Lneg, iteration = Ws2DDetermination(
        InitBox, b_intervals, I_intervals, eps_pos, eps_ori, zi, Angle_interval
    )

    print(f"Completed after {iteration} iterations")
    print(f"Boxes in workspace: {len(Lin)}")
    print(f"Boxes out of workspace: {len(Lout)}")
    print(f"Undetermined boxes: {len(Lneg)}")

    # Plotting
    print("Creating plot...")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()

    # Plot rectangle in workspace (green)
    for i in range(len(Lin)):
        Box = Lin[i]
        xo = Box[0][0][0][0]
        yo = Box[1][0][0][0]
        dx = Box[0][0][0][1] - xo
        dy = Box[1][0][0][1] - yo
        rect = patches.Rectangle((xo, yo), dx, dy, linewidth=1, edgecolor='black',
                               facecolor='lightgreen', alpha=0.5)
        ax.add_patch(rect)

    # Plot rectangle undetermined (gray)
    for i in range(len(Lneg)):
        Box = Lneg[i]
        xo = Box[0][0][0][0]
        yo = Box[1][0][0][0]
        dx = Box[0][0][0][1] - xo
        dy = Box[1][0][0][1] - yo
        rect = patches.Rectangle((xo, yo), dx, dy, linewidth=1, edgecolor='black',
                               facecolor='gray', alpha=0.2)
        ax.add_patch(rect)

    # Plot rectangle out of workspace (red)
    for i in range(len(Lout)):
        Box = Lout[i]
        xo = Box[0][0][0][0]
        yo = Box[1][0][0][0]
        dx = Box[0][0][0][1] - xo
        dy = Box[1][0][0][1] - yo
        rect = patches.Rectangle((xo, yo), dx, dy, linewidth=1, edgecolor='black',
                               facecolor='salmon', alpha=0.2)
        ax.add_patch(rect)

    # Format plot
    ax.axis('equal')
    ax.set_title('Workspace determination by affine analysis\n(with flexible constraints and orientation intervals)')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.5, label='In workspace'),
                      Patch(facecolor='gray', alpha=0.2, label='Undetermined'),
                      Patch(facecolor='red', alpha=0.2, label='Out of workspace')]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.show()

    print("Analysis complete!")

if __name__ == "__main__":
    main()
