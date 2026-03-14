import numpy as np
from scipy.optimize import linprog
from custom_interval import interval, imath
import math
import multiprocessing as mp
from functools import partial
import concurrent.futures
import threading
from itertools import product

def CreateFieldCombinationMatrix(n):
    """Create combination matrix to test combination of field"""
    nums = np.arange(2**n)
    M = ((nums.reshape(-1,1) & (2**np.arange(n))) != 0).astype(int)
    return M

# ============================================================================
# Original Functions (for compatibility)
# ============================================================================

def RobustFeasible(Ainterval, I_interval, b_interval):
    """
    Provides a fast implementation of the original Feasible check using a
    single, equivalent linear program. This avoids iterating through all 2^d
    vertices and should be significantly more efficient while producing the
    same result.

    The method verifies that for each of the 2^d vertices of the uncertainty space,
    a valid current vector exists.
    """
    Ainterval_obj = np.asarray(Ainterval)
    I_interval_obj = np.asarray(I_interval)
    b_interval_obj = np.asarray(b_interval)

    d = Ainterval_obj.shape[0]
    n = Ainterval_obj.shape[1]

    # Extract bounds from interval objects
    A_lower = np.array([[Ainterval_obj[i, j].lower for j in range(n)] for i in range(d)])
    A_upper = np.array([[Ainterval_obj[i, j].upper for j in range(n)] for i in range(d)])
    b_min = np.array([b.lower for b in b_interval_obj])
    b_max = np.array([b.upper for b in b_interval_obj])
    I_max_v = np.array([i.upper for i in I_interval_obj])
    I_min_v = np.array([i.lower for i in I_interval_obj])

    # Generate all 2^d vertices of the d-dimensional hypercube {-1, 1}^d
    Y = np.array(list(np.ndindex(*((2,)*d)))) * 2 - 1

    # Number of vertices
    num_vertices = 2**d
    
    # The variables for the single LP will be a concatenation of the variables
    # for each of the 2^d original LPs.
    # Each original LP has 2*n variables (I_plus, I_minus).
    # Total variables = num_vertices * 2 * n
    c = np.zeros(num_vertices * 2 * n)

    # We will build the block-diagonal constraint matrix for the single LP.
    A_eq_blocks = []
    b_eq_concat = np.array([])
    
    for k in range(num_vertices):
        y_k = Y[k, :]
        
        # Construct the vertex matrices Ay, Amy and vector by for this vertex
        Ay_k = np.zeros((d, n))
        Amy_k = np.zeros((d, n))
        by_k = np.zeros(d)
        
        for i in range(d):
            if y_k[i] == 1:
                Ay_k[i, :] = A_lower[i, :]
                Amy_k[i, :] = A_upper[i, :]
                by_k[i] = b_max[i]
            else: # y_k[i] == -1
                Ay_k[i, :] = A_upper[i, :]
                Amy_k[i, :] = A_lower[i, :]
                by_k[i] = b_min[i]
        
        # Form the equality constraint block for this vertex
        A_eq_k = np.hstack([Ay_k, -Amy_k])
        A_eq_blocks.append(A_eq_k)
        b_eq_concat = np.concatenate([b_eq_concat, by_k])

    # Create the large block-diagonal matrix for equality constraints
    from scipy.linalg import block_diag
    A_eq = block_diag(*A_eq_blocks)

    # Upper bound constraints for currents (for all vertices)
    # I_plus - I_minus <= I_max
    # -I_plus + I_minus <= -I_min
    I_block = np.hstack([np.eye(n), -np.eye(n)])
    A_ub_I_max = np.kron(np.eye(num_vertices), I_block)
    b_ub_I_max = np.tile(I_max_v, num_vertices)

    I_block_neg = np.hstack([-np.eye(n), np.eye(n)])
    A_ub_I_min = np.kron(np.eye(num_vertices), I_block_neg)
    b_ub_I_min = np.tile(-I_min_v, num_vertices)
    
    A_ub = np.vstack([A_ub_I_max, A_ub_I_min])
    b_ub = np.hstack([b_ub_I_max, b_ub_I_min])

    # All variables must be non-negative
    bounds = (0, None)
    
    # Solve the single large LP
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq_concat, method='highs')
        return res.success
    except Exception as e:
        # If solver fails, it is likely infeasible
        return False

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
    Check if the system is outside the workspace using interval constraint propagation.
    Simplified implementation based on the check_outside reference function.
    """
    import itertools
    
    d = np.shape(Ainter)[0]  # dimension (should be 2 for 2D)
    nc = len(Ainter[0])      # number of coils
    
    # Get field bounds from b_intervals
    field_bounds = []
    for i in range(d):
        b_interval = b_intervals[i]
        bmin_i = b_interval[0][0]
        bmax_i = b_interval[0][1]
        field_bounds.append([bmin_i, bmax_i])
    
    # Test all combinations of field corner values
    for b_tuple in itertools.product(*field_bounds):
        b_desired = np.array(b_tuple)
        
        # Initialize current intervals for all coils
        current_I = []
        for k in range(nc):
            I_interval = I_intervals[k]
            Imin_k = I_interval[0][0]
            Imax_k = I_interval[0][1]
            current_I.append(interval[Imin_k, Imax_k])
        
        # Iterative constraint propagation (max 5 iterations)
        for _ in range(5):
            changed = False
            
            # For each coil (variable) k
            for k in range(nc):
                # For each equation (constraint) i
                for i in range(d):
                    # Get coefficient A[i,k]
                    A_ik = Ainter[i][k]
                    A_ik_interval = interval[A_ik[0][0], A_ik[0][1]]
                    
                    # Skip if coefficient is zero
                    if A_ik[0][1] - A_ik[0][0] == 0 and A_ik[0][0] == 0:
                        continue
                    
                    # Calculate sum of other terms: sum(A_ij * I_j for j != k)
                    sum_part = interval[0, 0]
                    for j in range(nc):
                        if j != k:
                            A_ij = Ainter[i][j]
                            A_ij_interval = interval[A_ij[0][0], A_ij[0][1]]
                            sum_part = sum_part + A_ij_interval * current_I[j]
                    
                    # Calculate right-hand side: b_i - sum_part
                    rhs = b_desired[i] - sum_part
                    
                    # Calculate what I_k should be: I_k = rhs / A_ik
                    try:
                        new_I_k = rhs / A_ik_interval
                    except:
                        continue  # Skip if division fails
                    
                    # Store original width
                    original_width = current_I[k][0][1] - current_I[k][0][0]
                    
                    # Intersect with current interval
                    current_I[k] = current_I[k] & new_I_k
                    
                    # Check if interval became empty
                    bounds = current_I[k][0]
                    if bounds[0] > bounds[1] or math.isnan(bounds[0]) or math.isnan(bounds[1]):
                        return True  # System is infeasible - box is outside workspace
                    
                    # Check if interval width decreased (convergence detection)
                    new_width = bounds[1] - bounds[0]
                    if new_width < original_width - 1e-12:
                        changed = True
            
            # If no changes occurred, we've converged
            if not changed:
                break
    
    # If we reach here without finding infeasibility, the box is not definitively outside
    return False

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
        I_intervals = create_I_intervals(10, custom_bounds={8: (-10, 12)})
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

def get_interval_rotation_matrix_2d(beta):
    c_beta, s_beta = imath.cos(beta), imath.sin(beta)
    return np.array([[-s_beta, c_beta], [c_beta, s_beta]])


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

def Create3DBox(xi, yi, zi):
    """Create a 3D box for interval analysis"""
    # Initialize
    Box = [[0 for c in range(1)] for r in range(3)]
    
    xmin = xi[0][0]
    xmax = xi[0][1]
    ymin = yi[0][0]
    ymax = yi[0][1]
    zmin = zi[0][0]
    zmax = zi[0][1]
    
    Box[0][0] = interval[xmin, xmax] 
    Box[1][0] = interval[ymin, ymax]
    Box[2][0] = interval[zmin, zmax]
    
    return Box