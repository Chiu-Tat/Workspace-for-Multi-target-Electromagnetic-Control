import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, QhullError
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import combinations
from scipy.linalg import null_space
from custom_interval import interval, imath, fpu
import math

# 10 coils data 3D
params_list = [
    np.array([-13.04945069, -4.41557229, 6.47376799, 0.12129096, 0.00466922, -0.0174842]),  # coil 1
    np.array([-5.10083416, 13.54294901, 7.85474539, 0.05834654, -0.11165548, -0.01850546]),  # coil 2
    np.array([4.05088788, 14.23365818, 6.44760956, -0.05903076, -0.11020417, -0.01488244]),  # coil 3
    np.array([13.89011305, -0.06092074, 4.77365608, -0.12306086, -0.00085745, -0.01378161]), # coil 4
    np.array([11.44363813, -9.40543896, 4.46367162, -0.06806179, 0.1024875, -0.01397152]),  # coil 5
    np.array([-9.00577939, -12.78905365, 5.98650851, 0.06473315, 0.10618968, -0.0151172]),  # coil 6
    np.array([0.92820081, 8.54965337, 8.72298349, -0.00381254, -0.08845466, -0.08874662]),   # coil 7
    np.array([8.7302819, -4.90773115, 7.00109937, -0.07977306, 0.04481733, -0.08536032]),   # coil 8
    np.array([-7.68962762, -6.83258326, 8.12112247, 0.07498008, 0.04542436, -0.08696975]),   # coil 9
    np.array([2.35614001, -1.11370036, 14.00304846, -0.00722183, 0.00029277, -0.12482979])   # coil 10
]

num_coils = len(params_list)

def get_required_components(target_points):
    """Determine which components are actually needed across all target points"""
    required = {
        'Bx': False, 'By': False, 'Bz': False,
        'Bx_dx': False, 'Bx_dy': False, 'Bx_dz': False,
        'By_dy': False, 'By_dz': False,
        'fx': False, 'fy': False, 'fz': False,
        'tx': False, 'ty': False, 'tz': False
    }
    
    for target_point in target_points:
        for component in required.keys():
            if target_point.get(component, False):
                required[component] = True
    
    return required

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
            By_dy = mu0_4pi * (3 * dot_product / r5 + 3 * dy * m1 / r5 - 15 * dy**2 * dot_product / r7) * currents[i]
            results['By_dy'] = results['By_dy'] + By_dy
            
        if required_derivs['By_dz']:
            By_dz = mu0_4pi * (3 * dy * m2 / r5 + 3 * m1 * dz / r5 - 15 * dy * dz * dot_product / r7) * currents[i]
            results['By_dz'] = results['By_dz'] + By_dz

    return results

def calculate_Force_and_Torque_selective_interval(currents, X, Y, Z, m, alpha, beta, required_FT, B_results, deriv_results):
    """Calculate only the required force and torque components with interval inputs"""
    results = {}
    
    # Skip if no forces or torques are needed
    if not any(required_FT[key] for key in ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']):
        return results
    
    # Calculate magnetic moments only if needed
    need_moments = any(required_FT[key] for key in ['fx', 'fy', 'fz', 'tx', 'ty', 'tz'])
    if need_moments:
        mx = imath.sin(alpha) * imath.cos(beta)
        my = imath.sin(alpha) * imath.sin(beta)
        mz = imath.cos(alpha)
    
    # Calculate forces only if needed
    if any(required_FT[key] for key in ['fx', 'fy', 'fz']):
        if required_FT['fx']:
            fx = m * (mx * deriv_results.get('Bx_dx', interval[0, 0]) + 
                     my * deriv_results.get('Bx_dy', interval[0, 0]) + 
                     mz * deriv_results.get('Bx_dz', interval[0, 0]))
            results['fx'] = fx
            
        if required_FT['fy']:
            fy = m * (mx * deriv_results.get('Bx_dy', interval[0, 0]) + 
                     my * deriv_results.get('By_dy', interval[0, 0]) + 
                     mz * deriv_results.get('By_dz', interval[0, 0]))
            results['fy'] = fy
            
        if required_FT['fz']:
            fz = m * (-mz * deriv_results.get('Bx_dx', interval[0, 0]) + 
                     mx * deriv_results.get('Bx_dz', interval[0, 0]) - 
                     mz * deriv_results.get('By_dy', interval[0, 0]) + 
                     my * deriv_results.get('By_dz', interval[0, 0]))
            results['fz'] = fz

    # Calculate torques only if needed
    if any(required_FT[key] for key in ['tx', 'ty', 'tz']):
        if required_FT['tx']:
            tx = m * (-mz * B_results.get('By', interval[0, 0]) + 
                     my * B_results.get('Bz', interval[0, 0]))
            results['tx'] = tx
            
        if required_FT['ty']:
            ty = m * (mz * B_results.get('Bx', interval[0, 0]) - 
                     mx * B_results.get('Bz', interval[0, 0]))
            results['ty'] = ty
            
        if required_FT['tz']:
            tz = m * (-my * B_results.get('Bx', interval[0, 0]) + 
                     mx * B_results.get('By', interval[0, 0]))
            results['tz'] = tz

    return results

def calculate_B_and_derivatives_magnet_selective_interval(magnet_point, target_X, target_Y, target_Z, required_components):
    """Calculate only required magnetic field components from one magnet to target point with intervals"""
    results = {}
    
    # Skip if no components are needed
    if not any(required_components.values()):
        return results
    
    magnet_X = magnet_point['X']
    magnet_Y = magnet_point['Y']
    magnet_Z = magnet_point['Z']
    moment = magnet_point['m']
    alpha = magnet_point['alpha']
    beta = magnet_point['beta']

    # Calculate the magnetic moment of the magnet
    mx = imath.sin(alpha) * imath.cos(beta) * moment
    my = imath.sin(alpha) * imath.sin(beta) * moment
    mz = imath.cos(alpha) * moment

    # Calculate displacement
    dx = target_X - magnet_X
    dy = target_Y - magnet_Y
    dz = target_Z - magnet_Z
    
    # Calculate distance
    r = imath.sqrt(dx**2 + dy**2 + dz**2) + interval[1e-9, 1e-9]
    
    # Calculate dot product
    dot_product = mx * dx + my * dy + mz * dz
    
    # Magnetic field constants
    mu0_4pi = (4 * math.pi * 1e-7) / (4 * math.pi)
    
    # Calculate only required magnetic field components
    if required_components.get('Bx', False):
        results['Bx'] = mu0_4pi * (3 * dx * dot_product / (r**5) - mx / (r**3))
    if required_components.get('By', False):
        results['By'] = mu0_4pi * (3 * dy * dot_product / (r**5) - my / (r**3))
    if required_components.get('Bz', False):
        results['Bz'] = mu0_4pi * (3 * dz * dot_product / (r**5) - mz / (r**3))
    
    # Calculate partial derivatives only if needed
    need_derivatives = any(required_components.get(key, False) for key in ['Bx_dx', 'Bx_dy', 'Bx_dz', 'By_dy', 'By_dz'])
    if need_derivatives:
        r5 = r**5
        r7 = r**7
        
        if required_components.get('Bx_dx', False):
            results['Bx_dx'] = mu0_4pi * (3 * dot_product / r5 + 6 * dx * mx / r5 - 15 * dx**2 * dot_product / r7)
        if required_components.get('Bx_dy', False):
            results['Bx_dy'] = mu0_4pi * (3 * dx * my / r5 + 3 * mx * dy / r5 - 15 * dx * dy * dot_product / r7)
        if required_components.get('Bx_dz', False):
            results['Bx_dz'] = mu0_4pi * (3 * dx * mz / r5 + 3 * mx * dz / r5 - 15 * dx * dz * dot_product / r7)
        if required_components.get('By_dy', False):
            results['By_dy'] = mu0_4pi * (3 * dot_product / r5 + 3 * dy * my / r5 - 15 * dy**2 * dot_product / r7)
        if required_components.get('By_dz', False):
            results['By_dz'] = mu0_4pi * (3 * dy * mz / r5 + 3 * my * dz / r5 - 15 * dy * dz * dot_product / r7)

    return results

def calculate_Force_and_Torque_magnet_selective_interval(magnet_point1, magnet_point2, required_FT):
    """Calculate only required force and torque between two magnets with intervals"""
    results = {}
    
    # Skip if no forces or torques are needed
    if not any(required_FT.values()):
        return results
    
    X1 = magnet_point1['X']
    Y1 = magnet_point1['Y']
    Z1 = magnet_point1['Z']
    m1 = magnet_point1['m']
    alpha1 = magnet_point1['alpha']
    beta1 = magnet_point1['beta']
    
    # Determine what B and derivative components we need
    required_components = {}
    
    # For forces, we need derivatives
    if any(required_FT.get(key, False) for key in ['fx', 'fy', 'fz']):
        required_components.update({'Bx_dx': True, 'Bx_dy': True, 'Bx_dz': True, 'By_dy': True, 'By_dz': True})
    
    # For torques, we need B field components
    if any(required_FT.get(key, False) for key in ['tx', 'ty', 'tz']):
        required_components.update({'Bx': True, 'By': True, 'Bz': True})
    
    magnet_results = calculate_B_and_derivatives_magnet_selective_interval(magnet_point2, X1, Y1, Z1, required_components)

    mx = imath.sin(alpha1) * imath.cos(beta1)
    my = imath.sin(alpha1) * imath.sin(beta1)
    mz = imath.cos(alpha1)

    # Calculate only required forces
    if required_FT.get('fx', False):
        fx = m1 * (mx * magnet_results.get('Bx_dx', interval[0, 0]) + 
                  my * magnet_results.get('Bx_dy', interval[0, 0]) + 
                  mz * magnet_results.get('Bx_dz', interval[0, 0]))
        results['fx'] = fx
        
    if required_FT.get('fy', False):
        fy = m1 * (mx * magnet_results.get('Bx_dy', interval[0, 0]) + 
                  my * magnet_results.get('By_dy', interval[0, 0]) + 
                  mz * magnet_results.get('By_dz', interval[0, 0]))
        results['fy'] = fy
        
    if required_FT.get('fz', False):
        fz = m1 * (-mz * magnet_results.get('Bx_dx', interval[0, 0]) + 
                  mx * magnet_results.get('Bx_dz', interval[0, 0]) - 
                  mz * magnet_results.get('By_dy', interval[0, 0]) + 
                  my * magnet_results.get('By_dz', interval[0, 0]))
        results['fz'] = fz

    # Calculate only required torques
    if required_FT.get('tx', False):
        tx = m1 * (-mz * magnet_results.get('By', interval[0, 0]) + 
                  my * magnet_results.get('Bz', interval[0, 0]))
        results['tx'] = tx
        
    if required_FT.get('ty', False):
        ty = m1 * (mz * magnet_results.get('Bx', interval[0, 0]) - 
                  mx * magnet_results.get('Bz', interval[0, 0]))
        results['ty'] = ty
        
    if required_FT.get('tz', False):
        tz = m1 * (-my * magnet_results.get('Bx', interval[0, 0]) + 
                  mx * magnet_results.get('By', interval[0, 0]))
        results['tz'] = tz
    
    return results

def Map_I2H_Interval_Fast(target_points):
    """Create mapping matrix with interval arithmetic - optimized version that only calculates needed components"""
    
    # Determine which components are needed globally
    required = get_required_components(target_points)
    
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
            
            # Only calculate what we need for this specific component
            if component in ['Bx', 'By', 'Bz']:
                required_B = {comp: (comp == component) for comp in ['Bx', 'By', 'Bz']}
                B_results = calculate_B_selective_interval(currents, X, Y, Z, required_B)
                coil_effect = B_results.get(component, interval[0, 0])
                
            elif component in ['Bx_dx', 'Bx_dy', 'Bx_dz', 'By_dy', 'By_dz']:
                required_derivs = {comp: (comp == component) for comp in ['Bx_dx', 'Bx_dy', 'Bx_dz', 'By_dy', 'By_dz']}
                deriv_results = calculate_derivatives_selective_interval(currents, X, Y, Z, required_derivs)
                coil_effect = deriv_results.get(component, interval[0, 0])
                
            elif component in ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']:
                # For forces and torques, we need to calculate required B and derivatives
                required_B = {}
                required_derivs = {}
                required_FT = {comp: (comp == component) for comp in ['fx', 'fy', 'fz', 'tx', 'ty', 'tz']}
                
                # Determine what B components we need for this force/torque
                if component in ['tx', 'ty', 'tz']:
                    required_B = {'Bx': True, 'By': True, 'Bz': True}
                else:
                    required_B = {'Bx': False, 'By': False, 'Bz': False}
                
                # Determine what derivatives we need for this force/torque
                if component in ['fx', 'fy', 'fz']:
                    required_derivs = {'Bx_dx': True, 'Bx_dy': True, 'Bx_dz': True, 'By_dy': True, 'By_dz': True}
                else:
                    required_derivs = {'Bx_dx': False, 'Bx_dy': False, 'Bx_dz': False, 'By_dy': False, 'By_dz': False}
                
                B_results = calculate_B_selective_interval(currents, X, Y, Z, required_B)
                deriv_results = calculate_derivatives_selective_interval(currents, X, Y, Z, required_derivs)
                FT_results = calculate_Force_and_Torque_selective_interval(currents, X, Y, Z, m, alpha, beta, required_FT, B_results, deriv_results)
                coil_effect = FT_results.get(component, interval[0, 0])
            
            current_row.append(coil_effect)
        
        # Add this row to the matrix
        A_matrix.append(current_row)
    
    return np.array(A_matrix), row_mapping

def HyperPlaneShiftingMethod(A,Imin,Imax):
    #Create permutation matrix for the selection of unitary actuation fields
    def CreatePermuationMatrix(A):
        # A: Jacobian matrix
        # M: permutation matrix
        d = np.shape(A)[0] #dimension of output space (if field, this is 3)
        n = np.shape(A)[1] #number of coils
        comb = combinations(np.arange(n), d-1) 
        M = np.asarray(list(comb))  
        return M
    #Create combination matrix to test combination of field
    def CreateFieldCombinationMatrix(n):
        # n: dimension of the combination matrix
        # M: combination matrix
        nums = np.arange(2**n)
        M = ((nums.reshape(-1,1) & (2**np.arange(n))) != 0).astype(int)
        return M
    # Imin: minimum current (scalar) in A
    # Imax: maximum current (scalar) in A
    # J: Jacobian matrix of the eMNS
    # N, d_vec: Hyperplane representation of the zonotope
    # Imin = -10
    # Imax = 10
    dI = Imax - Imin
    M = CreatePermuationMatrix(A)
    nb_comb = np.shape(M)[0] #number of combination
    
    d = np.shape(A)[0] #dimension of output space (if field, this is 3)
    n_coils = np.shape(A)[1] #number of coils
    
    #Initialize matrix and vector for hyperplane representation
    N = np.zeros((2*nb_comb,d))
    d_vec = np.zeros((2*nb_comb,1))
    bmin = np.matmul(A,Imin*np.ones((n_coils,1)))
    
    #Iterate on the combination of unitary fields
    for i in range(nb_comb):
        # Step 1: define initial hyperplane
        #Define the set of vectors to be orthogonal with
        W = A[:,M[i,:]]
        
        #Get the orthogonal vector using the nullspace of W^T
        Wns = null_space(np.transpose(W))
        v = Wns[:,0]

        # Step 2: shift intial hyperplane
        temp = v / np.linalg.norm(v)
        n = temp.reshape((-1,1))
        
        # Step 3: build projections   
        lj_arr = np.zeros((n_coils-(d-1),1))
        k = 0
        h = 0. 
        for j in range(n_coils):
            if not(j in M[i,:]):
                lj = np.dot(np.transpose(A[:,j]),n)
                lj_arr[k,0] = lj
                k += 1

        C = CreateFieldCombinationMatrix(n_coils-(d-1))
        h = np.matmul(C,dI*lj_arr)
        hp = np.max(h)
        hm = np.min(h)
        
        #Step 4: compute hyperplane support
        pp = hp*n + bmin
        pm = hm*n + bmin
        
        # Step 5: build hyperplane representation
        N[i,:] = n.T
        N[i+nb_comb,:] = -n.T
        d_vec[i,:] = np.dot(n.T,pp)
        d_vec[i+nb_comb,:] = np.dot(-n.T,pm)
    return N, d_vec


def HyperPlaneShiftingMethod_Interval(Aint,Imin,Imax):
    """
    Interval arithmetic version of HyperPlaneShiftingMethod
    
    Args:
        Aint: Interval matrix (Jacobian matrix with interval entries)
        Imin: minimum current (scalar) in A
        Imax: maximum current (scalar) in A
    
    Returns:
        N: Hyperplane normals (interval matrix)
        d_vec: Hyperplane distances (interval vector)
    """
    #Create permutation matrix for the selection of unitary actuation fields
    def CreatePermuationMatrix(A):
        # A: Jacobian matrix
        # M: permutation matrix
        d = np.shape(A)[0] #dimension of output space (if field, this is 3)
        n = np.shape(A)[1] #number of coils
        comb = combinations(np.arange(n), d-1) 
        M = np.asarray(list(comb))  
        return M
    
    #Create combination matrix to test combination of field
    def CreateFieldCombinationMatrix(n):
        # n: dimension of the combination matrix
        # M: combination matrix
        nums = np.arange(2**n)
        M = ((nums.reshape(-1,1) & (2**np.arange(n))) != 0).astype(int)
        return M
    
    # Convert scalar inputs to intervals if needed
    if not isinstance(Imin, type(interval[0, 0])):
        Imin = interval[Imin, Imin]
    if not isinstance(Imax, type(interval[0, 0])):
        Imax = interval[Imax, Imax]
    
    dI = Imax - Imin
    M = CreatePermuationMatrix(Aint)
    nb_comb = np.shape(M)[0] #number of combination
    
    d = np.shape(Aint)[0] #dimension of output space (if field, this is 3)
    n_coils = np.shape(Aint)[1] #number of coils
    
    # Initialize matrix and vector for hyperplane representation with intervals
    N = np.empty((2*nb_comb, d), dtype=object)
    d_vec = np.empty((2*nb_comb, 1), dtype=object)
    
    # Calculate bmin with interval arithmetic
    ones_interval = np.array([Imin for _ in range(n_coils)])
    bmin = np.zeros(d, dtype=object)
    for i in range(d):
        bmin[i] = interval[0, 0]
        for j in range(n_coils):
            bmin[i] = bmin[i] + Aint[i, j] * ones_interval[j]
    bmin = bmin.reshape((-1, 1))
    
    #Iterate on the combination of unitary fields
    for i in range(nb_comb):
        # Step 1: define initial hyperplane
        # Define the set of vectors to be orthogonal with
        W = Aint[:, M[i, :]]
        
        # For interval matrices, we need to work with the structure
        # We'll use the midpoint for null space calculation and then apply interval arithmetic
        W_mid = np.zeros((d, d-1))
        for row in range(d):
            for col in range(d-1):
                W_mid[row, col] = (W[row, col].lower + W[row, col].upper) / 2
        
        # Get the orthogonal vector using the nullspace of W^T
        Wns = null_space(np.transpose(W_mid))
        v = Wns[:, 0]

        # Step 2: shift initial hyperplane
        temp = v / np.linalg.norm(v)
        n = temp.reshape((-1, 1))
        
        # Convert to interval vector
        n_interval = np.array([interval[val, val] for val in n.flatten()]).reshape((-1, 1))
        
        # Step 3: build projections with interval arithmetic
        lj_arr = np.zeros((n_coils-(d-1), 1), dtype=object)
        k = 0
        
        for j in range(n_coils):
            if not(j in M[i, :]):
                # Calculate dot product with interval arithmetic
                lj = interval[0, 0]
                for row in range(d):
                    lj = lj + Aint[row, j] * n_interval[row, 0]
                lj_arr[k, 0] = lj
                k += 1

        C = CreateFieldCombinationMatrix(n_coils-(d-1))
        
        # Calculate h with interval arithmetic - CORRECTED VERSION
        # In the original: h = np.matmul(C, dI*lj_arr)
        # This means h[i] = sum over j of C[i,j] * dI * lj_arr[j,0]
        h = np.zeros(C.shape[0], dtype=object)
        for row in range(C.shape[0]):
            h[row] = interval[0, 0]
            for col in range(C.shape[1]):
                h[row] = h[row] + interval[C[row, col], C[row, col]] * dI * lj_arr[col, 0]
        
        # Find max and min over all intervals in h
        # This is where we need to be careful - we want the maximum upper bound and minimum lower bound
        hp_candidates = []
        hm_candidates = []
        for h_val in h:
            hp_candidates.append(h_val.upper)
            hm_candidates.append(h_val.lower)
        
        hp = max(hp_candidates)
        hm = min(hm_candidates)
        
        # Convert back to intervals
        hp_interval = interval[hp, hp]
        hm_interval = interval[hm, hm]
        
        #Step 4: compute hyperplane support with interval arithmetic
        pp = np.zeros((d, 1), dtype=object)
        pm = np.zeros((d, 1), dtype=object)
        
        for row in range(d):
            pp[row, 0] = hp_interval * n_interval[row, 0] + bmin[row, 0]
            pm[row, 0] = hm_interval * n_interval[row, 0] + bmin[row, 0]
        
        # Step 5: build hyperplane representation
        for col in range(d):
            N[i, col] = n_interval[col, 0]
            N[i+nb_comb, col] = -n_interval[col, 0]
        
        # Calculate dot products for d_vec
        d_val_pos = interval[0, 0]
        d_val_neg = interval[0, 0]
        
        for col in range(d):
            d_val_pos = d_val_pos + n_interval[col, 0] * pp[col, 0]
            d_val_neg = d_val_neg + (-n_interval[col, 0]) * pm[col, 0]
        
        d_vec[i, 0] = d_val_pos
        d_vec[i+nb_comb, 0] = d_val_neg
    
    return N, d_vec


if __name__ == "__main__":
    # Test with the same example
    target_points = [
        {'X': interval[0, 0.02], 'Y': interval[0, 0], 'Z': interval[-0.04, -0.02], 'm': interval[0.00, 0.00], 'alpha': interval[math.pi/2, math.pi/2], 'beta': interval[0, 0], 'Bx': True, 'By': None, 'Bz': True, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': None, 'fy': None, 'fz': None, 'tx': None, 'ty': None, 'tz': None},
    ]

    print("Testing fast implementation...")
    A_fast, row_mapping = Map_I2H_Interval_Fast(target_points)
    print(f"Fast mapping matrix shape: {A_fast.shape}")
    print(f"Row mapping: {row_mapping}")
    print(f"Matrix values: {A_fast}")
    
    Imin, Imax = -15, 15
    N, d_vec = HyperPlaneShiftingMethod_Interval(A_fast, Imin, Imax)
    print(f"Hyperplane normals (N) shape: {N.shape}")
    print(f"Hyperplane distances (d_vec) shape: {d_vec.shape}")
    print(f"First hyperplane distance interval: {d_vec}")
    all_intervals = d_vec.flatten()
    # Find the maximum of all upper bounds, which represents the maximum reach.
    max_upper_value = max(iv.upper for iv in all_intervals)
    print(f"Maximum value (from upper bounds) in d_vec: {max_upper_value}")
    # You can also find the specific interval that contains this maximum value
    min_lower_value = min(iv.lower for iv in all_intervals)
    print(f"Minimum value (from lower bounds) in d_vec: {min_lower_value}")
    # interval_with_max_upper = max(all_intervals, key=lambda x: x.upper)
    # print(f"Interval containing the maximum value: {interval_with_max_upper}")

    # # Compare with original (if needed)
    # from WS_lib_Interval import Extract_Map_I2H, Map_I2H_Interval
    # A_original = Extract_Map_I2H(target_points) @ Map_I2H_Interval(target_points)
    # print(f"Original mapping matrix shape: {A_original.shape}")
    
    # print("✓ Fast implementation test completed!") 