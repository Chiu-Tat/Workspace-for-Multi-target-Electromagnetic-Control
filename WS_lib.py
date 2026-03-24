import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, QhullError
from pypoman import compute_polytope_vertices
import sympy as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from itertools import combinations
from scipy.linalg import null_space

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

# Define the symbols
m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z = sp.symbols('m0 m1 m2 r0_0 r0_1 r0_2 X Y Z')

# Constants
mu0 = 4 * sp.pi * 1e-7

# Calculate displacement vector
dx = X - r0_0
dy = Y - r0_1
dz = Z - r0_2

# Calculate distance to the coordinate point
r = sp.sqrt(dx**2 + dy**2 + dz**2) + 1e-9  # Add a small constant to avoid division by zero

# Calculate dot product of displacement vector and magnetic dipole moment
dot_product = m0 * dx + m1 * dy + m2 * dz

# Calculate magnetic field components
model_Bx = (mu0 / (4 * sp.pi)) * (3 * dx * dot_product / r**5 - m0 / r**3)
model_By = (mu0 / (4 * sp.pi)) * (3 * dy * dot_product / r**5 - m1 / r**3)
model_Bz = (mu0 / (4 * sp.pi)) * (3 * dz * dot_product / r**5 - m2 / r**3)

# Convert the symbolic functions to numerical functions
dipole_model_Bx = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_Bx, 'numpy')
dipole_model_By = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_By, 'numpy')
dipole_model_Bz = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_Bz, 'numpy')

# Calculate the partial derivatives
model_Bx_dx = (mu0 / (4 * sp.pi)) * (
    3 * dot_product / r**5 + 
    6 * dx * m0 / r**5 - 
    15 * dx**2 * dot_product / r**7
)

model_Bx_dy = (mu0 / (4 * sp.pi)) * (
    3 * dx * m1 / r**5 +
    3 * m0 * dy / r**5 - 
    15 * dx * dy * dot_product / r**7
)

model_Bx_dz = (mu0 / (4 * sp.pi)) * (
    3 * dx * m2 / r**5 +
    3 * m0 * dz / r**5 - 
    15 * dx * dz * dot_product / r**7
)

model_By_dy = (mu0 / (4 * sp.pi)) * (
    3 * dot_product / r**5 + 
    3 * dy * m1 / r**5 - 
    15 * dy**2 * dot_product / r**7
)

model_By_dz = (mu0 / (4 * sp.pi)) * (
    3 * dy * m2 / r**5 +
    3 * m1 * dz / r**5 - 
    15 * dy * dz * dot_product / r**7
)

# Convert the symbolic functions to numerical functions
dipole_model_Bx_dx = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_Bx_dx, 'numpy')
dipole_model_Bx_dy = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_Bx_dy, 'numpy')
dipole_model_Bx_dz = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_Bx_dz, 'numpy')
dipole_model_By_dy = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_By_dy, 'numpy')
dipole_model_By_dz = sp.lambdify((m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z), model_By_dz, 'numpy')

# Calculate the magnetic field and its partial derivatives at given points
def calculate_B_and_derivatives(currents, X, Y, Z):
    Bx_total = 0
    By_total = 0
    Bz_total = 0
    Bx_dx_total = 0
    Bx_dy_total = 0
    Bx_dz_total = 0
    By_dy_total = 0
    By_dz_total = 0
    Bz_dz_total = 0

    # Loop over the coils
    for i in range(num_coils):
        # Get the coil parameters
        params = params_list[i]
        m0, m1, m2, r0_0, r0_1, r0_2 = params
        # Calculate the magnetic field produced by this coil
        Bx = dipole_model_Bx(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]
        By = dipole_model_By(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]
        Bz = dipole_model_Bz(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]

        # Add to the total magnetic field
        Bx_total += Bx
        By_total += By
        Bz_total += Bz

        # Calculate the partial derivatives
        Bx_dx = dipole_model_Bx_dx(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]
        Bx_dy = dipole_model_Bx_dy(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]
        Bx_dz = dipole_model_Bx_dz(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]
        By_dy = dipole_model_By_dy(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]
        By_dz = dipole_model_By_dz(m0, m1, m2, r0_0, r0_1, r0_2, X, Y, Z) * currents[i]

        # Add to the total partial derivatives
        Bx_dx_total += Bx_dx
        Bx_dy_total += Bx_dy
        Bx_dz_total += Bx_dz
        By_dy_total += By_dy
        By_dz_total += By_dz

    return np.array([Bx_total, By_total, Bz_total, Bx_dx_total, Bx_dy_total, Bx_dz_total, By_dy_total, By_dz_total])

# Calculate the Force and Torque at given points and orientations
def calculate_Force_and_Torque(currents, X, Y, Z, m, alpha, beta):
    
    Bx_total, By_total, Bz_total, Bx_dx_total, Bx_dy_total, Bx_dz_total, By_dy_total, By_dz_total = calculate_B_and_derivatives(currents, X, Y, Z)

    G = [Bx_dx_total, Bx_dy_total, Bx_dz_total, By_dy_total, By_dz_total]
    B = [Bx_total, By_total, Bz_total]

    mx = np.sin(alpha) * np.cos(beta)
    my = np.sin(alpha) * np.sin(beta)
    mz = np.cos(alpha)

    # Calculate the Force
    force_matrix = np.array([[mx, my, mz, 0, 0],
                      [0, mx, 0, my, mz],
                      [-mz, 0, mx, -mz, my]])
    Force = m * np.dot(force_matrix, G)

    # Calculate the Torque
    torque_matrix = np.array([[0, -mz, my],
                      [mz, 0, -mx],
                      [-my, mx, 0]])
    Torque = m * np.dot(torque_matrix, B)

    return Force, Torque
# Calculate the magnetic field and its partial derivatives at given points and given magnet
def calculate_B_and_derivatives_magnet(magnet_point, target_X, target_Y, target_Z):
    # Input parameters: magnet point X, Y, Z, magnet moment, alpha, beta
    # and the target point X, Y, Z in the world frame
    magnet_X = magnet_point['X']
    magnet_Y = magnet_point['Y']
    magnet_Z = magnet_point['Z']
    moment = magnet_point['m']
    alpha = magnet_point['alpha']
    beta = magnet_point['beta']

    # Calculate the magnetic moment of the magnet
    mx = np.sin(alpha) * np.cos(beta) * moment
    my = np.sin(alpha) * np.sin(beta) * moment
    mz = np.cos(alpha) * moment

    # Calculate the magnetic field produced by this magnet
    Bx = dipole_model_Bx(mx, my, mz, magnet_X, magnet_Y, magnet_Z, target_X, target_Y, target_Z)
    By = dipole_model_By(mx, my, mz, magnet_X, magnet_Y, magnet_Z, target_X, target_Y, target_Z)
    Bz = dipole_model_Bz(mx, my, mz, magnet_X, magnet_Y, magnet_Z, target_X, target_Y, target_Z)

    # Calculate the partial derivatives
    Bx_dx = dipole_model_Bx_dx(mx, my, mz, magnet_X, magnet_Y, magnet_Z, target_X, target_Y, target_Z)
    Bx_dy = dipole_model_Bx_dy(mx, my, mz, magnet_X, magnet_Y, magnet_Z, target_X, target_Y, target_Z)
    Bx_dz = dipole_model_Bx_dz(mx, my, mz, magnet_X, magnet_Y, magnet_Z, target_X, target_Y, target_Z)
    By_dy = dipole_model_By_dy(mx, my, mz, magnet_X, magnet_Y, magnet_Z, target_X, target_Y, target_Z)
    By_dz = dipole_model_By_dz(mx, my, mz, magnet_X, magnet_Y, magnet_Z, target_X, target_Y, target_Z)

    return np.array([Bx, By, Bz, Bx_dx, Bx_dy, Bx_dz, By_dy, By_dz])

# Calculate the Force and Torque for two magnets
def calculate_Force_and_Torque_magnet(magnet_point1, magnet_point2):
    # Calculate the force and torque applied on magnet 1 by magnet 2
    X1 = magnet_point1['X']
    Y1 = magnet_point1['Y']
    Z1 = magnet_point1['Z']
    m1 = magnet_point1['m']
    alpha1 = magnet_point1['alpha']
    beta1 = magnet_point1['beta']
    X2 = magnet_point2['X']
    Y2 = magnet_point2['Y']
    Z2 = magnet_point2['Z']
    m2 = magnet_point2['m']
    alpha2 = magnet_point2['alpha']
    beta2 = magnet_point2['beta']
    Bx, By, Bz, Bx_dx, Bx_dy, Bx_dz, By_dy, By_dz = calculate_B_and_derivatives_magnet(magnet_point2, X1, Y1, Z1)

    G = [Bx_dx, Bx_dy, Bx_dz, By_dy, By_dz]
    B = [Bx, By, Bz]

    mx = np.sin(alpha1) * np.cos(beta1)
    my = np.sin(alpha1) * np.sin(beta1)
    mz = np.cos(alpha1)

    # Calculate the Force
    force_matrix = np.array([[mx, my, mz, 0, 0],
                      [0, mx, 0, my, mz],
                      [-mz, 0, mx, -mz, my]])
    Force = m1 * np.dot(force_matrix, G)

    # Calculate the Torque
    torque_matrix = np.array([[0, -mz, my],
                      [mz, 0, -mx],
                      [-my, mx, 0]])
    Torque = m1 * np.dot(torque_matrix, B)
    return Force, Torque


# Create the mapping matrix between the currents with the magnetic field and wrench
# target_points = [{'X': , 'Y': , 'Z': , 'm': , 'alpha': , 'beta': , 'Bx': , 'By': , 'Bz': , 'Bx_dx': , 'Bx_dy': , 'Bx_dz': , 'By_dy': , 'By_dz': , 'fx': , 'fy': , 'fz': , 'tx': , 'ty': , 'tz': }] <<-- the first six are required parameters, the rest are optional depending on the need for workspace analysis
def Map_I2H(target_points):
    # Initialize the matrix as a zero matrix with num_coils+1 columns
    A = np.zeros((14 * len(target_points), num_coils + 1))

    # Loop over the target points
    for i, target_point in enumerate(target_points):
        X = target_point['X']
        Y = target_point['Y']
        Z = target_point['Z']
        m = target_point['m']
        alpha = target_point['alpha']
        beta = target_point['beta']

        # Calculate interaction effects from other magnets
        b_magnet = np.zeros(14)
        for k, other_target in enumerate(target_points):
            if k != i:  # Exclude self-interaction
                Bx, By, Bz, Bx_dx, Bx_dy, Bx_dz, By_dy, By_dz = calculate_B_and_derivatives_magnet(other_target, X, Y, Z)
                Force_magnet, Torque_magnet = calculate_Force_and_Torque_magnet(target_point, other_target)
                
                b_magnet += np.concatenate(([Bx, By, Bz], 
                                          [Bx_dx, Bx_dy, Bx_dz], 
                                          [By_dy, By_dz], 
                                          Force_magnet, Torque_magnet))

        # Set the magnet interaction column (last column)
        A[14*i:14*(i+1), -1] = b_magnet

        # Loop over the coils
        for j in range(num_coils):
            # Calculate the magnetic field and its derivatives for a unit current in the j-th coil
            currents = np.zeros(num_coils)
            currents[j] = 1
            Bx_total, By_total, Bz_total, Bx_dx_total, Bx_dy_total, Bx_dz_total, By_dy_total, By_dz_total = calculate_B_and_derivatives(currents, X, Y, Z)
            Force, Torque = calculate_Force_and_Torque(currents, X, Y, Z, m, alpha, beta)

            # Set the corresponding rows in the A matrix
            A[14*i:14*(i+1), j] = np.concatenate(([Bx_total, By_total, Bz_total], 
                                                  [Bx_dx_total, Bx_dy_total, Bx_dz_total], 
                                                  [By_dy_total, By_dz_total], 
                                                  Force, Torque))
    return A


def Extract_Map_I2H(target_points):
    row_selection_matrix = np.zeros((14 * len(target_points), 14 * len(target_points)))

    for i, target_point in enumerate(target_points): 
        if target_point['Bx'] is not None:
            row_selection_matrix[14*i, 14*i] = 1
        if target_point['By'] is not None:
            row_selection_matrix[14*i+1, 14*i+1] = 1
        if target_point['Bz'] is not None:
            row_selection_matrix[14*i+2, 14*i+2] = 1
        if target_point['Bx_dx'] is not None:
            row_selection_matrix[14*i+3, 14*i+3] = 1
        if target_point['Bx_dy'] is not None:
            row_selection_matrix[14*i+4, 14*i+4] = 1
        if target_point['Bx_dz'] is not None:
            row_selection_matrix[14*i+5, 14*i+5] = 1
        if target_point['By_dy'] is not None:
            row_selection_matrix[14*i+6, 14*i+6] = 1
        if target_point['By_dz'] is not None:
            row_selection_matrix[14*i+7, 14*i+7] = 1
        if target_point['fx'] is not None:
            row_selection_matrix[14*i+8, 14*i+8] = 1
        if target_point['fy'] is not None:
            row_selection_matrix[14*i+9, 14*i+9] = 1
        if target_point['fz'] is not None:
            row_selection_matrix[14*i+10, 14*i+10] = 1
        if target_point['tx'] is not None:
            row_selection_matrix[14*i+11, 14*i+11] = 1
        if target_point['ty'] is not None:
            row_selection_matrix[14*i+12, 14*i+12] = 1
        if target_point['tz'] is not None:
            row_selection_matrix[14*i+13, 14*i+13] = 1

    return row_selection_matrix[np.sum(row_selection_matrix, axis=1) != 0]


def Combined_Map_I2H(target_points):
    """
    Combined function that creates the mapping matrix and applies row selection in one step.
    
    Args:
        target_points: List of dictionaries containing target point information
        
    Returns:
        A_final: The final mapping matrix after row selection
    """
    # Initialize lists to track which rows to keep and store the matrix data
    selected_rows = []
    A_rows = []
    
    # Loop over the target points
    for i, target_point in enumerate(target_points):
        X = target_point['X']
        Y = target_point['Y']
        Z = target_point['Z']
        m = target_point['m']
        alpha = target_point['alpha']
        beta = target_point['beta']

        # Calculate interaction effects from other magnets
        b_magnet = np.zeros(14)
        for k, other_target in enumerate(target_points):
            if k != i:  # Exclude self-interaction
                Bx, By, Bz, Bx_dx, Bx_dy, Bx_dz, By_dy, By_dz = calculate_B_and_derivatives_magnet(other_target, X, Y, Z)
                Force_magnet, Torque_magnet = calculate_Force_and_Torque_magnet(target_point, other_target)
                
                b_magnet += np.concatenate(([Bx, By, Bz], 
                                          [Bx_dx, Bx_dy, Bx_dz], 
                                          [By_dy, By_dz], 
                                          Force_magnet, Torque_magnet))

        # Create the full row for this target point
        A_full_rows = np.zeros((14, num_coils + 1))
        
        # Set the magnet interaction column (last column)
        A_full_rows[:, -1] = b_magnet

        # Loop over the coils
        for j in range(num_coils):
            # Calculate the magnetic field and its derivatives for a unit current in the j-th coil
            currents = np.zeros(num_coils)
            currents[j] = 1
            Bx_total, By_total, Bz_total, Bx_dx_total, Bx_dy_total, Bx_dz_total, By_dy_total, By_dz_total = calculate_B_and_derivatives(currents, X, Y, Z)
            Force, Torque = calculate_Force_and_Torque(currents, X, Y, Z, m, alpha, beta)

            # Set the corresponding column in the A matrix
            A_full_rows[:, j] = np.concatenate(([Bx_total, By_total, Bz_total], 
                                              [Bx_dx_total, Bx_dy_total, Bx_dz_total], 
                                              [By_dy_total, By_dz_total], 
                                              Force, Torque))

        # Check which rows to keep and add them to the final matrix
        row_indices = [
            (0, 'Bx'), (1, 'By'), (2, 'Bz'), (3, 'Bx_dx'), (4, 'Bx_dy'), 
            (5, 'Bx_dz'), (6, 'By_dy'), (7, 'By_dz'), (8, 'fx'), (9, 'fy'), 
            (10, 'fz'), (11, 'tx'), (12, 'ty'), (13, 'tz')
        ]
        
        for row_idx, field_name in row_indices:
            if target_point[field_name] is not None:
                A_rows.append(A_full_rows[row_idx])
                selected_rows.append(14*i + row_idx)
    
    # Convert to numpy array
    if len(A_rows) > 0:
        A_final = np.array(A_rows)
    else:
        A_final = np.zeros((0, num_coils + 1))
    
    return A_final

#Compute the hyperplane representation of the zonotope
# def HyperPlaneShiftingMethod(A,Imin,Imax):
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
                lj = float(np.dot(np.transpose(A[:,j]), n).squeeze())
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

def ModifiedHyperplaneShiftingMethod(A_hat, Imin, Imax):
    """
    Computes the H-representation of a translated zonotope.

    The system is defined by B_hat = A_hat * [I; 1], where the currents in I
    vary between Imin and Imax.

    Args:
        A_hat (np.ndarray): The d x (n+1) augmented Jacobian matrix.
        Imin (float): The minimum current for the first n coils.
        Imax (float): The maximum current for the first n coils.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - N_hat (np.ndarray): The (m x d) matrix of hyperplane normal vectors.
            - d_vec_hat (np.ndarray): The (m x 1) vector of hyperplane distances.
    """
    # Step 1: Separate the original matrix A from the translation vector c.
    # The original A matrix consists of all columns except the last one.
    A = A_hat[:, :-1]
    
    # The translation vector c is the last column of A_hat.
    c = A_hat[:, -1].reshape(-1, 1)

    # Step 2: Call the original function with the original matrix A.
    # This computes the H-representation of the un-shifted zonotope.
    # The shape and orientation are determined by A, Imin, and Imax.
    N, d_vec = HyperPlaneShiftingMethod(A, Imin, Imax)

    # The normal vectors N are the same for the translated zonotope.
    N_hat = N

    # Step 3: Adjust the distances d_vec to account for the translation.
    # As derived, d_hat = d + (n · c). We can do this for all normals at once.
    # np.matmul(N, c) calculates the dot product of each normal vector in N with c.
    translation_offset = np.matmul(N_hat, c)
    # print(f"Translation offset: {translation_offset}")
    d_vec_hat = d_vec + translation_offset

    return N_hat, d_vec_hat

# Create the mapping matrix between the currents and the magnetic field
# target_points = [{'X': , 'Y': , 'Z': , 'Bx': , 'By': , 'Bz': , 'Bx_dx': , 'Bx_dy': , 'Bx_dz': , 'By_dy': , 'By_dz': }] <<-- the first three are required parameters, the rest are optional depending on the need for workspace analysis
def Map_I2B(target_points):

    # Initialize the matrix as a zero matrix
    A = np.zeros((8 * len(target_points), num_coils))

    # Loop over the target points
    for i, target_point in enumerate(target_points): 
        X = target_point['X']
        Y = target_point['Y']
        Z = target_point['Z']

        # Loop over the coils
        for j in range(num_coils):
            # Calculate the magnetic field and its derivatives for a unit current in the j-th coil
            currents = np.zeros(num_coils)
            currents[j] = 1
            Bx_total, By_total, Bz_total, Bx_dx_total, Bx_dy_total, Bx_dz_total, By_dy_total, By_dz_total = calculate_B_and_derivatives(currents, X, Y, Z)

            # Set the corresponding rows in the A matrix
            A[8*i:8*(i+1), j] = [Bx_total, By_total, Bz_total, Bx_dx_total, Bx_dy_total, Bx_dz_total, By_dy_total, By_dz_total]
    return A

# Create the extraction matrix between the currents and the magnetic field
# target_points = [{'X': , 'Y': , 'Z': , 'Bx': , 'By': , 'Bz': , 'Bx_dx': , 'Bx_dy': , 'Bx_dz': , 'By_dy': , 'By_dz': }] <<-- the first three are required parameters, the rest are optional depending on the need for workspace analysis
def Extract_Map_I2B(target_points):
    row_selection_matrix = np.zeros((8 * len(target_points), 8 * len(target_points)))

    for i, target_point in enumerate(target_points): 
        if target_point['Bx'] is not None:
            row_selection_matrix[8*i, 8*i] = 1
        if target_point['By'] is not None:
            row_selection_matrix[8*i+1, 8*i+1] = 1
        if target_point['Bz'] is not None:
            row_selection_matrix[8*i+2, 8*i+2] = 1
        if target_point['Bx_dx'] is not None:
            row_selection_matrix[8*i+3, 8*i+3] = 1
        if target_point['Bx_dy'] is not None:
            row_selection_matrix[8*i+4, 8*i+4] = 1
        if target_point['Bx_dz'] is not None:
            row_selection_matrix[8*i+5, 8*i+5] = 1
        if target_point['By_dy'] is not None:
            row_selection_matrix[8*i+6, 8*i+6] = 1
        if target_point['By_dz'] is not None:
            row_selection_matrix[8*i+7, 8*i+7] = 1
    return row_selection_matrix[np.sum(row_selection_matrix, axis=1) != 0]

def one_point_rotating_radius(target_points):
    A = Extract_Map_I2B(target_points) @ Map_I2B(target_points)
    G, k = HyperPlaneShiftingMethod(A, -15, 15)
    radius = np.min(k)
    return radius

def multi_point_rotating_radius(target_points):
    r_values = []
    A = Extract_Map_I2B(target_points) @ Map_I2B(target_points)
    G, k = HyperPlaneShiftingMethod(A, -15, 15)
    if G.shape[1] % 2 != 0:
        raise ValueError("Expected an even number of columns in G (Bx/By pairs per target point)")

    for j in range(G.shape[0]):
        g_row_pairs = G[j, :].reshape(-1, 2)
        denom = np.sum(np.linalg.norm(g_row_pairs, axis=1))
        r_val = k[j] / denom
        r_values.append(r_val)
    radius = np.min(r_values)
    return radius

if __name__ == "__main__":
    # Interval example usage
    target_points = [
        {'X': 0.0, 'Y': 0.013, 'Z': 0.0, 'm': 0.1, 'alpha': np.pi/2, 'beta': np.pi, 'Bx': None, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': True, 'fy': None, 'fz': None, 'tx': None, 'ty': None, 'tz': None},
        {'X': -0.01, 'Y': 0.01, 'Z': 0.0, 'm': 0.1,'alpha': np.pi/2, 'beta': np.pi, 'Bx': None, 'By': None, 'Bz': None, 'Bx_dx': None, 'Bx_dy': None, 'Bx_dz': None, 'By_dy': None, 'By_dz': None, 'fx': True, 'fy': None, 'fz': None, 'tx': None, 'ty': None, 'tz': None}
        ]

    A = Combined_Map_I2H(target_points)
    print("Mapping matrix with intervals:", A)
    print("shape of A:", A.shape)
    # I_11 = np.array([15, 15, 15, -15, -15, -15, 15, -15, -15, 15, 1])
    # Result = np.dot(A, I_11)
    # print("Result of mapping with intervals:", Result)