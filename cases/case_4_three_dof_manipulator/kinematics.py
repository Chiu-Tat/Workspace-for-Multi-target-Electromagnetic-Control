import numpy as np
# Run from the repository root: python -m cases.case_4_three_dof_manipulator.kinematics
import math

def robot_arm_kinematics(alpha, beta, gamma):
    # Forward kinematics calculations
    x_base = 0
    y_base = 0
    z_base = 0
    # L1_m = 0.01173
    # L1 = 0.01973
    # L2_m = 0.005
    # L2 = 0.008
    L1_m = 0.5
    L1 = 1
    L2_m = 0.5
    L2 = 1

    # position of the first magnet
    X1 = L1_m * math.sin(alpha) * math.cos(beta)
    Y1 = L1_m * math.sin(alpha) * math.sin(beta)
    Z1 = L1_m * math.cos(alpha)
    X1 += x_base
    Y1 += y_base
    Z1 += z_base

    # # position of the second magnet
    # X2 = L1 * math.cos(beta1) * math.sin(alpha1) + L2_m * math.cos(beta1) * math.sin(alpha1) * math.cos(beta2b) - L2_m * math.sin(beta1) * math.sin(beta2b)
    # Y2 = L1 * math.sin(beta1) * math.sin(alpha1) + L2_m * math.sin(beta1) * math.sin(alpha1) * math.cos(beta2b) + L2_m * math.cos(beta1) * math.sin(beta2b)
    # Z2 = L1 * math.cos(alpha1) + L2_m * math.cos(alpha1) * math.cos(beta2b)

    # # Calculate the angles for the second magnet
    # cos_alpha2 = np.cos(alpha1) * np.cos(beta2b)
    # sin_alpha2 = np.sqrt(1 - cos_alpha2**2)
    # alpha2w = np.arccos(cos_alpha2)

    # x = -np.sin(beta1) * np.sin(beta2b) + np.cos(beta1) * np.sin(alpha1) * np.cos(beta2b)
    # y = np.cos(beta1) * np.sin(beta2b) + np.sin(beta1) * np.sin(alpha1) * np.cos(beta2b)
    # beta2w = np.arctan2(y, x)

    # position of the second magnet
    X2 = (L1 * math.sin(alpha) + L2_m * math.sin(alpha + gamma)) * math.cos(beta)
    Y2 = (L1 * math.sin(alpha) + L2_m * math.sin(alpha + gamma)) * math.sin(beta)
    Z2 = L1 * math.cos(alpha) + L2_m * math.cos(alpha + gamma)
    X2 += x_base
    Y2 += y_base
    Z2 += z_base

    # Calculate the angles for the second magnet
    alpha2 = np.arccos(math.cos(alpha + gamma))
    beta2 = beta
    return X1, Y1, Z1, X2, Y2, Z2, alpha2, beta2

# for case 5
def arm_end_positions(beta1, alpha2):
    # Forward kinematics calculations
    # x_base = 0
    # y_base = 0
    # z_base = 0
    # L1_m = 0.5
    # L1 = 1
    # L2_m = 0.5
    # L2 = 1
    x_base = 0
    y_base = 0.023
    z_base = 0.079
    L1_m = 0.011
    L1 = 0.023
    L2_m = 0.011
    L2 = 0.029

    beta1 = beta1 + np.pi/2
    alpha2 = alpha2 - np.pi/2

    # position of the first magnet
    p1_x = L1_m * np.sin(beta1)
    p1_y = -L1_m * np.cos(beta1)
    p1_z = 0.0
    p1_x += x_base
    p1_y += y_base
    p1_z += z_base

    # position of the second magnet
    p2_x = (L1 + L2_m * np.cos(alpha2)) * np.sin(beta1)
    p2_y = -(L1 + L2_m * np.cos(alpha2)) * np.cos(beta1)
    p2_z = -L2_m * np.sin(alpha2)
    p2_x += x_base
    p2_y += y_base
    p2_z += z_base

    return p1_x, p1_y, p1_z, p2_x, p2_y, p2_z

if __name__ == "__main__":
    # Test the robot arm kinematics
    alpha1 = math.radians(90)
    beta1 = math.radians(-90)
    beta2b = math.radians(0)

    positions = robot_arm_kinematics(alpha1, beta1, beta2b)
    print("Positions and angles:", positions)
