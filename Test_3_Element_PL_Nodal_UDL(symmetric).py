"""
Created on Thu Jul 23 15:56:13 2026

@author: pwi72
"""

import numpy as np
import matplotlib as plt

#MATERIAL AND SECTION PROPERTIES FOR EACH BEAM (edit these individually)

E1 = 200e9
E2 = 200e9
E3 = 200e9

Area1 = 5e-4  #Beam 1 cross-sectional area in m^2
Area2 = 5e-4  #Beam 2 cross-sectional area in m^2
Area3 = 5e-4  #Beam 3 cross-sectional area in m^2

I1 = 1e-5  #Beam 1 second moment of area in m^4
I2 = 1e-5  #Beam 2 second moment of area in m^4
I3 = 1e-5  #Beam 3 second moment of area in m^4

#BEAM LENGTHS

L1 = 3
L2 = 4.5
L3 = 3
pi = np.pi

#Angle relitive to +ve x-axis in radians

alpha1 = (90*pi)/180
alpha2 = 0
alpha3 = (-90*pi)/180

#UDL ON EACH BEAM (Direction of UDL is downwards, hence negative. Set to 0 if none)

UDL1 = 0
UDL2 = -10000
UDL3 = 0

#POINT LOAD ON EACH BEAM (Direction of Point Load is downwards, hence negative. Set to 0 if none)
#Position (a) is measured from LOCAL NODE 1 (start of the beam) in metres, so the load
#does NOT need to be at the centre of the beam (0 <= a <= L).

Point_Load1 = 0
a1 = 0        #position of Point_Load1 along beam 1, measured from node 1

Point_Load2 = -50000
a2 = L2/2     #position of Point_Load2 along beam 2, measured from node 1 (here: centre, matches original)

Point_Load3 = 0
a3 = 0        #position of Point_Load3 along beam 3, measured from node 1

# Assembly matrices for each beam element ROWS ARE DEGREES OF FREEDOM, COLUMNS ARE ELEMENT NODES
# degrees of freedom may have to be added depending on the number of nodes in the structure, 
# and the assembly matrices will have to be edited accordingly.

# If beams join using a pin there will need to be another degree of freedom for the rotation at that node, 
# and the assembly matrices will have to be edited accordingly.
A1_Matrix = np.zeros((6, 6))
A1_Matrix[0, 3] = 1
A1_Matrix[1, 4] = 1
A1_Matrix[2, 5] = 1
print(A1_Matrix)

A2_Matrix = np.identity(6)
print(A2_Matrix)

A3_Matrix = np.zeros((6, 6))
A3_Matrix[3, 0] = 1
A3_Matrix[4, 1] = 1
A3_Matrix[5, 2] = 1
print(A3_Matrix)

#Nodal point loads in the global reference frame, make sure to count degrees of freedom
# whatever node I assign as node 1 in the free body diagram will be the first node in the global reference frame, and the degrees of freedom will be counted from that node.

Q_nodal1 = 10000
Q_nodal2 = 0
Q_nodal3 = 0
Q_nodal4 = 10000
Q_nodal5 = 0
Q_nodal6 = 0

Q_nodal = np.array([[Q_nodal1],
                    [Q_nodal2], 
                    [Q_nodal3], 
                    [Q_nodal4], 
                    [Q_nodal5], 
                    [Q_nodal6]])
print(Q_nodal)




def Local_BAR(E, A, L, I):
    """returns the local bar stiffness matrix """
    Beta = (A * L**2) / I
    K_local = np.array([[Beta , 0  , 0    , -Beta, 0   , 0    ], 
                        [0    , 12 , 6*L  , 0    , -12 , 6*L  ], 
                        [0    , 6*L, 4*L**2, 0    , -6*L, 2*L**2], 
                        [-Beta, 0  , 0    , Beta , 0   , 0    ], 
                        [0    , -12, -6*L , 0    , 12  , -6*L ], 
                        [0    , 6*L, 2*L**2, 0    , -6*L, 4*L**2]])
    
    K_e = ((E * I) / L**3) * K_local
    return K_e

def global_bar(K_e, alpha):
    """returns the local stiffness matrix in the global reference frame"""
    c = np.cos(alpha)
    s = np.sin(alpha)
    Lamda_Matrix = np.array([[ c, s, 0], 
                             [-s, c, 0], 
                             [ 0, 0, 1]])
    Transform_Matrix = np.zeros((6, 6))
    Transform_Matrix[:3, :3] = Lamda_Matrix
    Transform_Matrix[3:, 3:] = Lamda_Matrix
    print(Transform_Matrix)
    K_e_hat = np.transpose(Transform_Matrix) @ K_e @ Transform_Matrix
    return K_e_hat

#Local stiffness matrices for each beam element 
K1 = Local_BAR(E1, Area1, L1, I1)
print(f"K1 = 1 x 10^7 x \n{K1/1e7}\n")

K2 = Local_BAR(E2, Area2, L2, I2)
print(f"K2 = 1 x 10^7 x \n{K2/1e7}\n")

K3 = Local_BAR(E3, Area3, L3, I3)
print(f"K3 = 1 x 10^7 x \n{K3/1e7}\n")

#Local stiffness matrices in the global reference frame for each beam element
K1hat = global_bar(K1, alpha1)
print(f"K1hat = 1 x 10^8 x \n{K1hat/1e8}\n")
    
K2hat = global_bar(K2, alpha2)
print(f"K2hat = 1 x 10^8 x \n{K2hat/1e8}\n")

K3hat = global_bar(K3, alpha3)
print(f"K3hat = 1 x 10^7 x \n{K3hat/1e7}\n")

# Global stiffness matrices for each beam element, using the assembly matrix for each element 
# to describe where the element stiffness matrix is located in the global stiffness matrix.
KG1 = A1_Matrix @ K1hat @ np.transpose(A1_Matrix)
print(f"KG1 = 1 x 10^7 x \n{KG1/1e7}\n")

KG2 = A2_Matrix @ K2hat @ np.transpose(A2_Matrix)
print(f"KG2 = 1 x 10^8 x \n{KG2/1e8}\n")

KG3 = A3_Matrix @ K3hat @ np.transpose(A3_Matrix)
print(f"KG3 = 1 x 10^8 x \n{KG3/1e8}\n")


# Global stiffness matrix for the entire structure, by summing the global stiffness matrices for each beam element.
KG = KG1 + KG2 + KG3
print(f"KG = 1 x 10^7 x \n{KG/1e7}\n")

def force_eq_local_UDL(UDL, L):
    """calculates and returns the eqivalent nodal point loading for a UDL"""
    f_eq = np.array([[0             ],
                     [(UDL * L) / 2 ], 
                     [(UDL * L**2) / 12], 
                     [0], 
                     [(UDL * L) / 2], 
                     [-((UDL * L**2) / 12)]])
    return f_eq


#Calculates the equivalent nodal point loading for a UDL on each beam in the local reference frame.
f1_eq_UDL_local = force_eq_local_UDL(UDL1, L1)
f2_eq_UDL_local = force_eq_local_UDL(UDL2, L2)
f3_eq_UDL_local = force_eq_local_UDL(UDL3, L3)

def transformation_matrix(alpha):
    """returns the element transformation matrix"""
    c = np.cos(alpha)
    s = np.sin(alpha)
    Lamda_Matrix = np.array([[ c, s, 0], 
                             [-s, c, 0], 
                             [ 0, 0, 1]])
    Transform_Matrix = np.zeros((6, 6))
    Transform_Matrix[:3, :3] = Lamda_Matrix
    Transform_Matrix[3:, 3:] = Lamda_Matrix
    return Transform_Matrix

#Calculates the equivalent nodal point loading for a UDL on each beam in the global reference frame.
F1_eq_UDL_global = np.transpose(transformation_matrix(alpha1)) @ f1_eq_UDL_local
F2_eq_UDL_global = np.transpose(transformation_matrix(alpha2)) @ f2_eq_UDL_local
F3_eq_UDL_global = np.transpose(transformation_matrix(alpha3)) @ f3_eq_UDL_local


# Calculates the equivalent nodal point loading for a UDL on each beam in the global reference frame, 
# using the assembly matrix for each beam to extract where the equivalent nodal point loading is located in the global equivalent nodal point loading vector.
Q1_eq_UDL = A1_Matrix @ F1_eq_UDL_global
Q2_eq_UDL = A2_Matrix @ F2_eq_UDL_global
Q3_eq_UDL = A3_Matrix @ F3_eq_UDL_global

print(f"{Q1_eq_UDL}\n")
print(f"{Q2_eq_UDL}\n")
print(f"{Q3_eq_UDL}\n")

def force_eq_local_PL(P, L, a):
    """calculates and returns the eqivalent nodal point loading for a Point Load AT ANY POSITION 'a'
    along the beam, measured from local node 1 (0 <= a <= L). Reduces to the original
    mid-span-only formula when a = L/2."""
    b = L - a
    f_eq = np.array([[0                          ],
                     [(P * b**2 * (3*a + b)) / L**3], 
                     [(P * a * b**2) / L**2       ], 
                     [0                          ], 
                     [(P * a**2 * (a + 3*b)) / L**3], 
                     [-((P * a**2 * b) / L**2)     ]])
    return f_eq

#Calculates the equivalent nodal point loading for a Point Load on each beam in the LOCAL reference frame.
f1_eq_PL_local = force_eq_local_PL(Point_Load1, L1, a1)
f2_eq_PL_local = force_eq_local_PL(Point_Load2, L2, a2)
f3_eq_PL_local = force_eq_local_PL(Point_Load3, L3, a3)

#Calculates the equivalent nodal point loading for a Point Load on each beam in the GLOBAL reference frame.
F1_eq_PL_global = np.transpose(transformation_matrix(alpha1)) @ f1_eq_PL_local
F2_eq_PL_global = np.transpose(transformation_matrix(alpha2)) @ f2_eq_PL_local
F3_eq_PL_global = np.transpose(transformation_matrix(alpha3)) @ f3_eq_PL_local

# Calculates the equivalent nodal point loading for a Point Load on each beam in the GLOBAL reference frame, 
# using the assembly matrix for each beam to extract where the equivalent nodal point loading is located in the global equivalent nodal point loading vector.
Q1_eq_PL = A1_Matrix @ F1_eq_PL_global
Q2_eq_PL = A2_Matrix @ F2_eq_PL_global
Q3_eq_PL = A3_Matrix @ F3_eq_PL_global

print(f"{Q1_eq_PL}\n")
print(f"{Q2_eq_PL}\n")
print(f"{Q3_eq_PL}\n")

#calculates the total equivalent nodal point loading for the entire structure, by summing the equivalent nodal point loadings for the UDLs, Point Loads and Nodal Point Loads on all three beams.
Q_total = (Q1_eq_UDL + Q2_eq_UDL + Q3_eq_UDL) + (Q1_eq_PL + Q2_eq_PL + Q3_eq_PL) + Q_nodal
print(Q_total)

#Solves for the global nodal displacements by inverting the global stiffness matrix and multiplying it by the total equivalent nodal point loading vector.
q = np.linalg.solve(KG, Q_total)
print(f"q = 1 x 10^-3 x \n{q * 1e3}\n")

#Reaction Forces.
#Calculates the global nodal forces for each beam element by multiplying the global stiffness matrix for each beam element by the global nodal displacements vector.
F1 = K1hat @ np.transpose(A1_Matrix) @ q
print(f"F1 = 1 x 10^3 x \n{F1/1e3}\n")

F2 = K2hat @ np.transpose(A2_Matrix) @ q
print(f"F2 = 1 x 10^3 x \n{F2/1e3}\n")

F3 = K3hat @ np.transpose(A3_Matrix) @ q
print(f"F3 = 1 x 10^3 x \n{F3/1e3}\n")