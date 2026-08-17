"""
Created on Thu Jul 23 15:56:13 2026

@author: pwi72

LINEARLY VARYING LOAD VERSION.
In addition to the UDL and Point Load on each beam, each beam can also carry a
linearly varying (trapezoidal) transverse load, defined by its intensity at
local node 1 (w1_beam) and its intensity at local node 2 (w2_beam), in N/m.
Setting w1_beam = w2_beam reproduces a plain UDL (use the dedicated UDL
variables for that instead, or set both here equal to the same value).
Setting either one to 0 gives a pure triangular load rising/falling from 0.
Downwards loading is negative, matching the sign convention used for UDL/Point_Load.
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

#LINEARLY VARYING LOAD ON EACH BEAM (trapezoidal load, intensity w1 at local node 1
#ramping linearly to intensity w2 at local node 2, in N/m). Downwards is negative.
#Set both to 0 if a beam has no linearly varying load.

w1_beam1 = 0
w2_beam1 = 0

w1_beam2 = 0
w2_beam2 = 0

w1_beam3 = 0
w2_beam3 = -8000     #e.g. load ramping from 0 at node 1 to -8000 N/m at node 2

#Assembly matrices for each beam element
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

#Nodal point loads in the global reference frame, make sure to count degrees of freedom.

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

def force_eq_local_LinearLoad(w1, w2, L):
    """calculates and returns the equivalent nodal point loading for a LINEARLY VARYING
    (trapezoidal) load, of intensity w1 at local node 1 ramping to intensity w2 at local
    node 2. Reduces to the plain UDL formula when w1 = w2."""
    f_eq = np.array([[0                                ],
                     [(L * (7*w1 + 3*w2)) / 20          ], 
                     [(L**2 * (w1/20 + w2/30))          ], 
                     [0                                ], 
                     [(L * (3*w1 + 7*w2)) / 20          ], 
                     [-(L**2 * (w1/30 + w2/20))         ]])
    return f_eq

#Calculates the equivalent nodal point loading for a Linearly Varying Load on each beam in the LOCAL reference frame.
f1_eq_Lin_local = force_eq_local_LinearLoad(w1_beam1, w2_beam1, L1)
f2_eq_Lin_local = force_eq_local_LinearLoad(w1_beam2, w2_beam2, L2)
f3_eq_Lin_local = force_eq_local_LinearLoad(w1_beam3, w2_beam3, L3)

#Calculates the equivalent nodal point loading for a Point Load on each beam in the LOCAL reference frame.
f1_eq_PL_local = force_eq_local_PL(Point_Load1, L1, a1)
f2_eq_PL_local = force_eq_local_PL(Point_Load2, L2, a2)
f3_eq_PL_local = force_eq_local_PL(Point_Load3, L3, a3)

#Calculates the equivalent nodal point loading for a Linearly Varying Load on each beam in the GLOBAL reference frame.
F1_eq_Lin_global = np.transpose(transformation_matrix(alpha1)) @ f1_eq_Lin_local
F2_eq_Lin_global = np.transpose(transformation_matrix(alpha2)) @ f2_eq_Lin_local
F3_eq_Lin_global = np.transpose(transformation_matrix(alpha3)) @ f3_eq_Lin_local

# Calculates the equivalent nodal point loading for a Linearly Varying Load on each beam in the GLOBAL reference frame, 
# using the assembly matrix for each beam to extract where the equivalent nodal point loading is located in the global equivalent nodal point loading vector.
Q1_eq_Lin = A1_Matrix @ F1_eq_Lin_global
Q2_eq_Lin = A2_Matrix @ F2_eq_Lin_global
Q3_eq_Lin = A3_Matrix @ F3_eq_Lin_global

print(f"{Q1_eq_Lin}\n")
print(f"{Q2_eq_Lin}\n")
print(f"{Q3_eq_Lin}\n")

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

#calculates the total equivalent nodal point loading for the entire structure, by summing the equivalent nodal point loadings for the UDLs, Point Loads, Linearly Varying Loads and Nodal Point Loads on all three beams.
Q_total = (Q1_eq_UDL + Q2_eq_UDL + Q3_eq_UDL) + (Q1_eq_PL + Q2_eq_PL + Q3_eq_PL) + (Q1_eq_Lin + Q2_eq_Lin + Q3_eq_Lin) + Q_nodal
print(Q_total)

#Solves for the global nodal displacements by inverting the global stiffness matrix and multiplying it by the total equivalent nodal point loading vector.
q = np.linalg.solve(KG, Q_total)
print(f"q = 1 x 10^-3 x \n{q * 1e3}\n")

#Calculates the global nodal forces for each beam element by multiplying the global stiffness matrix for each beam element by the global nodal displacements vector.
F1 = K1hat @ np.transpose(A1_Matrix) @ q
print(f"F1 = 1 x 10^3 x \n{F1/1e3}\n")

F2 = K2hat @ np.transpose(A2_Matrix) @ q
print(f"F2 = 1 x 10^3 x \n{F2/1e3}\n")

F3 = K3hat @ np.transpose(A3_Matrix) @ q
print(f"F3 = 1 x 10^3 x \n{F3/1e3}\n")

#Reaction forces at the supports, in the global reference frame.
#Beam 1's local node 1 (top of the structure) and Beam 3's local node 2 (bottom of
#the structure) are the two fixed supports, so the reactions are simply the
#corresponding end forces already calculated above for F1 and F3.

Reaction1 = F1[0:3]   #Reaction at support 1 (Fx, Fy, M), taken from beam 1's local node 1 end forces
print(f"Reaction1 (Fx, Fy, M) = \n{Reaction1}\n")

Reaction2 = F3[3:6]   #Reaction at support 2 (Fx, Fy, M), taken from beam 3's local node 2 end forces
print(f"Reaction2 (Fx, Fy, M) = \n{Reaction2}\n")

#Check: sum of reactions plus applied nodal loads should balance the sum of all
#applied UDL/point/linearly varying loads on the structure (equilibrium check).
#NOTE: this check resolves each beam's local transverse load into the GLOBAL
#frame using its own alpha, since a beam's "downward local-y" load only lines
#up with global vertical for a HORIZONTAL beam (alpha = 0). For beams angled
#at +-90 degrees (like beam 1 and beam 3 here), the same local load instead
#acts in the global X direction, so it correctly does NOT appear in Total_Reaction_Fy.
def resultant_local_to_global(udl, pl, w1, w2, L, alpha):
    fy_local_total = udl*L + pl + (w1+w2)*L/2
    c, s = np.cos(alpha), np.sin(alpha)
    fx_global = s * fy_local_total
    fy_global = c * fy_local_total
    return fx_global, fy_global

fx1, fy1 = resultant_local_to_global(UDL1, Point_Load1, w1_beam1, w2_beam1, L1, alpha1)
fx2, fy2 = resultant_local_to_global(UDL2, Point_Load2, w1_beam2, w2_beam2, L2, alpha2)
fx3, fy3 = resultant_local_to_global(UDL3, Point_Load3, w1_beam3, w2_beam3, L3, alpha3)

Total_Reaction_Fy = Reaction1[1] + Reaction2[1]
Applied_Fy = fy1 + fy2 + fy3
print(f"Total vertical reaction = {Total_Reaction_Fy[0]:.2f} N, Total applied vertical load (resolved to global) = {Applied_Fy:.2f} N\n")