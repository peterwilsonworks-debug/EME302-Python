# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 15:56:13 2026

@author: pwi72
"""

import numpy as np
import matplotlib as plt

E = 200e9
E3 = 70e9

Area = 2e-4
Area3 = 400e-6

L1 = 5
L2 = 2.5
L3 = 2
pi = np.pi
alpha1 = 0
alpha2 = 0
alpha3 = (-90*pi)/180

I = 640e-6
I3 = 1e-10

A1_Matrix = np.zeros((6, 6))
A1_Matrix[0, 3] = 1
A1_Matrix[1, 4] = 1
A1_Matrix[2, 5] = 1
print(A1_Matrix)

A2_Matrix = np.identity(6)
print(A2_Matrix)

A3_Matrix = np.zeros((6, 6))
A3_Matrix[0, 0] = 1
A3_Matrix[1, 1] = 1
A3_Matrix[2, 2] = 1
print(A3_Matrix)

Q1 = 0
Q2 = 0
Q3 = 0
Q4 = 0
Q5 = -150000
Q6 = 0

Q = np.array([[Q1], [Q2], [Q3], [Q4], [Q5], [Q6]])
print(Q)




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
    Lamda_Matrix = np.array([[ c, s, 0], [-s, c, 0], [ 0, 0, 1]])
    Transform_Matrix = np.zeros((6, 6))
    Transform_Matrix[:3, :3] = Lamda_Matrix
    Transform_Matrix[3:, 3:] = Lamda_Matrix
    print(Transform_Matrix)
    K_e_hat = np.transpose(Transform_Matrix) @ K_e @ Transform_Matrix
    return K_e_hat
    
K1 = Local_BAR(E, Area, L1, I)
print(f"K1 = 1 x 10^7 x \n{K1/1e7}\n")


K2 = Local_BAR(E, Area, L2, I)
print(f"K2 = 1 x 10^7 x \n{K2/1e7}\n")

K3 = Local_BAR(E3, Area3, L3, I3)
print(f"K3 = 1 x 10^7 x \n{K3/1e7}\n")


K1hat = global_bar(K1, alpha1)
print(f"K1hat = 1 x 10^8 x \n{K1hat/1e8}\n")
    
K2hat = global_bar(K2, alpha2)
print(f"K2hat = 1 x 10^8 x \n{K2hat/1e8}\n")

K3hat = global_bar(K3, alpha3)
print(f"K3hat = 1 x 10^7 x \n{K3hat/1e7}\n")

KG1 = A1_Matrix @ K1hat @ np.transpose(A1_Matrix)
print(f"KG1 = 1 x 10^7 x \n{KG1/1e7}\n")

KG2 = A2_Matrix @ K2hat @ np.transpose(A2_Matrix)
print(f"KG2 = 1 x 10^8 x \n{KG2/1e8}\n")

KG3 = A3_Matrix @ K3hat @ np.transpose(A3_Matrix)
print(f"KG3 = 1 x 10^8 x \n{KG3/1e8}\n")

KG = KG1 + KG2 + KG3
print(f"KG = 1 x 10^8 x \n{KG/1e8}\n")

q = np.linalg.solve(KG, Q)
print(f"q = 1 x 10^-3 x \n{q * 1e3}\n")

F1 = K1hat @ np.transpose(A1_Matrix) @ q
print(f"F1 = 1 x 10^5 x \n{F1/1e5}\n")

F2 = K2hat @ np.transpose(A2_Matrix) @ q
print(f"F2 = 1 x 10^5 x \n{F2/1e5}\n")




