# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 15:56:13 2026

@author: pwi72
"""

import numpy as np
import matplotlib as plt

E1 = 200e9
E2 = 200e9

Area1 = 400e-6
Area2 = 600e-6

L1 = 1.1
L2 = 0.8

alpha1 = 0
alpha2 = (55 * np.pi) / 180

A1_Matrix = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
print(A1_Matrix)
A2_Matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
print(A2_Matrix)

Q1 = 0
Q2 = -20000

Q = np.array([[Q1], [Q2]])
print(Q)

def Local_BAR(E, A, L):
    """returns the local bar stiffness matrix """
    K_e = ((E * A) / L) * np.array([[1, -1], [-1, 1]])
    return K_e
    

def global_bar(K_e, alpha):
    """returns the local stiffness matrix in the global reference frame"""
    c = np.cos(alpha)
    s = np.sin(alpha)
    Lamda_Matrix = np.array([[c, s, 0, 0], [0, 0, c, s]])
    K_e_hat = np.transpose(Lamda_Matrix) @ K_e @ Lamda_Matrix
    return K_e_hat
    
K1 = Local_BAR(E1, Area1, L1)
print(f"K1 = 1 x 10^7 x \n{K1/1e7}\n")


K2 = Local_BAR(E2, Area2, L2)
print(f"K2 = 1 x 10^7 x \n{K2/1e7}\n")


K1hat = global_bar(K1, alpha1)
print(f"K1hat = 1 x 10^7 x \n{K1hat/1e7}\n")
    
K2hat = global_bar(K2, alpha2)
print(f"K2hat = 1 x 10^8 x \n{K2hat/1e8}\n")

KG1 = A1_Matrix @ K1hat @ np.transpose(A1_Matrix)
print(f"KG1 = 1 x 10^7 x \n{KG1/1e7}\n")

KG2 = A2_Matrix @ K2hat @ np.transpose(A2_Matrix)
print(f"KG2 = 1 x 10^8 x \n{KG2/1e8}\n")

KG = KG1 + KG2
print(f"KG = 1 x 10^8 x \n{KG/1e8}\n")

q = np.linalg.solve(KG, Q)
print(f"q = 1 x 10^-3 x \n{q * 1e3}\n")

F1 = K1hat @ np.transpose(A1_Matrix) @ q
print(f"F1 = 1 x 10^4 x \n{F1/1e4}\n")

F2 = K2hat @ np.transpose(A2_Matrix) @ q
print(f"F2 = 1 x 10^4 x \n{F2/1e4}\n")



