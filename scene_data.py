# -*- coding: utf-8 -*-

#Scene data for standard gprmax simu

import numpy as np
from scipy.constants import c 
#SNR
snr_db = 30
snr = 10**(snr_db/10)

#Dimension de la scène en m²
dim_scene = np.array([5.0,5.0]) 
dim_img = np.array([35,35])
nx,nz = dim_img
dx,dz = dim_scene / dim_img

px = np.array([2.6])
pz = np.array([4.1])


#fréquences du stepped frequency signal
fmin = 1.01973677e+09
fmax = 2.99922581e+09
B = fmax - fmin
M = 100
w = 2*np.pi*np.linspace(fmin,fmax,M)

# Nombre de position du SAR et espacement entre deux positions
dn = 0.02
N = 67

#Positions des émetteurs SAR
posinitx = 1.824 
xn = np.linspace(posinitx,posinitx + dn*N-dn,N)
zn = np.zeros(xn.shape[0])
recs = np.vstack([xn,zn]).T

#Caracteristiques du mur
zoff = 1.2
d = 0.5
eps = 4.5

#Positions en cross range et réfléctivité des murs interieurs left/right
xw_l = 0
xw_r = dim_scene[0]

#nombres de reflections dans le mur pour wall ringing
iw = 1

#Nb of multipaths considered
R_mp = 2