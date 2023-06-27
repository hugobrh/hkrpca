#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 19:34:11 2022

@author: hugobrehier
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

import jax.numpy as jnp
import numpy as np

from Functions.srcs import srcs,plot_srcs
from Functions.helpers import gen_dict,make_img,vec,add_noise
from Functions.wall_mitigation import wall_mitigation_simple
from scene_data import N,px,pz


PSI = gen_dict("PSI_gpr.npy")
Psi_A = jnp.vstack(jnp.hsplit(PSI,N))
Psi_A = jnp.array(Psi_A,dtype=jnp.complex64)

#Init stepsize
#eigvals = jnp.linalg.eigvalsh(PhiPsi.conj().T @ PhiPsi)
eigvals = jnp.linalg.eigvalsh(Psi_A.conj().T @ Psi_A)
Lip_cst = jnp.max(jnp.abs(eigvals))

#%%

t_df = 2.1
snr_db = 12
snr = 10**(snr_db/10)

noise_type = 'pt'
noise_dist = 'student'
nb_outs = 0

np.random.seed(4)
y_mp = np.load('TTW_Bscan_no_interior_walls_ricker_merged_3mm_curated_resampled.npy')
y_mp = add_noise(y_mp, snr, t_df,noise_type=noise_type ,noise_dist=noise_dist)
y_mp = jnp.array(y_mp,dtype=jnp.complex64)

#Wall filtering
y_bar = wall_mitigation_simple(y_mp)
yvec = vec(y_bar)
yvec = yvec/np.max(np.abs(yvec))

#%% Algo

l=1
rvec = srcs(yvec,Psi_A,l=l,t=1/Lip_cst,nits=20)
r=make_img(np.array(rvec))
plot_srcs(r, l)
