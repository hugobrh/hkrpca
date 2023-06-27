#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 18:53:46 2022

@author: hugobrehier
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

from scene_data import M,N,px,pz,dx,dz,dim_img
from Functions.helpers import gen_dict,make_img,add_noise,vec
from Functions.hkrpca_sd import hkrpca_sd,inds,build_psi_grad,plot_hkrpca_sd

import jax.numpy as jnp
import numpy as np

PSI = gen_dict("PSI_gpr.npy")
PSI = jnp.array(PSI,dtype=jnp.complex64)

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
y_mp = y_mp/np.max(np.abs(y_mp))

# #Profiling
# load_ext line_profiler
# lprun -f hkrpca_sd hkrpca_sd(y_mp,PSI,lbd,mu,nu,c,bs,t_r=1e-5,nits=40,psis_gr_mat=psis_gr_mat)


#%% blocksize preprocess

bs = (1,1)

indices = inds(bs,y_mp.shape)
psis_gr = build_psi_grad(np.hsplit(np.array(PSI),N),indices)
psis_gr_mat = jnp.array(np.hstack(psis_gr),dtype=jnp.complex64)
del(indices,psis_gr)

#%% Algo with ROC metrics

c  =  0.1
lbd = 1
mu  = 10
nu  = 1

L,K,R = hkrpca_sd(y_mp,PSI,lbd,mu,nu,c,bs,t_r=1e-5,nits=20,psis_gr_mat=psis_gr_mat)
r=make_img(np.array(R))
plot_hkrpca_sd(r,c,lbd,mu,nu,plt_rect=False)

#%% ROC
# from matplotlib import pyplot as plt
# from skimage.measure import block_reduce
# from Functions.measures import True_Map,Estimated_Map
# from sklearn.metrics import roc_curve
# TrueMap = True_Map(px, pz, dx, dz, dim_img,br=(3,3))
# EstimMap = block_reduce(r,(3,3),np.mean)
# fpr, tpr, thresholds = roc_curve(vec(TrueMap), vec(EstimMap))
# plt.plot(fpr,tpr,linestyle='--',marker='o')
