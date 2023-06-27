#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 19:19:09 2022

@author: hugobrehier
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

from scene_data import (M,N,px,pz)
from Functions.helpers import gen_dict,make_img,vec,unvec,add_noise
from Functions.hkrpca_fd import hkrpca_fd,plot_hkrpca_fd

import jax.numpy as jnp
import numpy as np

PSI = gen_dict("PSI_gpr.npy")
PSI = jnp.array(PSI,dtype=jnp.complex64)

#%conda info
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

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

#%% Algo
#params

bs = (1,1)

c  =  0.1
lbd = 1
mu  = 10
nu  = 1
eta = 1e10

L,K,R,S = hkrpca_fd(y_mp,PSI,lbd,mu,nu,eta,c,bs,nits=20)

r=make_img(np.array(R))
plot_hkrpca_fd(r, c, lbd, mu, nu, eta,plt_rect=False)
s=make_img(np.array(S))
plot_hkrpca_fd(s, c, lbd, mu, nu, eta,plt_rect=False)


#%% Bayesian hypeparams tuning ?

from Functions.measures import True_Map,Estimated_Map
from scene_data import dim_img,dx,dz
from bayes_opt import BayesianOptimization,UtilityFunction
from sklearn.metrics import f1_score

TrueMap = True_Map(px, pz, dx, dz, dim_img,br=(3,3))

def black_box_function(lbd,mu,nu,eta,c):
    L,K,R,S = hkrpca_fd(y_mp,PSI,lbd,mu,nu,eta,c,bs,nits=20)
    r=make_img(np.array(R))
    EstimMap_R = Estimated_Map(np.array(r),thr=0.75,br=(3,3))
    return f1_score(vec(TrueMap), vec(EstimMap_R))
                        
    # s=make_img(np.array(S))
    # EstimMap_S = Estimated_Map(np.array(s),thr=0.25,br=(3,3))
    # return 0.5*f1_score(vec(TrueMap), vec(EstimMap_R)) + 0.5*f1_score(vec(TrueMap), vec(EstimMap_S)) 


# Bounded region of parameter space
pbounds = {'lbd': (0, 10), 'mu': (0, 10), 'nu': (0,10), 'eta': (0,1e10), 'c' : (0, 1)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=42,
    verbose=2)

# optimizer.probe(
#     params={"lbd": 1,'mu': 10, 'nu': 1, 'eta': 1e10, 'c' : 0.1},
#     lazy=True)

optimizer.maximize(
    acquisition_function=UtilityFunction(kind='ei', kappa=3),
    init_points=5,
    n_iter=20)

print("Best result {} with f1_score= {}".format(optimizer.max["params"],optimizer.max["target"]))


c,eta,lbd,mu,nu = optimizer.max["params"].values()
L,K,R,S = hkrpca_fd(y_mp,PSI,lbd,mu,nu,eta,c,bs,nits=40)

r=make_img(np.array(R))
plot_hkrpca_fd(r, c, lbd, mu, nu, eta,plt_rect=False)
s=make_img(np.array(S))
plot_hkrpca_fd(s, c, lbd, mu, nu, eta,plt_rect=False)
