#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 18:06:35 2022

@author: hugobrehier
"""

import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"

from scene_data import N,px,pz

from Functions.helpers import gen_dict,make_img,add_noise,vec
from Functions.krpca import krpca,plot_krpca

import jax.numpy as jnp
import numpy as np

PSI = gen_dict("PSI_gpr.npy")
PSI = jnp.array(PSI,dtype=jnp.complex64)
Ps = jnp.vstack(jnp.hsplit(PSI,N))
Ps = Ps.conj().T @ Ps
Ps = jnp.array(Ps,dtype=jnp.complex64)
eigsP = jnp.linalg.eigvalsh(Ps)


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
# %load_ext line_profiler
# %lprun -f krpca krpca(Y=y_mp,Psi=PSI,lbd=lbd,mu=mu,Ps=Ps,t=t,nits=20)

#%% Test algo and metrics for ROC

lbd = 1
mu = 10
t = 1 / (mu*jnp.max(jnp.abs(eigsP)))

L,r_vec = krpca(Y=y_mp,Psi=PSI,lbd=lbd,mu=mu,Ps=Ps,t=t,nits=20)
r=make_img(np.array(r_vec))
plot_krpca(r,lbd,mu)
#back_projection(np.array(L), M, N, B, dim_scene, dx, dz, xn, nx, nz);

#%% Bayesian hypeparams tuning ?

from Functions.measures import True_Map,Estimated_Map
from scene_data import dim_img,dx,dz
from bayes_opt import BayesianOptimization, UtilityFunction
from sklearn.metrics import f1_score

TrueMap = True_Map(px, pz, dx, dz, dim_img,br=(3,3))

def black_box_function(lbd,mu):
    t = 1 / (mu*jnp.max(jnp.abs(eigsP)))
    L,r_vec = krpca(Y=y_mp,Psi=PSI,lbd=lbd,mu=mu,Ps=Ps,t=t,nits=20)
    r=make_img(np.array(r_vec))
    EstimMap = Estimated_Map(np.array(r),thr=0.75,br=(3,3))
    return f1_score(vec(TrueMap), vec(EstimMap))


# Bounded region of parameter space
pbounds = {'lbd': (0, 25), 'mu': (0, 25)}

optimizer = BayesianOptimization(
    f=black_box_function,
    pbounds=pbounds,
    random_state=42,
    verbose=2)


optimizer.maximize(
    acquisition_function= UtilityFunction(kind="ucb", kappa=10),
    init_points=3,
    n_iter=20)

print("Best result {} with f1_score= {}".format(optimizer.max["params"],optimizer.max["target"]))

lbd,mu = optimizer.max["params"].values()

L,r_vec = krpca(Y=y_mp,Psi=PSI,lbd=lbd,mu=mu,Ps=Ps,t=t,nits=40)
r=make_img(np.array(r_vec))
plot_krpca(r,lbd,mu)
