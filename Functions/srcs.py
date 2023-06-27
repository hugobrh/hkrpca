#%%
# -*- coding: utf-8 -*-
import time
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scene_data import px,pz,dx,dz,dim_img,dim_scene,nx,nz
from Functions.helpers import vec,unvec,row_thres

ratio_dims = dim_scene/dim_img
targets = [(px[i],pz[i]) for i in range(len(px))]


def srcs(y,Psi_A,l,t,eps=1e-3,nits=None,AhA=np.zeros(1),print_conv = False):
    '''proximal gradient descent (with acceleration via FISTA) 
    for the regularized regression problem:
    0.5*||y-Ax||^2 +  l*||x||_2,1
    thus the update rule is (until convergence):
    x_{k+1} = prox_{h,lt}(x_{k} - t * grad{g}(x_k)) 
    where:
    g(x) = 0.5*||y-Ax||^2 , h(x) = l*||x||_2,1
    t = stepsize, positive scalar (possibly set to 1/L,
    with L the spectral radius of the hessian of g which is A^H A) 
    grad(g)(x) = gradient of g at x : -A^H(y-Ax)
    prox_h,lt(x) = proximal of h at x with threshold parameter l*t
    Note that the proximal of h is row-wise thresholding.
    '''
    
    #Keep memory of x_k-1 and x_k-2 for acceleration
    
    x = jnp.zeros(Psi_A.shape[1],dtype=np.complex64)
    x_prev = x.copy()
    k=1
    
    if not nits==None:
        it = 0
     
    Psi_Ah = Psi_A.conj().T
    
    if AhA.all() == 0 : 
        Psi_A_normal = Psi_Ah @ Psi_A
        
    res_hist = []
    time_hist = []
    
    cond = False
    
    t0 = time.time()

    while cond == False or k<=0:
        #Compute momentum value
        w= k/(k+3)
        v = x + w*(x - x_prev)
        x_prev = x.copy()
        
        #Compute proximal
        x = vec(row_thres(unvec(v + t* (Psi_Ah @ y - Psi_A_normal @ v),(nx*nz,-1)),l*t))

        k+=1
        res = jnp.linalg.norm(y-Psi_A @ x) / jnp.linalg.norm(y)
        
        print(res, " || " , eps)
        
        if not nits==None:
            cond = it >= nits
            it += 1
        else:
            cond = res <= eps
         
        res_hist.append(res)
        time_hist.append(time.time())
        
        
    if print_conv:
        res_hist = np.array(res_hist) - res_hist[-1]  
        time_hist = np.array(time_hist) - t0
        
        return x,res_hist,time_hist
    
    return x

def plot_srcs(r,lbd,plt_rect=False):
    #Plot        
    fig, ax = plt.subplots()
    im = ax.imshow(np.abs(r)/np.max(np.abs(r)),aspect='auto',
               extent = [0,dim_scene[0],0,dim_scene[1]])
    
    if plt_rect:
        for i in range(len(targets)):
            rect = Rectangle(targets[i] - 2/2 * np.array([dx,dz]), 2*dx, 2*dz,
                              linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    
    ax.set_xticks(np.arange(0,dim_scene[0],0.5))
    ax.set_yticks(np.arange(0,dim_scene[1],0.5))
    ax.set_title(r"$\lambda$={}".format(lbd))
    fig.colorbar(im)
    plt.show()
