# -*- coding: utf-8 -*-

import time
from Functions.measures import True_Map,Estimated_Map
from scene_data import dim_img,dim_scene,N,nx,nz,px,pz,dx,dz,R_mp
from Functions.helpers import (csgn,vec,unvec,row_thres,svd_thres,blockshaped,unblockshaped,hub,
                               prox_huber,prox_huber_norm,make_img)
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
targets = [(px[i],pz[i]) for i in range(len(px))]
ratio_dims = dim_scene/dim_img

def fobj_r(r,Psi_A,L,Y,c,block_size):
    ''' objective fct for r-step '''
    Psi_I_kron_r = unvec(Psi_A @ r, Y.shape)       
    Es = blockshaped(Y-L-Psi_I_kron_r,*(block_size))
    Es = Es.reshape(Es.shape[0],-1)
    Es_n = jnp.sqrt(jnp.apply_along_axis(jnp.sum,1,jnp.abs(Es)**2))
    Es_hub = hub(Es_n,c)
    sum_es = jnp.sum(Es_hub)
    return sum_es


def hkrpca_fd(Y,Psi,lbd,mu,nu,eta,c,block_size:tuple,eps=1e-3,nits=None,print_conv=False):
    """
    Huber RPCA (with Kronecker structured sparsity) via decoupling & ADMM.

    Parameters
    ----------
    Y : array
        data matrix.
    Psi : array
        dictionary.
    lbd : scalar
        sparsity parameter.
    mu : scalar
        rank parameter.
    nu : scalar
        first equality constraint.
    eta : scalar
        second equality constraint.
    c : scalar
        Huber fct threshold.
    block_size : tuple
        size of a block considered by the Huber fct.
    nits : integer, optional
        number of iteration for the algo.
    r_nits : integer, optional
        number of iteration for r-step GD.

    Returns
    -------
    L : array
        low rank matrix recovered.
    r : array
        sparse vector recovered.
    """
    
    ##Initialize variables
    #Primal variables
    L = jnp.zeros(Y.shape,dtype=Y.dtype)
    R = jnp.zeros((nx*nz,R_mp),dtype=Y.dtype)
    r = vec(R)
    #Secondary variable
    K = jnp.zeros(Y.shape,dtype=Y.dtype) #instead of M in the pdf
    S = jnp.zeros(R.shape,dtype=R.dtype)
    #Dual variable
    U = jnp.zeros(Y.shape,dtype=Y.dtype)
    V = jnp.zeros(R.shape,dtype=R.dtype)

    
    Psi_A = jnp.vstack(jnp.hsplit(Psi,N))     
    
    Psi_I_kron_r = unvec(Psi_A @ r, Y.shape)

    if not nits==None:
        it = 0
        
    res_hist = []
    time_hist = []

    
    cond=False
    
    t0 = time.time()

    while cond==False:

        #L-step
       
        if block_size == (1,1):
            Arg = K + U/nu - Y + Psi_I_kron_r
            L = csgn(Arg)*prox_huber(jnp.abs(Arg),c,mu/(2*nu)) + Y - Psi_I_kron_r
        else:
            Mat1 = blockshaped(K + U/nu ,*(block_size))
            Mat2 = blockshaped(-Y + Psi_I_kron_r ,*(block_size))
            
            L= []
            for i in range(Mat1.shape[0]):
                L.append(prox_huber_norm(Mat1[i,:,:] + Mat2[i,:,:],c,mu/(2*nu)) - Mat2[i,:,:])
            L = jnp.array(L)
            L = unblockshaped(L,*Y.shape)

        
           
        #R-step (MM)
        for itMM in range(1):
            Psi_I_kron_r = unvec(Psi_A @ r, Y.shape)
            
            E = Y-L-Psi_I_kron_r
            E = blockshaped(E,*(block_size))
            E = E.reshape(E.shape[0],np.prod(block_size))
            
            nE = jnp.apply_along_axis(jnp.linalg.norm,1, E)
            sqrtE = jnp.sqrt(c/nE)
            
            W = jnp.where(nE<c,1,sqrtE) 
            W = jnp.outer(W,jnp.ones(np.prod(block_size)))
            W = W.reshape(W.shape[0],*(block_size))
            W = unblockshaped(W,*Y.shape)
    
            Psi_AW = (jnp.outer(vec(W),jnp.ones(Psi_A.shape[1]))) * Psi_A
            YW = Y * W
            LW = L * W
            
            rMat = (mu/2)*Psi_AW.conj().T @ Psi_AW + eta * jnp.identity(Psi_AW.shape[1])
            rVec = (mu/2)*Psi_AW.conj().T @ vec(YW-LW) + vec(eta*S+V)
            
            r = jnp.linalg.solve(rMat,rVec)
            
        R = unvec(r,R.shape) 
        
        #Secondary variables
        Kprev = K
        K = svd_thres(L - U/nu, 1/nu)
        
        Sprev = S
        S = row_thres(R - V/eta, lbd/eta)
                
        #Dual variables
        U = U + nu*(K-L)
        V = V + eta*(S-R)
        
        #Residual balancing
        prim_resL = jnp.linalg.norm(K-L)
        dual_resL = nu*jnp.linalg.norm(Kprev - K)
        
        if prim_resL > 10*dual_resL:
            nu = 2*nu
        elif dual_resL > 10*prim_resL:
            nu = nu/2
        
        prim_resR = jnp.linalg.norm(S-R)
        dual_resR = eta*jnp.linalg.norm(Sprev - S)
        
        if prim_resR > 10*dual_resR:
            eta = 2*eta
        elif dual_resR > 10*prim_resR:
            eta = eta/2
            
        Psi_I_kron_r = unvec(Psi_A @ r, Y.shape)

        #Convergence check
        res = (fobj_r(r,Psi_A,L,Y,c,block_size) + jnp.linalg.norm(S-R) +
               jnp.linalg.norm(K-L))/jnp.linalg.norm(Y)
               
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
        return L,K,R,S,res_hist,time_hist 


    plt.plot(res_hist)
    plt.show()
    
    return L,K,R,S

    
def plot_hkrpca_fd(r,c,lbd,mu,nu,eta,plt_rect=False):
    #Plot        
    fig, ax = plt.subplots()
    im = ax.imshow(np.abs(r),aspect='auto',
               extent = [0,dim_scene[0],0,dim_scene[1]])
    
    if plt_rect == True :
        for i in range(len(targets)):
            rect = Rectangle(targets[i] - 2/2 * np.array([dx,dz]), 2*dx, 2*dz,
                             linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    
    ax.set_xticks(np.arange(0,dim_scene[0],0.5))
    ax.set_yticks(np.arange(0,dim_scene[1],0.5))
    ax.set_title(r"$c$={}, $\lambda$={}, $\mu$={}, $\nu$={},$\eta$={}".format(c,lbd,mu,nu,eta))
    fig.colorbar(im)
    plt.show()
    
