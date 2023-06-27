# -*- coding: utf-8 -*-

import time

from Functions.measures import True_Map,Estimated_Map
from Functions.helpers import (csgn,vec,unvec,row_thres,svd_thres,blockshaped,unblockshaped,hub,dhub,
                               prox_huber,prox_huber_norm,make_img)
from scene_data import dim_img,dim_scene,nx,nz,px,pz,dx,dz,R_mp,N

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
targets = [(px[i],pz[i]) for i in range(len(px))]
ratio_dims = dim_scene/dim_img

def inds(block_size:tuple,mat_size:tuple):
    '''Construct indices list for each block'''
    
    #nb of blocks verically and horizontally
    nv  = mat_size[0] // block_size[0]
    nh  = mat_size[1] // block_size[1]
    
    #vertical and horizontal indices of upper left pts
    nvs = [i*block_size[0] for i in np.arange(nv)]    
    nhs = [i*block_size[1] for i in np.arange(nh)]
    
    #list of upper left indices
    upperleftindices = []
    for nv in nvs:
        for nh in nhs:
            upperleftindices.append([nv,nh])
    
    #create list of indices for each block
    indices = []
    for upleftind in upperleftindices:
        blockinds = []
        for j in range(block_size[0]):
           for k in range(block_size[1]):
               pt = tuple(np.array(upleftind) + np.array([j,k]))
               blockinds.append(pt)
        indices.append(blockinds)
        
    return indices

def build_psi_grad(Psis,inds):
    dictmat = []
    for i in range(len(inds)):
        mat = []
        for ind in inds[i]:
            j,k = ind
            mat.append(Psis[k][j,:])
        mat = np.vstack(mat)
        dictmat.append(mat.conj().T)
    return dictmat

def fobj_r(r,Psi_A,L,Y,c,block_size):
    ''' objective fct for r-step '''
    Psi_I_kron_r = unvec(Psi_A @ r, Y.shape)       
    Es = blockshaped(Y-L-Psi_I_kron_r,*(block_size))
    Es = Es.reshape(Es.shape[0],-1)
    Es_n = jnp.sqrt(jnp.apply_along_axis(jnp.sum,1,jnp.abs(Es)**2))
    Es_hub = hub(Es_n,c)
    sum_es = jnp.sum(Es_hub)
    return sum_es


def grad_fobj_r(r,Psi_A,psis_gr_mat,L,Y,c,block_size):
    ''' gradient (ie 2df/dr*) of obj fct wrt r'''
    Psi_I_kron_r = unvec(Psi_A @ r, Y.shape)       
    Es = blockshaped(Y-L-Psi_I_kron_r,*(block_size))
    Es = Es.reshape(Es.shape[0],-1)
    Es_n = jnp.sqrt(jnp.apply_along_axis(jnp.sum,1,jnp.abs(Es)**2))
    Es_dhub = dhub(Es_n,c)
    Es_gr= jnp.kron(Es_dhub.ravel()/Es_n.ravel(),np.ones(Es.shape[1])) * Es.ravel()  
    grad = -psis_gr_mat @ Es_gr
    return grad


# from jax.scipy.linalg import block_diag
# def grad_fobj_r2(r,Psi_A,psis_gr_mat,L,Y,c,block_size):
#     ''' gradient (ie 2df/dr*) of obj fct wrt r'''
#     Psi_I_kron_r = unvec(Psi_A @ r, Y.shape)       
#     Es = blockshaped(Y-L-Psi_I_kron_r,*(block_size))
#     Es = Es.reshape(Es.shape[0],-1)
    
#     Es_n = jnp.sqrt(jnp.apply_along_axis(jnp.sum,1,jnp.abs(Es)**2))
#     Es_dhub = dhub(Es_n,c)
#     Es_tilde = Es_dhub.ravel()/Es_n.ravel()
    
#     nblocks = int(np.prod(Y.shape)/np.prod(block_size))
#     Es = np.split(Es,nblocks,axis=0)
#     Es_bdiag = block_diag(*Es).T
    
#     if block_size == (1,1):
#         Es_gr = jnp.diag(Es_bdiag) * Es_tilde
#     else:
#         Es_gr = Es_bdiag @ Es_tilde
        
#     grad = -psis_gr_mat @ Es_gr
    
#     return grad

def btls(f,x,grad_fx,a = 1,c1 = 1e-4,t = 0.1,args=()):
    '''back-tracking line-search (armijo condition) '''
    while f(x - a*grad_fx,*args) > f(x,*args) - a*c1*jnp.linalg.norm(grad_fx)**2:
        a = a * t
    return a

def hkrpca_sd(Y,Psi,lbd,mu,nu,c,block_size:tuple,t_r=None,eps=1e-3,nits=None,psis_gr_mat=None,print_conv=False):
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
    K = jnp.zeros(Y.shape,dtype=Y.dtype) #K instead of M in the pdf
    #Dual variable
    U = jnp.zeros(Y.shape,dtype=Y.dtype)
    
    Psi_A = jnp.vstack(jnp.hsplit(Psi,N))

    #step size 
    a=1
    
    #sub-dictionaries    
    if psis_gr_mat is None:
        indices = inds(block_size,Y.shape)
        psis_gr = build_psi_grad(jnp.hsplit(Psi,N),indices)
        psis_gr_mat = jnp.hstack(psis_gr)
        del(indices,psis_gr)
    
    if t_r is None:
        do_btls = True
    else:
        do_btls = False

    if not nits==None:
        it = 0
    


    res_hist = []
    time_hist = []
    
    Psi_I_kron_r = unvec(Psi_A @ r, Y.shape)
    # assert np.allclose(Psi_I_kron_r,Psi @ jnp.kron(jnp.identity(N),r).T) , 'sparse matrice not good'


    cond=False
    t0 = time.time()

    while cond==False:
        #Primal steps
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
            
            # L2 = jnp.apply_along_axis(prox_huber_norm, 0, Mat1+Mat2, c=c,a=1/nu) - Mat2
            # L2 = unblockshaped(L2,*Y.shape)
            # assert np.allclose(L, L2) ,'Ls are different' #this is false

        #R-step  
        #APGD with linesearch        
        k = 0
        r_prev = r.copy()
        for itPGD in range(1):
            w= k/(k+3)
            s = r + w*(r - r_prev)
            r_prev = r.copy()
            
            grad_r = grad_fobj_r(r,Psi_A,psis_gr_mat,L,Y,c,block_size)
            # grad_r2 = grad_fobj_r2(r,Psi_A,psis_gr_mat,L,Y,c,block_size)
            # assert np.allclose(grad_r, grad_r2) , 'grads are different' #this is true
            
            if do_btls:  
                t_r = btls(fobj_r,r,grad_r,a=a,args=(Psi_A,L,Y,c,block_size))
                print(f't_r:{t_r}')
            R = row_thres(unvec(s - t_r*(mu/2)*grad_r,R.shape),t_r*lbd)
            r = vec(R)
            
            k += 1
        print(f'r-loops: {k}')
        
        #Secondary variable
        Kprev = K
        K = svd_thres(L - U/nu, 1/nu)
                
        #Dual variable
        U = U + nu*(K-L)
        
        #Residual balancing
        prim_res = jnp.linalg.norm(K-L)
        dual_res = nu*jnp.linalg.norm(Kprev -K)
        if prim_res > 10*dual_res:
            nu = 2*nu
        elif dual_res > 10*prim_res:
            nu = nu/2
            
        Psi_I_kron_r = unvec(Psi_A @ r, Y.shape)
        
           
        res = (fobj_r(r,Psi_A,L,Y,c,block_size)+
               jnp.linalg.norm(K-L))/jnp.linalg.norm(Y)
                
        print(res, " || " , eps)
        
        #MSE compare
        
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
        return L,K,R,res_hist,time_hist 
    
    # plt.plot(res_hist)
    # plt.show()
    
    
    return L,K,R

   
def plot_hkrpca_sd(r,c,lbd,mu,nu,plt_rect=True):
    #Plot        
    fig, ax = plt.subplots()
    im = ax.imshow(np.abs(r),aspect='auto',
               extent = [0,dim_scene[0],0,dim_scene[1]])
    
    if plt_rect == True:
        for i in range(len(targets)):
            rect = Rectangle(targets[i] - 2/2 * np.array([dx,dz]), 2*dx, 2*dz,
                             linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
    
    ax.set_xticks(np.arange(0,dim_scene[0],0.5))
    ax.set_yticks(np.arange(0,dim_scene[1],0.5))
    ax.set_title(r"$c={},\lambda$={}, $\mu$={}, $\nu$={}".format(c,lbd,mu,nu))
    fig.colorbar(im)
    plt.show()
    
