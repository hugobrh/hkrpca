#%%
import time
from Functions.helpers import vec,unvec,row_thres,svd_thres
from scene_data import nx,nz,R_mp,N,dim_img,dim_scene,px,pz,dx,dz

import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
targets = [(px[i],pz[i]) for i in range(len(px))]
ratio_dims = dim_scene/dim_img


def krpca(Y,Psi,lbd,mu,Ps=None,t=None,eps=1e-3,nits=None,print_conv = False):
    '''
    Decompose Y in : Y = L + Psi(I kron r)
    
    where L is low-rank and r sparse, Psi a given dictionnary.
    
    Parameters
    ----------
    Y : 2D array.
        Matrix to decompose.
    Psi : 2D array.
        Dictionnary.
    mu : positive scalar.
        tuning parameter for reconstruction error (l2 norm).
    lbd : positive scalar.
        tuning parameter for proximal term (l1 norm).
    eps : small positive scalar.
        tolerance for stopping criterion.
    nits : integer. Supplants eps if not None.
        number of iterations.    
    
    Returns
    -------
    L : 2D array.
        Recovered low-rank matrix.
    S : 2D array s.t. S= I kron r .
        Recovered sparse matrix.

    '''
     
    L= jnp.zeros(Y.shape,dtype=Y.dtype)
    r = jnp.zeros(nx*nz*R_mp,dtype=Y.dtype)
    U = jnp.zeros(Y.shape,dtype=Y.dtype)

    Psi_A = jnp.vstack(jnp.hsplit(Psi,N))
    
    if Ps is None:
        Ps = Psi_A.conj().T @ Psi_A
    
    if t is None :
        eigsP = jnp.linalg.eigvalsh(mu*Ps)
        t = 1 / (jnp.max(jnp.abs(eigsP)))
    
    res_hist = []
    time_hist = []
    
    if not nits==None:
        it = 0
    
    Psi_I_kron_r = unvec(Psi_A @ r, Y.shape)
    #assert np.allclose(Psi_I_kron_r,Psi @ jnp.kron(jnp.identity(N),r).T) , 'sparse matrice not good'

    cond=False
    
    t0 = time.time()

    while cond==False:        

        #L step
        L = svd_thres(Y - Psi_I_kron_r + (1/mu)*U, 1/mu)
        
        #r step : PGD
        G= L - Y - U/mu
        n = Psi_A.conj().T @ vec(G)

        k = 0
        r_prev = r.copy()
        for itPGD in range(1):
            w= k/(k+3)
            s = r + w*(r - r_prev)
            r_prev = r.copy()

            r = vec(row_thres(unvec(s - t*mu*(n + Ps @ s),(nx*nz,R_mp)),t*lbd)) #L2,1 prox.
            k += 1
                        
        Psi_I_kron_r = unvec(Psi_A @ r, Y.shape)
        
        #U step (dual variable)        
        U = U + mu *(Y-L-Psi_I_kron_r)
        
        res = jnp.linalg.norm(Y-L-Psi_I_kron_r) / jnp.linalg.norm(Y)
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
        
        return L,r,res_hist,time_hist
    
    # plt.plot(res_hist)
    # plt.show()
    
    return L,r


def plot_krpca(r,lbd,mu,plt_rect=False):
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
    ax.set_title(r"$\lambda$={}, $\mu$={}".format(lbd,mu))
    fig.colorbar(im)
    plt.show()

