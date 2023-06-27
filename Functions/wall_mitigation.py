# -*- coding: utf-8 -*-


import jax.numpy as jnp
from skimage.filters import threshold_otsu 
from scipy.stats.mstats import gmean
from scene_data import c,M,N,B

def wall_mitigation_simple(y_orig):
    dR = c/(2*B)
    U,sv,Vh = jnp.linalg.svd(y_orig)
    Dsv = jnp.array([sv[i] - sv[i+1] for i in range(sv.shape[0]-1)])
    Pw = jnp.zeros(y_orig.shape,dtype=y_orig.dtype)
    boundary = jnp.argmax(Dsv)+1
    print(f'boundary: {boundary}th singular value')
    for i in jnp.arange(boundary):
        Pw += jnp.outer(U[:,i], Vh[:,i])
    return (jnp.identity(M) - Pw @ Pw.conj().T) @ y_orig


def wall_mitigation(y_orig):
    
    dR = c/(2*B)
    U,sv,Vh = jnp.linalg.svd(y_orig)
    V = Vh.conj().T
        
    #Threshold for classif wall vs target 
     
    delta = threshold_otsu(sv,nbins=256*4)
    Cw = sv[sv >= delta]
    Ct = sv[sv < delta]
    
    #Wall range calculation
    
    h = jnp.zeros(len(Cw))
    for i in range(len(Cw)):
        Psi = sv[i]* U[:,i].reshape(-1,1) @ V[:,i].conj().T.reshape(1,-1)
        ri = jnp.zeros(Psi.shape[0],dtype=jnp.complex64)
        for j,col in enumerate(Psi.T):
            ri += jnp.fft.ifft(col)
        ri = ri / len(sv)   
        ind_max = jnp.argmax(jnp.abs(ri))
        h[i] = jnp.arange(0,dR*M,dR)[ind_max]
    
    n = jnp.max(h)
    
    #Pass on all singular values for classif wall/target
    
    h = jnp.zeros(len(sv))
    for i in range(len(sv)):
        Psi = sv[i]* U[:,i].reshape(-1,1) @ V[:,i].conj().T.reshape(1,-1)
        ri = jnp.zeros(Psi.shape[0],dtype=jnp.complex64)
        for j,col in enumerate(Psi.T):
            ri += jnp.fft.ifft(col)
        ri = ri / len(sv)
        ind_max = jnp.argmax(jnp.abs(ri))
        h[i] = jnp.arange(0,dR*M,dR)[ind_max]
        
    Cw = jnp.arange(len(sv))[h <= n]
    Ct = jnp.arange(len(sv))[h > n]
    
    #Removal of mean signal (across transceiver positions)    
    eT = jnp.ones(N).reshape(1,-1)
    m = jnp.mean(y_orig,axis=1).reshape(-1,1)
    y_tilde = y_orig - m @ eT
    
    U,sv,Vh = jnp.linalg.svd(y_tilde)
    V = Vh.conj().T
    
    #Removal of wall returns by projection on orthogonal complement
    
    Pw = jnp.zeros(y_orig.shape,dtype=jnp.complex64)
    for i in Cw:
        Pw += U[:,i].reshape(-1,1) @ V[:,i].conj().T.reshape(1,-1)
    
    y_hat = (jnp.identity(M) - Pw @ Pw.conj().T) @ y_tilde
    
    U,sv,Vh = jnp.linalg.svd(y_hat)
    V = Vh.conj().T
    
    #Removal of noise subspace
    
    AICs = [N*(M-i)*jnp.log(jnp.mean(sv[i+1:])/gmean(sv[i+1:])) + (2*M-i)*i for i in range(len(sv)-1)] 
    i_opt = jnp.argmin(AICs)
                      
    Pn = jnp.zeros(y_orig.shape,dtype=jnp.complex64)
    
    for i in range(i_opt,len(sv)):
        Pn += U[:,i].reshape(-1,1) @ V[:,i].conj().T.reshape(1,-1)
          
    y_bar = (jnp.identity(M) - Pn @ Pn.conj().T) @ y_hat


    return y_bar


