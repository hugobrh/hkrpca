#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 18:07:37 2022

@author: hugobrehier
"""


from scene_data import nx,nz,d,zoff,M,N,recs,R_mp,w,xw_l,xw_r,dim_img,dim_scene
from Functions.delay_propagation import (delai_propag,delai_propag_interior_wall,
                                         delai_propag_ttw,delai_propag_wall_ringing)
import jax.numpy as jnp
import numpy as np
from scipy.linalg import block_diag

ratio_dims = dim_scene/dim_img

import itertools,os

def unique_file(basename, ext):
    actualname = "%s.%s" % (basename, ext)
    c = itertools.count()
    while os.path.exists(actualname):
        actualname = "%s_(%d).%s" % (basename, next(c), ext)
    return actualname

def add_noise(y_mp,snr,t_df,noise_type,noise_dist):
    
    P_y = np.mean(np.abs(y_mp)**2)
    sig_n = np.sqrt(P_y/snr)  
    
    cgn = sig_n/np.sqrt(2) * np.random.randn(M,N) + 1j*sig_n/np.sqrt(2) * np.random.randn(M,N)

    if noise_type=='pt':
        if noise_dist=='gaussian':
            text = 1    
        
        elif noise_dist=='student':
            x = np.random.gamma(shape=t_df/2,scale=2,size= (M,N))
            text = t_df/x
        
        elif noise_dist=='k':
            text = np.random.gamma(shape=t_df,scale=1/t_df,size= (M,N))
        
        else:
            raise ValueError("check noise dist")

        y_mp += np.sqrt(text)*cgn
        
    elif noise_type=='col':
        if noise_dist=='gaussian':
            text = np.ones(N)
        
        elif noise_dist=='student':
            x = np.random.gamma(shape=t_df/2,scale=2,size= N)
            text = t_df/x
        
        elif noise_dist=='k':
            text = np.random.gamma(shape=t_df,scale=1/t_df,size=N)
        
        else:
            raise ValueError("check noise dist")

        y_mp += cgn @ np.diag(np.sqrt(text))
    
    else:
        raise ValueError("check noise type")

    return y_mp

def vec(Y):
    return Y.ravel('F')

def unvec(v,dims:tuple): 
    return v.reshape(dims, order='F')

def csgn(X):
    return jnp.where(X!=0,X/jnp.abs(X),0)

def soft_thres(Y,l):
    return csgn(Y) * jnp.maximum(0,jnp.abs(Y) - l)

def svd_thres(Y,l):
    U, s, Vh = jnp.linalg.svd(Y, full_matrices=False)
    return U @ jnp.diag(soft_thres(s, l)) @ Vh

def row_thres(U,l):
    '''Row-wise Thresholding (mixed l2/l1 norm proximal)'''
    nU = jnp.sqrt(jnp.diag(U @ U.conj().T))
    Ut = (jnp.maximum(0,1-(l/nU)) * U.T).T
    return Ut
    
def hub(x,c):
    ''' Huber fct '''  
    return ((jnp.abs(x)<=c)*0.5*jnp.abs(x)**2 + (jnp.abs(x)>c)*c*(jnp.abs(x)-0.5*c))
    
def dhub(x,c):
    ''' Huber fct '''    
    return jnp.where(jnp.abs(x)>c,c*csgn(x),x)

def prox_huber(x,c,a):
    return jnp.where(jnp.abs(x)<=c*(a+1),x/(a+1),x-c*a*csgn(x))

def prox_huber_norm(X,c,a):
    """proximal of the a-scaled Huber function composed with norm, of threshold c and matrix argument X """
    nX = jnp.linalg.norm(X)
    #cast nan to zero this wqy so it can be compiled by jax
    return jnp.where(jnp.abs(X/nX) < jnp.inf, prox_huber(nX, c, a) * X/nX, jnp.zeros(X.shape))
    
def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array looks like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def unblockshaped(arr, h, w):
    """
    Return an array of shape (h, w) where
    h * w = arr.size

    If arr is of shape (n, nrows, ncols), n sublocks of shape (nrows, ncols),
    then the returned array preserves the "physical" layout of the sublocks.
    """
    n, nrows, ncols = arr.shape
    return (arr.reshape(h//nrows, -1, nrows, ncols)
               .swapaxes(1,2)
               .reshape(h, w))

def make_img(R):
    r_vec = vec(R)
    #Unvectorize solution
    r_mat = unvec(r_vec,(nx*nz,-1))
    
    #combine all R sub-images 
    r_comb = np.zeros(nx*nz,dtype=np.complex64)
    for pix in range(nx*nz):
        r_comb[pix] = np.linalg.norm(r_mat[pix,:])
    
    #Unvectorize to form image
    r = unvec(r_comb,(nx,nz))
    r = np.rot90(r)
    
    return r

#Multipath dictionary
def gen_dict(path):
    try:
        PSI = np.load(path)
    except FileNotFoundError:
        PSI = None
        print('Saved dictionary PSI not found')
        
    
    if not PSI is None:
        print("PSI already loaded from hard drive")
    else:
        print('Creating dictionary PSI')

        PSI = []
        
        #create one dico of delay for each SAR position
        for n in range(N):
            delais_simple_ttw = np.zeros([nx,nz],dtype=np.complex64)
            points = np.zeros([nx,nz,2])
            
            for i in range(nx):
                for j in range(nz):
                    points[i,j] = np.array([i,j]) * ratio_dims
                    if points[i,j][1] <= (zoff+d):
                        delais_simple_ttw[i,j] = delai_propag(points[i,j],recs[n,:])
                    else:
                        _,delais_simple_ttw[i,j] = delai_propag_ttw(points[i,j],recs[n,:])
                        
            #create one dico of delay for each multipath            
            for r in range(R_mp):
                psi_r = np.zeros([M,nx*nz],dtype=np.complex64)
                
                tau = np.zeros([nx,nz],dtype=np.complex64)    
                for i in range(nx):
                    for j in range(nz):
                        if points[i,j][1] <= (zoff+d): #zone avant mur
                            tau[i,j] = delais_simple_ttw[i,j]
                        else: 
                            if r == 0:
                                tau[i,j] = delais_simple_ttw[i,j]
                            elif r == 1:
                                tau[i,j] = delais_simple_ttw[i,j]/2 + delai_propag_wall_ringing(points[i,j],recs[n,:],1)
                            elif r == 2:
                                tau[i,j] = delais_simple_ttw[i,j]/2 + delai_propag_interior_wall(points[i,j],recs[n,:],xw_r)
                            elif r == 3:
                                tau[i,j] = delais_simple_ttw[i,j]/2 + delai_propag_interior_wall(points[i,j],recs[n,:],xw_l)   
                tau = vec(tau)
                for m in range(M):
                    for pos in range(nx*nz):
                        psi_r[m,pos] = np.exp(-1j*w[m]*tau[pos])
                
                if r==0:
                    Psi_n = psi_r
                else:
                    Psi_n = np.hstack([Psi_n,psi_r])
            
            PSI.append(Psi_n)
        PSI = np.hstack(PSI)
        np.save(path,PSI)
        print(f"Dictionary saved in relative path : {path}")
    return PSI




# def CS(PSI,path):
    
#     try:
#         Phi = np.load(path)
#     except FileNotFoundError:
#         Phi = None
#         print('Saved compressive matrix Phi not found')
    
#     if Phi is not None:
#         print("Phi already loaded from hard drive")
#     else:
        
#         Q1 = round(0.25 * M)
#         Q2 = round(1 * N)
        
#         theta = np.zeros([Q2,N])
#         indsq2 = np.sort(np.random.choice(N,Q2,replace=False))
#         for i,row in enumerate(theta):
#             row[indsq2[i]] = 1
        
#         Iq1 = np.identity(Q1)
#         diag_phis = []
#         indsq1 = np.sort(np.random.choice(M,Q1,replace=False))
#         for n in range(N):
#             phi_n = np.zeros([Q1,M])
#             for i,row in enumerate(phi_n):
#                 row[indsq1[i]] = 1
            
#             diag_phis.append(phi_n)
#         diag_phis = block_diag(*diag_phis)
#         Phi = np.kron(theta,Iq1) @ diag_phis
    
#     #np.save("../Phi_csmp",Phi)
    
#     try:
#         Phi_small = np.load("Phismall_csmp.npy")
#     except FileNotFoundError:
#         Phi_small = None
#         print('Saved compressive matrix Phi_small not found')
    
#     if Phi_small is not None:
#         print("Phi_small already loaded from hard drive")
#     else:
#         Phi_small = np.zeros([Q1,M])
#         for i,row in enumerate(Phi_small):
#             row[indsq1[i]] = 1
            
#     #np.save("../Phismall_csmp",Phi_small)
    
#     #Create the reduced dictionnary by applying Phi to each of the R Psi
#     try:
#         PhiPsi = np.load("PhiPsi_csmp_R2.npy")
#     except FileNotFoundError:
#         PhiPsi = None
#         print('Saved compressed dict PhiPsi not found')
    
#     if PhiPsi is not None:
#         print("PhiPsi already loaded from hard drive")
#     else:
#         PhiPsi = [Phi @ Psi for Psi in PSI]
#         PhiPsi = np.hstack(PhiPsi)
    
#     #np.save("PhiPsi_csmp_R2",PhiPsi)
#     Phi = jnp.array(Phi,dtype=jnp.complex64)
#     PhiPsi = jnp.array(PhiPsi,dtype=jnp.complex64)
