# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:59:01 2021

@author: hbreh
"""

import numpy as np
from Functions.delay_propagation import (delai_propag_ttw,delai_propag_wall_ringing,
                               delai_propag_interior_wall)

#signal recu en multipath (interior wall, wall ringing)

def gen_multipath_signal(M,N,P,px,pz,recs,wall_refl,refl,w,tau_wall,snr,
                         xw_l,xw_r,refl_virt_l,refl_virt_r,iw,refl_wr,t_df,do_pt_outliers = True):
    
    
    y_mp = np.zeros([M,N],dtype=np.complex64)
    
    for n in range(N):
        #Init delays
        tau = np.zeros(P)
        tau_int_wall_l = np.zeros(P)
        tau_int_wall_r = np.zeros(P)
        tau_wall_ring = np.zeros((P,iw))
        
        for p in range(P):
            #Direct path ttw delay
            _,tau[p] = delai_propag_ttw((px[p],pz[p]),recs[n,:])
            
            #Interior wall propag delay
            tau_int_wall_r[p] = tau[p]/2 +  delai_propag_interior_wall((px[p],pz[p]), recs[n,:], xw_r)
            tau_int_wall_l[p] = tau[p]/2 +  delai_propag_interior_wall((px[p],pz[p]), recs[n,:], xw_l)
            
            #Wall ringing propag delay
            for i in range(iw):
                tau_wall_ring[p,i] = tau[p]/2 + delai_propag_wall_ringing((px[p],pz[p]),recs[n,:],i+1)
    
        #Compute returns
        #front wall returns
        y_mp[:,n] += wall_refl*np.exp(-1j*w*tau_wall)
        
        #target returns : direct, interior walls left/right, wall ringing
        for p in range(P):
            y_mp[:,n] += refl[p]*np.exp(-1j*w*tau[p])
            #y_mp[:,n] += refl_virt_r*refl[p]*np.exp(-1j*w*tau_int_wall_r[p])
            #y_mp[:,n] += refl_virt_l*refl[p]*np.exp(-1j*w*tau_int_wall_l[p])
            for i in range(iw):
                y_mp[:,n] += refl_wr[i]*refl[p] * np.exp(-1j*w*tau_wall_ring[p,i])
                    
    #calcul de la puissance du bruit par le SNR.  n ~ sig_n * CN(0,1)
    # SNR := Py / Pn avec Pn := E(n²) = Var(n) := sig²_n , Py:= tr(YY^H)/(M*N) 
    P_y = np.mean(np.abs(y_mp)**2)
    sig_n = np.sqrt(P_y/snr)  
    
    #assert t_df > 2 , f'Number of d.f. {t_df} should be > 2 for finite 2nd order moments and centering E[texture]=1'

    if do_pt_outliers:
        cgn = sig_n/np.sqrt(2) * np.random.randn(M,N) + 1j*sig_n/np.sqrt(2) * np.random.randn(M,N)
        
        # #gaussian
        # text = 1
        
        #t-dist 
        x = np.random.gamma(shape=t_df/2,scale=2,size= (M,N))
        text = t_df/x
        
        # #k-dist
        # text = np.random.gamma(shape=t_df,scale=1/t_df,size= (M,N))
        
        #compute
        y_mp += np.sqrt(text)*cgn
        
    else:
        cgn = sig_n/np.sqrt(2) * np.random.randn(M,N) + 1j*sig_n/np.sqrt(2) * np.random.randn(M,N)
        
        # #gaussian
        # text = np.ones(N)
        
        #t-dist
        x = np.random.gamma(shape=t_df/2,scale=2,size= N)
        text = t_df/x
        
        # #k-dist
        # text = np.random.gamma(shape=t_df,scale=1/t_df,size=N)
        
        #compute
        y_mp += cgn @ np.diag(np.sqrt(text))
        #y_mp += cgn


    return y_mp
