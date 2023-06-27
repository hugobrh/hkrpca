# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:59:01 2021

@author: hbreh
"""

import numpy as np
from Functions.delay_propagation import delai_propag_ttw

def gen_ttw_signal(M,N,P,px,pz,recs,wall_refl,refl,w,tau_wall,snr):
    #signal recu en through wall
    y_ttw = np.zeros([M,N],dtype=np.complex_)
    for n in range(N):
        tau = np.zeros(P)
        for p in range(P):
            _,tau[p] = delai_propag_ttw((px[p],pz[p]),recs[n,:])
        for m in range(M):
            y_ttw[m,n] += wall_refl*np.exp(-1j*w[m]*tau_wall)
            for p in range(P):
                y_ttw[m,n] += refl[p]*np.exp(-1j*w[m]*tau[p])
    
    
    #calcul de la puissance du bruit par le SNR.  n ~ sig_n * CN(0,1)
    # SNR := Py / Pn avec Pn := E(n²) = Var(n) := sig²_n , Py:= tr(YY^H)/(M*N) 
    P_y = (np.trace(y_ttw.conj().T @ y_ttw) / np.prod(np.array(y_ttw.shape))).real
    sig_n = np.sqrt(P_y/snr)  
    
    #Ajout du bruit : y = signal + noise
    np.random.seed(123)
    
    cgn = sig_n/np.sqrt(2) * np.random.randn(M,N) + 1j*sig_n/np.sqrt(2) * np.random.randn(M,N)
    y_ttw += cgn
    
    return y_ttw
