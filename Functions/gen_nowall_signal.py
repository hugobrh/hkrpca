# -*- coding: utf-8 -*-

#Packages
import numpy as np
from Functions.delay_propagation import delai_propag

def gen_no_wall_signal(M,N,P,px,pz,recs,refl,w):
    #signal recu en free space
    y_nowall = np.zeros([M,N],dtype=np.complex_)
    for n in range(N): 
        tau = np.zeros(P)
        for p in range(P):
            tau[p] = delai_propag([px[p],pz[p]],recs[n,:])
        for m in range(M):
            for p in range(P):
                y_nowall[m,n] += refl[p]*np.exp(-1j*w[m]*tau[p])
                
    return y_nowall

