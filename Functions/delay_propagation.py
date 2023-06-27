# -*- coding: utf-8 -*-
"""
Created on Fri May 14 10:07:44 2021

@author: hbreh
"""

import numpy as np
import matplotlib.pyplot as plt
from scene_data import c,zoff,d,eps

#délai propagation émetteur n à cible p (sans mur)
def delai_propag(p,n):
    dist = np.linalg.norm(p-n)
    return 2*dist/c

#Delai propag avec mur
def sec(x):
    '''secant function : sec(x) = 1/cos(x)'''
    return 1/np.cos(x)

def csc(x):
    '''cosecant function : csc(x) = 1/sin(x)'''
    return 1/np.sin(x)

def cot(x):
    '''cotangent function : cot(x) = cos(x)/sin(x)'''
    return np.cos(x)/np.sin(x)

def f1(x,Xq,Xtm):
    '''
    Transcendental function whose roots to find where x := theta in Amin's paper.
    '''
    f1_x = (
        (Xq[0]-(Xtm[0]+zoff*np.tan(x)))**2 + Xq[1]**2 -
        (d/(np.cos(np.arcsin(np.sin(x)/np.sqrt(eps)))))**2 -
        ((Xq[1]-d)/np.cos(x))**2 + 2*(d/(np.cos(np.arcsin(np.sin(x)/np.sqrt(eps))))) *
        ((Xq[1]-d)/np.cos(x)) * np.cos(np.pi - x + np.arcsin(np.sin(x)/np.sqrt(eps)))
        )   
    return f1_x
                
def df1(x,Xq,Xtm):
    '''
    df1/dx. First derivative of f1 wrt x := theta in Amin's paper.
    '''
    term1 = -2*zoff*(sec(x))**2 *(Xq[0]-Xtm[0]-zoff*np.tan(x))
    term2 = (d**2 * eps * np.sin(2*x))/((eps - np.sin(x)**2)**2)
    term3 = 2*(Xq[1]-d)**2 * sec(x)**2 *np.tan(x)
    term4 = ((1/np.sqrt(eps))*d*(d - Xq[1])*sec(x)*sec(x/np.sqrt(eps))*
            (-2*np.cos(x + np.arccos(np.sin(x/np.sqrt(eps)))) +
            2*np.sqrt(np.cos(x/np.sqrt(eps))**2)*(np.sqrt(eps)*sec(x) +
            sec(x/np.sqrt(eps)) *np.sin(x + np.arccos(np.sin(x/np.sqrt(eps)))))*np.tan(x/np.sqrt(eps))))
    df1_x = term1 - term2 - term3 + term4 
    return df1_x

def d2f1(x,Xq,Xtm):
    '''
    d²f1/dx². Second derivative of f1 wrt x := theta in Amin's paper.
    '''
    term1 = 2*zoff*sec(x)**2 *(zoff* sec(x)**2 + 2*np.tan(x)* (-Xq[0] + Xtm[0] + zoff* np.tan(x)))
    term2 = (d**2 * eps *(3 + (-2 + 4 * eps)* np.cos(2*x) - np.cos(4*x)))/(2*(eps - np.sin(x)**2)**3)
    term3 = -2* (Xq[1] - d)**2 *(-2 + np.cos(2*x))* sec(x)**4
    term4 = ((1/(eps*np.sqrt(np.cos(x/np.sqrt(eps))**2))) *2*d*(d - Xq[1])*sec(x)*
             sec(x/np.sqrt(eps))*(np.sqrt(eps)*sec(x)*(2 + np.sqrt(eps)*np.sin((2*x)/np.sqrt(eps))*np.tan(x))+
                                  2*np.sin(x)*np.tan(x/np.sqrt(eps))))
    d2f1_x = term1 - term2 - term3 + term4
    return d2f1_x

        
from scipy.optimize import newton
        
def delai_propag_ttw(p,n,halley=False,fullout=False):
    '''
    TWO WAY Propagation delay Through the Wall. The calculation of theta
    can be achieved through Newton-Raphson algorithm or Halley's method 
    which expands it to incorporate the second derivative.
    
    Parameters
    ----------
    p : tuple of length 2.
        (x,y) coordinates of a target
    n : tuple of length 2.
        (x,y) coordinates of a sar position.
    halley : boolean, optional
        Whether to use Halley's method. The default is False.
    fullout : boolean, optional
        Whether to print full details of convergence. The default is False.

    Returns
    -------
    theta : scalar
        angle in radians.
    delai_prop_ttw : scalar
        Propagation delay in seconds between the emiter and the target .

    '''
    
    #Centrer l'axe des ordonnées sur la position du mur
    Xtm = np.array(n) - np.array([0,zoff])
    Xq = np.array(p) - np.array([0,zoff])
    
    #Valeur initiale pour theta: angle entre Xq,Xtm et le point (xq,ztm)
    xo = np.arctan2(Xq[0]-Xtm[0],Xq[1]-Xtm[1])    

    if halley==False:
        if fullout == True:
            theta,outputs = newton(func=f1,x0=xo,args=(Xq,Xtm),fprime=df1,
                                   full_output=fullout)
            
            print('-------------------------------------------')
            print(outputs)
            print('theta = ' + str(theta*180/np.pi) + '°')
            
        else:
            theta = newton(func=f1,x0=xo,args=(Xq,Xtm),fprime=df1,
                          full_output=fullout)
            
    else:
        if fullout == True:
            theta,outputs = newton(func=f1,x0=xo,args=(Xq,Xtm),fprime=df1,
                                   fprime2=d2f1,full_output=fullout)
            
            print('-------------------------------------------')
            print(outputs)
            print('theta = ' + str(theta*180/np.pi) + '°')
            
        else:
            theta = newton(func=f1,x0=xo,args=(Xq,Xtm),fprime=df1,
                           fprime2=d2f1,full_output=fullout)
        
        
        
    
    lnp_air1 = zoff / np.cos(theta)
    lnp_wall = d/(np.cos(np.arcsin(np.sin(theta)/np.sqrt(eps))))
    lnp_air2 = (Xq[1]-d)/np.cos(theta)
    
    delai_prop_ttw = 2*lnp_air1/c + 2*lnp_wall/(c/np.sqrt(eps)) + 2*lnp_air2/c 

    if fullout == True:
        xa = Xtm[0] + zoff * np.tan(theta)
        Xa = np.array([xa,0])
        xb = Xa[0] + lnp_wall * np.sin(np.arcsin(np.sin(theta)/np.sqrt(eps)))
        Xb = np.array([xb,d])
        
        plt.plot([Xtm[0],Xa[0]],[Xtm[1],Xa[1]],'bo-')
        plt.plot([Xa[0],Xb[0]],[Xa[1],Xb[1]],'bo-')
        plt.plot([Xb[0],Xq[0]],[Xb[1],Xq[1]],'bo-')
        plt.axhline(y=0,linestyle='-')
        plt.axhline(y=d,linestyle='-')
        plt.show()

    return theta,delai_prop_ttw


def delai_propag_interior_wall(Xq,Xtm,xw):
    """
    ONE WAY propagation delay in interior wall scenario
    """
    _,delai = delai_propag_ttw((2*xw-Xq[0],Xq[1]),Xtm)
    return delai/2

def Fct(x,Xq,Xtm,iw):
    '''
    vector function for multipath wall ringing.
    Solvable by multivariate root finding algorithms.
    x = [theta_air,theta_wall]
    '''
    
    dx = np.abs(Xtm[0] - Xq[0])
    dz = np.abs(Xtm[1] - Xq[1]) 
    
    F1_x = (dz-d)*np.tan(x[0]) + d*(1+2*iw)*np.tan(x[1]) - dx
    F2_x = np.sin(x[0])/np.sin(x[1]) - np.sqrt(eps)
    
    return [F1_x,F2_x]

def J_Fct(x,Xq,Xtm,iw):
    '''
    Jacobian of Fct. 
    '''

    dz = np.abs(Xtm[1] - Xq[1]) 
    
    DF1_x0 = (c-dz)*sec(x[0])**2  
    DF1_x1 = d*(1+iw)*sec(x[1])**2 
    DF2_x0 = np.cos(x[0]) * csc(x[1])
    DF2_x1 = -cot(x[1])*csc(x[1])*np.sin(x[0])
    
    return np.array([[DF1_x0,DF1_x1],[DF2_x0,DF2_x1]])

from scipy.optimize import root

def delai_propag_wall_ringing(Xq,Xtm,iw,fullout=False):
    """
    ONE WAY propagation delay in wall ringing scenario
    """
    
    dz = np.abs(Xtm[1] - Xq[1]) 
    
    th_a_init = np.arctan2(Xq[0]-Xtm[0],Xq[1]-Xtm[1])  
    th_w_init = np.arcsin(np.sin(th_a_init)/np.sqrt(eps))
    xo = [th_a_init,th_w_init]
    
    sol = root(Fct, xo, jac=J_Fct,args=(Xq,Xtm,iw), method='hybr')
    th_a,th_w = sol.x
    tau = (dz-d)/(c*np.cos(th_a)) + (np.sqrt(eps)*d*(1+2*iw))/(c*np.cos(th_w))
        
    if fullout == True:
        long_wall = d/np.cos(th_w)
        xa_o = Xtm[0] + zoff * np.tan(th_a)      
        dw = long_wall * np.sin(th_w)
        
        
        xa_prime = np.arange(iw+1)
        xb_prime = np.arange(iw+1)
        
        xa_prime = 2*dw * xa_prime
        xb_prime = 2*dw * xb_prime
        
        xa_prime += xa_o
        xb_prime += xa_o+dw
        
        plt.plot([Xtm[0],xa_prime[0]],[Xtm[1],zoff],'bo-')
        
        for i in range(iw+1):
            plt.plot([xa_prime[i],xb_prime[i]],[zoff,zoff+d],'bo-')

            if i>0:
                plt.plot([xb_prime[i-1],xa_prime[i]],[zoff+d,zoff],'bo-')


        plt.plot([xb_prime[-1],Xq[0]],[zoff+d,Xq[1]],'bo-')

        plt.axhline(y=zoff,linestyle='-')
        plt.axhline(y=zoff+d,linestyle='-')
        plt.show()
    
    
    return tau
