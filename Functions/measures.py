# -*- coding: utf-8 -*-

import numpy as np
from skimage.measure import block_reduce



#True detection map
def True_Map(px,pz,dx,dz,dim_img,br=(1,1)):
    ts = np.array([(px[i]/dx,pz[i]/dz) for i in range(len(px))],dtype='int')
    R_0 = np.zeros(dim_img)
    for i in range(ts.shape[0]):
        R_0[ts[i,0],ts[i,1]] = 1
    R_0 = np.rot90(R_0)
    img_0 = block_reduce(R_0, br, np.max)
    return img_0

#Return estimated detection map ready for classification metrics
def Estimated_Map(R_a,thr=0.01,br=(1,1)):
    img_0 = np.nan_to_num(np.abs(R_a) / np.max(np.abs(R_a)))
    img_0 = block_reduce(img_0,br, np.max)
    assert thr != None
    img= np.zeros(img_0.shape)
    img[np.where(img_0>thr)] = 1
    return img


#Error = difference in support
def MSE(R_a,R_0,thr=0.05):
    img = np.abs(R_a) / np.max(np.abs(R_a))
    img = block_reduce(img, (2,2), np.max)

    img_a = np.zeros(img.shape)
    img_a[np.where(img>thr)] = 1

    img_0 = np.abs(R_0) / np.max(np.abs(R_0))
    img_0 = block_reduce(img_0, (2,2), np.max)
    
    return np.linalg.norm(img_a - img_0)**2

#Target to Clutter Ratio : ratio of average intensities in region of target and clutter
def TCR(R_a,R_0,thr=0.05):
    img_a = block_reduce(R_a, (2,2), np.mean)
    img_0 = block_reduce(R_0, (2,2), np.max)
    
    n_t = (np.where(img_0==1)[0]).size
    n_c = (np.where(img_0!=1)[0]).size
       
    I_t = np.linalg.norm(img_a[np.where(img_0==1)])**2/n_t
    I_c = np.linalg.norm(img_a[np.where(img_0!=1)])**2/n_c
      
    return 10*np.log10(I_t/I_c)

#Iintensity of target region
def I_t(R_a,R_0,thr=0.05):
    img_a = block_reduce(R_a, (2,2), np.mean)
    img_0 = block_reduce(R_0, (2,2), np.max)
    
    n_t = (np.where(img_0==1)[0]).size       
    I_t = np.linalg.norm(img_a[np.where(img_0==1)])**2/n_t
      
    return I_t
