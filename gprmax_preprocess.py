#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import matplotlib.pyplot as plt
import numpy as np
import shutil

# %matplotlib

gprmax_dir = "../../gprMax/ttw_model/"
filename = "TTW_Bscan_no_interior_walls_ricker_merged_3mm"

#%%
f = h5py.File(gprmax_dir+filename + '.out', 'r')
# list(f.keys())
# list(f.attrs.keys())
# list(f.attrs.values())

dt = f.attrs["dt"]
its = f.attrs["Iterations"]
T = its*dt

f.close()

#%%
f = h5py.File(gprmax_dir+filename + '.out', 'r')
Bscan = np.array(f['rxs']["rx1"]["Ez"])
f.close()
plt.imshow(Bscan,aspect='auto')
plt.show()

# remove coupling 
t_crosstalk = 1*2/3e8 
its_crosstalk = int(t_crosstalk/dt)
Bscan[0:its_crosstalk,:] = 0
plt.imshow(Bscan,aspect='auto')
plt.show()

# from Functions.rpca import R_pca
# rpca = R_pca(Bscan)
# L, S = rpca.fit(max_iter=300, iter_print=100)

# plt.imshow(L,aspect='auto')
# plt.show()

# plt.imshow(S,aspect='auto')
# plt.show()

output_file = shutil.copy(src=gprmax_dir+filename+ '.out',dst=gprmax_dir+filename+'_curated.out')
output_file = h5py.File(output_file, 'r+')
output_file['rxs']["rx1"]["Ez"][...] = Bscan
output_file.close()

#%%

output_file = h5py.File(gprmax_dir+filename+'_curated.out', 'r')
Bscan = np.array(output_file['rxs']["rx1"]["Ez"])
output_file.close()

embed_Bscan = Bscan
itse = embed_Bscan.shape[0] 
plt.plot(embed_Bscan[:,0])
plt.show()

#%%

freqs = np.fft.fftfreq(n=itse,d=dt)[:itse//2]
select_freqs = (freqs>=1e9)&(freqs<=3e9)

#%%

Bscan_fft = np.array([np.fft.fft(embed_Bscan[:,i]) for i in range(embed_Bscan.shape[1])]).T
plt.plot(freqs[select_freqs],np.abs(Bscan_fft[:,0][:itse//2][select_freqs]))
plt.show()

plt.plot(np.real(np.fft.ifft(Bscan_fft[:,0])))
plt.show()

Bscan_fft_curated = np.array([Bscan_fft[:,i][:itse//2][select_freqs] for i in range(embed_Bscan.shape[1])]).T
plt.imshow(np.abs(Bscan_fft_curated),aspect='auto')

#%%
np.save(gprmax_dir+filename+'_curated_resampled',Bscan_fft_curated)
