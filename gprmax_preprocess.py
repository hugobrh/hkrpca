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

pulse_filename = 'ricker.txt'
#pulse_filename = 'chirp.txt'

pulse = np.loadtxt(gprmax_dir+pulse_filename,skiprows=1)
embed_pulse = np.zeros(itse)
embed_pulse[:len(pulse)] = pulse
plt.plot(embed_pulse)
plt.show()

#%%

freqs = np.fft.fftfreq(n=itse,d=dt)[:itse//2]
select_freqs = (freqs>=1e9)&(freqs<=3e9)

embed_pulse_fft = np.fft.fft(embed_pulse)
psd= np.abs(embed_pulse_fft)[:len(embed_pulse_fft)//2]
psd = psd/np.max(psd)
psd_log = 20*np.log10(psd)

plt.plot(freqs[select_freqs],psd_log[select_freqs])
plt.axhline(y=-100,ls='--')
plt.show()

#%%

Bscan_fft = np.array([np.fft.fft(embed_Bscan[:,i]) * np.conj(embed_pulse_fft) for i in range(embed_Bscan.shape[1])]).T
plt.plot(freqs[select_freqs],np.abs(Bscan_fft[:,0][:itse//2][select_freqs]))
plt.show()

plt.plot(np.real(np.fft.ifft(Bscan_fft[:,0])))
plt.show()

Bscan_fft_curated = np.array([Bscan_fft[:,i][:itse//2][select_freqs] for i in range(embed_Bscan.shape[1])]).T
plt.imshow(np.abs(Bscan_fft_curated),aspect='auto')

#%%
np.save(gprmax_dir+filename+'_curated_resampled',Bscan_fft_curated)
