**Repo for HKRPCA**

Associated paper is [Through the Wall Radar Imaging via Kronecker-structured Huber-type RPCA](https://www.sciencedirect.com/science/article/pii/S016516842300302X)

1. Folder `Functions/` contains the implementation of the different methods.
2. Folder `Call/` contains the code to run an example with the provided scene:
	- The needed dictionary (mapping the scene) will be created (once then stored) with scene described by `scene_data.py`
	- The radar data in `TTW_Bscan_no_interior_walls_ricker_merged_3mm_curated_resampled.npy` is simulated from GprMax and pretreated 
3. Yml files are provided to create a suitable conda environnment 
