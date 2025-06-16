#%%
from cil.utilities import dataexample
from cil.optimisation.functions import TotalVariation
from cil.plugins.ccpi_regularisation.functions import FGP_TV
from cil.utilities.display import show2D

import time

N = 10
num_iter = 1000
isotropic = True

#%%
# get an example image

data = {'numpy': dataexample.CAMERA.get(size=(128, 128))}
geom = data['numpy'].geometry.copy()

import numpy as np
import torch as to
import cupy as cp
from cil.framework import ImageData


data['torch'] = ImageData(to.from_numpy(data['numpy'].as_array()), 
                          geometry=geom.copy())


data['cupy'] = ImageData(cp.asarray(data['numpy'].as_array()), 
                          geometry=geom.copy())

data['c'] = data['numpy']

#%% create a TotalVariation object
TV = {}
TV['cupy']  = TotalVariation(max_iteration=num_iter, isotropic=isotropic, backend='numpy')
TV['torch'] = TotalVariation(max_iteration=num_iter, isotropic=isotropic, backend='numpy')
TV['numpy'] = TotalVariation(max_iteration=num_iter, isotropic=isotropic, backend='numpy')
TV['c'] = TotalVariation(max_iteration=num_iter, isotropic=isotropic, backend='c')



#%%
fgp = FGP_TV(max_iteration=num_iter, isotropic=isotropic, device='gpu')

#%%
d1 = fgp.proximal(data['numpy'], tau=1)
t0 = time.time()
for _ in range(N):
    d1 = fgp.proximal(data['numpy'], tau=1, out=d1)
t1 = time.time()
dt_fgp = t1-t0
print (f"Elapsed time: {dt_fgp:.6f} seconds")

#%%
# d3 = cTV.proximal(x, tau=1)

#%%
# cupy

def run_TV(data, TV, backend, tau=1, N=10):
    d = TV[backend].proximal(data[backend], tau=tau)
    t0 = time.time()
    for _ in range(N):
        d = TV[backend].proximal(data[backend], tau=tau, out=d)
    t1 = time.time()
    dt = t1-t0
    print (f"Elapsed time: {dt:.6f} seconds")
    return ( d.as_array(), dt )


#%%

d2 , dt_2 = run_TV(data, TV, 'cupy', tau=1, N=N)
#%%
# d3 , dt_3 = run_TV(data, TV, 'torch', tau=1, N=N)
# #%%
# d4 , dt_4 = run_TV(data, TV, 'numpy', tau=1, N=N)
# #%%
# d5 , dt_5 = run_TV(data, TV, 'c', tau=1, N=N)

# #%%
# show2D(d4, title=f'np TotalVariation {dt_4}', origin='upper', cmap='inferno')

# #%%
# show2D([d1, d5, d4, d3.numpy(), cp.asnumpy(d2)], title=[f'FGP_TV {dt_fgp/N:.3f}s', 
#                                  f'C TotalVariation {dt_5/N:.3f}s',
#                                  f'np TotalVariation {dt_4/N:.3f}s',
#                                  f'torch TotalVariation {dt_3/N:.3f}s',
#                                  f'cp TotalVariation {dt_2/N:.3f}s',
#                                  ], origin='upper', cmap='inferno',
#                                  num_cols=2)
# # %%
# import array_api_compat
# np_array = array_api_compat.to_numpy(data['cupy'].as_array())

# # %%
