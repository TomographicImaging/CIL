#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from functools import partial
from os import path
import os
from glob import glob
from docopt import docopt
import matplotlib.pyplot as plt
import numpy as np

from sirf.Utilities import error, show_2D_array, examples_data_path
import sirf.Reg as reg
import sirf.STIR as pet
from ccpi.optimisation.algorithms import PDHG, SPDHG
from ccpi.optimisation.functions import     KullbackLeibler, BlockFunction, IndicatorBox, MixedL21Norm
from ccpi.optimisation.operators import     CompositionOperator, BlockOperator, LinearOperator
from ccpi.plugins.regularisers import FGP_TV
from ccpi.filters import regularisers
from ccpi.utilities import NUM_THREADS
from ccpi.utilities.display import plotter2D

from ccpi.optimisation.operators import Gradient

pet.AcquisitionData.set_storage_scheme('memory')
pet.set_verbosity(0)


# In[ ]:


# load data
template_acq_data = pet.AcquisitionData('Siemens_mMR', span=11, max_ring_diff=15, view_mash_factor=1)
# data_path = '/home/sirfuser/devel/buildVM/sources/SIRF/data/examples/PET/mMR'
# acq_data = pet.AcquisitionData('{}/sino_f1g1d0b0.hs'.format(data_path))
seconds = 600
data_path = '/home/edo/scratch/code/PETMR/install/share/sirf/NEMA'
os.chdir(os.path.abspath(data_path))
acq_data = pet.AcquisitionData('NEMA_sino_0-{}s.hs'.format(seconds))

# fix a problem with the header which doesn't allow
# to do algebra with randoms and sinogram
# rand_arr = pet.AcquisitionData('{}/sino_randoms_f1g1d0b0.hs'.format(data_path)).as_array()
rand_arr = pet.AcquisitionData('NEMA_randoms_0-{}s.hs'.format(seconds))
rand = acq_data * 0
rand.fill(rand_arr)

attns = pet.ImageData('mu_map.hv')
asm_norm = pet.AcquisitionSensitivityModel('norm.n.hdr')


# In[ ]:


# get initial estimate
#image = get_initial_estimate(acq_data)
vaggelis = True
if vaggelis:
    image = acq_data.create_uniform_image(0., (127, 220, 220))
    image.initialise(dim=(127, 220, 220), vsize=(2.03125, 1.7080754, 1.7080754))
    # create a shape
    shape = pet.EllipticCylinder()
    shape.set_length(400)
    shape.set_radii((40, 100))
    shape.set_origin((10, 60, 0))

    # add the shape to the image
    image.add_shape(shape, scale = 1)

    # add another shape
    shape.set_radii((30, 30))
    shape.set_origin((10, -30, 60))
    image.add_shape(shape, scale = 1.5)

    # add another shape
    shape.set_origin((10, -30, -60))
    image.add_shape(shape, scale = 0.75)

    G = Gradient(image, backend='c', correlation='SpaceChannel')
    out = G.direct(image)

    plotter2D([image.as_array()[70], 
               out.get_item(0).as_array()[70], 
               out.get_item(1).as_array()[70],
               out.get_item(2).as_array()[70]], cmap='inferno', 
               titles=['ImageData', 'grad_x', 'grad_y', 'grad_z'])

    raise RuntimeError('Stop it')
else:
    nxny = 127
    image = acq_data.create_uniform_image(0.0, (nxny, nxny))


# In[ ]:


def get_asm_attn(sino, attn, acq_model):
    """Get attn ASM from sino, attn image and acq model."""
    asm_attn = pet.AcquisitionSensitivityModel(attn, acq_model)
    # temporary fix pending attenuation offset fix in STIR:
    # converting attenuation into 'bin efficiency'
    asm_attn.set_up(sino)
    bin_eff = pet.AcquisitionData(sino)
    bin_eff.fill(1.0)
    asm_attn.unnormalise(bin_eff)
    asm_attn = pet.AcquisitionSensitivityModel(bin_eff)
    return asm_attn


# In[ ]:


# set up the acquisition model
am = pet.AcquisitionModelUsingRayTracingMatrix()
# ASM norm already there
asm_attn = get_asm_attn(acq_data,attns,am)
# Get ASM dependent on attn and/or norm
asm = pet.AcquisitionSensitivityModel(asm_norm, asm_attn)
am.set_acquisition_sensitivity(asm)
am.set_up(acq_data, image)


# In[ ]:





# In[ ]:


acq_data.dimensions()


# In[ ]:


from ccpi.optimisation.algorithms import PDHG, SPDHG
from ccpi.optimisation.functions import KullbackLeibler, L2NormSquared, LeastSquares, IndicatorBox,                                         BlockFunction, Function, ScaledFunction
from ccpi.optimisation.operators import LinearOperator, BlockOperator, Gradient, ScaledOperator
from ccpi.plugins.regularisers import FGP_TV 
from ccpi.framework import ImageGeometry, BlockGeometry
from numbers import Number


# In[ ]:


ImageData = am.domain_geometry()
print(ImageData.voxel_sizes())


# In[ ]:


domain_cil = ImageGeometry(voxel_num_x = 127, voxel_num_y = 127, voxel_num_z = 127,
                       voxel_size_x = 0.04681, voxel_size_z = 0.02031,
                       voxel_size_y = 0.04681)

domain_sirf = ImageData


# In[ ]:


def PowerMethod(operator, iterations, x_init=None):
    '''Power method to calculate iteratively the Lipschitz constant
    
    :param operator: input operator
    :type operator: :code:`LinearOperator`
    :param iterations: number of iterations to run
    :type iteration: int
    :param x_init: starting point for the iteration in the operator domain
    :returns: tuple with: L, list of L at each iteration, the data the iteration worked on.
    '''
    
    # Initialise random
    if x_init is None:
        x0 = operator.domain_geometry().allocate('random')
    else:
        x0 = x_init.copy()
        
    x1 = operator.domain_geometry().allocate()
    y_tmp = operator.range_geometry().allocate()
    s = []
    # Loop
    i = 0
    while i < iterations:
        operator.direct(x0,out=y_tmp)
        operator.adjoint(y_tmp,out=x1)
        x1norm = x1.norm()
        if hasattr(x0, 'squared_norm'):
            s.append( x1.dot(x0) / x0.squared_norm() )
        else:
            x0norm = x0.norm()
            s.append( x1.dot(x0) / (x0norm * x0norm) ) 
        x1.multiply((1.0/x1norm), out=x0)
        print ("current squared norm: {}".format(s[-1]))
        i += 1
        #if i == iterations:
            #cont=input("Continue with {} iterations?[y/n]".format(iterations))
            #if cont == 'y':
               # i = 0
    return np.sqrt(s[-1]), [np.sqrt(si) for si in s], x0


# In[ ]:
do_pdhg = False

if do_pdhg:
    #am_norm = PowerMethod(am, 20)[0]
    am_norm = np.sqrt(4479081.53395)


    # # PDHG without regularization

    # In[ ]:


    # reg parameter
    alpha = 0.001


    # explicit case

    # rescale KL
    # KL(lambda *x + eta, b) = lambda * KL(x + eta/lambda, b/lambda)
    f1 = ScaledFunction( KullbackLeibler(b =  (1/am_norm)*acq_data, eta = (1/am_norm)*rand ), am_norm)



    F = f1
    G = IndicatorBox(lower=0)

    # rescale operators
    am_rescaled = ScaledOperator(am, (1/am_norm))

    K = am_rescaled


    # In[ ]:


    sigma = 1.0
    tau = 1.0
    pet.set_max_omp_threads(15)


    # In[ ]:


    pdhg = PDHG(f = F, g = G, operator=K, sigma=sigma, tau=tau,
                max_iteration=1000, update_objective_interval = 10)
    pdhg.run(20,very_verbose = True)


    # In[ ]:


    plt.figure()
    plt.imshow(pdhg.get_output().as_array()[75,:,:], cmap="inferno")
    plt.colorbar()
    plt.show()


# # SPDHG without regularization

# In[ ]:


num_subsets = 10
ams = []
fs = []
# set the image to 1 to be able to get the mask
image.fill(1)
# duplicate by the number of sinograms
for k in range(num_subsets):
    amk = pet.AcquisitionModelUsingRayTracingMatrix()
    amk.set_acquisition_sensitivity(asm)
    # XXX
    amk.set_up(acq_data, image)
    amk.num_subsets = num_subsets
    amk.subset_num = k
    ams.append(amk)

    mask = amk.direct(image)
    fk = KullbackLeibler(b =  acq_data, eta = rand, mask=mask.as_array(), use_numba=True )
    fs.append(fk)
    


# In[ ]:


Fs = BlockFunction(*fs)
Gs = IndicatorBox(lower=0)
Ks = BlockOperator(*ams)
rho = 0.99
# normK = PowerMethod(Ks, 6)[0]
# normK = 7978677.9511325015
normK = 1491.1303497291835
print ("NORMK", normK)

sigma = [rho/normK] * num_subsets
tau = rho/normK

# In[ ]:


import time
data = acq_data * 0
data += 1e-2

t0 = time.time()
Fs[0].proximal_conjugate(data, tau)
t1 = time.time()
print ("prox_conj", t1-t0)
# raise RuntimeError("stop here")

# In[ ]:


# from ccpi.utilities.jupyter import islicer, link_islicer
# image.fill(1)
# subsets_bdc = Ks.direct(image)
# # get mask
# mask = []
# for i in range(len(subsets_bdc)):
#      mask.append( subsets_bdc.get_item(i).as_array() > 0 )
# print("how many masks", len(mask))
# import time
# t0 = time.time()
# a = subsets_bdc + acq_data
# t1 = time.time()
# print ("a = subset_bdc + acq_data ", t1-t0)

# a_mask = subsets_bdc.copy()
# a_mask *= 0
# t = [0,0,0]
# data1 = acq_data.as_array()
# tt0 = time.time()
# for i in range(len(subsets_bdc)):
#     t0 = time.time()
#     # arr = a_mask.get_item(i).as_array()
#     t1 = time.time()
#     t[0] += t1 - t0
#     arr = np.add(subsets_bdc.get_item(i).as_array(), data1, where=mask[i])
#     t2 = time.time()
#     t[1] += t2 - t1
#     a_mask.get_item(i).fill( arr )
#     t[2] += time.time() - t2
        
# tt1 = time.time()

# print ("b = subset_bdc[mask] + acq_data[mask] as_array", t[0])
# print ("b = subset_bdc[mask] + acq_data[mask] np algebra", t[1])
# print ("b = subset_bdc[mask] + acq_data[mask] fill", t[2])
# print ("b = subset_bdc[mask] + acq_data[mask] total", t[0]+t[1]+t[2], tt1-tt0)


# In[ ]:


pet.set_max_omp_threads(25)

def update_no_objective(algo, iteration, objective, solution):
    print(algo.verbose_output(verbose=False))
from functools import partial


spdhg = SPDHG(f = Fs, g = Gs, operator=Ks, sigma=sigma, tau=tau,
            max_iteration=1000, update_objective_interval = 10, use_axpby=True)
spdhg.run(num_subsets*50,print_interval=1, verbose = 2)


# In[ ]:





# In[ ]:


pet.set_max_omp_threads(15)
# spdhg.update_objective_interval = 200
# # spdhg.run(3000,verbose = False, callback = lambda x,y,z: print (x))
# spdhg.run(1000, verbose = 2)


# In[ ]:

spdhg.get_output().write('spdhg_out.hs')
print (spdhg.iterations, spdhg.objective)
plt.figure()
plt.imshow(spdhg.get_output().as_array()[75,:,:], cmap="inferno")
plt.colorbar()
plt.show()


# In[ ]:


print (spdhg.iteration)
9 * 3000 / 60 / 60


# In[ ]:




