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
# from ccpi.utilities.multiprocessing import NUM_THREADS
from ccpi.utilities.display import plotter2D
import time

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

image = acq_data.create_uniform_image(1., (127, 220, 220))
image.initialise(dim=(127, 220, 220), vsize=(2.03125, 1.7080754, 1.7080754))

attns = pet.ImageData('mu_map.hv')
asm_norm = pet.AcquisitionSensitivityModel('norm.n.hdr')


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


f_numba = KullbackLeibler(b = acq_data, eta = rand , use_numba=True)
f_numpy = KullbackLeibler(b = acq_data, eta = rand , use_numba=False)


fake_data = am.direct(image)
t0 = time.time()
res_numba = f_numba(fake_data)
t1 = time.time()
dt_numba = t1-t0

t0 = time.time()
res_numpy = f_numpy(fake_data)
t1 = time.time()
dt_numpy = t1-t0
print ("call took", t1-t0)

print ("numba {} {}s ".format(res_numba, dt_numba))
print ("numpy {} {}s ".format(res_numpy, dt_numpy))
# 
tau = 1.

print ("TEST proximal_conjugate")
# proximal_conjugate
out_nb = f_numba.proximal_conjugate(fake_data, tau=tau)
out_np = f_numpy.proximal_conjugate(fake_data, tau=tau)


t0 = time.time()
out_nb = f_numba.proximal_conjugate(fake_data, tau=tau)
t1 = time.time()
dt_numba = t1-t0

t0 = time.time()
out_np = f_numpy.proximal_conjugate(fake_data, tau=tau)
t1 = time.time()
dt_numpy = t1-t0

print ("out=None numba {} {}s ".format(res_numba, dt_numba))
print ("out=None numpy {} {}s ".format(res_numpy, dt_numpy))

np.testing.assert_allclose(out_np.as_array(), out_nb.as_array(), rtol=2e-5)

out_np = fake_data * 0.
out_nb = fake_data * 0.
f_numba.proximal_conjugate(fake_data, tau=tau,  out=out_nb)
f_numpy.proximal_conjugate(fake_data, tau=tau, out=out_np)


t0 = time.time()
f_numba.proximal_conjugate(fake_data, tau=tau,  out=out_nb)
t1 = time.time()
dt_numba = t1-t0

t0 = time.time()
f_numpy.proximal_conjugate(fake_data, tau=tau, out=out_np)
t1 = time.time()
dt_numpy = t1-t0

print ("out= numba {} {}s ".format(res_numba, dt_numba))
print ("out= numpy {} {}s ".format(res_numpy, dt_numpy))

np.testing.assert_allclose(out_np.as_array(), out_nb.as_array(), rtol=2e-5)



diff = out_np - out_nb
# plotter2D([el.as_array()[0][80] for el in [out_nb, out_np, diff]], 
#            titles=['numba', 'numpy', 'diff'], cmap='viridis')

print ("TEST proximal")
# proximal
out_np = fake_data * 0.
out_nb = fake_data * 0.
tau = 1.
f_numba.proximal(fake_data, tau=tau, out=out_nb)
f_numpy.proximal(fake_data, tau=tau, out=out_np)


t0 = time.time()
f_numba.proximal(fake_data, tau=tau, out=out_nb)
t1 = time.time()
dt_numba = t1-t0

t0 = time.time()
f_numpy.proximal(fake_data, tau=tau, out=out_np)
t1 = time.time()
dt_numpy = t1-t0

print ("numba {} {}s ".format('proximal', dt_numba))
print ("numpy {} {}s ".format('proximal', dt_numpy))
np.testing.assert_allclose(out_np.as_array(), out_nb.as_array(), rtol=2e-5)

f_numba.proximal(fake_data, tau=tau, out=out_nb)
f_numpy.proximal(fake_data, tau=tau, out=out_np)


t0 = time.time()
out_nb = f_numba.proximal(fake_data, tau=tau)
t1 = time.time()
dt_numba = t1-t0

t0 = time.time()
out_np = f_numpy.proximal(fake_data, tau=tau)
t1 = time.time()
dt_numpy = t1-t0

print ("out None numba {} {}s ".format('proximal', dt_numba))
print ("out None numpy {} {}s ".format('proximal', dt_numpy))
np.testing.assert_allclose(out_np.as_array(), out_nb.as_array(), rtol=2e-5)


diff = out_np - out_nb
# plotter2D([el.as_array()[0][80] for el in [out_nb, out_np, diff]], 
#            titles=['numba', 'numpy', 'diff'], cmap='viridis')

np.testing.assert_allclose(out_np.as_array(), out_nb.as_array(), rtol=2e-5)


print ("TEST convex_conjugate")
# convex_conjugate

t0 = time.time()
out_nb = f_numba.convex_conjugate(fake_data)
t1 = time.time()
dt_numba = t1-t0

t0 = time.time()
out_np = f_numpy.convex_conjugate(fake_data)
t1 = time.time()
dt_numpy = t1-t0

print ("{} numba {} {}s ".format(out_nb, 'convex_conjugate', dt_numba))
print ("{} numpy {} {}s ".format(out_np, 'convex_conjugate', dt_numpy))


# call with mask

am.num_subsets = 10
am.subset_num = 1
image.fill(1)

fake_data = am.direct(image)
mask = fake_data.as_array() > 0
print ("has mask some values? {}/{}".format(np.sum(mask), mask.size))
f_numba_mask = KullbackLeibler(b = acq_data, eta = rand , use_numba=True, mask=mask)

t0 = time.time()
res_numba = f_numba(fake_data)
t1 = time.time()
dt_numba = t1-t0

t0 = time.time()
res_numpy = f_numpy(fake_data)
t1 = time.time()
dt_numpy = t1-t0
print ("call took", t1-t0)

print ("numba {} {}s ".format(res_numba, dt_numba))
print ("numpy {} {}s ".format(res_numpy, dt_numpy))



