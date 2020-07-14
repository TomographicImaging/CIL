from ccpi.astra.operators import AstraProjectorSimple
from ccpi.framework import BlockDataContainer, AcquisitionData, AcquisitionGeometry, ImageData, ImageGeometry
from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.optimisation.functions import BlockFunction, KullbackLeibler, MixedL21Norm, IndicatorBox
from ccpi.optimisation.algorithms import SPDHG, PDHG
# Fast Gradient Projection algorithm for Total Variation(TV)
from ccpi.optimisation.functions import TotalVariation
from ccpi.framework import TestData
import numpy as np
import tomophantom, os
from tomophantom import TomoP2D
from ccpi.astra.processors import *
model = 12 # select a model number from the library
N = 128 * 1 # set dimension of the phantom
device = 'gpu'
path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")

phantom = TomoP2D.Model(model, N, path_library2D) 

# Define image geometry.
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N, 
                   voxel_size_x = 0.1,
                   voxel_size_y = 0.1)
im_data = ig.allocate()
im_data.fill(phantom)

# show(im_data, title = 'TomoPhantom', cmap = 'inferno')
# Create AcquisitionGeometry and AcquisitionData 
detectors = N
angles = np.linspace(0, np.pi, 180, dtype='float32')
ag = AcquisitionGeometry('parallel','2D', angles, detectors,
                        pixel_size_h = 0.1)

# Create projection operator using Astra-Toolbox. Available CPU/CPU
Aop = AstraProjectorSimple(ig, ag, device = device)
data = Aop.direct(im_data)

# loader = TestData()
# data = loader.load(TestData.SIMPLE_PHANTOM_2D, size=(128,128))
# print ("here")
# ig = data.geometry
# ig.voxel_size_x = 0.1
# ig.voxel_size_y = 0.1
    
# detectors = ig.shape[0]
# angles = np.linspace(0, np.pi, 180)
# ag = AcquisitionGeometry('parallel','2D',angles, detectors, pixel_size_h = 0.1)
# # Select device
# # device = input('Available device: GPU==1 / CPU==0 ')
# # if device=='1':
# #     dev = 'gpu'
# # else:
# #     dev = 'cpu'
# dev = 'gpu'

# Aop = AstraProjectorSimple(ig, ag, dev)

# sin = Aop.direct(data)
# # Create noisy data. Apply Gaussian noise
# noises = ['gaussian', 'poisson']
# noise = noises[1]
# if noise == 'poisson':
#     np.random.seed(10)
#     scale = 5
#     eta = 0
#     noisy_data = AcquisitionData(np.random.poisson( scale * (eta + sin.as_array()))/scale, ag)
# elif noise == 'gaussian':
#     np.random.seed(10)
#     n1 = np.random.normal(0, 0.1, size = ag.shape)
#     noisy_data = AcquisitionData(n1 + sin.as_array(), ag)
    
# else:
#     raise ValueError('Unsupported Noise ', noise)

noisy_data = data

# Create BlockOperator
operator = Aop 
f = KullbackLeibler(b=noisy_data)        
alpha = 0.5
g =  TotalVariation(alpha, 50, 1e-4, lower=0)   
normK = operator.norm()
sigma = 1/normK
tau = 1/normK
    
# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 1000
pdhg.update_objective_interval = 200
#pdhg.run(200, very_verbose = True)

#%% 'implicit' PDHG, preconditioned step-sizes


#normK = operator.norm()
tau_tmp = 1
sigma_tmp = 1
tau = sigma_tmp / operator.adjoint(tau_tmp * operator.range_geometry().allocate(1.))
sigma = tau_tmp / operator.direct(sigma_tmp * operator.domain_geometry().allocate(1.))
x_init = operator.domain_geometry().allocate()

# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 1000
pdhg.update_objective_interval = 200
# pdhg.run(1000, very_verbose = True)


###############################################################################################
### SPDHG Factory
class SPDHGFactory(object):
    
    @staticmethod
    def get_instance(self, f=None, g=None, operator=None, num_physical_subsets=1, tau=None, sigma=None,
                 x_init=None, use_axpby=True, *args, **kwargs):

        
        return SPDHG()
class SubsetKullbackLeibler(KullbackLeibler):
    def __init__(self, b, eta=None):
        if eta is not None:
            super(SubsetKullbackLeibler, self).__init__(b=b, eta=eta)
        else:
            super(SubsetKullbackLeibler, self).__init__(b=b)
        
    def notify_new_subset(self, subset_id, number_of_subsets):
        self.b.geometry.subset_id = subset_id
    def __getitem__(self, index):
        self.notify_new_subset(index, self.b.geometry.number_of_subsets)
        return self

class AstraSubsetProjectorSimple(AstraProjectorSimple):
    
    def __init__(self, geomv, geomp, device, **kwargs):
        kwargs = {'indices':None, 
                  'subset_acquisition_geometry':None,
                  #'subset_id' : 0,
                  #'number_of_subsets' : kwargs.get('number_of_subsets', 1)
                  }
        # This does not forward to its parent class :(
        super(AstraSubsetProjectorSimple, self).__init__(geomv, geomp, device)
        number_of_subsets = kwargs.get('number_of_subsets',1)
        # self.sinogram_geometry.generate_subsets(number_of_subsets, 'random')
        if geomp.number_of_subsets > 1:
            self.notify_new_subset(0, geomp.number_of_subsets)
    def __getitem__(self, index):
        self.notify_new_subset(index, self.range_geometry().number_of_subsets)
        return self
    def notify_new_subset(self, subset_id, number_of_subsets):
        # print ('AstraSubsetProjectorSimple notify_new_subset')
        # updates the sinogram geometry and updates the projectors
        self.subset_id = subset_id
        self.number_of_subsets = number_of_subsets

        # self.sinogram_geometry.subset_id = subset_id

        #self.indices = self.sinogram_geometry.subsets[subset_id]
        device = self.fp.device
        # this will only copy the subset geometry
        ag = self.range_geometry().copy()
        #print (ag.shape)
        
        self.fp = AstraForwardProjector(volume_geometry=self.domain_geometry(),
                                        sinogram_geometry=ag,
                                        proj_id = None,
                                        device=device)

        self.bp = AstraBackProjector(volume_geometry = self.domain_geometry(),
                                        sinogram_geometry = ag,
                                        proj_id = None,
                                        device = device)
    def __len__(self):
        return self.number_of_subsets
subsets = 10
alpha = 0.5
G = TotalVariation(alpha, 50, 1e-4, lower=0) 

if False:
    size_of_subsets = int(len(angles)/subsets)
    # take angles and create uniform subsets in uniform+sequential setting
    list_angles = [angles[i:i+size_of_subsets] for i in range(0, len(angles), size_of_subsets)]
    # create acquisitioin geometries for each the interval of splitting angles
    list_geoms = [AcquisitionGeometry('parallel','2D',list_angles[i], detectors, pixel_size_h = 0.1) 
                    for i in range(len(list_angles))]
    # create with operators as many as the subsets
    A = BlockOperator(*[AstraProjectorSimple(ig, list_geoms[i], dev) for i in range(subsets)])
    ## number of subsets
    #(sub2ind, ind2sub) = divide_1Darray_equally(range(len(A)), subsets)
    #
    ## acquisisiton data
    g = BlockDataContainer(*[AcquisitionData(noisy_data.as_array()[i:i+size_of_subsets,:])
                                for i in range(0, len(angles), size_of_subsets)])
    ## block function
    F = BlockFunction(*[KullbackLeibler(b=g[i]) for i in range(subsets)]) 

else:
    noisy_data.geometry.generate_subsets(subsets,'uniform')
    A = AstraSubsetProjectorSimple(ig, noisy_data.geometry, device = 'gpu')
    F = SubsetKullbackLeibler(b=noisy_data)
    prob = [1/len(A)]*len(A)
    spdhg = SPDHG(f=F,g=G,operator=A, 
            max_iteration = 1000,
            update_objective_interval=10, prob = prob)
    def save_previous_iteration(self, index):
        print("EEE")
        self._y.update_indices()
    # spdhg.save_previous_iteration = save_previous_iteration




spdhg.run(30, very_verbose = True)
# from ccpi.utilities.quality_measures import mae, mse, psnr
# qm = (mae(spdhg.get_output(), pdhg.get_output()),
#     mse(spdhg.get_output(), pdhg.get_output()),
#     psnr(spdhg.get_output(), pdhg.get_output())
#     )
# print ("Quality measures", qm)

from ccpi.utilities.display import plotter2D

plotter2D(spdhg.x)