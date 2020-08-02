from ccpi.astra.operators import AstraProjectorSimple
from ccpi.framework import BlockDataContainer, AcquisitionData, AcquisitionGeometry, ImageData, ImageGeometry
from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.optimisation.functions import BlockFunction, KullbackLeibler, MixedL21Norm, IndicatorBox
from ccpi.optimisation.algorithms import SPDHG, PDHG
from ccpi.framework import TestData
import numpy as np
from ccpi.utilities.display import plotter2D

loader = TestData()
data = loader.load(TestData.SIMPLE_PHANTOM_2D, size=(128,128))
print ("here")
ig = data.geometry
ig.voxel_size_x = 0.1
ig.voxel_size_y = 0.1
    
detectors = ig.shape[0]
angles = np.linspace(0, np.pi, 180)
ag = AcquisitionGeometry('parallel','2D',angles, detectors, pixel_size_h = 0.1)
# Select device
# device = input('Available device: GPU==1 / CPU==0 ')
# if device=='1':
#     dev = 'gpu'
# else:
#     dev = 'cpu'
dev = 'gpu'

Aop = AstraProjectorSimple(ig, ag, dev)

sin = Aop.direct(data)
# Create noisy data. Apply Gaussian noise
noises = ['gaussian', 'poisson']
noise = noises[1]
if noise == 'poisson':
    np.random.seed(10)
    scale = 5
    eta = 0
    noisy_data = AcquisitionData(np.random.poisson( scale * (eta + sin.as_array()))/scale, ag)
elif noise == 'gaussian':
    np.random.seed(10)
    n1 = np.random.normal(0, 0.1, size = ag.shape)
    noisy_data = AcquisitionData(n1 + sin.as_array(), ag)
    
else:
    raise ValueError('Unsupported Noise ', noise)

#%% 'explicit' SPDHG, scalar step-sizes
subsets = 10
size_of_subsets = int(len(angles)/subsets)
# create Gradient operator
op1 = Gradient(ig)
# take angles and create uniform subsets in uniform+sequential setting
list_angles = [angles[i:i+size_of_subsets] for i in range(0, len(angles), size_of_subsets)]
# create acquisitioin geometries for each the interval of splitting angles
list_geoms = [AcquisitionGeometry('parallel','2D',list_angles[i], detectors, pixel_size_h = 0.1) 
for i in range(len(list_angles))]
# create with operators as many as the subsets
A = BlockOperator(*[AstraProjectorSimple(ig, list_geoms[i], dev) for i in range(subsets)] + [op1])
## number of subsets
#(sub2ind, ind2sub) = divide_1Darray_equally(range(len(A)), subsets)
#
## acquisisiton data
g = BlockDataContainer(*[AcquisitionData(noisy_data.as_array()[i:i+size_of_subsets,:]) for i in range(0, len(angles), size_of_subsets)])
alpha = 0.5
## block function
F = BlockFunction(*[*[KullbackLeibler(b=g[i]) for i in range(subsets)] + [alpha * MixedL21Norm()]]) 
G = IndicatorBox(lower=0)

print ("here")
prob = [1/(2*subsets)]*(len(A)-1) + [1/2]

# norms = [A[i].norm() for i in range(len(A))]
# rho = 0.99
# gamma = 1
# finfo = np.finfo(dtype = np.float32)

# sigma = [rho * gamma / ni for ni in norms]
# sigma = [si * (1 - 1000 * finfo.eps) for si in sigma]
# sigma = [1e-3 for _ in sigma]
sigma = None

algos = []
algos.append( SPDHG(f=F,g=G,operator=A, 
            max_iteration = 1000000,
            update_objective_interval=1000, prob = prob, use_axpby=True,
            sigma = sigma)
)
algos[0].run(10000, very_verbose = True)

dA = BlockOperator(op1, Aop)
dF = BlockFunction(alpha * MixedL21Norm(), KullbackLeibler(b=noisy_data))
algos.append( PDHG(f=dF,g=G,operator=dA, 
            max_iteration = 1000000,
            update_objective_interval=1000, use_axpby=True)
)
algos[1].run(10000, very_verbose = True)


# np.testing.assert_array_almost_equal(algos[0].get_output().as_array(), algos[1].get_output().as_array())
from ccpi.utilities.quality_measures import mae, mse, psnr
qm = (mae(algos[0].get_output(), algos[1].get_output()),
    mse(algos[0].get_output(), algos[1].get_output()),
    psnr(algos[0].get_output(), algos[1].get_output())
    )
print ("Quality measures", qm)
plotter2D([algos[0].get_output(), algos[1].get_output()], titles=['axpby True eps' , 'axpby True'], cmap='viridis')