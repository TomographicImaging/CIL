from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, BlockDataContainer, AcquisitionData

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.optimisation.functions import ZeroFunction, L2NormSquared, \
                      MixedL21Norm, BlockFunction, KullbackLeibler, IndicatorBox, TotalVariation
                      
from ccpi.astra.operators import AstraProjectorSimple

import os, sys
import tomophantom
from tomophantom import TomoP2D

#%%
# user supplied input
if len(sys.argv) > 1:
    which_noise = int(sys.argv[1])
else:
    which_noise = 1
    
model = 1 # select a model number from the library
N = 128 # set dimension of the phantom
path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")

phantom_2D = TomoP2D.Model(model, N, path_library2D)    
data = ImageData(phantom_2D)
ig = ImageGeometry(voxel_num_x=N, voxel_num_y=N)

# Create acquisition data and geometry
detectors = N
angles = np.linspace(0, np.pi, 180)
ag = AcquisitionGeometry('parallel','2D',angles, detectors)

# Select device
device = input('Available device: GPU==1 / CPU==0 ')
if device=='1':
    dev = 'gpu'
else:
    dev = 'cpu'
    
Aop = AstraProjectorSimple(ig, ag, dev)
sin = Aop.direct(data)

#%%

# Create noisy data. Apply Gaussian noise
noises = ['gaussian', 'poisson']
noise = noises[which_noise]

if noise == 'poisson':
    scale = 5
    eta = 0
    noisy_data = AcquisitionData(np.random.poisson( scale * (eta + sin.as_array()))/scale, ag)
elif noise == 'gaussian':
    n1 = np.random.normal(0, 1, size = ag.shape)
    noisy_data = AcquisitionData(n1 + sin.as_array(), ag)
    
else:
    raise ValueError('Unsupported Noise ', noise)

# Show Ground Truth and Noisy Data
plt.figure(figsize=(10,10))
plt.subplot(1,2,2)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(1,2,1)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.show()

#%%

print(" ##################################################### ")
print(" ##################################################### ")
print(" Setup Explicit PDHG with and without preconditioning ")
print(" ##################################################### ")
print(" ##################################################### ")      

      
#%%      
# Create operators
op1 = Gradient(ig)
op2 = Aop

# Create BlockOperator
operator = BlockOperator(op1, op2, shape=(2,1) ) 

# Compute operator Norm
normK = operator.norm()

# Create functions
if noise == 'poisson':
    
    alpha = 2
    f2 = KullbackLeibler(b=noisy_data)  
    g =  IndicatorBox(lower=0)        
        
elif noise == 'gaussian':   
    
    alpha = 10
    f2 = 0.5 * L2NormSquared(b=noisy_data)                                         
    g = ZeroFunction()
    
sigma = 1./normK
tau = 1./normK     
    
f1 = alpha * MixedL21Norm() 
f = BlockFunction(f1, f2)   

print(" Run PDHG without preconditioning ")
pdhg_noprecond = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma,
                      max_iteration = 1000,
                      update_objective_interval = 200)
pdhg_noprecond.run(1000, very_verbose = True)

#%%
print(" Choose diagonal precond for tau/sigma ")

tau = 1. / (op2.adjoint(op2.range_geometry().allocate(1.)) + 4.)
sigma2 = 1. / op2.direct(op2.domain_geometry().allocate(1.))
sigma1 =  op1.range_geometry().allocate(2.)
sigma = BlockDataContainer(sigma1, sigma2)


def PDHG_new_update(self):
     """Modify the PDHG update to allow preconditioning"""
     # save previous iteration
     self.x_old.fill(self.x)
     self.y_old.fill(self.y)
     # Gradient ascent for the dual variable
     self.operator.direct(self.xbar, out=self.y_tmp)
     self.y_tmp *= self.sigma
     self.y_tmp += self.y_old
     self.f.proximal_conjugate(self.y_tmp, self.sigma, out=self.y)
     # Gradient descent for the primal variable
     self.operator.adjoint(self.y, out=self.x_tmp)
     self.x_tmp *= -1*self.tau
     self.x_tmp += self.x_old
     self.g.proximal(self.x_tmp, self.tau, out=self.x)
     # Update
     self.x.subtract(self.x_old, out=self.xbar)
     self.xbar *= self.theta    
     self.xbar += self.x
    
PDHG.update = PDHG_new_update

print(" Run PDHG with preconditioning ")
pdhg_precond = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma,
                    max_iteration = 1000,
                    update_objective_interval = 1)
pdhg_precond.run(1000, very_verbose = True)

#%%

print(" Compare reconstructions ")

plt.figure(figsize=(15,15))
plt.subplot(2,2,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(2,2,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.subplot(2,2,3)
plt.imshow(pdhg_noprecond.get_output().as_array())
plt.title('PDHG Explicit no precond')
plt.colorbar()
plt.subplot(2,2,4)
plt.imshow(pdhg_precond.get_output().as_array())
plt.title('PDHG Explicit precond')
plt.colorbar()
plt.show()

print(" ##################################################### ")
print(" ##################################################### ")
print(" End Explicit PDHG with and without preconditioning ")
print(" ##################################################### ")
print(" ##################################################### ")   
      
#%%      
      
print(" ##################################################### ")
print(" ##################################################### ")
print(" Setup implicit PDHG with and without preconditioning ")
print(" ##################################################### ")
print(" ##################################################### ")    
   
      
operator = Aop
f = KullbackLeibler(b=noisy_data)      
g = TotalVariation(alpha, 100, tolerance = None, lower = 0)
normK = operator.norm()
sigma = 1./normK
tau = 1./normK

pdhg_implicit_noprecond = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma,
                      max_iteration = 1000,
                      update_objective_interval = 200)
pdhg_implicit_noprecond.run(1000, very_verbose = True)      


#%%

print(" Choose diagonal precond for tau/sigma ")

tau = 1. / (operator.adjoint(operator.range_geometry().allocate(1.)))
sigma = 1. / operator.direct(operator.domain_geometry().allocate(1.))

pdhg_implicit_precond = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma,
                      max_iteration = 1000,
                      update_objective_interval = 200)
pdhg_implicit_precond.run(1000, very_verbose = True)   

#%%

print(" Compare reconstructions ")

plt.figure(figsize=(15,15))
plt.subplot(2,2,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(2,2,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.subplot(2,2,3)
plt.imshow(pdhg_implicit_noprecond.get_output().as_array())
plt.title('PDHG Implicit no precond')
plt.colorbar()
plt.subplot(2,2,4)
plt.imshow(pdhg_implicit_precond.get_output().as_array())
plt.title('PDHG Implicit precond')
plt.colorbar()
plt.show()
