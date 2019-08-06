#========================================================================
# Copyright 2019 Science Technology Facilities Council
# Copyright 2019 University of Manchester
#
# This work is part of the Core Imaging Library developed by Science Technology
# Facilities Council and University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#=========================================================================

""" 

Total Variation Denoising using PDHG algorithm:


Problem:     min_{u},   \alpha * ||\nabla u||_{2,1} + Fidelity(u, g)

             \alpha: Regularization parameter
             
             \nabla: Gradient operator 
             
             g: Noisy Data 
                          
             Fidelity =  1) L2NormSquarred ( \frac{1}{2} * || u - g ||_{2}^{2} ) if Noise is Gaussian
                         2) L1Norm ( ||u - g||_{1} )if Noise is Salt & Pepper
                         3) Kullback Leibler (\int u - g * log(u) + Id_{u>0})  if Noise is Poisson
                                                       
             Method = 0 ( PDHG - split ) :  K = [ \nabla,
                                                 Identity]
                          
                                                                    
             Method = 1 (PDHG - explicit ):  K = \nabla    
             
             
             Default: ROF denoising
             noise = Gaussian
             Fidelity = L2NormSquarred 
             method = 0
             
             
"""

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import PDHG

from ccpi.optimisation.operators import BlockOperator, Identity, Gradient
from ccpi.optimisation.functions import ZeroFunction, L1Norm, \
                      MixedL21Norm, BlockFunction, L2NormSquared,\
                          KullbackLeibler
from ccpi.framework import TestData
import os, sys
#import scipy.io
if int(numpy.version.version.split('.')[1]) > 12:
    from skimage.util import random_noise
else:
    from demoutil import random_noise


loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
data = loader.load(TestData.SHAPES, size=(50,50))
ig = data.geometry
ag = ig


from ccpi.framework import ImageData, ImageGeometry
from ccpi.optimisation.operators import LinearOperator
import scipy

class Convolution(LinearOperator):


    def __init__(self, kernel, gm_domain):

        super(Convolution, self).__init__()
        self.kernel = kernel
        self.gm_domain = gm_domain
        self.gm_range = gm_domain

    def direct(self, x, out=None):

        tmp = scipy.signal.fftconvolve(x.as_array(), self.kernel, mode='same')       
        if out is not None:            
            out.fill(ImageData(tmp))
        else:
            return ImageData(tmp)

    def adjoint(self, x, out=None):

        tmp = scipy.signal.fftconvolve(x.as_array(), self.kernel[::-1, ::-1], mode='same')
        if out is not None:
            out.fill(ImageData(tmp))
        else:
            return ImageData(tmp)

    def domain_geometry(self):
        return self.gm_domain
    
    def range_geometry(self):
        return self.gm_range
        
        
#kernel = np.outer(signal.gaussian(10, 3), signal.gaussian(10, 3))

space = odl.uniform_discr([-1, -1], [1, 1], [200, 300])
kernel = odl.phantom.cuboid(space, [-0.05, -0.05], [0.05, 0.05])


A = Convolution(kernel, ig)    
    
blurred_data = A.direct(data)    
noisy_data = blurred_data + ImageData(np.random.normal(
                    loc=0, scale=0.02, size=data.shape))

#%%


u = ig.allocate('random_int')
w = ig.allocate('random_int')

rhs = A.direct(u).dot(w)
lhs = u.dot(A.adjoint(w))
print(rhs, lhs) 



#%%
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.imshow(data.as_array())
plt.title('Ground Truth')
plt.colorbar()
plt.subplot(1,2,2)
plt.imshow(noisy_data.as_array())
plt.title('Noisy Data')
plt.colorbar()
plt.show()

alpha = 500
f2 = 0.5 * L2NormSquared(b=noisy_data)
    
method = '0'      

if method == '0':

    # Create operators
    op1 = Gradient(ig, correlation=Gradient.CORRELATION_SPACE)
    op2 = A

    # Create BlockOperator
    operator = BlockOperator(op1, op2, shape=(2,1) ) 

    # Create functions      
    f = BlockFunction(alpha * MixedL21Norm(), f2) 
    g = ZeroFunction()
    
else:
    
    operator = Gradient(ig)
    f =  alpha * MixedL21Norm()
    g = f2
        
    
# Compute operator Norm
normK = operator.norm()

# Primal & dual stepsizes
sigma = 0.1
tau = 1/(sigma*normK**2)


# Setup and run the PDHG algorithm
pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
pdhg.max_iteration = 5000
pdhg.update_objective_interval = 500
pdhg.run(5000)

plt.imshow(pdhg.get_output().as_array())
plt.title('TV Reconstruction')
plt.colorbar()



#%%
if data.geometry.channels > 1:
    plt.figure(figsize=(20,15))
    for row in range(data.geometry.channels):
        
        plt.subplot(3,4,1+row*4)
        plt.imshow(data.subset(channel=row).as_array())
        plt.title('Ground Truth')
        plt.colorbar()
        plt.subplot(3,4,2+row*4)
        plt.imshow(noisy_data.subset(channel=row).as_array())
        plt.title('Noisy Data')
        plt.colorbar()
        plt.subplot(3,4,3+row*4)
        plt.imshow(pdhg.get_output().subset(channel=row).as_array())
        plt.title('TV Reconstruction')
        plt.colorbar()
        plt.subplot(3,4,4+row*4)
        plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), data.subset(channel=row).as_array()[int(N/2),:], label = 'GTruth')
        plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), pdhg.get_output().subset(channel=row).as_array()[int(N/2),:], label = 'TV reconstruction')
        plt.legend()
        plt.title('Middle Line Profiles')
    plt.show()
    
else:
    plt.figure(figsize=(20,5))
    plt.subplot(2,2,1)
    plt.imshow(data.subset(channel=0).as_array())
    plt.title('Ground Truth')
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.imshow(noisy_data.subset(channel=0).as_array())
    plt.title('Noisy Data')
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.imshow(pdhg.get_output().subset(channel=0).as_array())
    plt.title('TV Reconstruction')
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), data.as_array()[int(ig.shape[0]/2),:], label = 'GTruth')
    plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), pdhg.get_output().as_array()[int(ig.shape[0]/2),:], label = 'TV reconstruction')
    plt.legend()
    plt.title('Middle Line Profiles')
    plt.show()


#%% Check with CVX solution

from ccpi.optimisation.operators import SparseFiniteDiff

try:
    from cvxpy import *
    cvx_not_installable = True
except ImportError:
    cvx_not_installable = False


if cvx_not_installable:

    ##Construct problem    
    u = Variable(ig.shape)
    
    DY = SparseFiniteDiff(ig, direction=0, bnd_cond='Neumann')
    DX = SparseFiniteDiff(ig, direction=1, bnd_cond='Neumann')
    
    # Define Total Variation as a regulariser
    regulariser = alpha * sum(norm(vstack([Constant(DX.matrix()) * vec(u), Constant(DY.matrix()) * vec(u)]), 2, axis = 0))
    
    # choose solver
    if 'MOSEK' in installed_solvers():
        solver = MOSEK
    else:
        solver = SCS      

    # fidelity
    if noise == 's&p':
        fidelity = pnorm( u - noisy_data.as_array(),1)
    elif noise == 'poisson':
        fidelity = sum(kl_div(noisy_data.as_array(), u)) 
        solver = SCS
    elif noise == 'gaussian':
        fidelity = 0.5 * sum_squares(noisy_data.as_array() - u)
                
    obj =  Minimize( regulariser +  fidelity)
    prob = Problem(obj)
    result = prob.solve(verbose = True, solver = solver)
    
    diff_cvx = numpy.abs( pdhg.get_output().as_array() - u.value )
        
    plt.figure(figsize=(15,15))
    plt.subplot(3,1,1)
    plt.imshow(pdhg.get_output().as_array())
    plt.title('PDHG solution')
    plt.colorbar()
    plt.subplot(3,1,2)
    plt.imshow(u.value)
    plt.title('CVX solution')
    plt.colorbar()
    plt.subplot(3,1,3)
    plt.imshow(diff_cvx)
    plt.title('Difference')
    plt.colorbar()
    plt.show()    
    
    plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), pdhg.get_output().as_array()[int(ig.shape[0]/2),:], label = 'PDHG')
    plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), u.value[int(ig.shape[0]/2),:], label = 'CVX')
    plt.legend()
    plt.title('Middle Line Profiles')
    plt.show()
            
    print('Primal Objective (CVX) {} '.format(obj.value))
    print('Primal Objective (PDHG) {} '.format(pdhg.objective[-1][0]))
