

import numpy as np
from ccpi.optimisation.operators import *
from ccpi.optimisation.algorithms import *
from ccpi.optimisation.functions import *
from ccpi.framework import *
from cvxpy import *

m = 5
n = 500

np.random.seed(10)
# Create matrix A and data b
Amat = np.asarray( np.random.randn(m, n), dtype=numpy.float32)
bmat = np.asarray( np.random.randn(m), dtype=numpy.float32)

# Geometries for data and solution
vgb = VectorGeometry(m)
vgx = VectorGeometry(n)

b = vgb.allocate(0, dtype=numpy.float32)
b.fill(bmat)
A = LinearOperatorMatrix(Amat)


def cgls_new(x0, operator, data, it):
    
    x = operator.domain_geometry().allocate()

    
    r = data - operator.direct(x)
    s = operator.adjoint(r)
    
    p = s
    norms0 = s.norm()
    gamma = norms0**2
    normx = x.norm()
    xmax = normx
    
    for _ in range(it):
        
        q = operator.direct(p)
        delta = q.squared_norm()
        alpha = gamma/delta
        
        x += alpha * p
        r -= alpha * q
        
        s = operator.adjoint(r)
        
        norms = s.norm()
        gamma1 = gamma
        gamma = norms**2
        beta = gamma/gamma1
        p = s + beta * p   
        
        normx = x.norm()
        xmax = np.maximum(xmax, normx)
        
        
        if gamma<=1e-6:
            break    
        
        
    return x
        
#x0 = A.domain_geometry().allocate()        
#x = cgls_new(x0, A, b, 1000)
#
#
#x1 = Variable(n)        
#obj =  Minimize( sum_squares(A.A*x1 - b.as_array()))
#prob = Problem(obj)
#result = prob.solve(verbose = True, solver = MOSEK)    
#
#print((VectorData(x1.value) - x).squared_norm())

#%%

from ccpi.framework import AcquisitionGeometry, BlockDataContainer, AcquisitionData

import numpy as np 
import numpy                          
import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import CGLS
from ccpi.optimisation.operators import BlockOperator, Gradient
       
from ccpi.framework import TestData
import os, sys
from ccpi.astra.ops import AstraProjectorSimple 

# Load Data  
loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))                 
N = 64
M = 64
data = loader.load(TestData.SIMPLE_PHANTOM_2D, size=(N,M), scale=(0,1))

ig = data.geometry

detectors = N
angles = np.linspace(0, np.pi, N, dtype=np.float32)

ag = AcquisitionGeometry('parallel','2D', angles, detectors)

dev = 'cpu'

Aop = AstraProjectorSimple(ig, ag, dev)    
sin = Aop.direct(data)

noisy_data = AcquisitionData( sin.as_array() + np.random.normal(0,3,ig.shape))


# Setup and run the CGLS algorithm  
alpha = 5
Grad = Gradient(ig)
#
## Form Tikhonov as a Block CGLS structure
op_CGLS = BlockOperator( Aop, alpha * Grad, shape=(2,1))
block_data = BlockDataContainer(noisy_data, Grad.range_geometry().allocate())

x2 = op_CGLS.domain_geometry().allocate()
x3 = cgls_new(x2, op_CGLS, block_data, 1000)
plt.imshow(x3.as_array())
plt.colorbar()
plt.show()
        
    
    
    
    