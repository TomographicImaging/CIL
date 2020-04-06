# -*- coding: utf-8 -*-
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ccpi.optimisation.operators import LinearOperator
from ccpi.optimisation.functions import Function
import warnings


class LeastSquares(Function):
    r"""Least Squares function
    
    .. math:: F(x) = c\|Ax-b\|_2^2
    
    Parameters:
        
        A : Operator
        
        c : Scaling Constant
        
        b : Data
        
    L : Lipshitz Constant of the gradient of :math:`F` which is :math:`2c||A||_2^2 = 2s1(A)^2`,
    
    where s1(A) is the largest singular value of A.
        
    
    """
    
    def __init__(self, A, b, c=1.0, estimate_Lipschitz = False):
        super(LeastSquares, self).__init__()
    
        self.A = A  # Should be an operator, default identity
        self.b = b  # Default zero DataSet?
        self.c = c  # Default 1.
        self.range_tmp = A.range_geometry().allocate()
        
        if estimate_Lipschitz:
            # Compute the Lipschitz parameter from the operator if possible
            # Leave it initialised to None otherwise
            try:
                self.L = 2.0*self.c*(self.A.norm()**2)
            except AttributeError as ae:
                if self.A.is_linear():
                    Anorm = LinearOperator.PowerMethod(self.A, 10)[0]
                    self.L = 2.0 * self.c * (Anorm*Anorm)
                else:
                    warnings.warn('{} could not calculate Lipschitz Constant. {}'.format(
                    self.__class__.__name__, ae))
                
            except NotImplementedError as noe:
                warnings.warn('{} could not calculate Lipschitz Constant. {}'.format(
                    self.__class__.__name__, noe))
        
    def __call__(self, x):
        
        r""" Returns the value of :math:`F(x) = c\|Ax-b\|_2^2`
        """

        y = self.A.direct(x)
        y.subtract(self.b, out=y)
        try:
            return y.squared_norm() * self.c
        except AttributeError as ae:
            # added for compatibility with SIRF
            warnings.warn('squared_norm method not found! Proceeding with norm.')
            yn = y.norm()
            if self.c == 1:
                return yn * yn
            return (yn * yn) * self.c
    
    def gradient(self, x, out=None):
        
        r""" Returns the value of the gradient of :math:`F(x) = c*\|A*x-b\|_2^2`
        
             .. math:: F'(x) = 2cA^T(Ax-b)

        """
        
        if out is not None:
            self.A.direct(x, out=self.range_tmp)
            self.range_tmp.subtract(self.b , out=self.range_tmp)
            self.A.adjoint(self.range_tmp, out=out)
            out.multiply (self.c * 2.0, out=out)
        else:
            return (2.0*self.c)*self.A.adjoint(self.A.direct(x) - self.b)
        

        
        
if __name__ == '__main__':
    
    
    
    print("Check LeastSquares")
    from ccpi.framework import ImageGeometry
    from ccpi.optimisation.operators import Identity
    import numpy
    
    ig = ImageGeometry(4,5)
    Aop = 2*Identity(ig)
    data = ig.allocate('random')
    
    alpha = 0.4
    f = LeastSquares(Aop, data, c = alpha)
    x = ig.allocate('random')
    
    res1 = f(x)
    res2 = alpha * ((Aop.direct(x) - data)**2).sum()
    numpy.testing.assert_almost_equal(res1, res2) 
    print("Checking call .... OK ")
    
    res1 = f.gradient(x)
    res2 = 2*alpha * Aop.adjoint(Aop.direct(x) - data)
    numpy.testing.assert_almost_equal(res1.as_array(), res2.as_array(), decimal=4) 
    print("Checking gradient .... OK ")
    
    
    
    
    
    
    

    
    ####################################################
    ############ CGLS ##################################
    ###################################################
#    from ccpi.framework import ImageGeometry, ImageData, \
#        AcquisitionGeometry, BlockDataContainer, AcquisitionData
#    
#    import numpy as np 
#    import numpy                          
#    import matplotlib.pyplot as plt
#    
#    from ccpi.optimisation.algorithms import CGLS
#    from ccpi.optimisation.operators import BlockOperator, Gradient, Identity
#           
#    import tomophantom
#    from tomophantom import TomoP2D
#    from ccpi.astra.operators import AstraProjectorSimple 
#    import os
#    
#    
#    # Load  Shepp-Logan phantom 
#    model = 1 # select a model number from the library
#    N = 64 # set dimension of the phantom
#    path = os.path.dirname(tomophantom.__file__)
#    path_library2D = os.path.join(path, "Phantom2DLibrary.dat")
#    phantom_2D = TomoP2D.Model(model, N, path_library2D)
#    
#    ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N)
#    data = ImageData(phantom_2D)
#    
#    detectors =  N
#    angles = np.linspace(0, np.pi, 180, dtype=np.float32)
#    
#    ag = AcquisitionGeometry('parallel','2D', angles, detectors)
#    
#    device = input('Available device: GPU==1 / CPU==0 ')
#    
#    if device =='1':
#        dev = 'gpu'
#    else:
#        dev = 'cpu'
#    
#    Aop = AstraProjectorSimple(ig, ag, dev)    
#    sin = Aop.direct(data)
#    
#    np.random.seed(10)
#    noisy_data = AcquisitionData( sin.as_array() + np.random.normal(0,1,ag.shape))
#    
#    # Show Ground Truth and Noisy Data
#    plt.figure(figsize=(10,10))
#    plt.subplot(2,1,1)
#    plt.imshow(data.as_array())
#    plt.title('Ground Truth')
#    plt.colorbar()
#    plt.subplot(2,1,2)
#    plt.imshow(noisy_data.as_array())
#    plt.title('Noisy Data')
#    plt.colorbar()
#    plt.show()
#    
#    # Setup and run the simple CGLS algorithm  
#    x_init = ig.allocate()  
#    
#    cgls1 = CGLS(x_init = x_init, operator = Aop, data = sin)
#    cgls1.max_iteration = 20
#    cgls1.update_objective_interval = 5
#    cgls1.run(20, verbose = True)
#    
#    plt.imshow(cgls1.get_output().as_array())
#    plt.show()       
    
    
    
    
#    print("Check LeastSquares with FunctionOperatorComposition")
#    
#    from ccpi.framework import ImageGeometry
#    from ccpi.optimisation.functions import FunctionOperatorComposition, L2NormSquared
#    from ccpi.optimisation.operators import Identity, DiagonalOperator, CompositionOperator
#    import numpy
#    
#    ig = ImageGeometry(4,5)
#    Aop = 2 * Identity(ig)
#    data = ig.allocate('random')
#    
#    x = ig.allocate('random')
#    alpha = 5
#    tmp = alpha * L2NormSquared(b=data)
#    f1 = FunctionOperatorComposition(tmp, Aop)
#    f2 = LeastSquares(Aop, data, c = alpha)
#    res1 = f1(x)
#    res2 = f2(x)
#    
#    numpy.testing.assert_almost_equal(res1, res2)   
#    
#    res1 = f1.gradient(x)
#    res2 = f2.gradient(x)
#    numpy.testing.assert_array_almost_equal(res1.as_array(), res2.as_array())
#    
#    print("Check WeightLeastSquares")
#    
#    weight = ig.allocate('random', seed = 10)
#    tmp1 = weight.sqrt() * (Aop.direct(x) - data)
#    res1 = alpha *  tmp1.dot(tmp1)    
#    
#    f = alpha * WeightedLeastSquares(Aop, data, c=alpha, weight = weight)    
#    res2 = f(x)    
#    numpy.testing.assert_almost_equal(res1, res2) 
#    
#    #### test
#    D = DiagonalOperator(weight.sqrt())
#    f = L2NormSquared()
#    res1 = f(D.direct(x))
#    
#    
##    from ccpi.framework 
#    
    
    
    
    

    


    
    
    
    
    
    
    
    

    
    
    
    
        

    
    
    
