# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 12:36:40 2019

@author: ofn77899
"""
#from ccpi.optimisation.ops import Operator
import numpy
from numbers import Number
class Operator(object):
    '''Operator that maps from a space X -> Y'''
    def is_linear(self):
        '''Returns if the operator is linear'''
        return False
    def direct(self,x, out=None):
        raise NotImplementedError
    def size(self):
        # To be defined for specific class
        raise NotImplementedError
    def norm(self):
        raise NotImplementedError
    def allocate_direct(self):
        '''Allocates memory on the Y space'''
        raise NotImplementedError
    def allocate_adjoint(self):
        '''Allocates memory on the X space'''
        raise NotImplementedError
    def range_dim(self):
        raise NotImplementedError
    def domain_dim(self):
        raise NotImplementedError

class LinearOperator(Operator):
    '''Operator that maps from a space X -> Y'''
    def is_linear(self):
        '''Returns if the operator is linear'''
        return True
    def adjoint(self,x, out=None):
        raise NotImplementedError
        
class CompositeDataContainer(object):
    '''Class to hold a composite operator'''
    def __init__(self, *args):
        self.containers = args
        self.index = 0
    def __iter__(self):
        return self
    def next(self):
        '''python2 backwards compatibility'''
        return self.__next__()
    def __next__(self):
        try:
            out = self[self.index]
        except IndexError as ie:
            raise StopIteration()
        self.index+=1
        return out
    
    def is_compatible(self, other):
        '''basic check if the size of the 2 objects fit'''
        if isinstance(other, Number):
            return True   
        elif isinstance(other, list) or isinstance(other, numpy.ndarray):
            # TODO look elements should be numbers
            for ot in other:
                if not isinstance(ot, Number):
                    raise ValueError('List/ numpy array can only contain numbers')
            return len(self.containers) == len(other)
        return len(self.containers) == len(other.containers)
    def __getitem__(self, index):
        return self.containers[index]
                
    def add(self, other, out=None, *args, **kwargs):
        assert self.is_compatible(other)
        if isinstance(other, Number):
            return type(self)(*[ el.add(other, out, *args, **kwargs) for el in self.containers])
        elif isinstance(other, list):
            return type(self)(*[ el.add(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other)])
        return type(self)(*[ el.add(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other.containers)])
        
    def subtract(self, other, out=None , *args, **kwargs):
        assert self.is_compatible(other)
        if isinstance(other, Number):
            return type(self)(*[ el.subtract(other, out, *args, **kwargs) for el in self.containers])
        elif isinstance(other, list):
            return type(self)(*[ el.subtract(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other)])
        return type(self)(*[ el.subtract(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other.containers)])

    def multiply(self, other , out=None, *args, **kwargs):
        assert self.is_compatible(other)
        if isinstance(other, Number):
            return type(self)(*[ el.multiply(other, out, *args, **kwargs) for el in self.containers])
        elif isinstance(other, list):
            return type(self)(*[ el.multiply(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other)])
        return type(self)(*[ el.multiply(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other.containers)])
    
    def divide(self, other , out=None ,*args, **kwargs):
        assert self.is_compatible(other)
        if isinstance(other, Number):
            return type(self)(*[ el.divide(other, out, *args, **kwargs) for el in self.containers])
        elif isinstance(other, list):
            return type(self)(*[ el.divide(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other)])
        return type(self)(*[ el.divide(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other.containers)])
    
    def power(self, other , out=None, *args, **kwargs):
        assert self.is_compatible(other)
        if isinstance(other, Number):
            return type(self)(*[ el.power(other, out, *args, **kwargs) for el in self.containers])
        elif isinstance(other, list):
            return type(self)(*[ el.power(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other)])
        return type(self)(*[ el.power(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other.containers)])
    
    def maximum(self,other, out=None, *args, **kwargs):
        assert self.is_compatible(other)
        if isinstance(other, Number):
            return type(self)(*[ el.maximum(other, out, *args, **kwargs) for el in self.containers])
        elif isinstance(other, list):
            return type(self)(*[ el.maximum(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other)])
        return type(self)(*[ el.maximum(ot, out, *args, **kwargs) for el,ot in zip(self.containers,other.containers)])
    
    ## unary operations    
    def abs(self, out=None, *args,  **kwargs):
        return type(self)(*[ el.abs(out, *args, **kwargs) for el in self.containers]) 
    def sign(self, out=None, *args,  **kwargs):
        return type(self)(*[ el.sign(out, *args, **kwargs) for el in self.containers])
    def sqrt(self, out=None, *args,  **kwargs):
        return type(self)(*[ el.sqrt(out, *args, **kwargs) for el in self.containers])
    
    ## reductions
    def sum(self, out=None, *args, **kwargs):
        return [ el.sum(*args, **kwargs) for el in self.containers]
    
    def copy(self):
        '''alias of clone'''    
        return self.clone()
    def clone(self):
        return type(self)(*[el.copy() for el in self.containers])
    
    def __add__(self, other):
        return self.add( other )
    # __radd__
    
    def __sub__(self, other):
        return self.subtract( other )
    # __rsub__
    
    def __mul__(self, other):
        return self.multiply(other)
    # __rmul__
    
    def __div__(self, other):
        return self.divide(other)
    # __rdiv__
    def __truediv__(self, other):
        return self.divide(other)
    
    def __pow__(self, other):
        return self.power(other)
    # reverse operand
    def __radd__(self, other):
        return self + other
    # __radd__
    
    def __rsub__(self, other):
        return (-1 * self) + other
    # __rsub__
    
    def __rmul__(self, other):
        return self * other
    # __rmul__
    
    def __rdiv__(self, other):
        print ("call __rdiv__")
        return pow(self / other, -1)
    # __rdiv__
    def __rtruediv__(self, other):
        return self.__rdiv__(other)
    
    def __rpow__(self, other):
        return other.power(self)
    
    def __iadd__(self, other):
        if isinstance (other, CompositeDataContainer):
            for el,ot in zip(self.containers, other.containers):
                el += ot
        elif isinstance(other, Number):
            for el in self.containers:
                el += other
        elif isinstance(other, list) or isinstance(other, numpy.ndarray):
            assert self.is_compatible(other)
            for el,ot in zip(self.containers, other):
                el += ot
        return self
    # __radd__
    
    def __isub__(self, other):
        if isinstance (other, CompositeDataContainer):
            for el,ot in zip(self.containers, other.containers):
                el -= ot
        elif isinstance(other, Number):
            for el in self.containers:
                el -= other
        elif isinstance(other, list) or isinstance(other, numpy.ndarray):
            assert self.is_compatible(other)
            for el,ot in zip(self.containers, other):
                el -= ot
        return self
    # __rsub__
    
    def __imul__(self, other):
        if isinstance (other, CompositeDataContainer):
            for el,ot in zip(self.containers, other.containers):
                el *= ot
        elif isinstance(other, Number):
            for el in self.containers:
                el *= other
        elif isinstance(other, list) or isinstance(other, numpy.ndarray):
            assert self.is_compatible(other)
            for el,ot in zip(self.containers, other):
                el *= ot
        return self
    # __rmul__
    
    def __idiv__(self, other):
        if isinstance (other, CompositeDataContainer):
            for i,el,ot in enumerate(zip(self.containers, other.containers)):
                print ('__idiv__', i, el.as_array()[0][0][0], ot.as_array()[0][0][0])
                el /= ot
                print ("fatto", el.as_array()[0][0][0])
        elif isinstance(other, Number):
            for el in self.containers:
                print ("prima", el)
                print ('__idiv__', el.as_array()[0][0][0], other)
                el /= other
                print ("fatto", el.as_array()[0][0][0])
        elif isinstance(other, list) or isinstance(other, numpy.ndarray):
            assert self.is_compatible(other)
            for el,ot in zip(self.containers, other):
                print ('__idiv__', el.as_array()[0][0][0], ot[0][0][0])
                el /= ot
                print ("fatto", el.as_array()[0][0][0])
        return self
    # __rdiv__
    def __itruediv__(self, other):
        return self.__idiv__(other)

import time
from ccpi.optimisation.funcs import ZeroFun

class Algorithm(object):
    '''Base class for iterative algorithms

      provides the minimal infrastructure.
      Algorithms are iterables so can be easily run in a for loop. They will
      stop as soon as the stop cryterion is met.
      The user is required to implement the set_up, __init__, update and
      should_stop and update_objective methods
   '''

    def __init__(self):
        self.iteration = 0
        self.stop_cryterion = 'max_iter'
        self.__max_iteration = 0
        self.__loss = []
        self.memopt = False
        self.timing = []
    def set_up(self, *args, **kwargs):
        raise NotImplementedError()
    def update(self):
        raise NotImplementedError()
    
    def should_stop(self):
        '''stopping cryterion'''
        raise NotImplementedError()
    
    def __iter__(self):
        return self
    def next(self):
        '''python2 backwards compatibility'''
        return self.__next__()
    def __next__(self):
        if self.should_stop():
            raise StopIteration()
        else:
            time0 = time.time()
            self.update()
            self.timing.append( time.time() - time0 )
            self.update_objective()
            self.iteration += 1
    def get_output(self):
        '''Returns the solution found'''
        return self.x
    def get_current_loss(self):
        '''Returns the current value of the loss function'''
        return self.__loss[-1]
    def update_objective(self):
        raise NotImplementedError()
    @property
    def loss(self):
        return self.__loss
    @property
    def max_iteration(self):
        return self.__max_iteration
    @max_iteration.setter
    def max_iteration(self, value):
        assert isinstance(value, int)
        self.__max_iteration = value
    
class GradientDescent(Algorithm):
    '''Implementation of a simple Gradient Descent algorithm
    '''

    def __init__(self, **kwargs):
        '''initialisation can be done at creation time if all 
        proper variables are passed or later with set_up'''
        super(GradientDescent, self).__init__()
        self.x = None
        self.rate = 0
        self.objective_function = None
        self.regulariser = None
        args = ['x_init', 'objective_function', 'rate']
        for k,v in kwargs.items():
            if k in args:
                args.pop(args.index(k))
        if len(args) == 0:
            return self.set_up(x_init=kwargs['x_init'],
                               objective_function=kwargs['objective_function'],
                               rate=kwargs['rate'])
    
    def should_stop(self):
        '''stopping cryterion, currently only based on number of iterations'''
        return self.iteration >= self.max_iteration
    
    def set_up(self, x_init, objective_function, rate):
        '''initialisation of the algorithm'''
        self.x = x_init.copy()
        if self.memopt:
            self.x_update = x_init.copy()
        self.objective_function = objective_function
        self.rate = rate
        self.loss.append(objective_function(x_init))
        
    def update(self):
        '''Single iteration'''
        if self.memopt:
            self.objective_function.gradient(self.x, out=self.x_update)
            self.x_update *= -self.rate
            self.x += self.x_update
        else:
            self.x += -self.rate * self.objective_function.grad(self.x)

    def update_objective(self):
        self.loss.append(self.objective_function(self.x))

    
class CompositeOperator(Operator):
    '''Class to hold a composite operator'''
    def __init__(self, *args):
        self.operators = args
        
    def norm(self):
        return [op.norm() for op in self.operators]
    
    def direct(self, x, out=None):
        return CompositeDataContainer(*[op.direct(X) for op,X in zip(self.operators, x)])
    
    def adjoint(self, x, out=None):
        return CompositeDataContainer(*[op.adjoint(X) for op,X in zip(self.operators, x)])
    
    
if __name__ == '__main__':
    #from ccpi.optimisation.Algorithms import GradientDescent
    from ccpi.plugins.ops import CCPiProjectorSimple
    from ccpi.optimisation.ops import TomoIdentity, PowerMethodNonsquare
    from ccpi.optimisation.funcs import Norm2sq, Norm1
    from ccpi.framework import ImageGeometry, ImageData, AcquisitionGeometry
    import matplotlib.pyplot as plt

    ig0 = ImageGeometry(2,3,4)
    ig1 = ImageGeometry(12,42,55,32)
    
    data0 = ImageData(geometry=ig0)
    data1 = ImageData(geometry=ig1) + 1
    
    data2 = ImageData(geometry=ig0) + 2
    data3 = ImageData(geometry=ig1) + 3
    
    cp0 = CompositeDataContainer(data0,data1)
    cp1 = CompositeDataContainer(data2,data3)
#    
    a = [ (el, ot) for el,ot in zip(cp0.containers,cp1.containers)]
    print  (a[0][0].shape)
    #cp2 = CompositeDataContainer(*a)
    cp2 = cp0.add(cp1)
    assert (cp2[0].as_array()[0][0][0] == 2.)
    assert (cp2[1].as_array()[0][0][0] == 4.)
    
    cp2 = cp0 + cp1 
    assert (cp2[0].as_array()[0][0][0] == 2.)
    assert (cp2[1].as_array()[0][0][0] == 4.)
    cp2 = cp0 + 1 
    numpy.testing.assert_almost_equal(cp2[0].as_array()[0][0][0] , 1. , decimal=5)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0] , 2., decimal = 5)
    cp2 = cp0 + [1 ,2]
    numpy.testing.assert_almost_equal(cp2[0].as_array()[0][0][0] , 1. , decimal=5)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0] , 3., decimal = 5)
    cp2 += cp1
    numpy.testing.assert_almost_equal(cp2[0].as_array()[0][0][0] , +3. , decimal=5)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0] , +6., decimal = 5)
    
    cp2 += 1
    numpy.testing.assert_almost_equal(cp2[0].as_array()[0][0][0] , +4. , decimal=5)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0] , +7., decimal = 5)
    
    cp2 += [-2,-1]
    numpy.testing.assert_almost_equal(cp2[0].as_array()[0][0][0] , 2. , decimal=5)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0] , 6., decimal = 5)
    
    
    cp2 = cp0.subtract(cp1)
    assert (cp2[0].as_array()[0][0][0] == -2.)
    assert (cp2[1].as_array()[0][0][0] == -2.)
    cp2 = cp0 - cp1
    assert (cp2[0].as_array()[0][0][0] == -2.)
    assert (cp2[1].as_array()[0][0][0] == -2.)
    
    cp2 = cp0 - 1 
    numpy.testing.assert_almost_equal(cp2[0].as_array()[0][0][0] , -1. , decimal=5)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0] , 0, decimal = 5)
    cp2 = cp0 - [1 ,2]
    numpy.testing.assert_almost_equal(cp2[0].as_array()[0][0][0] , -1. , decimal=5)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0] , -1., decimal = 5)
    
    cp2 -= cp1
    numpy.testing.assert_almost_equal(cp2[0].as_array()[0][0][0] , -3. , decimal=5)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0] , -4., decimal = 5)
    
    cp2 -= 1
    numpy.testing.assert_almost_equal(cp2[0].as_array()[0][0][0] , -4. , decimal=5)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0] , -5., decimal = 5)
    
    cp2 -= [-2,-1]
    numpy.testing.assert_almost_equal(cp2[0].as_array()[0][0][0] , -2. , decimal=5)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0] , -4., decimal = 5)
    
    
    cp2 = cp0.multiply(cp1)
    assert (cp2[0].as_array()[0][0][0] == 0.)
    assert (cp2[1].as_array()[0][0][0] == 3.)
    cp2 = cp0 * cp1
    assert (cp2[0].as_array()[0][0][0] == 0.)
    assert (cp2[1].as_array()[0][0][0] == 3.)
    
    cp2 = cp0 * 2 
    numpy.testing.assert_almost_equal(cp2[0].as_array()[0][0][0] , 0. , decimal=5)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0] , 2, decimal = 5)
    cp2 = cp0 * [3 ,2]
    numpy.testing.assert_almost_equal(cp2[0].as_array()[0][0][0] , 0. , decimal=5)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0] , 2., decimal = 5)
    
    cp2 *= cp1
    numpy.testing.assert_almost_equal(cp2[0].as_array()[0][0][0] , 0 , decimal=5)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0] , +6., decimal = 5)
    
    cp2 *= 1
    numpy.testing.assert_almost_equal(cp2[0].as_array()[0][0][0] , 0. , decimal=5)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0] , +6., decimal = 5)
    
    cp2 *= [-2,-1]
    numpy.testing.assert_almost_equal(cp2[0].as_array()[0][0][0] , 0. , decimal=5)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0] , -6., decimal = 5)
    
    
    cp2 = cp0.divide(cp1)
    assert (cp2[0].as_array()[0][0][0] == 0.)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0], 1./3., decimal=4)
    cp2 = cp0/cp1
    assert (cp2[0].as_array()[0][0][0] == 0.)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0], 1./3., decimal=4)
    
    cp2 = cp0 / 2 
    numpy.testing.assert_almost_equal(cp2[0].as_array()[0][0][0] , 0. , decimal=5)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0] , 0.5, decimal = 5)
    cp2 = cp0 / [3 ,2]
    numpy.testing.assert_almost_equal(cp2[0].as_array()[0][0][0] , 0. , decimal=5)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0] , 0.5, decimal = 5)
    
    cp2 += 1
    print ("cp2" , cp2[0].as_array()[0][0][0],cp2[1].as_array()[0][0][0])
    print ("cp1" , cp1[0].as_array()[0][0][0],cp1[1].as_array()[0][0][0])
    #cp2 /= cp1
    # TODO fix inplace division
    cp2 /= 0.5
    print (cp2[0].as_array()[0][0][0],cp2[1].as_array()[0][0][0])
    
    numpy.testing.assert_almost_equal(cp2[0].as_array()[0][0][0] , 1./0.5 , decimal=5)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0] , .2, decimal = 5)
    
    cp2 /= 1
    numpy.testing.assert_almost_equal(cp2[0].as_array()[0][0][0] , 0. , decimal=5)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0] , 0.5/3., decimal = 5)
    
    cp2 /= [-2,-1]
    numpy.testing.assert_almost_equal(cp2[0].as_array()[0][0][0] , 0. , decimal=5)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0] , -0.5/3., decimal = 5)
    
    
    cp2 = cp0.power(cp1)
    assert (cp2[0].as_array()[0][0][0] == 0.)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0], 1., decimal=4)
    cp2 = cp0**cp1
    assert (cp2[0].as_array()[0][0][0] == 0.)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0], 1., decimal=4)
    
    cp2 = cp0 ** 2 
    numpy.testing.assert_almost_equal(cp2[0].as_array()[0][0][0] , 0., decimal=5)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0] , 1., decimal = 5)
    
    cp2 = cp0.maximum(cp1)
    assert (cp2[0].as_array()[0][0][0] == cp1[0].as_array()[0][0][0])
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0], cp2[1].as_array()[0][0][0], decimal=4)
    
    
    cp2 = cp0.abs()
    numpy.testing.assert_almost_equal(cp2[0].as_array()[0][0][0], 0., decimal=4)
    numpy.testing.assert_almost_equal(cp2[1].as_array()[0][0][0], 1., decimal=4)
    
    cp2 = cp0.subtract(cp1)
    s = cp2.sign()
    numpy.testing.assert_almost_equal(s[0].as_array()[0][0][0], -1., decimal=4)
    numpy.testing.assert_almost_equal(s[1].as_array()[0][0][0], -1., decimal=4)
    
    cp2 = cp0.add(cp1)
    s = cp2.sqrt()
    numpy.testing.assert_almost_equal(s[0].as_array()[0][0][0], numpy.sqrt(2), decimal=4)
    numpy.testing.assert_almost_equal(s[1].as_array()[0][0][0], numpy.sqrt(4), decimal=4)
    
    s = cp0.sum()
    numpy.testing.assert_almost_equal(s[0], 0, decimal=4)
    s0 = 1
    s1 = 1
    for i in cp0[0].shape:
        s0 *= i
    for i in cp0[1].shape:
        s1 *= i
        
    numpy.testing.assert_almost_equal(s[1], cp0[0].as_array()[0][0][0]*s0 +cp0[1].as_array()[0][0][0]*s1, decimal=4)
    
    # Set up phantom size N x N x vert by creating ImageGeometry, initialising the 
    # ImageData object with this geometry and empty array and finally put some
    # data into its array, and display one slice as image.
    
    # Image parameters
    N = 128
    vert = 4
    
    # Set up image geometry
    ig = ImageGeometry(voxel_num_x=N,
                       voxel_num_y=N, 
                       voxel_num_z=vert)
    
    # Set up empty image data
    Phantom = ImageData(geometry=ig,
                        dimension_labels=['horizontal_x',
                                          'horizontal_y',
                                          'vertical'])
    
    # Populate image data by looping over and filling slices
    i = 0
    while i < vert:
        if vert > 1:
            x = Phantom.subset(vertical=i).array
        else:
            x = Phantom.array
        x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
        x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 0.98
        if vert > 1 :
            Phantom.fill(x, vertical=i)
        i += 1
    
    # Display slice of phantom
    if vert > 1:
        plt.imshow(Phantom.subset(vertical=0).as_array())
    else:
        plt.imshow(Phantom.as_array())
    plt.show()
    
    
    # Set up AcquisitionGeometry object to hold the parameters of the measurement
    # setup geometry: # Number of angles, the actual angles from 0 to 
    # pi for parallel beam, set the width of a detector 
    # pixel relative to an object pixe and the number of detector pixels.
    angles_num = 20
    det_w = 1.0
    det_num = N
    
    angles = numpy.linspace(0,numpy.pi,angles_num,endpoint=False,dtype=numpy.float32)*\
                 180/numpy.pi
    
    # Inputs: Geometry, 2D or 3D, angles, horz detector pixel count, 
    #         horz detector pixel size, vert detector pixel count, 
    #         vert detector pixel size.
    ag = AcquisitionGeometry('parallel',
                             '3D',
                             angles,
                             N, 
                             det_w,
                             vert,
                             det_w)
    
    # Set up Operator object combining the ImageGeometry and AcquisitionGeometry
    # wrapping calls to CCPi projector.
    A = CCPiProjectorSimple(ig, ag)
    
    # Forward and backprojection are available as methods direct and adjoint. Here 
    # generate test data b and do simple backprojection to obtain z. Display all
    #  data slices as images, and a single backprojected slice.
    b = A.direct(Phantom)
    z = A.adjoint(b)
    
    for i in range(b.get_dimension_size('vertical')):
        plt.imshow(b.subset(vertical=i).array)
        plt.show()
    
    plt.imshow(z.subset(vertical=0).array)
    plt.title('Backprojected data')
    plt.show()
    
    # Using the test data b, different reconstruction methods can now be set up as
    # demonstrated in the rest of this file. In general all methods need an initial 
    # guess and some algorithm options to be set. Note that 100 iterations for 
    # some of the methods is a very low number and 1000 or 10000 iterations may be
    # needed if one wants to obtain a converged solution.
    x_init = ImageData(geometry=ig, 
                       dimension_labels=['horizontal_x','horizontal_y','vertical'])
    x_init1 = ImageData(geometry=ig, 
                       dimension_labels=['horizontal_x','horizontal_y','vertical'])
    X_init = CompositeDataContainer(x_init, x_init1)
    B = CompositeDataContainer(b, 
                               ImageData(geometry=ig, dimension_labels=['horizontal_x','horizontal_y','vertical']))
    
    # setup a tomo identity
    I = TomoIdentity(geometry=ig)
    
    # composite operator
    K = CompositeOperator(A, I)
    
    out = K.direct(X_init)
    
#    f = Norm2sq(K,B)
#    f.L = 0.001
#    
#    gd = GradientDescent()
#    gd.set_up(X_init, f, 0.001 )
#    gd.max_iteration = 2
#    
#    for _ in gd:
#        pass
    
        
    