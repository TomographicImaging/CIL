# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 12:36:40 2019

@author: ofn77899
"""
#from ccpi.optimisation.ops import Operator
import numpy
from numbers import Number
import functools
from ccpi.framework import AcquisitionData, ImageData, BlockDataContainer
from ccpi.optimisation.operators import Operator, LinearOperator


       
class BlockOperator(Operator):
    '''Class to hold a block operator'''
    def __init__(self, *args, shape=None):
        self.operators = args
        if shape is None:
            shape = (len(args),1)
        self.shape = shape
        n_elements = functools.reduce(lambda x,y: x*y, shape, 1)
        if len(args) != n_elements:
            raise ValueError(
                    'Dimension and size do not match: expected {} got {}'
                    .format(n_elements,len(args)))
    def get_item(self, row, col):
        if row > self.shape[0]:
            raise ValueError('Requested row {} > max {}'.format(row, self.shape[0]))
        if col > self.shape[1]:
            raise ValueError('Requested col {} > max {}'.format(col, self.shape[1]))
        
        index = row*self.shape[1]+col
        return self.operators[index]
    
    def norm(self):
        norm = [op.norm() for op in self.operators]
        b = []
        for i in range(self.shape[0]):
            b.append([])
            for j in range(self.shape[1]):
                b[-1].append(norm[i*self.shape[1]+j])
        return numpy.asarray(b)
    
    def direct(self, x, out=None):
        shape = self.get_output_shape(x.shape)
        res = []
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                if col == 0:
                    prod = self.get_item(row,col).direct(x.get_item(col))
                else:
                    prod += self.get_item(row,col).direct(x.get_item(col))
            res.append(prod)
        return BlockDataContainer(*res, shape=shape)
    
    def adjoint(self, x, out=None):
        shape = self.get_output_shape(x.shape, adjoint=True)
        res = []
        for row in range(self.shape[1]):
            for col in range(self.shape[0]):
                if col == 0:
                    prod = self.get_item(row,col).adjoint(x.get_item(col))
                else:
                    prod += self.get_item(row,col).adjoint(x.get_item(col))
            res.append(prod)
        return BlockDataContainer(*res, shape=shape)
    
    def get_output_shape(self, xshape, adjoint=False):
        sshape = self.shape[1]
        oshape = self.shape[0]
        if adjoint:
            sshape = self.shape[0]
            oshape = self.shape[1]
        if sshape != xshape[0]:
            raise ValueError('Incompatible shapes {} {}'.format(self.shape, xshape))
        return (oshape, xshape[-1])
    
'''    
    def direct(self, x, out=None):
        
        out = [None]*self.dimension[0]
        for i in range(self.dimension[0]):
            z1 = ImageData(np.zeros(self.compMat[i][0].range_dim()))
            for j in range(self.dimension[1]):
                z1 += self.compMat[i][j].direct(x[j])
            out[i] = z1    
                                
        return out          
        
    
    def adjoint(self, x, out=None):        
        
        out = [None]*self.dimension[1]
        for i in range(self.dimension[1]):
            z2 = ImageData(np.zeros(self.compMat[0][i].domain_dim()))
            for j in range(self.dimension[0]):
                z2 += self.compMat[j][i].adjoint(x[j])
            out[i] = z2
'''
from ccpi.optimisation.algorithms import CGLS

    
if __name__ == '__main__':
    #from ccpi.optimisation.Algorithms import GradientDescent
    from ccpi.plugins.ops import CCPiProjectorSimple
    from ccpi.optimisation.ops import PowerMethodNonsquare
    from ccpi.optimisation.ops import TomoIdentity
    from ccpi.optimisation.funcs import Norm2sq, Norm1
    from ccpi.framework import ImageGeometry, AcquisitionGeometry
    from ccpi.optimisation.Algorithms import GradientDescent
    #from ccpi.optimisation.Algorithms import CGLS
    import matplotlib.pyplot as plt

    ig0 = ImageGeometry(2,3,4)
    ig1 = ImageGeometry(12,42,55,32)
    
    data0 = ImageData(geometry=ig0)
    data1 = ImageData(geometry=ig1) + 1
    
    data2 = ImageData(geometry=ig0) + 2
    data3 = ImageData(geometry=ig1) + 3
    
    cp0 = BlockDataContainer(data0,data1)
    cp1 = BlockDataContainer(data2,data3)
#    
    a = [ (el, ot) for el,ot in zip(cp0.containers,cp1.containers)]
    print  (a[0][0].shape)
    #cp2 = BlockDataContainer(*a)
    cp2 = cp0.add(cp1)
    assert (cp2.get_item(0,0).as_array()[0][0][0] == 2.)
    assert (cp2.get_item(1,0).as_array()[0][0][0] == 4.)
    
    cp2 = cp0 + cp1 
    assert (cp2.get_item(0,0).as_array()[0][0][0] == 2.)
    assert (cp2.get_item(1,0).as_array()[0][0][0] == 4.)
    cp2 = cp0 + 1 
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 1. , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 2., decimal = 5)
    cp2 = cp0 + [1 ,2]
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 1. , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 3., decimal = 5)
    cp2 += cp1
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , +3. , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , +6., decimal = 5)
    
    cp2 += 1
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , +4. , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , +7., decimal = 5)
    
    cp2 += [-2,-1]
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 2. , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 6., decimal = 5)
    
    
    cp2 = cp0.subtract(cp1)
    assert (cp2.get_item(0,0).as_array()[0][0][0] == -2.)
    assert (cp2.get_item(1,0).as_array()[0][0][0] == -2.)
    cp2 = cp0 - cp1
    assert (cp2.get_item(0,0).as_array()[0][0][0] == -2.)
    assert (cp2.get_item(1,0).as_array()[0][0][0] == -2.)
    
    cp2 = cp0 - 1 
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , -1. , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 0, decimal = 5)
    cp2 = cp0 - [1 ,2]
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , -1. , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , -1., decimal = 5)
    
    cp2 -= cp1
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , -3. , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , -4., decimal = 5)
    
    cp2 -= 1
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , -4. , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , -5., decimal = 5)
    
    cp2 -= [-2,-1]
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , -2. , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , -4., decimal = 5)
    
    
    cp2 = cp0.multiply(cp1)
    assert (cp2.get_item(0,0).as_array()[0][0][0] == 0.)
    assert (cp2.get_item(1,0).as_array()[0][0][0] == 3.)
    cp2 = cp0 * cp1
    assert (cp2.get_item(0,0).as_array()[0][0][0] == 0.)
    assert (cp2.get_item(1,0).as_array()[0][0][0] == 3.)
    
    cp2 = cp0 * 2 
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0. , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 2, decimal = 5)
    cp2 = 2 * cp0  
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0. , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 2, decimal = 5)
    cp2 = cp0 * [3 ,2]
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0. , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 2., decimal = 5)
    cp2 = cp0 * numpy.asarray([3 ,2])
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0. , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 2., decimal = 5)
    
    cp2 = [3,2] * cp0 
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0. , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 2., decimal = 5)
    cp2 = numpy.asarray([3,2]) * cp0 
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0. , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 2., decimal = 5)
    cp2 = [3,2,3] * cp0 
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0. , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 2., decimal = 5)
    
    cp2 *= cp1
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0 , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , +6., decimal = 5)
    
    cp2 *= 1
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0. , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , +6., decimal = 5)
    
    cp2 *= [-2,-1]
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0. , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , -6., decimal = 5)
    
    
    cp2 = cp0.divide(cp1)
    assert (cp2.get_item(0,0).as_array()[0][0][0] == 0.)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0], 1./3., decimal=4)
    cp2 = cp0/cp1
    assert (cp2.get_item(0,0).as_array()[0][0][0] == 0.)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0], 1./3., decimal=4)
    
    cp2 = cp0 / 2 
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0. , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 0.5, decimal = 5)
    cp2 = cp0 / [3 ,2]
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0. , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 0.5, decimal = 5)
    cp2 = cp0 / numpy.asarray([3 ,2])
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0. , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 0.5, decimal = 5)
    cp3 = numpy.asarray([3 ,2]) / (cp0+1)
    numpy.testing.assert_almost_equal(cp3.get_item(0,0).as_array()[0][0][0] , 3. , decimal=5)
    numpy.testing.assert_almost_equal(cp3.get_item(1,0).as_array()[0][0][0] , 1, decimal = 5)
    
    cp2 += 1
    cp2 /= cp1
    # TODO fix inplace division
     
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 1./2 , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 1.5/3., decimal = 5)
    
    cp2 /= 1
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0.5 , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 0.5, decimal = 5)
    
    cp2 /= [-2,-1]
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , -0.5/2. , decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , -0.5, decimal = 5)
    ####
    
    cp2 = cp0.power(cp1)
    assert (cp2.get_item(0,0).as_array()[0][0][0] == 0.)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0], 1., decimal=4)
    cp2 = cp0**cp1
    assert (cp2.get_item(0,0).as_array()[0][0][0] == 0.)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0], 1., decimal=4)
    
    cp2 = cp0 ** 2 
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0] , 0., decimal=5)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0] , 1., decimal = 5)
    
    cp2 = cp0.maximum(cp1)
    assert (cp2.get_item(0,0).as_array()[0][0][0] == cp1.get_item(0,0).as_array()[0][0][0])
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0], cp2.get_item(1,0).as_array()[0][0][0], decimal=4)
    
    
    cp2 = cp0.abs()
    numpy.testing.assert_almost_equal(cp2.get_item(0,0).as_array()[0][0][0], 0., decimal=4)
    numpy.testing.assert_almost_equal(cp2.get_item(1,0).as_array()[0][0][0], 1., decimal=4)
    
    cp2 = cp0.subtract(cp1)
    s = cp2.sign()
    numpy.testing.assert_almost_equal(s.get_item(0,0).as_array()[0][0][0], -1., decimal=4)
    numpy.testing.assert_almost_equal(s.get_item(1,0).as_array()[0][0][0], -1., decimal=4)
    
    cp2 = cp0.add(cp1)
    s = cp2.sqrt()
    numpy.testing.assert_almost_equal(s.get_item(0,0).as_array()[0][0][0], numpy.sqrt(2), decimal=4)
    numpy.testing.assert_almost_equal(s.get_item(1,0).as_array()[0][0][0], numpy.sqrt(4), decimal=4)
    
    s = cp0.sum()
    numpy.testing.assert_almost_equal(s[0], 0, decimal=4)
    s0 = 1
    s1 = 1
    for i in cp0.get_item(0,0).shape:
        s0 *= i
    for i in cp0.get_item(1,0).shape:
        s1 *= i
        
    numpy.testing.assert_almost_equal(s[1], cp0.get_item(0,0).as_array()[0][0][0]*s0 +cp0.get_item(1,0).as_array()[0][0][0]*s1, decimal=4)
    
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
    Phantom += 0.05
    # Populate image data by looping over and filling slices
    i = 0
    while i < vert:
        if vert > 1:
            x = Phantom.subset(vertical=i).array
        else:
            x = Phantom.array
        x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 0.5
        x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 0.94
        if vert > 1 :
            Phantom.fill(x, vertical=i)
        i += 1
    
    
    perc = 0.02
    # Set up empty image data
    noise = ImageData(numpy.random.normal(loc = 0.04 ,
                             scale = perc , 
                             size = Phantom.shape), geometry=ig,
                        dimension_labels=['horizontal_x',
                                          'horizontal_y',
                                          'vertical'])
    Phantom += noise
    
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
    # generate test data b and some noise
    
    b = A.direct(Phantom)
    
    
    #z = A.adjoint(b)
    
    
    # Using the test data b, different reconstruction methods can now be set up as
    # demonstrated in the rest of this file. In general all methods need an initial 
    # guess and some algorithm options to be set. Note that 100 iterations for 
    # some of the methods is a very low number and 1000 or 10000 iterations may be
    # needed if one wants to obtain a converged solution.
    x_init = ImageData(geometry=ig, 
                       dimension_labels=['horizontal_x','horizontal_y','vertical'])
    X_init = BlockDataContainer(x_init)
    B = BlockDataContainer(b, 
                               ImageData(geometry=ig, dimension_labels=['horizontal_x','horizontal_y','vertical']))
    
    # setup a tomo identity
    Ibig = 1e5 * TomoIdentity(geometry=ig)
    Ismall = 1e-5 * TomoIdentity(geometry=ig)
    
    # composite operator
    Kbig = BlockOperator(A, Ibig, shape=(2,1))
    Ksmall = BlockOperator(A, Ismall, shape=(2,1))
    
    #out = K.direct(X_init)
    
    f = Norm2sq(Kbig,B)
    f.L = 0.00003
    
    fsmall = Norm2sq(Ksmall,B)
    f.L = 0.00003
    
    simplef = Norm2sq(A, b)
    simplef.L = 0.00003
    
    gd = GradientDescent( x_init=x_init, objective_function=simplef,
                         rate=simplef.L)
    gd.max_iteration = 10
    
    cg = CGLS()
    cg.set_up(X_init, Kbig, B )
    cg.max_iteration = 1
    
    cgsmall = CGLS()
    cgsmall.set_up(X_init, Ksmall, B )
    cgsmall.max_iteration = 1
    
    
    cgs = CGLS()
    cgs.set_up(x_init, A, b )
    cgs.max_iteration = 6
#    
    #out.__isub__(B)
    #out2 = K.adjoint(out)
    
    #(2.0*self.c)*self.A.adjoint( self.A.direct(x) - self.b )
    
    for _ in gd:
        print ("iteration {} {}".format(gd.iteration, gd.get_current_loss()))
    
    cg.run(10, lambda it,val: print ("iteration {} objective {}".format(it,val)))
    
    cgs.run(10, lambda it,val: print ("iteration {} objective {}".format(it,val)))
    
    cgsmall.run(10, lambda it,val: print ("iteration {} objective {}".format(it,val)))
    cgsmall.run(10, lambda it,val: print ("iteration {} objective {}".format(it,val)))
#    for _ in cg:
#        print ("iteration {} {}".format(cg.iteration, cg.get_current_loss()))
#    
#    fig = plt.figure()
#    plt.imshow(cg.get_output().get_item(0,0).subset(vertical=0).as_array())
#    plt.title('Composite CGLS')
#    plt.show()
#    
#    for _ in cgs:
#        print ("iteration {} {}".format(cgs.iteration, cgs.get_current_loss()))
#    
    fig = plt.figure()
    plt.subplot(1,5,1)
    plt.imshow(Phantom.subset(vertical=0).as_array())
    plt.title('Simulated Phantom')
    plt.subplot(1,5,2)
    plt.imshow(gd.get_output().subset(vertical=0).as_array())
    plt.title('Simple Gradient Descent')
    plt.subplot(1,5,3)
    plt.imshow(cgs.get_output().subset(vertical=0).as_array())
    plt.title('Simple CGLS')
    plt.subplot(1,5,4)
    plt.imshow(cg.get_output().get_item(0,0).subset(vertical=0).as_array())
    plt.title('Composite CGLS\nbig lambda')
    plt.subplot(1,5,5)
    plt.imshow(cgsmall.get_output().get_item(0,0).subset(vertical=0).as_array())
    plt.title('Composite CGLS\nsmall lambda')
    plt.show()