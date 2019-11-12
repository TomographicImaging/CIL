#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Imports
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ccpi.framework import ImageData, TestData, ImageGeometry, AcquisitionGeometry, AcquisitionData, BlockDataContainer

from ccpi.optimisation.functions import L2NormSquared, ZeroFunction, L1Norm, BlockFunction, MixedL21Norm, IndicatorBox, FunctionOperatorComposition
from ccpi.optimisation.operators import Gradient, BlockOperator
from ccpi.optimisation.algorithms import PDHG, SIRT, CGLS

from ccpi.astra.operators import AstraProjectorSimple, AstraProjector3DSimple
from ccpi.astra.processors import FBP, AstraForwardProjector, AstraBackProjector

import tomophantom
from tomophantom import TomoP2D
import os, sys, time

import matplotlib.pyplot as plt


import numpy as np

# from utilities import islicer, link_islicer, psnr, plotter2D
from ccpi.utilities.show_utilities import show

from ccpi.optimisation.algorithms import Algorithm, GradientDescent   
import numpy


from ccpi.optimisation.functions import Norm2Sq
from ccpi.utilities import plotter2D

# get_ipython().magic(u'matplotlib inline')


# In[4]:


model = 12 # select a model number from the library
N = 512 # set dimension of the phantom
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
A = AstraProjectorSimple(ig, ag, device = device)
data = A.direct(im_data)


# In[ ]:




class StochasticAlgorithm(Algorithm):
    def __init__(self, **kwargs):
        super(StochasticAlgorithm, self).__init__(**kwargs)
        self.epoch = 0
        self.number_of_subsets = kwargs.get('number_of_subsets', 1)
        self.current_subset_id = 0
        self.update_subset_interval = kwargs.get('update_subset_interval' , 1)
        
    def update_subset(self):
        if self.iteration % self.update_subset_interval == 0:
            # increment epoch
            self.epoch += 1
            
        self.current_subset_id += 1
        # this callback must be defined by the concrete implementation of the 
        # algorithm to link to the appropriate object dealing with subsets
        self.notify_new_subset(self.current_subset_id, self.number_of_subsets)
        
    def should_stop(self):
        '''default stopping cryterion: number of iterations
        
        The user can change this in concrete implementatition of iterative algorithms.'''
        return self.max_epoch_stop_cryterion()
    
    def max_epoch_stop_cryterion(self):
        '''default stop cryterion for iterative algorithm: max_iteration reached'''
        return self.epoch >= self.max_iteration
    def notify_new_subset(self, subset_id, number_of_subsets):
        raise NotImplemented('This callback must be implemented by the concrete algorithm')
    
    def __next__(self):
        super(StochasticAlgorithm, self).__next__()
        self.update_subset()
        
class StochasticGradientDescent(StochasticAlgorithm, GradientDescent):
    def __init__(self, **kwargs):
        super(StochasticGradientDescent, self).__init__(**kwargs)
        
    def notify_new_subset(self, subset_id, number_of_subsets):
        self.objective_function.notify_new_subset(subset_id, number_of_subsets)
        




f = Norm2Sq(A, data)



class AcquisitionGeometrySubsetGenerator(object):
    RANDOM='random'
    UNIFORM_SAMPLING='uniform'
    
    ### Changes in the Operator required to work as OS operator
    @staticmethod
    def generate_subset(ag, subset_id, number_of_subsets, method='random'):
        ags = ag.clone()
        angles = ags.angles
        if method == 'random':
            indices = AcquisitionGeometrySubsetGenerator.random_indices(angles, subset_id, number_of_subsets)
        else:
            raise ValueError('Can only do '.format('random'))
        ags.angles = ags.angles[indices]
        return ags , indices
    @staticmethod
    def random_indices(angles, subset_id, number_of_subsets):
        N = int(numpy.floor(float(len(angles))/float(number_of_subsets)))
        indices = numpy.asarray(range(len(angles)))
        numpy.random.shuffle(indices)
        indices = indices[:N]
        ret = numpy.asarray(numpy.zeros_like(angles), dtype=numpy.bool)
        for i,el in enumerate(indices):
            ret[el] = True
        return ret

class AstraSubsetProjectorSimple(AstraProjectorSimple):
    
    def __init__(self, geomv, geomp, device, **kwargs):
        kwargs = {'indices':None, 
                  'subset_acquisition_geometry':None,
                  'subset_id' : 0,
                  'number_of_subsets' : kwargs.get('number_of_subsets', 1)
                  }
        # This does not forward to its parent class :(
        super(AstraSubsetProjectorSimple, self).__init__(geomv, geomp, device)
        self.notify_new_subset(0, kwargs.get('number_of_subsets',1))
        
    def notify_new_subset(self, subset_id, number_of_subsets):
        # print ('AstraSubsetProjectorSimple notify_new_subset')
        # updates the sinogram geometry and updates the projectors
        self.subset_id = subset_id
        self.number_of_subsets = number_of_subsets

        ag , indices = AcquisitionGeometrySubsetGenerator.generate_subset(
            self.sinogram_geometry, 
            subset_id, 
            number_of_subsets,
            AcquisitionGeometrySubsetGenerator.RANDOM)

        self.indices = indices
        device = self.fp.device
        self.subset_acquisition_geometry = ag
        
        self.fp = AstraForwardProjector(volume_geometry=self.volume_geometry,
                                        sinogram_geometry=ag,
                                        proj_id = None,
                                        device=device)

        self.bp = AstraBackProjector(volume_geometry = self.volume_geometry,
                                        sinogram_geometry = ag,
                                        proj_id = None,
                                        device = device)
        self.subs = self.subset_acquisition_geometry.allocate(0)
        

    def direct(self, image_data, out=None):
        self.fp.set_input(image_data)
        ret = self.fp.get_output()
            
        if out is None:
            out = self.sinogram_geometry.allocate(0)
            #print (self.indices)
            out.as_array()[self.indices] = ret.as_array()[:]
            return out
        else:
            out.as_array()[self.indices] = ret.as_array()[:]
            

    def adjoint(self, acquisition_data, out=None):
        self.subs.fill(acquisition_data.as_array()[self.indices])
        self.bp.set_input(self.subs)
        
        if out is None:
            return self.bp.get_output()
        else:
            # out.as_array()[self.indices] = ret.as_array()[:]
            out += self.bp.get_output()
    
# In[ ]:


# Create projection operator using Astra-Toolbox. Available CPU/CPU
#%%
            
subs = 10
A_os = AstraSubsetProjectorSimple(ig, ag, device = 'gpu', number_of_subsets=10)
A_os.notify_new_subset(1, 10)
data_os = A_os.direct(im_data)
im_b = A_os.adjoint(data)

A_os.notify_new_subset(2, 10)
data_os_1 = A_os.direct(im_data)

im_b_1 = A_os.adjoint(data)

A_os1 = AstraSubsetProjectorSimple(ig, ag, device = 'gpu')
A_os1.notify_new_subset(1, 20)
data_os1 = A_os1.direct(im_data)

im_b = A_os1.adjoint(data)

bp = A_os.bp

im_back = A.adjoint(data)


#plotter2D([data, data_os1, data_os, data_os_1], 
#  titles=['No subsets', '1 subset' , 'subset {} / {} subsets'.format(1,subs),'subset {} / {} subsets'.format(2,subs) ],
#  cmap='viridis')

plotter2D([im_back, im_b, im_b_1], titles=['Full data', '20 subsets', '10 subsets'])

#%%
l2 = Norm2Sq(A=A, b=data)
gd = GradientDescent(x_init=im_data*0., objective_function=l2, rate=1e-4 , 
     update_objective_interval=10, max_iteration=100)
tgd0 = time.time()
gd.run()
tgd1 = time.time()
#%%
print ("######################################")

class StochasticNorm2Sq(Norm2Sq):
    def __init__(self, A, b, c=1.0, number_of_subsets=1):
        super(StochasticNorm2Sq, self).__init__(A, b, c)
#        self.number_of_subsets = number_of_subsets
#        self.notify_new_subset(0,number_of_subsets)
        
    def notify_new_subset(self, subset_id, number_of_subsets):
        self.A.notify_new_subset(subset_id, number_of_subsets)
        
#    
#    def gradient(self, x, out=None):
#        if out is not None:
#            #return 2.0*self.c*self.A.adjoint( self.A.direct(x) - self.b )
#            self.A.direct(x, out=self.range_tmp)
#            self.range_tmp.subtract(self.b , out=self.range_tmp)
#            self.A.adjoint(self.range_tmp, out=out)
#            #self.direct_placehold.multiply(2.0*self.c, out=out)
#            out.multiply (self.c * 2.0, out=out)
#        else:
#            return (2.0*self.c)*self.A.adjoint(self.A.direct(x) - self.b)

#    def __call__(self, x):
#        self.notify_new_subset(0, 1)
#        return super(StochasticNorm2Sq, self).__call__(x)

nsubs = 10
theA = AstraSubsetProjectorSimple(ig, ag, device = device, number_of_subsets=10)

theA.notify_new_subset(0,nsubs)
sl2 = StochasticNorm2Sq(A=theA,
                        b=data, number_of_subsets=nsubs)


sl2.memopt = True


rnd_data = im_data.copy()
grad = im_data * 0.
N = 10
rnd_data = im_data.copy()
numpy.random.shuffle(rnd_data.as_array())

solution = rnd_data *0.
rate = 1e-3
t0 = time.time()
for i in range (N):
    if i % 100==0:
        plotter2D([solution])
    sl2.A.direct(solution, out=sl2.range_tmp)
    #plotter2D([sl2.A.direct(rnd_data), sl2.range_tmp])
    sl2.range_tmp.subtract(sl2.b, out=sl2.range_tmp)
    #plotter2D([sl2.A.direct(rnd_data), sl2.range_tmp])
    sl2.A.adjoint(sl2.range_tmp, out=grad)
    #plotter2D([out, sl2.range_tmp])
    grad.multiply (sl2.c * 2.0, out=grad)
    #plotter2D([grad, sl2.range_tmp])
    grad *= -rate
    solution += grad
    
    sl2.notify_new_subset(i,nsubs)
    
print ("decomposed ", time.time()-t0)
#%%

sl2_a = StochasticNorm2Sq(A=A_os,
                          b=data, number_of_subsets=10)
sl2_a.memopt = True

def update_sgd(x, x_update, objective_function, rate):
    objective_function.gradient(x, out=x_update)
    x_update *= -rate
    x += x_update
    


niter=N
nsubs=10
sol = im_data * 0.
tmp = im_data * 0.
import time
t0 = time.time()
for i in range(niter):
    #update_sgd(sol, tmp, sl2_a, 1e-3)
    sl2.gradient(sol, out=tmp)
    #plotter2D([grad, sl2.range_tmp])
    tmp *= -rate
    sol += tmp
    sl2.notify_new_subset(i, nsubs)
    if i % 10 == 0:
        print ("iter {} objective {:.3e}".format(i, sl2(sol)))
        # plotter2D([sol])
        #print ("id(sol) {} id(tmp) {}".format(id(sol), id(tmp)))
t1 = time.time()

#%%
sl2 = StochasticNorm2Sq(A=AstraSubsetProjectorSimple(ig, ag, device = 'gpu'),
                        b=data, number_of_subsets=nsubs)
sgd = StochasticGradientDescent(x_init=im_data*0., 
                                objective_function=sl2, rate=1e-3, 
                                update_objective_interval=10, max_iteration=100, 
                                number_of_subsets=10)

#tsgd0 = time.time()
#sgd.run()
#tsgd1 = time.time()



#sgd = GradientDescent(x_init=im_data*0., 
#                    objective_function=sl2, rate=1e-3, 
#                    update_objective_interval=10, max_iteration=1000)

tsgd0 = time.time()
sgd.run()
tsgd1 = time.time()
#%%
plotter2D([im_data, gd.get_output(), sgd.get_output(), sol], titles=['ground truth', 
           'gd {}'.format(tgd1-tgd0), 'sgd {}'.format(tsgd1-tsgd0), 'hack {}'.format((t1-t0))])


def print_minmax(x):
    print (x.as_array().min(), x.as_array().max())
    

#%%

import time

N = 10
x0 = im_data *0.
t0 = time.time()
for i in range(N):
    sl2_a.A.adjoint(
            sl2_a.A.direct(im_data)
            )
    sl2_a.A.notify_new_subset(i, 10)
#    sl2.gradient(x0)
t1 = time.time()
for i in range(N):
    sl2.A.adjoint(
            sl2.A.direct(im_data)
            )
#    l2.gradient(x0)
t2 = time.time()

print ("Norm2Sq {}\nStochasticNorm2Sq {}".format(t2-t1, t1-t0))


    

