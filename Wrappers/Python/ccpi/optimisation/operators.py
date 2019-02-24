#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from ccpi.optimisation.ops import PowerMethodNonsquare
from ccpi.framework import DataContainer, ImageData, ImageGeometry
from numbers import Number
import sys
#sys.path.insert(0, '/Users/evangelos/Desktop/Projects/CCPi/edo_CompOpBranch/CCPi-Framework/Wrappers/Python/ccpi/optimisation/operators')
#from CompositeOperator import *



###############################################################################


class Operator(object):
    '''Operator that maps from a space X -> Y'''
    def __init__(self, **kwargs):
        self.scalar = 1
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
    def __rmul__(self, other):
        assert isinstance(other, Number)
        self.scalar = other
        return self    
        
class LinearOperator(Operator):
    '''Operator that maps from a space X -> Y'''
    def is_linear(self):
        '''Returns if the operator is linear'''
        return True
    def adjoint(self,x, out=None):
        raise NotImplementedError        
                                
class CompositeOperator_new(Operator):
    
    def __init__(self, dimension, *operators):
        self.dimension = dimension
        self.operators = operators
                
        n = self.dimension[0] * self.dimension[1]
        assert n == len (self.operators)
        self.compMat = [[ None for i in range(self.dimension[1])] for j in range(self.dimension[0])]
        s = 0
        for i in range(self.dimension[0]):
            for j in range(self.dimension[1]):
                self.compMat[i][j] = self.operators[s]
                s +=1
                                        
    def opMatrix(self):
        
        n = self.dimension[0] * self.dimension[1]
        assert n == len (self.operators)
        out = [[ None for i in range(self.dimension[1])] for j in range(self.dimension[0])]
        s = 0
        for i in range(self.dimension[0]):
            for j in range(self.dimension[1]):
                out[i][j] = self.operators[s]
                s +=1
        return out    
    
    def norm(self):
        tmp = 0
        for i in range(self.dimension[0]):
            for j in range(self.dimension[1]):
                tmp += self.compMat[i][j].norm()**2
        return np.sqrt(tmp)        
        
    
    def direct(self, x, out=None):
        
        out = [None]*self.dimension[0]
        for i in range(self.dimension[0]):                        
            z1 = ImageData(np.zeros(self.compMat[i][0].range_dim()))
            for j in range(self.dimension[1]):
                z1 += self.compMat[i][j].direct(x.get_item(j))
            out[i] = z1    
                                
        return CompositeDataContainer(*out)          
        
    
    def adjoint(self, x, out=None):        
        
        out = [None]*self.dimension[1]
        for i in range(self.dimension[1]):
            z2 = ImageData(np.zeros(self.compMat[0][i].domain_dim()))
            for j in range(self.dimension[0]):
                z2 += self.compMat[j][i].adjoint(x.get_item(j))
            out[i] = z2
                                
        return CompositeDataContainer(*out)   
        
    def range_dim(self):
        tmp = [] 
        for i in range(self.dimension[0]):
            tmp.append(self.compMat[i][0].range_dim())
        return tmp
    
    def domain_dim(self):
        tmp = [] 
        for i in range(self.dimension[1]):
            tmp.append(self.compMat[0][i].domain_dim())
        return tmp
            
      
                
#%%        
        
def finite_diff(x, direction = 0, method = 'for', out=None):
    
    x_asarr = x
    x_sz = len(x.shape)
    if out is None:        
        out = np.zeros(x.shape)
        
    fd_arr = out        
                            
    if x_sz == 2:
        
        if method == 'for':
            
            if direction == 1:                            
                np.subtract( x_asarr[:,1:], x_asarr[:,0:-1], out = fd_arr[:,0:-1] )
                
            if direction == 0:
                np.subtract( x_asarr[1:], x_asarr[0:-1], out = fd_arr[0:-1,:] ) 
                
        elif method == 'back':
#            
            if direction == 1:
#                
                np.subtract( x_asarr[:,1:], x_asarr[:,0:-1], out = fd_arr[:,1:] )
                np.subtract( x_asarr[:,0], 0, out = fd_arr[:,0] )
                np.subtract( -x_asarr[:,-2], 0, out = fd_arr[:,-1] )
#                
            if direction == 0:
#                
                np.subtract( x_asarr[1:,:], x_asarr[0:-1,:], out = fd_arr[1:,:] )
                np.subtract( x_asarr[0,:], 0, out = fd_arr[0,:] )
                np.subtract( -x_asarr[-2,:], 0, out = fd_arr[-1,:] )  
                
        elif method == 'central':
            pass
                

    elif x_sz == 3:
        
        if method == 'for':
            
            if direction == 0:                            
                np.subtract( x_asarr[1:,:,:], x_asarr[0:-1,:,:], out = fd_arr[0:-1,:,:] )
                
            if direction == 2:
                np.subtract( x_asarr[:,1:,:], x_asarr[:,0:-1,:], out = fd_arr[:,0:-1,:] ) 
                
            if direction == 1:
                np.subtract( x_asarr[:,:,1:], x_asarr[:,:,0:-1], out = fd_arr[:,:,0:-1] )  
                
        elif method == 'back':
             
            if direction == 0:                            
                np.subtract( x_asarr[1:,:,:], x_asarr[0:-1,:,:], out = fd_arr[1:,:,:] )
                np.subtract( x_asarr[0,:,:], 0, out = fd_arr[0,:,:] )
                np.subtract( -x_asarr[-2,:,:], 0, out = fd_arr[-1,:,:] )
                
            if direction == 2:
                np.subtract( x_asarr[:,1:,:], x_asarr[:,0:-1,:], out = fd_arr[:,1:,:] )
                np.subtract( x_asarr[:,0,:], 0, out = fd_arr[:,0,:] )
                np.subtract( -x_asarr[:,-2,:], 0, out = fd_arr[:,-1,:] )
                
            if direction == 1:
                np.subtract( x_asarr[:,:,1:], x_asarr[:,:,0:-1], out = fd_arr[:,:,1:] ) 
                np.subtract( x_asarr[:,:,0], 0, out = fd_arr[:,:,0] ) 
                np.subtract( -x_asarr[:,:,-2], 0, out = fd_arr[:,:,-1] ) 
                                             
                
        elif method == 'central':
            pass
                
    return out


###############################################################################
    
class Gradient(Operator):
    
    def __init__(self, gm_domain, gm_range=None, bnd_cond = 'Neumann', **kwargs):
        
        self.gm_domain = gm_domain
        self.gm_range = gm_range
        self.bnd_cond = bnd_cond
        
        if self.gm_range is None:
            self.gm_range = ((len(list(self.gm_domain)),) + self.gm_domain)
            
        self.memopt = kwargs.get('memopt',False)
        self.correlation = kwargs.get('correlation','Space') 
                            
        super(Gradient, self).__init__()  
        
    def direct(self, x, out=None):
        
        res = []
        for i in range(self.gm_range[0]):
            tmp = FiniteDiff(self.gm_domain, direction = i, bnd_cond = self.bnd_cond).direct(x)
            res.append(tmp)
        return CompositeDataContainer(*res)  
    
    def adjoint(self, x, out=None):
             #        
        res = []
        for i in range(len(self.gm_domain)):
            tmp = FiniteDiff(self.gm_domain, direction = i, bnd_cond = self.bnd_cond).adjoint(x)
            res.append(tmp)
        return CompositeDataContainer(*res) 
                       
    def norm(self):
#        return np.sqrt(4*len(self.domainDim()))        
        #TODO this takes time for big ImageData
        # for 2D ||grad|| = sqrt(8), 3D ||grad|| = sqrt(12)        
        x0 = ImageData(np.random.random_sample(self.domainDim()), geometry = self.geometry)
        self.s1, sall, svec = PowerMethodNonsquare(self, 25, x0)
        return self.s1
    
class gradient(Operator):
            
    def __init__(self, gm_domain, gm_range=None, **kwargs):
        
        self.gm_domain = gm_domain
        self.gm_range = gm_range
        
        if self.gm_range is None:
            self.gm_range = self.range_dim()
        
        self.memopt = kwargs.get('memopt',False)
        self.correlation = kwargs.get('correlation','Space')  
        
        super(gradient, self).__init__()  
                                      
    def domain_dim(self):       
        return self.gm_domain
                   
    def range_dim(self):                 
        return (len(self.domain_dim()),) + self.domain_dim() 
                                         
    def direct(self, x, out=None):
                         
        grad = np.zeros(self.range_dim())

        for i in range(self.range_dim()[0]):
            grad[i]=finite_diff(x.as_array(), i, method='for')
        
        return type(x)(grad)
#    
    def adjoint(self, x, out=None):
        
        div = np.zeros(x.shape[1:])
        for i in range(len(x.shape[1:])):
            div += finite_diff(x.as_array()[i], direction = i, method = 'back')
        
        return type(x)(-div)
    
    def norm(self):
#        return np.sqrt(4*len(self.domainDim()))        
        #TODO this takes time for big ImageData
        # for 2D ||grad|| = sqrt(8), 3D ||grad|| = sqrt(12)        
        x0 = ImageData(np.random.random_sample(self.domain_dim()))
        self.s1, sall, svec = PowerMethodNonsquare(self, 25, x0)
        return self.s1    

#class gradient_old(Operator):
#    
#    def __init__(self, geometry, memopt = False):
#        self.memopt = memopt
#        self.geometry = geometry
#        super(gradient_old, self).__init__()
#             
#    def direct(self, x, out=None):
#        
#        shape = [len(x.shape), ] + list(x.shape)
#        gradient = np.zeros(shape, dtype=x.as_array().dtype)
#        slice_all = [0, slice(None, -1),]
#        for d in range(len(x.shape)):
#            gradient[slice_all] = np.diff(x.as_array(), axis=d)
#            slice_all[0] = d + 1
#            slice_all.insert(1, slice(None))  
#        if self.memopt:    
#            out.fill(type(x)(gradient, geometry = x.geometry))  
#        else:
#            return type(x)(gradient, geometry = x.geometry)
#                                    
#    def adjoint(self, x, out = None):
#        
#        res = np.zeros(x.shape[1:])
#        for d in range(x.shape[0]):
#            this_grad = np.rollaxis(x.as_array()[d], d)
#            this_res = np.rollaxis(res, d)
#            this_res[:-1] += this_grad[:-1]
#            this_res[1:-1] -= this_grad[:-2]
#            this_res[-1] -= this_grad[-2]
#            
#        if self.memopt:    
#            out.fill(type(x)(-res, geometry = x.geometry))  
#        else:
#            return type(x)(-res, geometry = x.geometry) 
#        
#    def domainDim(self):       
#        return ImageData(geometry = self.geometry).shape         
#        
#    def rangeDim(self):
#        return (len(self.domainDim()),) + self.domainDim() 
#             
#    def get_max_sing_val(self):
#        
#        #TODO this takes time for big ImageData
#        # for 2D ||grad|| = sqrt(8), 3D ||grad|| = sqrt(12)        
#        x0 = ImageData(np.random.random_sample(self.domainDim()), geometry = self.geometry)
#        self.s1, sall, svec = PowerMethodNonsquare(self, 25, x0)
#        return self.s1

###############################################################################

class test_sym_gradient(Operator):
    
#    def __init__(self, memopt = False):
    def __init__(self, geometry, memopt = False):
        self.memopt = memopt
        self.geometry = geometry
        super(sym_gradient, self).__init__()
        
    def domain_dim(self):       
        return self.geometry
#        return ImageData(geometry = self.geometry).shape         
        
    def range_dim(self):
        return (len(self.domain_dim()),) + self.domain_dim()[1:]        
        
    def direct(self, x, out=None):
                
        grad = np.zeros((len(x.shape),)+x.shape[1:])
        
        grad[0] = finite_diff(x.as_array()[0], direction = 1, method = 'back')
        grad[1] = finite_diff(x.as_array()[1], direction = 0, method = 'back')
        grad[2] = 0.5 * (finite_diff(x.as_array()[0], direction = 0, method = 'back') + \
                      finite_diff(x.as_array()[1], direction = 1, method = 'back') )
                            
        return type(x)(grad)
    
    def adjoint(self, x, out=None):
        
        div = np.zeros((len(x.shape[1:]),) + x.shape[1:])
        
        div[0] = finite_diff(x.as_array()[0], direction = 1, method = 'for') + \
                 finite_diff(x.as_array()[2], direction = 0, method = 'for')
         
        div[1] = finite_diff(x.as_array()[2], direction = 1, method = 'for') + \
                 finite_diff(x.as_array()[1], direction = 0, method = 'for')
                 
        return type(x)(-div) 
    
    def norm(self):
#        return np.sqrt(4*len(self.domainDim()))        
        #TODO this takes time for big ImageData
        # for 2D ||grad|| = sqrt(8), 3D ||grad|| = sqrt(12)        
        x0 = ImageData(np.random.random_sample(self.domainDim()), geometry = self.geometry)
        self.s1, sall, svec = PowerMethodNonsquare(self, 25, x0)
        return self.s1


class sym_gradient(Operator):
    
#    def __init__(self, memopt = False):
    def __init__(self, geometry, memopt = False):
        self.memopt = memopt
        self.geometry = geometry
        super(sym_gradient, self).__init__()
        
    def domain_dim(self):       
        return self.geometry
#        return ImageData(geometry = self.geometry).shape         
        
    def range_dim(self):
        return (len(self.domain_dim()),) + self.domain_dim()[1:]        
        
    def direct(self, x, out=None):
                
        grad = np.zeros((len(x.shape),)+x.shape[1:])
        
        grad[0] = finite_diff(x.as_array()[0], direction = 1, method = 'back')
        grad[1] = finite_diff(x.as_array()[1], direction = 0, method = 'back')
        grad[2] = 0.5 * (finite_diff(x.as_array()[0], direction = 0, method = 'back') + \
                      finite_diff(x.as_array()[1], direction = 1, method = 'back') )
                            
        return type(x)(grad)
    
    def adjoint(self, x, out=None):
        
        div = np.zeros((len(x.shape[1:]),) + x.shape[1:])
        
        div[0] = finite_diff(x.as_array()[0], direction = 1, method = 'for') + \
                 finite_diff(x.as_array()[2], direction = 0, method = 'for')
         
        div[1] = finite_diff(x.as_array()[2], direction = 1, method = 'for') + \
                 finite_diff(x.as_array()[1], direction = 0, method = 'for')
                 
        return type(x)(-div) 
    
    def norm(self):
#        return np.sqrt(4*len(self.domainDim()))        
        #TODO this takes time for big ImageData
        # for 2D ||grad|| = sqrt(8), 3D ||grad|| = sqrt(12)        
        x0 = ImageData(np.random.random_sample(self.domain_dim()), geometry = self.geometry)
        self.s1, sall, svec = PowerMethodNonsquare(self, 25, x0)
        return self.s1
    
###############################################################################    

class ZeroOp(Operator):
    
    def __init__(self, gm_domain, gm_range):
        self.gm_domain = gm_domain
        self.gm_range = gm_range
        super(ZeroOp, self).__init__()
        
    def direct(self,x,out=None):
        if out is None:
            return ImageData(np.zeros(self.gm_range))
        else:
            return ImageData(np.zeros(self.gm_range))
    
    def adjoint(self,x, out=None):
        if out is None:
            return ImageData(np.zeros(self.gm_domain))
        else:
            return ImageData(np.zeros(self.gm_domain))
        
    def norm(self):
        return 0
    
    def domain_dim(self):       
        return self.gm_domain  
        
    def range_dim(self):
        return self.gm_range
    
###############################################################################    
            
class MyTomoIdentity(Operator):
    
    def __init__(self, gm_domain, gm_range, scalar = 1):
        self.s1 = 1.0
        self.gm_domain = gm_domain
        self.gm_range = gm_range    
        self.scalar = scalar
        super(MyTomoIdentity, self).__init__()
        
    def direct(self,x,out=None):
        if out is None:
            return self.scalar * x.copy()
        else:
            out.fill(self.scalar * x)
    
    def adjoint(self,x, out=None):
        if out is None:
            return self.scalar * x.copy()
        else:
            out.fill(self.scalar * x)
    
    def size(self):
        return NotImplemented
    
    def norm(self):
        return self.s1
    
    def allocate_direct(self):
        if issubclass(type(self.geometry), ImageGeometry):
            return ImageData(geometry=self.geometry)
        elif issubclass(type(self.geometry), AcquisitionGeometry):
            return AcquisitionData(geometry=self.geometry)
        else:
            raise ValueError("Wrong geometry type: expected ImageGeometry of AcquisitionGeometry, got ", type(self.geometry))
    def allocate_adjoint(self):
        return self.allocate_direct()    
    
    def domain_dim(self):       
        return self.gm_domain
        
    def range_dim(self):
        return self.gm_range     
    
###############################################################################
    
    











