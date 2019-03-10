#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy
from ccpi.framework import DataContainer, ImageData, ImageGeometry, AcquisitionData
from ccpi.astra.processors import AstraForwardProjector, AstraBackProjector
from ccpi.optimisation.ops import PowerMethodNonsquare

from Operators.operators import Operator
from Operators.CompositeDataContainer import CompositeDataContainer

from numbers import Number
import functools


#%%
class CompositeOperator(Operator):
    
    def __init__(self, shape, *operators):
        self.shape = shape
        self.operators = operators
                
        n = self.shape[0] * self.shape[1]
        assert n == len (self.operators)
        self.compMat = [[ None for i in range(self.shape[1])] for j in range(self.shape[0])]
        s = 0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                self.compMat[i][j] = self.operators[s]
                s +=1
                                        
    def opMatrix(self):
        
        n = self.shape[0] * self.shape[1]
        assert n == len (self.operators)
        out = [[ None for i in range(self.shape[1])] for j in range(self.shape[0])]
        s = 0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                out[i][j] = self.operators[s]
                s +=1
        return out    
    
    def norm(self):
        tmp = 0
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                tmp += self.compMat[i][j].norm()**2
        return np.sqrt(tmp)   

    def direct(self, x, out=None):
        
        tmp = [None]*self.shape[0]
        for i in range(self.shape[0]):                        
            z1 = ImageData(np.zeros(self.compMat[i][0].range_dim()))
            for j in range(self.shape[1]):
                z1 += self.compMat[i][j].direct(x.get_item(j))
            tmp[i] = z1    
        if out is None:                        
            return CompositeDataContainer(*tmp)
        else:
            out = CompositeDataContainer(*tmp)
            return out
                                          
    def adjoint(self, x, out=None):        
        
        tmp = [None]*self.shape[1]
        for i in range(self.shape[1]):
            z2 = ImageData(np.zeros(self.compMat[0][i].domain_dim()))
            for j in range(self.shape[0]):
                z2 += self.compMat[j][i].adjoint(x.get_item(j))
            tmp[i] = z2
        if out is None:                        
            return CompositeDataContainer(*tmp)
        else:
            out = CompositeDataContainer(*tmp)
            return out          
        
            
    def range_dim(self):

        tmp = [ [] for i in range(self.shape[0])]
        for i in range(self.shape[0]):
            tmp[i]=self.compMat[i][0].range_dim()
#            if isinstance(tmp[i],tuple):
#                tmp[i]=[tmp[i]]
        return tmp
    
    def domain_dim(self):

        tmp = [[] for k in range(self.shape[1])]
        for i in range(self.shape[1]):
            tmp[i]=self.compMat[0][i].domain_dim()
        return tmp
    
    def alloc_domain_dim(self):     
        tmp = [[] for k in range(self.shape[1])] 
        for k in range(self.shape[1]):
            tmp[k] = ImageData(np.zeros(self.compMat[0][k].domain_dim()))
        return CompositeDataContainer(*tmp)
        
    
    def alloc_range_dim(self):
        tmp = [ [] for i in range(self.shape[0])]
        for k in range(self.shape[0]):            
            tmp[k] = ImageData(np.zeros(self.compMat[k][0].range_dim()))
        return CompositeDataContainer(*tmp) 

