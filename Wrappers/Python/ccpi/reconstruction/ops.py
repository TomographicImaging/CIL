# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018 Jakob Jorgensen, Daniil Kazantsev and Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy
from scipy.sparse.linalg import svds
from ccpi.framework import DataSet, VolumeData, SinogramData, DataSetProcessor

# Maybe operators need to know what types they take as inputs/outputs
# to not just use generic DataSet


class Operator(object):
    def direct(self,x):
        return x
    def adjoint(self,x):
        return x
    def size(self):
        # To be defined for specific class
        return None

# Or should we rather have an attribute isLinear instead of separate class?

#class OperatorLinear(Operator):
#    
#    def __init__():

class ForwardBackProjector(Operator):
    
    # The constructor should set up everything, ie at least hold equivalent of 
    # projection geometry and volume geometry, so that when calling direct and 
    # adjoint methods, only the volume/sinogram is needed as input. Quite 
    # similar to opTomo operator.
    
    def __init__(self):
        # do nothing
        i  = 1
        super(ForwardBackProjector, self).__init__()
    

class LinearOperatorMatrix(Operator):
    def __init__(self,A):
        self.A = A
        self.s1 = None   # Largest singular value, initially unknown
        super(LinearOperatorMatrix, self).__init__()
        
    def direct(self,x):
        return DataSet(numpy.dot(self.A,x.as_array()))
    
    def adjoint(self,x):
        return DataSet(numpy.dot(self.A.transpose(),x.as_array()))
    
    def size(self):
        return self.A.shape
    
    def get_max_sing_val(self):
        # If unknown, compute and store. If known, simply return it.
        if self.s1 is None:
            self.s1 = svds(self.A,1,return_singular_vectors=False)[0]
            return self.s1
        else:
            return self.s1

class Identity(Operator):
    def __init__(self):
        self.s1 = 1.0
        super(Identity, self).__init__()
        
    def direct(self,x):
        return x
    
    def adjoint(self,x):
        return x
    
    def size(self):
        return NotImplemented
    
    def get_max_sing_val(self):
        return self.s1


def PowerMethodNonsquare(op,numiters):
    # Initialise random
    inputsize = op.size()[1]
    x0 = DataSet(numpy.random.randn(inputsize[0],inputsize[1]))
    s = numpy.zeros(numiters)
    # Loop
    for it in numpy.arange(numiters):
        x1 = op.adjoint(op.direct(x0))
        x1norm = numpy.sqrt((x1**2).sum())
        s[it] = (x1*x0).sum() / (x0*x0).sum()
        x0 = (1.0/x1norm)*x1
    return numpy.sqrt(s[-1]), numpy.sqrt(s), x0

#def PowerMethod(op,numiters):
#    # Initialise random
#    x0 = np.random.randn(400)
#    s = np.zeros(numiters)
#    # Loop
#    for it in np.arange(numiters):
#        x1 = np.dot(op.transpose(),np.dot(op,x0))
#        x1norm = np.sqrt(np.sum(np.dot(x1,x1)))
#        s[it] = np.dot(x1,x0) / np.dot(x1,x0)
#        x0 = (1.0/x1norm)*x1
#    return s, x0