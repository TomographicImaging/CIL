import numpy
from scipy.sparse.linalg import svds
from ccpi.framework import DataContainer
from ccpi.framework import AcquisitionData
from ccpi.framework import ImageData
from ccpi.framework import ImageGeometry
from ccpi.framework import AcquisitionGeometry
from numbers import Number
from ccpi.optimisation.operators import LinearOperator
class LinearOperatorMatrix(LinearOperator):
    def __init__(self,A):
        self.A = A
        self.s1 = None   # Largest singular value, initially unknown
        super(LinearOperatorMatrix, self).__init__()
        
    def direct(self,x, out=None):
        if out is None:
            return type(x)(numpy.dot(self.A,x.as_array()))
        else:
            numpy.dot(self.A, x.as_array(), out=out.as_array())
            
    
    def adjoint(self,x, out=None):
        if out is None:
            return type(x)(numpy.dot(self.A.transpose(),x.as_array()))
        else:
            numpy.dot(self.A.transpose(),x.as_array(), out=out.as_array())
            
    
    def size(self):
        return self.A.shape
    
    def get_max_sing_val(self):
        # If unknown, compute and store. If known, simply return it.
        if self.s1 is None:
            self.s1 = svds(self.A,1,return_singular_vectors=False)[0]
            return self.s1
        else:
            return self.s1
    def allocate_direct(self):
        '''allocates the memory to hold the result of adjoint'''
        #numpy.dot(self.A.transpose(),x.as_array())
        M_A, N_A = self.A.shape
        out = numpy.zeros((N_A,1))
        return DataContainer(out)
    def allocate_adjoint(self):
        '''allocate the memory to hold the result of direct'''
        #numpy.dot(self.A.transpose(),x.as_array())
        M_A, N_A = self.A.shape
        out = numpy.zeros((M_A,1))
        return DataContainer(out)
