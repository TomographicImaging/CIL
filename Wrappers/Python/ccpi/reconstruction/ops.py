
import numpy as np
import astra
from scipy.sparse.linalg import svds
from ccpi.framework import DataSet, VolumeData, SinogramData, DataSetProcessor

# Maybe operators need to know what types they take as inputs/outputs
# to not just use generic DataSet


class Operator:
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
    

class LinearOperatorMatrix(Operator):
    def __init__(self,A):
        self.A = A
        self.s1 = None   # Largest singular value, initially unknown
        
    def direct(self,x):
        return DataSet(np.dot(self.A,x.as_array()))
    
    def adjoint(self,x):
        return DataSet(np.dot(self.A.transpose(),x.as_array()))
    
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
        
    def direct(self,x):
        return x
    
    def adjoint(self,x):
        return x
    
    def size(self):
        return NotImplemented
    
    def get_max_sing_val(self):
        return self.s1

class AstraProjector:
    """A simple 2D/3D parallel/fan beam projection/backprojection class based on ASTRA toolbox"""
    def __init__(self, DetWidth, DetectorsDim, SourceOrig, OrigDetec, AnglesVec, ObjSize, projtype, device):
        self.DetectorsDim = DetectorsDim
        self.AnglesVec = AnglesVec
        self.ProjNumb = len(AnglesVec)
        self.ObjSize = ObjSize
        if projtype == 'parallel':
            self.proj_geom = astra.create_proj_geom('parallel', DetWidth, DetectorsDim, AnglesVec)
        elif projtype == 'fanbeam':
            self.proj_geom = astra.create_proj_geom('fanflat', DetWidth, DetectorsDim, AnglesVec, SourceOrig, OrigDetec)
        else:
            print ("Please select for projtype between 'parallel' and 'fanbeam'")
        self.vol_geom = astra.create_vol_geom(ObjSize, ObjSize)
        if device == 'cpu':
            self.proj_id = astra.create_projector('line', self.proj_geom, self.vol_geom) # for CPU
            self.device = 1
        elif device == 'gpu':
            self.proj_id = astra.create_projector('cuda', self.proj_geom, self.vol_geom) # for GPU
            self.device = 0
        else:
            print ("Select between 'cpu' or 'gpu' for device")
        self.s1 = None
    def direct(self, IM):
        """Applying forward projection to IM [2D or 3D array]"""
        if np.ndim(IM.as_array()) == 3:
            slices = np.size(IM.as_array(),np.ndim(IM.as_array())-1)
            DATA = np.zeros((self.ProjNumb,self.DetectorsDim,slices), 'float32')
            for i in range(0,slices):
                sinogram_id, DATA[:,:,i] = astra.create_sino(IM[:,:,i].as_array(), self.proj_id)
                astra.data2d.delete(sinogram_id)
            astra.data2d.delete(self.proj_id)
        else:
            sinogram_id, DATA = astra.create_sino(IM.as_array(), self.proj_id)
            astra.data2d.delete(sinogram_id)
            astra.data2d.delete(self.proj_id)
        return SinogramData(DATA)
    def adjoint(self, DATA):
        """Applying backprojection to DATA [2D or 3D]"""
        if np.ndim(DATA) == 3:
           slices = np.size(DATA.as_array(),np.ndim(DATA.as_array())-1)
           IM = np.zeros((self.ObjSize,self.ObjSize,slices), 'float32')
           for i in range(0,slices):
               rec_id, IM[:,:,i] = astra.create_backprojection(DATA[:,:,i].as_array(), self.proj_id)
               astra.data2d.delete(rec_id)
           astra.data2d.delete(self.proj_id)
        else:
            rec_id, IM = astra.create_backprojection(DATA.as_array(), self.proj_id)        
            astra.data2d.delete(rec_id)
            astra.data2d.delete(self.proj_id)
        return VolumeData(IM)
    
    def delete(self):
        astra.data2d.delete(self.proj_id)
    
    def get_max_sing_val(self):
        self.s1, sall, svec = PowerMethodNonsquare(self,10)
        return self.s1
    
    def size(self):
        return ( (self.AnglesVec.size, self.DetectorsDim), \
                 (self.ObjSize, self.ObjSize) )

def PowerMethodNonsquare(op,numiters):
    # Initialise random
    inputsize = op.size()[1]
    x0 = DataSet(np.random.randn(inputsize[0],inputsize[1]))
    s = np.zeros(numiters)
    # Loop
    for it in np.arange(numiters):
        x1 = op.adjoint(op.direct(x0))
        x1norm = np.sqrt((x1**2).sum())
        s[it] = (x1*x0).sum() / (x0*x0).sum()
        x0 = (1.0/x1norm)*x1
    return np.sqrt(s[-1]), np.sqrt(s), x0

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