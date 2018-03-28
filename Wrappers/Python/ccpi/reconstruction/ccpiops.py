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
from ccpi.optimisation.ops import Operator, PowerMethodNonsquare
from ccpi.framework import ImageData, DataContainer
from ccpi.reconstruction.processors import CCPiBackwardProjector, CCPiForwardProjector

class LinearOperatorMatrix(Operator):
    def __init__(self,A):
        self.A = A
        self.s1 = None   # Largest singular value, initially unknown
        super(LinearOperatorMatrix, self).__init__()
        
    def direct(self,x):
        return DataContainer(numpy.dot(self.A,x.as_array()))
    
    def adjoint(self,x):
        return DataContainer(numpy.dot(self.A.transpose(),x.as_array()))
    
    def size(self):
        return self.A.shape
    
    def get_max_sing_val(self):
        # If unknown, compute and store. If known, simply return it.
        if self.s1 is None:
            self.s1 = svds(self.A,1,return_singular_vectors=False)[0]
            return self.s1
        else:
            return self.s1
        


class CCPiProjectorSimple(Operator):
    """ASTRA projector modified to use DataSet and geometry."""
    def __init__(self, geomv, geomp):
        super(CCPiProjectorSimple, self).__init__()
        
        # Store volume and sinogram geometries.
        self.acquisition_geometry = geomp
        self.volume_geometry = geomv
        
        self.fp = CCPiForwardProjector(image_geometry=geomv,
                                       acquisition_geometry=geomp,
                                       output_axes_order=['angle','vertical','horizontal'])
        
        self.bp = CCPiBackwardProjector(image_geometry=geomv,
                                    acquisition_geometry=geomp,
                                    output_axes_order=['horizontal_x','horizontal_y','vertical'])
                
        # Initialise empty for singular value.
        self.s1 = None
    
    def direct(self, image_data):
        self.fp.set_input(image_data)
        out = self.fp.get_output()
        return out
    
    def adjoint(self, acquisition_data):
        self.bp.set_input(acquisition_data)
        out = self.bp.get_output()
        return out
    
    #def delete(self):
    #    astra.data2d.delete(self.proj_id)
    
    def get_max_sing_val(self):
        a = PowerMethodNonsquare(self,10)
        self.s1 = a[0] 
        return self.s1
    
    def size(self):
        # Only implemented for 3D
        return ( (self.acquisition_geometry.angles.size, \
                  self.acquisition_geometry.pixel_num_v,
                  self.acquisition_geometry.pixel_num_h), \
                 (self.volume_geometry.voxel_num_x, \
                  self.volume_geometry.voxel_num_y,
                  self.volume_geometry.voxel_num_z) )
    def create_image_data(self):
        x0 = ImageData(geometry = self.volume_geometry, 
                       dimension_labels=self.bp.output_axes_order)#\
                       #.subset(['horizontal_x','horizontal_y','vertical'])
        print (x0)
        x0.fill(numpy.random.randn(*x0.shape))
        return x0