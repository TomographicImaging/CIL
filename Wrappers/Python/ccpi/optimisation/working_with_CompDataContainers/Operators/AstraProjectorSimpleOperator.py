#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 19:31:57 2019

@author: evangelos
"""

import numpy as np
from ccpi.framework import DataContainer, ImageData, ImageGeometry, AcquisitionData
from ccpi.astra.processors import AstraForwardProjector, AstraBackProjector
from ccpi.optimisation.ops import PowerMethodNonsquare
#from numbers import Number
#import functools
from operators import Operator

class AstraProjectorSimple(Operator):
    
    """ASTRA projector modified to use DataSet and geometry."""
    def __init__(self, geomv, geomp, device):
        super(AstraProjectorSimple, self).__init__()
        
        # Store volume and sinogram geometries.
        self.sinogram_geometry = geomp
        self.volume_geometry = geomv
        
        self.fp = AstraForwardProjector(volume_geometry=geomv,
                                        sinogram_geometry=geomp,
                                        proj_id=None,
                                        device=device)
        
        self.bp = AstraBackProjector(volume_geometry=geomv,
                                        sinogram_geometry=geomp,
                                        proj_id=None,
                                        device=device)
                
        # Initialise empty for singular value.
        self.s1 = None
    
    def direct(self, IM):
        self.fp.set_input(IM)
        out = self.fp.get_output()
        return out
    
    def adjoint(self, DATA):
        self.bp.set_input(DATA)
        out = self.bp.get_output()
        return out
    
    #def delete(self):
    #    astra.data2d.delete(self.proj_id)
    
    def domain_dim(self):
        return (self.volume_geometry.voxel_num_x, \
                  self.volume_geometry.voxel_num_y)
        
    def range_dim(self):  
        return (self.sinogram_geometry.angles.size, \
                  self.sinogram_geometry.pixel_num_h)    
    
    def norm(self):
        x0 = ImageData(np.random.random_sample(self.domain_dim()))
        self.s1, sall, svec = PowerMethodNonsquare(self,10,x0)
        return self.s1
    
    def size(self):
        # Only implemented for 2D
        return ( (self.sinogram_geometry.angles.size, \
                  self.sinogram_geometry.pixel_num_h), \
                 (self.volume_geometry.voxel_num_x, \
                  self.volume_geometry.voxel_num_y) )  