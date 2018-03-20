# -*- coding: utf-8 -*-
#    This work is independent part of the Core Imaging Library developed by
#    Visual Analytics and Imaging System Group of the Science Technology
#    Facilities Council, STFC
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ccpi.reconstruction.ops import Operator
import numpy
import astra
from ccpi.framework import AcquisitionData, ImageData, DataContainer
from ccpi.reconstruction.ops import PowerMethodNonsquare
from ccpi.astra.astra_processors import AstraForwardProjector, AstraBackProjector, \
     AstraForwardProjectorMC, AstraBackProjectorMC

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
    
    def get_max_sing_val(self):
        self.s1, sall, svec = PowerMethodNonsquare(self,10)
        return self.s1
    
    def size(self):
        # Only implemented for 2D
        return ( (self.sinogram_geometry.angles.size, \
                  self.sinogram_geometry.pixel_num_h), \
                 (self.volume_geometry.voxel_num_x, \
                  self.volume_geometry.voxel_num_y) )
    
    def create_image_data(self):
        inputsize = self.size()[1]
        return DataContainer(numpy.random.randn(inputsize[0],
                                                inputsize[1]))

class AstraProjectorMC(Operator):
    """ASTRA Multichannel projector"""
    def __init__(self, geomv, geomp, device):
        super(AstraProjectorMC, self).__init__()
        
        # Store volume and sinogram geometries.
        self.sinogram_geometry = geomp
        self.volume_geometry = geomv
        
        self.fp = AstraForwardProjectorMC(volume_geometry=geomv,
                                        sinogram_geometry=geomp,
                                        proj_id=None,
                                        device=device)
        
        self.bp = AstraBackProjectorMC(volume_geometry=geomv,
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
    
    def get_max_sing_val(self):
        if self.s1 is None:
            self.s1, sall, svec = PowerMethodNonsquare(self,10)
            return self.s1
        else:
            return self.s1
    
    def size(self):
        # Only implemented for 2D
        return ( (self.sinogram_geometry.angles.size, \
                  self.sinogram_geometry.pixel_num_h), \
                 (self.volume_geometry.voxel_num_x, \
                  self.volume_geometry.voxel_num_y) )
    
    def create_image_data(self):
        inputsize = self.size()[1]
        return DataContainer(numpy.random.randn(self.volume_geometry.channels,
                                                inputsize[0],
                                                inputsize[1]))
    
