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
import astra
import numpy
from ccpi.framework import SinogramData, VolumeData
from ccpi.reconstruction.ops import PowerMethodNonsquare

class AstraProjectorSimple(Operator):
    """ASTRA projector modified to use DataSet and geometry."""
    def __init__(self, geomv, geomp, device):
        super(AstraProjectorSimple, self).__init__()
        
        # Store volume and sinogram geometries.
        self.sinogram_geometry = geomp
        self.volume_geometry = geomv
        
        # Set up ASTRA Volume geometry, not stored
        if geomp.dimension == '2D':
            vol_geom = astra.create_vol_geom(geomv.voxel_num_x, 
                                             geomv.voxel_num_y, 
                                             geomv.getMinX(), 
                                             geomv.getMaxX(), 
                                             geomv.getMinY(), 
                                             geomv.getMaxY())
        elif geomp.dimension == '3D':
            vol_geom = astra.create_vol_geom(geomv.voxel_num_x, 
                                             geomv.voxel_num_y, 
                                             geomv.voxel_num_z, 
                                             geomv.getMinX(), 
                                             geomv.getMaxX(), 
                                             geomv.getMinY(), 
                                             geomv.getMaxY(), 
                                             geomv.getMinZ(), 
                                             geomv.getMaxZ())
        else:
            NotImplemented
            
        
        # ASTRA projection geometry
        if geomp.dimension == '2D':
            if geomp.geom_type == 'parallel':
                proj_geom = astra.create_proj_geom('parallel',
                                                   geomp.pixel_size_h,
                                                   geomp.pixel_num_h,
                                                   geomp.angles)
            elif geomp.geom_type == 'cone':
                proj_geom = astra.create_proj_geom('fanflat',
                                                   geomp.pixel_size_h,
                                                   geomp.pixel_num_h,
                                                   geomp.angles,
                                                   geomp.dist_source_center,
                                                   geomp.dist_center_detector)
            else:
                NotImplemented
        elif geomp.dimension == '3D':
            if geomp.proj_geom == 'parallel':
                proj_geom = astra.create_proj_geom('parallel3d',
                                                   geomp.pixel_size_h,
                                                   geomp.pixel_size_v,
                                                   geomp.pixel_num_v,
                                                   geomp.pixel_num_h,
                                                   geomp.angles)
            elif geomp.geom_type == 'cone':
                proj_geom = astra.create_proj_geom('cone',
                                                   geomp.pixel_size_h,
                                                   geomp.pixel_size_v,
                                                   geomp.pixel_num_v,
                                                   geomp.pixel_num_h,
                                                   geomp.angles,
                                                   geomp.dist_source_center,
                                                   geomp.dist_center_detector)
            else:
                NotImplemented
        else:
            NotImplemented
        
        # ASTRA projector, to be stored
        if device == 'cpu':
            # Note that 'line' is only for parallel (2D) and only one option
            self.proj_id = astra.create_projector('line', proj_geom, vol_geom) 
        elif device == 'gpu':
            self.proj_id = astra.create_projector('cuda', proj_geom, vol_geom) 
        else:
            NotImplemented
        
        # Initialise empty for singular value.
        self.s1 = None
    
    def direct(self, IM):
        
        sinogram_id, DATA = astra.create_sino(IM.as_array(), self.proj_id)
        astra.data2d.delete(sinogram_id)
        return SinogramData(DATA,geometry=self.sinogram_geometry)
    
    def adjoint(self, DATA):
        rec_id, IM = astra.create_backprojection(DATA.as_array(), self.proj_id)
        astra.data2d.delete(rec_id)
        return VolumeData(IM,geometry=self.volume_geometry)
    
    def delete(self):
        astra.data2d.delete(self.proj_id)
    
    def get_max_sing_val(self):
        self.s1, sall, svec = PowerMethodNonsquare(self,10)
        return self.s1
    
    def size(self):
        # Only implemented for 2D
        return ( (self.sinogram_geometry.angles.size, \
                  self.sinogram_geometry.pixel_num_h), \
                 (self.volume_geometry.voxel_num_x, \
                  self.volume_geometry.voxel_num_y) )
    

class AstraProjector(Operator):
    """A simple 2D/3D parallel/fan beam projection/backprojection class based 
    on ASTRA toolbox"""
    def __init__(self, DetWidth, DetectorsDim, SourceOrig, OrigDetec, 
                 AnglesVec, ObjSize, projtype, device):
        super(AstraProjector, self).__init__()
        self.DetectorsDim = DetectorsDim
        self.AnglesVec = AnglesVec
        self.ProjNumb = len(AnglesVec)
        self.ObjSize = ObjSize
        if projtype == 'parallel':
            self.proj_geom = astra.create_proj_geom('parallel', DetWidth, 
                                                    DetectorsDim, AnglesVec)
        elif projtype == 'fanbeam':
            self.proj_geom = astra.create_proj_geom('fanflat', DetWidth,
                                                    DetectorsDim, AnglesVec, 
                                                    SourceOrig, OrigDetec)
        else:
            print ("Please select for projtype between 'parallel' and 'fanbeam'")
        self.vol_geom = astra.create_vol_geom(ObjSize, ObjSize)
        if device == 'cpu':
            self.proj_id = astra.create_projector('line', self.proj_geom, 
                                                  self.vol_geom) # for CPU
            self.device = 1
        elif device == 'gpu':
            self.proj_id = astra.create_projector('cuda', self.proj_geom, 
                                                  self.vol_geom) # for GPU
            self.device = 0
        else:
            print ("Select between 'cpu' or 'gpu' for device")
        self.s1 = None
    def direct(self, IM):
        """Applying forward projection to IM [2D or 3D array]"""
        if numpy.ndim(IM.as_array()) == 3:
            slices = numpy.size(IM.as_array(),numpy.ndim(IM.as_array())-1)
            DATA = numpy.zeros((self.ProjNumb,self.DetectorsDim,slices), 
                               'float32')
            for i in range(0,slices):
                sinogram_id, DATA[:,:,i] = \
                    astra.create_sino(IM[:,:,i].as_array(), self.proj_id)
                astra.data2d.delete(sinogram_id)
            astra.data2d.delete(self.proj_id)
        else:
            sinogram_id, DATA = astra.create_sino(IM.as_array(), self.proj_id)
            astra.data2d.delete(sinogram_id)
            astra.data2d.delete(self.proj_id)
        return SinogramData(DATA)
    def adjoint(self, DATA):
        """Applying backprojection to DATA [2D or 3D]"""
        if numpy.ndim(DATA) == 3:
           slices = numpy.size(DATA.as_array(),numpy.ndim(DATA.as_array())-1)
           IM = numpy.zeros((self.ObjSize,self.ObjSize,slices), 'float32')
           for i in range(0,slices):
               rec_id, IM[:,:,i] = \
                   astra.create_backprojection(DATA[:,:,i].as_array(), 
                                               self.proj_id)
               astra.data2d.delete(rec_id)
           astra.data2d.delete(self.proj_id)
        else:
            rec_id, IM = astra.create_backprojection(DATA.as_array(), 
                                                     self.proj_id)        
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

