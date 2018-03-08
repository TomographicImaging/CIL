
class VolumeGeometry:
    
    def __init__(self, 
                 voxel_num_x=0, 
                 voxel_num_y=0, 
                 voxel_num_z=0, 
                 voxel_size_x=1, 
                 voxel_size_y=1, 
                 voxel_size_z=1, 
                 center_x=0, 
                 center_y=0, 
                 center_z=0, 
                 channels=1):
        
        self.voxel_num_x = voxel_num_x
        self.voxel_num_y = voxel_num_y
        self.voxel_num_z = voxel_num_z
        self.voxel_size_x = voxel_size_x
        self.voxel_size_y = voxel_size_y
        self.voxel_size_z = voxel_size_z
        self.center_x = center_x
        self.center_y = center_y
        self.center_z = center_z  
        self.channels = channels
        
    def getMinX(self):
        return self.center_x - 0.5*self.voxel_num_x*self.voxel_size_x
        
    def getMaxX(self):
        return self.center_x + 0.5*self.voxel_num_x*self.voxel_size_x
        
    def getMinY(self):
        return self.center_y - 0.5*self.voxel_num_y*self.voxel_size_y
        
    def getMaxY(self):
        return self.center_y + 0.5*self.voxel_num_y*self.voxel_size_y
        
    def getMinZ(self):
        if not voxel_num_z == 0:
            return self.center_z - 0.5*self.voxel_num_z*self.voxel_size_z
        else:
            return 0
        
    def getMaxZ(self):
        if not voxel_num_z == 0:
            return self.center_z + 0.5*self.voxel_num_z*self.voxel_size_z
        else:
            return 0
        
    
class SinogramGeometry:
    
    def __init__(self, 
                 geom_type, 
                 dimension, 
                 angles, 
                 pixel_num_h=0, 
                 pixel_size_h=1, 
                 pixel_num_v=0, 
                 pixel_size_v=1, 
                 dist_source_center=None, 
                 dist_center_detector=None, 
                 channels=1
                 ):
        """
        General inputs for standard type projection geometries
        detectorDomain or detectorpixelSize:
            If 2D
                If scalar: Width of detector or single detector pixel
                If 2-vec: Error
            If 3D
                If scalar: Width in both dimensions
                If 2-vec: Vertical then horizontal size
        grid
            If 2D
                If scalar: number of detectors
                If 2-vec: error
            If 3D
                If scalar: Square grid that size
                If 2-vec vertical then horizontal size
        cone or parallel
        2D or 3D
        parallel_parameters: ?
        cone_parameters:
            source_to_center_dist (if parallel: NaN)
            center_to_detector_dist (if parallel: NaN)
        standard or nonstandard (vec) geometry
        angles
        angles_format radians or degrees
        """
        self.geom_type = geom_type   # 'parallel' or 'cone'
        self.dimension = dimension # 2D or 3D
        self.angles = angles
        
        self.dist_source_center = dist_source_center
        self.dist_center_detector = dist_center_detector
        
        self.pixel_num_h = pixel_num_h
        self.pixel_size_h = pixel_size_h
        self.pixel_num_v = pixel_num_v
        self.pixel_size_v = pixel_size_v
        
        self.channels = channels

        
                
