
class VolumeGeometry:
    
    def __init__(self, \
                 voxel_num_x=None, \
                 voxel_num_y=None, \
                 voxel_num_z=None, \
                 voxel_size_x=None, \
                 voxel_size_y=None, \
                 voxel_size_z=None):
        
        self.voxel_num_x = voxel_num_x
        self.voxel_num_y = voxel_num_y
        self.voxel_num_z = voxel_num_z
        self.voxel_size_x = voxel_size_x
        self.voxel_size_y = voxel_size_y
        self.voxel_size_z = voxel_size_z    
        
    def get_min_x(self):
        return -0.5*self.voxel_num_x*self.voxel_size_x
        
    def get_max_x(self):
        return 0.5*self.voxel_num_x*self.voxel_size_x
        
    def get_min_y(self):
        return -0.5*self.voxel_num_y*self.voxel_size_y
        
    def get_max_y(self):
        return 0.5*self.voxel_num_y*self.voxel_size_y
        
    def get_min_z(self):
        return -0.5*self.voxel_num_z*self.voxel_size_z
        
    def get_max_z(self):
        return 0.5*self.voxel_num_z*self.voxel_size_z
        
    
class SinogramGeometry:
    
    def __init__(self, \
                 geom_type, \
                 dimension, \
                 angles, \
                 pixel_num_h=None, \
                 pixel_size_h=None, \
                 pixel_num_v=None, \
                 pixel_size_v=None, \
                 dist_source_center=None, \
                 dist_center_detector=None, \
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
        
        # Only implement 2D for now
        #if dimension == '2D':
        self.pixel_num_h = pixel_num_h
        self.pixel_size_h = pixel_size_h
        self.pixel_num_v = pixel_num_v
        self.pixel_size_v = pixel_size_v
        #else:
        #   NotImplemented
        
        
        #pixel_num_h
        #pixel_size_h
        #pixel_num_v
        #pixel_size_v
        
        #self.source_x = 0.0
        #self.detector_x = 0.0
        #self.pixel_h_size = 1.0
        #self.pixel_v_size = 1.0
        #self.num_of_vertical_pixels = 0
        #self.num_of_horizontal_pixels = 0
        
        #self.has_offsets = False 
        #self.m_radius = 1.0
        #self.experiment_name = ''
        #self.pixels_per_voxel = pixels_per_voxel #passed as argument
        
                
