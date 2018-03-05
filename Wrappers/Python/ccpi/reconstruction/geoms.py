
class VolumeGeometry:
    
    def __init__(self,grid,domain):
        self.domain = domain
        self.grid = grid
    
    
class SinogramGeometry:
    
    def __init__(self, \
                 geom_type, \
                 dimension, \
                 angles, \
                 grid, \
                 domain, \
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
        if dimension == '2D':
            self.grid = (grid,)  # grid assumed scalar in 2D
            self.domain = domain
        else:
            NotImplemented
        
                
