import astra

def convert_geometry_to_astra(volume_geometry, sinogram_geometry):
    # Set up ASTRA Volume and projection geometry, not stored
    if sinogram_geometry.dimension == '2D':
        vol_geom = astra.create_vol_geom(volume_geometry.voxel_num_x, 
                                         volume_geometry.voxel_num_y, 
                                         volume_geometry.get_min_x(), 
                                         volume_geometry.get_max_x(), 
                                         volume_geometry.get_min_y(), 
                                         volume_geometry.get_max_y())
        
        if sinogram_geometry.geom_type == 'parallel':
            proj_geom = astra.create_proj_geom('parallel',
                                               sinogram_geometry.pixel_size_h,
                                               sinogram_geometry.pixel_num_h,
                                               sinogram_geometry.angles)
        elif sinogram_geometry.geom_type == 'cone':
            proj_geom = astra.create_proj_geom('fanflat',
                                               sinogram_geometry.pixel_size_h,
                                               sinogram_geometry.pixel_num_h,
                                               sinogram_geometry.angles,
                                               sinogram_geometry.dist_source_center,
                                               sinogram_geometry.dist_center_detector)
        else:
            NotImplemented
            
    elif sinogram_geometry.dimension == '3D':
        vol_geom = astra.create_vol_geom(volume_geometry.voxel_num_x, 
                                         volume_geometry.voxel_num_y, 
                                         volume_geometry.voxel_num_z, 
                                         volume_geometry.get_min_x(), 
                                         volume_geometry.get_max_x(), 
                                         volume_geometry.get_min_y(), 
                                         volume_geometry.get_max_y(), 
                                         volume_geometry.get_min_z(), 
                                         volume_geometry.get_max_z())
        
        if sinogram_geometry.geom_type == 'parallel':
            proj_geom = astra.create_proj_geom('parallel3d',
                                               sinogram_geometry.pixel_size_h,
                                               sinogram_geometry.pixel_size_v,
                                               sinogram_geometry.pixel_num_v,
                                               sinogram_geometry.pixel_num_h,
                                               sinogram_geometry.angles)
        elif sinogram_geometry.geom_type == 'cone':
            proj_geom = astra.create_proj_geom('cone',
                                               sinogram_geometry.pixel_size_h,
                                               sinogram_geometry.pixel_size_v,
                                               sinogram_geometry.pixel_num_v,
                                               sinogram_geometry.pixel_num_h,
                                               sinogram_geometry.angles,
                                               sinogram_geometry.dist_source_center,
                                               sinogram_geometry.dist_center_detector)
        else:
            NotImplemented
            
    else:
        NotImplemented
    
    return vol_geom, proj_geom