#%%
from cil.framework import AcquisitionGeometry, ImageGeometry
from cil.utilities.display import plotter2D
import os, sys
# from tigre_geometry import TIGREProjectionOperator, make_circle
from cil.plugins.tigre import ProjectionOperator as TIGREProjectionOperator
import numpy as np
from cil.utilities.jupyter import islicer, link_islicer

#%% Setup Geometry
if __name__ == '__main__':

    # create a simple 
    voxel_num_xy = 255
    voxel_num_z = 15
    cs_ind = (voxel_num_z-1)//2

    mag = 2
    src_to_obj = 50
    src_to_det = src_to_obj * mag

    pix_size = 0.2
    det_pix_x = voxel_num_xy
    det_pix_y = voxel_num_z

    num_projections = 180
    angles = np.linspace(0, np.pi, num=num_projections, endpoint=False)

    ag = AcquisitionGeometry.create_Cone3D([0,-src_to_obj,0],[0,src_to_det-src_to_obj,0])\
                                    .set_angles(angles, angle_unit='radian')\
                                    .set_panel((det_pix_x,det_pix_y), (pix_size,pix_size))\
                                    .set_labels(['angle', 'vertical', 'horizontal'])

    ag_slice = AcquisitionGeometry.create_Cone2D([0,-src_to_obj],[0,src_to_det-src_to_obj])\
                                        .set_angles(angles, angle_unit='radian')\
                                        .set_panel(det_pix_x, pix_size)\
                                        .set_labels(['angle','horizontal'])

    ig_2D = ag_slice.get_ImageGeometry()
    ig_3D = ag.get_ImageGeometry()

    #%% Create phantom
    kernel_size = voxel_num_xy
    kernel_radius = (kernel_size - 1) // 2
    y, x = np.ogrid[-kernel_radius:kernel_radius+1, -kernel_radius:kernel_radius+1]

    circle1 = [5,0,0] #r,x,y
    dist1 = ((x - circle1[1])**2 + (y - circle1[2])**2)**0.5

    circle2 = [5,100,0] #r,x,y
    dist2 = ((x - circle2[1])**2 + (y - circle2[2])**2)**0.5

    circle3 = [25,0,100] #r,x,y
    dist3 = ((x - circle3[1])**2 + (y - circle3[2])**2)**0.5

    mask1 =(dist1 - circle1[0]).clip(0,1) 
    mask2 =(dist2 - circle2[0]).clip(0,1) 
    mask3 =(dist3 - circle3[0]).clip(0,1) 
    phantom = 1 - np.logical_and(np.logical_and(mask1, mask2),mask3)

    golden_data = ig_3D.allocate(0)
    for i in range(4):
        golden_data.fill(array=phantom, vertical=7+i)

    # golden_data_cs = golden_data.subset(vertical=cs_ind, force=True)
    golden_data_cs = ig_2D.allocate(0)
    
    #make_circle(20,60,10,golden_data_cs.as_array())
    golden_data_cs.fill(phantom)

    # 2D 
    print ("Create TIGRE Projection Operator")
    A2D = TIGREProjectionOperator(domain_geometry=ig_2D, range_geometry=ag_slice, \
         direct_method='interpolated', adjoint_method='matched')

    ad2D = A2D.direct(golden_data_cs)
    bck2D = A2D.adjoint(ad2D)

    #plotter2D([golden_data_cs, ad2D, bck2D], titles=['phantom', 'forward', 'backward'])


    A3D = TIGREProjectionOperator(domain_geometry=ig_3D, range_geometry=ag, \
         direct_method='interpolated', adjoint_method='matched')

    ad3D = A3D.direct(golden_data)
    bck3D = A3D.adjoint(ad3D)

    # height = 8
    # plotter2D([golden_data.subset(vertical=height, force=True), ad3D.subset(vertical=height, force=True), 
    # bck3D.subset(vertical=height, force=True)], 
    #     titles=['phantom', 'forward', 'backward'])
    islicer(ad3D, direction='vertical')
# %%
