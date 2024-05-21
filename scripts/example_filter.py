# cil imports
from cil.processors import TransmissionAbsorptionConverter

import os

from cil.io import ZEISSDataReader
from cil.processors import TransmissionAbsorptionConverter
from cil.recon import FDK
from cil.recon.FBP import GenericFilteredBackProjection

# External imports
import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.fft import fftfreq

from cil.framework import AcquisitionGeometry

path = r'c:/Users/zvm34551/Coding_environment/DATA/CIL-demos_Data/walnut/valnut/'
filename = os.path.join(path, "valnut_2014-03-21_643_28/tomo-A/valnut_tomo-A.txrm")
data = ZEISSDataReader(file_name=filename).read()

data = TransmissionAbsorptionConverter()(data)

data.reorder(order='tigre')
ig = data.geometry.get_ImageGeometry()

#filter custom

#%% Setup Geometry
voxel_num_xy = 16
voxel_num_z = 4

mag = 2
src_to_obj = 50
src_to_det = src_to_obj * mag

pix_size = 0.2
det_pix_x = voxel_num_xy
det_pix_y = voxel_num_z

num_projections = 36
angles = np.linspace(0, 360, num=num_projections, endpoint=False)

ag = AcquisitionGeometry.create_Cone2D([0,-src_to_obj],[0,src_to_det-src_to_obj])\
                                    .set_angles(angles)\
                                    .set_panel(det_pix_x, pix_size)\
                                    .set_labels(['angle','horizontal'])

ig = ag.get_ImageGeometry()

ag3D = AcquisitionGeometry.create_Cone3D([0,-src_to_obj,0],[0,src_to_det-src_to_obj,0])\
                                .set_angles(angles)\
                                .set_panel((det_pix_x,det_pix_y), (pix_size,pix_size))\
                                .set_labels(['angle','vertical','horizontal'])
ig3D = ag3D.get_ImageGeometry()

ad3D = ag3D.allocate('random')
ig3D = ag3D.get_ImageGeometry()
print(ad3D)

fdk = GenericFilteredBackProjection(ad3D)
#fdk = FDK(data, ig)
FBP_filter = 'hamming'#'hann'#'cosine'#'shepp-logan'#'ram-lak'#'custom'######'custom'#'hann'###'shepp-logan'#'ram-lak'## 'ram-lak'
cutoff = 0.5
if FBP_filter =='custom':
    filter_length = 256
    freq = fftfreq(filter_length)
    freq*=2
    ramp = abs(freq)
    ramp[ramp>cutoff]=0
    FBP_filter = ramp*(np.cos(freq*np.pi*4)+1*np.cos(1/5*freq*np.pi/2))/2

#recon = FBP(data, ig, FBP_filter)
fdk.set_filter(FBP_filter, cutoff)
plot = fdk.plot_filter()
print(plot)
plot.show()
#recon = fdk.run()
#show2D(recon, slice_list=[('vertical',512), ('horizontal_x', 512)], fix_range=(-0.01, 0.06))
#da = plot.data()
#print(da)