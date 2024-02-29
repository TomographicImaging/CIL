# cil imports
from cil.framework import ImageData, ImageGeometry
from cil.framework import AcquisitionGeometry, AcquisitionData

from cil.processors import Slicer, AbsorptionTransmissionConverter, TransmissionAbsorptionConverter

from cil.optimisation.functions import IndicatorBox
from cil.optimisation.algorithms import CGLS, SIRT

from cil.plugins.astra.operators import ProjectionOperator
from cil.plugins.astra.processors import FBP

from cil.plugins import TomoPhantom

from cil.utilities import dataexample
from cil.utilities.display import show2D, show1D, show_geometry

import os

from cil.io import ZEISSDataReader, TIFFWriter
from cil.processors import TransmissionAbsorptionConverter, Slicer
from cil.recon import FDK
from cil.utilities.display import show2D, show_geometry
from cil.utilities.jupyter import islicer, link_islicer

# External imports
import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.fft import fftfreq

path = r'c:/Users/zvm34551/Coding_environment/DATA/CIL-demos_Data/walnut/valnut/'
filename = os.path.join(path, "valnut_2014-03-21_643_28/tomo-A/valnut_tomo-A.txrm")
data = ZEISSDataReader(file_name=filename).read()

data = TransmissionAbsorptionConverter()(data)

data.reorder(order='tigre')
ig = data.geometry.get_ImageGeometry()

#filter custom

fdk = FDK(data, ig)
FBP_filter ='custom'#'hann'#'hamming'#'cosine'#'shepp-logan'#'ram-lak'## 'ram-lak'
cutoff = 0.4
if FBP_filter =='custom':
    filter_length = 2048
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
