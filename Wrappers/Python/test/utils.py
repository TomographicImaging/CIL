try:
    import tigre
    has_tigre = True
except ModuleNotFoundError:
    has_tigre = False
else:
    from cil.plugins.tigre import ProjectionOperator
    from tigre.utilities.errors import TigreCudaCallError

try:
    import astra
    has_astra = True
except ModuleNotFoundError:
    has_astra = False

import os
from cil.framework import AcquisitionGeometry, ImageGeometry
import numpy as np


def has_nvidia_smi():
    return os.system('nvidia-smi') == 0


def has_gpu_tigre():
    print ("has_gpu_tigre")
    if not has_nvidia_smi():
        return False
    N = 3
    angles = np.linspace(0, np.pi, 2, dtype='float32')

    ag = AcquisitionGeometry.create_Cone2D([0,-100],[0,200])\
                            .set_angles(angles, angle_unit='radian')\
                            .set_panel(N, 0.1)\
                            .set_labels(['angle', 'horizontal'])
    
    ig = ag.get_ImageGeometry()
    
    data = ig.allocate(1)

    Op = ProjectionOperator(ig, ag)

    has_gpu = True
    try:
        res = Op.direct(data)
    except TigreCudaCallError:
        has_gpu = False
    return has_gpu

def has_gpu_astra():
    print ("has_gpu_astra")
    if not has_nvidia_smi():
        return False
    has_gpu = True
    try:
        astra.test_CUDA()
    except:
        has_gpu = False
    return has_gpu

def has_ipp():
    print ("has_ipp")
    from cil.reconstructors import FBP
    try:
        fbp = FBP('fail')
        return True
    except ImportError:
        return False
    except:
        pass
