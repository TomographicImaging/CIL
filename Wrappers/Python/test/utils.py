from cil.framework import AcquisitionGeometry
from cil.framework import cilacc
import numpy as np
import os

#import cvxpy

try:
    import cvxpy
    has_cvxpy = True
except ModuleNotFoundError:
    has_cvxpy = False
print ("has_cvxpy\t{}".format(has_cvxpy))    

try:
    import tigre
    has_tigre = True
except ModuleNotFoundError:
    has_tigre = False
print ("has_tigre\t{}".format(has_tigre))


try:
    import astra
    has_astra = True
except ModuleNotFoundError:
    has_astra = False
print ("has_astra\t{}".format(has_astra))


try:
    cilacc.filter_projections_avh
    has_ipp = True
except:
    has_ipp = False
print ("has_ipp\t{}".format(has_ipp))


def has_nvidia_smi():
    return os.system('nvidia-smi') == 0

def has_gpu_tigre():

    if not has_tigre:
        return False

    has_gpu = True
    if has_nvidia_smi():
        from cil.plugins.tigre import ProjectionOperator
        from tigre.utilities.errors import TigreCudaCallError

        N = 3
        angles = np.linspace(0, np.pi, 2, dtype='float32')

        ag = AcquisitionGeometry.create_Cone2D([0,-100],[0,200])\
                                .set_angles(angles, angle_unit='radian')\
                                .set_panel(N, 0.1)\
                                .set_labels(['angle', 'horizontal'])
        
        ig = ag.get_ImageGeometry()
        
        data = ig.allocate(1)

        Op = ProjectionOperator(ig, ag)

        try:
            Op.direct(data)
            has_gpu = True
        except TigreCudaCallError:
            has_gpu = False
    else:
        has_gpu = False

    
    print ("has_gpu_tigre\t{}".format(has_gpu))
    return has_gpu


def has_gpu_astra():

    if not has_astra:
        return False

    has_gpu = False
    if has_nvidia_smi():
        try:
            astra.test_CUDA()
            has_gpu = True
        except:
            pass

    print ("has_gpu_astra\t{}".format(has_gpu))
    return has_gpu


