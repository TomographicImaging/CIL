from cil.framework import  AcquisitionGeometry
from cil.optimisation.algorithms import Algorithm
from cil.plugins.astra.operators import ProjectionOperator
from cil.plugins import TomoPhantom

import numpy as np

class Tomophantom(object):

    def  __init__(self, name, device):
        N = 256
        detectors =  N

        # Angles
        angles = np.linspace(0,180,180, dtype='float32')

        # Setup acquisition geometry
        ag = AcquisitionGeometry.create_Parallel2D()\
                                .set_angles(angles)\
                                .set_panel(detectors, pixel_size=0.1)
        # Get image geometry
        self.ig = ag.get_ImageGeometry()

        # Get phantom
        self.phantom = TomoPhantom.get_ImageData(12, self.ig)

        # Create projection operator using Astra-Toolbox. Available CPU/CPU
        self.A = ProjectionOperator(self.ig, ag, device=device)

        # Create an acqusition data (numerically)
        sino = self.A.direct(self.phantom)

        # Simulate Gaussian noise for the sinogram
        noise_scale = 1

        gaussian_var = 0.5 * noise_scale
        gaussian_mean = 0

        n1 = np.random.normal(gaussian_mean, gaussian_var, size=ag.shape)

        self.noisy_sino = ag.allocate()
        self.noisy_sino.fill(n1 + sino.array)
        self.noisy_sino.array[self.noisy_sino.array < 0] = 0

    def get_data(self):
        return self.noisy_sino
    
    def get_Fwd_Op(self):
        return self.A
    
    def get_GT(self):
        return self.phantom

    def get_init(self):
        return self.ig.allocate(0)

