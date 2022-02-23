'''

This is a test for SGD comparing to ISTA and SAGA on noisy CT example.
Seems that SGD can be even faster than SAGA in this example. (There is a further need of implementing shrinking step-sizes for SGD)

Here ISTA function is modified to take into account the n_subsets (default=1 for detereministic ISTA) to ensure right scaling factor

--Billy 23/2/2022

'''

# Import libraries

from cil.framework import  AcquisitionGeometry
from cil.optimisation.algorithms import Algorithm
from cil.optimisation.functions import Function, L2NormSquared, BlockFunction, MixedL21Norm, IndicatorBox, LeastSquares, TotalVariation
from cil.optimisation.operators import GradientOperator, BlockOperator
from cil.optimisation.algorithms import PDHG
from cil.plugins.astra.operators import ProjectionOperator
# from cil.plugins.tigre import ProjectionOperator
from cil.plugins.astra.processors import FBP
from cil.plugins import TomoPhantom
from cil.utilities.display import show2D, show_geometry
from cil.utilities import noise
from cil.processors import Slicer

import matplotlib.pyplot as plt

import numpy as np

import importlib

# Define SAGAFunction
import sys
cil_path = '/store/CIA/jt814/cil_codes/CIL-master/Wrappers'
fun_path = 'Wrappers/Python/cil/optimisation/functions'
sys.path.append(cil_path + fun_path)
import SubsetSumFunction


# Detectors
N = 256
detectors =  N

# Angles
angles = np.linspace(0,180,180, dtype='float32')

# Setup acquisition geometry
ag = AcquisitionGeometry.create_Parallel2D()\
                        .set_angles(angles)\
                        .set_panel(detectors, pixel_size=0.1)
# Get image geometry
ig = ag.get_ImageGeometry()

# Get phantom
phantom = TomoPhantom.get_ImageData(12, ig)

# Create projection operator using Astra-Toolbox. Available CPU/CPU
A = ProjectionOperator(ig, ag, device='gpu')

# Create an acqusition data (numerically)
sino = A.direct(phantom)

# Simulate Gaussian noise for the sinogram
noise_scale = 5
tv_reg_scale = 100

gaussian_var = 0.5 * noise_scale
gaussian_mean = 0

n1 = np.random.normal(gaussian_mean, gaussian_var, size=ag.shape)

noisy_sino = ag.allocate()
noisy_sino.fill(n1 + sino.array)
noisy_sino.array[noisy_sino.array < 0] = 0
# noisy_sino.fill(sino.array)

# Show numerical and noisy sinograms
show2D([phantom, sino, noisy_sino], title=['Ground Truth', 'Sinogram', 'Noisy Sinogram'], num_cols=3, cmap='inferno')


# Setup and run the FBP algorithm
fbp_recon = FBP(ig, ag,  device = 'gpu')(noisy_sino)

# Show reconstructions
show2D([phantom, fbp_recon],
       title = ['Ground Truth','FBP reconstruction'],
       cmap = 'inferno', fix_range=(0,1.), size=(10,10))


class GradientDescent(Algorithm):
    """
        Gradient Descent w/o Armijo rule
        Convergence guarantee: step-size < 2/L where L is the lipschitz constant
        of the gradient of the objective function
    """

    def __init__(self, initial=None, objective_function=None, step_size=1, **kwargs):
        super(GradientDescent, self).__init__(**kwargs)

        self.set_up(initial=initial, objective_function=objective_function, step_size=step_size)

    def set_up(self, initial, objective_function, step_size):
        self.x = initial.copy()
        self.objective_function = objective_function
        self.x_update = initial.copy()
        self.step_size = step_size
        self.update_objective()
        self.update_step_size = False
        self.configured = True

    def update(self):
        '''Single iteration'''

        self.x_update = self.objective_function.gradient(self.x)

        self.x_update *= -self.step_size
        self.x += self.x_update

    def update_objective(self):
        self.loss.append(self.objective_function(self.x))


importlib.reload(SubsetSumFunction)
from SubsetSumFunction import SumFunction


class ISTA(Algorithm):
    """
        Iterative Shrinkage-Thresholding Algorithm
        Goal: minimize f + g with f smooth, g simple

        Convergence guarantee:
        - f,g convex
        - step-size < 1/L where the gradient of f is L-Lipschitz

        See Section 3 of Beck, Amir, and Marc Teboulle,
        "A fast iterative shrinkage-thresholding algorithm for linear inverse problems."
        SIAM journal on imaging sciences 2.1 (2009)
    """

    def __init__(self, initial=None, f=None, g=None, step_size=1, n_subsets=1, **kwargs):
        super(ISTA, self).__init__(**kwargs)

        self.set_up(initial=initial, f=f, g=g, step_size=step_size, n_subsets=n_subsets)

    def set_up(self, initial, f, g, step_size, n_subsets):
        self.f = f
        self.g = g
        self.x = initial.copy()
        self.objective_function = SumFunction(f, g)
        self.x_update = initial.copy()
        self.step_size = step_size
        self.update_objective()
        self.update_step_size = False
        self.configured = True
        self.n_subsets = n_subsets

    def update(self):
        '''Single iteration'''

        self.x_update = self.f.gradient(self.x)
        self.x_update *= -self.step_size
        self.x += self.x_update
        self.x = self.g.proximal(self.x, self.step_size/self.n_subsets)

    def update_objective(self):
        self.loss.append(self.objective_function(self.x))



# gradient descent (no non-neg constraint)
# num_epochs = 20
# f_gd = LeastSquares(A, noisy_sino)
# initial = ig.allocate(0)
# step_size = 2 / f_gd.L
# gd = GradientDescent(initial=initial, objective_function=f_gd,
#                      step_size=0.0009, update_objective_interval=1,
#                      max_iteration=1000)
# gd.run(num_epochs, verbose=0)


# ISTA
num_epochs = 20
f_gd = LeastSquares(A, noisy_sino)
initial = ig.allocate(0)
step_size = 1 / f_gd.L
g = IndicatorBox(lower=0)
ista = ISTA(initial=initial, f=f_gd, g=g,
                     step_size=step_size, update_objective_interval=1,
                     max_iteration=1000)
ista.run(num_epochs, verbose=0)


# Define number of subsets
n_subsets = 10

# Initialize the lists containing the F_i's and A_i's
f_subsets = []
A_subsets = []

# Define F_i's and A_i's
for i in range(n_subsets):
    # Total number of angles
    n_angles = len(ag.angles)
    # Divide the data into subsets
    data_subset = Slicer(roi = {'angle' : (i,n_angles,n_subsets)})(noisy_sino)

    # Define A_i and put into list
    ageom_subset = data_subset.geometry
    Ai = ProjectionOperator(ig, ageom_subset)
    A_subsets.append(Ai)

    # Define F_i and put into list
    fi = LeastSquares(Ai, b = data_subset)
    f_subsets.append(fi)


# Define F and K
F = BlockFunction(*f_subsets)



importlib.reload(SubsetSumFunction)
from SubsetSumFunction import SumFunction
from SubsetSumFunction import SAGAFunction, SGDFunction

num_epochs = 20
F_saga = SAGAFunction(F)
# admissible step-size is gamma = 1/ (3 max_i L_i)
step_size = 1 / (3*F_saga.Lmax)
initial = ig.allocate(0)
F_saga.memory_reset()
saga = ISTA(initial=initial,
            f=F_saga,
            g=g,
            step_size=step_size, update_objective_interval=n_subsets,
            max_iteration=10000)
saga.run(num_epochs * n_subsets, verbose=0)

F_sgd = SGDFunction(F)
# admissible step-size is gamma = 1/ (3 max_i L_i)
step_size = 1 / (1*F_sgd.Lmax)
initial = ig.allocate(0)
#F_sgd.memory_reset()
sgd = ISTA(initial=initial,
            f=F_sgd,
            g=g,
            step_size=step_size, update_objective_interval=n_subsets,
            max_iteration=10000)
sgd.run(num_epochs * n_subsets, verbose=0)



# subsequent subsets: 1, 2, etc...
def subset_select_function(a,b):
    return (a+1)%b






# compare results
plt.figure()
plt.semilogy(ista.objective, label="ISTA")
plt.semilogy(saga.objective, label="SAGA")
plt.semilogy(sgd.objective, label="SGD")


plt.legend()
plt.ylabel('Epochs')
plt.xlabel('Objective function')
plt.show()

# ISTA
num_epochs = 60
f_gd = LeastSquares(A, noisy_sino)
initial = ig.allocate(0)
step_size = 1 / f_gd.L
lb = 0.01 * tv_reg_scale
g = lb * TotalVariation(lower=0)
ista_tv = ISTA(initial=initial, f=f_gd, g=g,
                     step_size=step_size, update_objective_interval=1,
                     max_iteration=3000)
ista_tv.run(num_epochs, verbose=0)


# SAGA
num_epochs = 60
F_saga = SAGAFunction(F)
# admissible step-size is gamma = 1/ (3 max_i L_i)
step_size = 1 / (3*F_saga.Lmax)
initial = ig.allocate(0)
F_saga.memory_reset()
saga_tv = ISTA(initial=initial,
            f=F_saga,
            g=g,
            step_size=step_size, n_subsets=n_subsets, update_objective_interval=n_subsets,
            max_iteration=30000)
saga_tv.run(num_epochs * n_subsets, verbose=0)


# SGD
num_epochs = 60
F_sgd = SGDFunction(F)
# admissible step-size is gamma = 1/ (3 max_i L_i)
step_size = 1 / (1*F_sgd.Lmax)
initial = ig.allocate(0)
#F_sgd.memory_reset()
sgd_tv = ISTA(initial=initial,
            f=F_sgd,
            g=g,
            step_size=step_size, n_subsets=n_subsets, update_objective_interval=n_subsets,
            max_iteration=30000)
sgd_tv.run(num_epochs * n_subsets, verbose=0)


# SGD2 
num_epochs = 60
F_sgd2 = SGDFunction(F)
# admissible step-size is gamma = 1/ (3 max_i L_i)
step_size2 = 1 / (3*F_sgd2.Lmax)
initial = ig.allocate(0)
#F_sgd.memory_reset()
sgd_tv2 = ISTA(initial=initial,
            f=F_sgd2,
            g=g,
            step_size=step_size2, n_subsets=n_subsets, update_objective_interval=n_subsets,
            max_iteration=30000)
sgd_tv2.run(num_epochs * n_subsets, verbose=0)


show2D([ista.solution, ista_tv.solution, saga.solution, saga_tv.solution, sgd.solution, sgd_tv.solution],
       title=["ISTA","ISTA TV", "SAGA","SAGA TV","SGD","SGD TV"],
       origin="upper",
       fix_range=(0,1), num_cols=2,
       cmap='inferno')

# compare results
plt.figure()
plt.semilogy(ista_tv.objective, label="ISTA TV")
plt.semilogy(saga_tv.objective, label="SAGA TV")
plt.semilogy(sgd_tv.objective, label="SGD TV (1/L)")
plt.semilogy(sgd_tv2.objective, label="SGD TV (1/3L)")
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Objective function')
plt.show()

plt.figure()
plt.semilogy(ista_tv.objective, label="ISTA TV")
plt.semilogy(saga_tv.objective, label="SAGA TV")
plt.semilogy(sgd_tv.objective, label="SGD TV (1/L)")
plt.semilogy(sgd_tv2.objective, label="SGD TV (1/3L)")
plt.ylim([250000, 500000])
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Objective function')
plt.show()


