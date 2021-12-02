# from cil.optimisation.algorithms import Algorithm
# from cil.optimisation.functions import IndicatorBox, Function
# from cil.optimisation.functions.LeastSquares import LeastSquares
# import numpy as np

# ##################################################
# ########### Design: PLAN A #######################
# ##################################################

# class SubsetGradientAlgorithm(Algorithm):
    
#     """
#         Generic SubsetGradientAlgorithm Class: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9226504
#         SAG, SAGA, SVRG
        
#         # Maybe create a class for the Gradient memory, to decide
#     """    

#     # precondition = True/False and how do we format it. Not implemented atm
    
#     def __init__(self, x0, function, step_size=1, **kwargs):
        
#         # inherits from Algorithm base                
#         super(SubsetGradientAlgorithm, self).__init__(**kwargs)        
        
#         # init calls set_up
#         self.set_up(x0 = x0, function = function, step_size=step_size, **kwargs)        
        
#     def set_up(self, x0, function, step_size, **kwargs):
        
#         # Need to check that this is a BlockFunction???
#         # assume f has gradient method implemented otherwise raise Error.
#         self.function = function
            
#         # Check that stepsize is positive
#         self.step_size = step_size
        
#         # Get num of subsets
#         self.num_subsets = len(function) 
        
#         # initial
#         self.x0 = x0
        
#         # non-negative constraint, for TV will be include, to decide
#         self.constraint = IndicatorBox(lower=0.0)
        
#         # call memory_init method
#         self.memory = self._memory_init()
        
            
#     def _memory_init(self):
#         """
#             init of v_i's and g_bar                    
#         """
#         raise NotImplemented
    
#     def _memory_update(self):
#         """
#             update of v_i's and g_bar            
#         """
#         raise NotImplemented
    
        
#     def _approx_gradient(self, i):
#         """
#          Defined by the specific algorithm: \tilde{\nabla} 
         
#          i: subset number
         
#         """
#         raise NotImplemented
                    
#     def update(self):
                
#         #choose random subset, int or not?
#         subset_num = int(np.random.choice(self.num_subsets))
                
#         sub_grad_new = self.function[subset_num].gradient(self.x0)        
                        
#         # gradient step, missing preconditioning 
#         self.x0 = self.constraint(self.x0 + self.step_size * self._approx_gradient(subset_num, sub_grad_new))
        
#         # memory step
#         self._memory_update(subset_num, sub_grad_new)
        
#     def update_objective(self):
        
#         """
#             objective value
            
#             # Call method of BlockFunction, needs a blockdatacontainer, instead we iterate on each
#             # function.
#         """
                
#         s = 0
#         for i in range(self.num_subsets):
#             s += self.function[i](self.x0)
#         return s
                 

# class SAGA(SubsetGradientAlgorithm):
    
    
#     """
#         SAGA algorithm
#     """
    
#     def __init__(self, **kwargs):
        
#         # inherits from StochasticAlgorithm base class and calls parent method: set_up
#         super(SAGA, self).__init__(**kwargs) 
        
#         # default gradient init to be zero
#         self.init_grad_zero = kwargs.get('init_grad_zero', True)
                     
#     def _memory_init(self):
        
#         """
        
#             initialize subset gradient and full gradient and store in memory
        
#         """
        
#         self.subset_gradients = []
#         self.full_gradient = self.x0.copy()*0.0
        
#         if self.init_grad_zero:
            
#             for i in range(self.num_subsets):
#                 self.subset_gradients.append(self.x0.copy()*0.0)
              
#         else:
            
#             for i in range(self.num_subsets):
                
#                 sub_grad = self.function[i].gradient(self.x0)
#                 self.subset_gradients.append(sub_grad)
#                 self.full_gradient += sub_grad
                
#             # average over self.num_subsets    
#             self.full_gradient /= self.num_subsets
                                                    
    
#     def _memory_update(self, subset_num, sub_grad_new):
        
#         """ 
        
#           subset_num: the number of the subset
          
#           update sub_grad of subset_num          
          
#           full_gradient = full_gradient - 1/num_subsets * sub_grad_old + 1/num_subsets * sub_grad_new
          
          
#         """
        
#         sub_grad_old = self.subset_gradients[subset_num]
        
#         self.full_gradient += (sub_grad_new - sub_grad_old)/self.num_subsets
#         self.subset_gradients[subset_num] = sub_grad_new
                
    
#     def _approx_gradient(self, subset_num, sub_grad_new):
        
        
#         return sub_grad_new - self.subset_gradients[subset_num] + self.full_gradient

        

##################################################
########### Design: PLAN B #######################
##################################################

# Create  GradientEstimator class
    
#     - has a __call__ method computing the sum etc
#     - has a "approx gradient" method, not Implemented by default
#     - has method that computes full gradient
#     - 

# Create a Child classs from GeneralisedFunction class, e.g., 
#  FunctionSAGA, decide on the name
#  FunctionSAG, decide on the name
#  FunctionSVRG, decide on the name

# Create accel version of each algos, e.g.,


#################################################################################
######## Implementation of PLAN B below #########################################
#################################################################################

# %%

from cil.optimisation.algorithms import Algorithm


class ApproximateGradientFunction(object):

    
    def __init__(self, function, **kwargs):

        self.function = function
        self.num_subsets = self.function.length
        
    def __call__(self, x):

        s = 0
        for i in range(self.num_subsets):
            s += self.function[i](x)
        return s    

    def _full_gradient(self, x, out=None):

        s = 0
        for i in range(self.num_subsets):
            s += self.function[i].gradient(x)
        return s                
    
    def gradient(self, x, out=None):
        """        
            maybe calls _approx_grad inside        
        """
        raise NotImplemented


    def subset_gradient(self, x, subset_num,  out=None):

        return self.function[subset_num].gradient(x)


    def next_subset(self):

        raise NotImplemented


class SAGA_Function(ApproximateGradientFunction):

    def __init__(self, function):

        self.initialise_gradients = False
        
        super(SAGA_Function, self).__init__(function)

    def gradient(self, x, out=None):

        if not self.initialise_gradients:
            self.memory_init(x) 

        # random choice of subset
        self.next_subset()

        subset_grad_old = self.subset_gradients[self.subset_num]
        full_grad_old = self.full_gradient

        # This is step 6 of the SAGA algo, and we multiply by the num_subsets to take care the (1/n) weight
        # step below to be optimised --> multiplication
        subset_grad = self.num_subsets * self.function[self.subset_num].gradient(x)

        tmp = subset_grad - subset_grad_old + full_grad_old

        
        self.memory_update(subset_grad)
     
        return tmp


    def memory_update(self, subset_grad):

        # step below to be optimised --> div
        self.full_gradient += (subset_grad - self.subset_gradients[self.subset_num])/self.num_subsets
        self.subset_gradients[self.subset_num] = subset_grad 
        

    def next_subset(self):
        
        self.subset_num = int(np.random.choice(self.num_subsets))

    def memory_init(self, x):
        
        """        
            initialize subset gradient (v_i_s) and full gradient (g_bar) and store in memory.

        """


        self.initialise_gradients = True

        # this is the memory init = subsets_gradients + full gradient
        self.subset_gradients = [x.copy()*0.0 for _ in range(self.num_subsets)]
        self.full_gradient = x.copy()*0.0

        # self.memory = [self.x.copy()*0.0 for _ in range(self.num_subsets)]
        
        # if self.init_grad_zero:
            
        #     for i in range(self.num_subsets):
        #         self.subset_gradients.append(self.x0.copy()*0.0)
              
        # else:
            
        #     for i in range(self.num_subsets):
                
        #         sub_grad = self.function[i].gradient(self.x0)
        #         self.subset_gradients.append(sub_grad)
        #         self.full_gradient += sub_grad
                
        #     # average over self.num_subsets    
        #     self.full_gradient /= self.num_subsets
                                                    



#%%
class GradientDescent(Algorithm):
    """
        Gradient Descent w/o Armijo rule
    """

    def  __init__(self, initial=None, objective_function=None, step_size=1, **kwargs):

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

# %%

# Import libraries

from cil.framework import  AcquisitionGeometry

from cil.optimisation.functions import L2NormSquared, BlockFunction, MixedL21Norm, IndicatorBox
from cil.optimisation.operators import GradientOperator, BlockOperator
from cil.optimisation.algorithms import PDHG
from cil.plugins.astra.operators import ProjectionOperator
from cil.plugins.astra.processors import FBP
from cil.plugins import TomoPhantom
from cil.utilities.display import show2D, show_geometry
from cil.utilities import noise

import matplotlib.pyplot as plt

import numpy as np
# %%


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

# %%

# Create projection operator using Astra-Toolbox. Available CPU/CPU
A = ProjectionOperator(ig, ag, device = 'gpu')

# Create an acqusition data (numerically)
sino = A.direct(phantom)

# Simulate Gaussian noise for the sinogram
gaussian_var = 0.5
gaussian_mean = 0

n1 = np.random.normal(gaussian_mean, gaussian_var, size = ag.shape)
                      
noisy_sino = ag.allocate()
noisy_sino.fill(n1 + sino.array)
noisy_sino.array[noisy_sino.array<0]=0

# Show numerical and noisy sinograms
show2D([phantom, sino, noisy_sino], title = ['Ground Truth','Sinogram','Noisy Sinogram'], num_cols=3, cmap = 'inferno')


# %%

from cil.processors import Slicer
from cil.optimisation.functions import LeastSquares

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


#%%

F_saga = SAGA_Function(F) 

initial = ig.allocate(0)
gd = GradientDescent(initial=initial, objective_function=F_saga,
                     step_size=0.0001, update_objective_interval=1, 
                     max_iteration=1000)
gd.run(200)


#%%
show2D(gd.solution, origin="upper")



# %%
from matplotlib.pyplot import plot
plt.figure()
plt.plot(gd.objective)
plt.show()
# %%

#####################################################################################
#####################################################################################
#####################################################################################

## Future Steps:

# on EstimatedGradientSumFunction class:
#       RedefinedGradientSumFunction?
#       ApproximatedGradientSumFunction?
#  - extend SumFunction to a list of functions (see issue #1091)
#  - implement ApproximatedGradientFunction as a child of SumFunction
#  - add more methods for the Lipschitz constant (list of each, average, max...)



#  ON SAGA
#  more options on initialisation (SAGA aglo)
#  more options on subset selection (SAGA aglo)
#  check (1/n) scale factor (SAGA aglo)
#  implement SVRG, SAG, SGD
#  epochs counter


# ON GradientDescent
#  implement non-negativity constraint

#  preconditioning : should be a part of ApproximatedGradientFunction or of Algorithm?

#  test warm start -> in Reconstruction class?



# Acceleration versions of algos above
#    - ISTA?
#    - FISTA-like 
#    - Katyusha
#    - Non-linear accel
#    ** We have GradientDescent (similar to GD algo of CIL), and FISTA








