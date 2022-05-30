# Copyright 2022 United Kingdom Research and Innovation
# Copyright 2022 The University of Manchester

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from cil.optimisation.functions import Function, BlockFunction, MixedL21Norm, ZeroFunction, L2NormSquared, IndicatorBox
from cil.optimisation.operators import GradientOperator, BlockOperator,IdentityOperator, ZeroOperator, SymmetrisedGradientOperator
from cil.optimisation.algorithms import PDHG
import numpy as np


class TotalGeneralisedVariation(Function):

    r""" Total Generalised Variation (TGV) Function, see :cite:`Bredies2010`.

        .. math:: \mathrm{TGV}_{\alpha, \beta}(u) := \underset{u}{\mathrm{argmin}}\,\alpha \|\nabla u - w\|_{2,1} + \beta\|\mathcal{E}w\|_{2,1}


        Notes
        -----
        The :code:`TotalGeneralisedVariation` (TGV) :code:`Function` acts as a compositite function, i.e.,
        the composition of a separable function 
        
        .. math:: f(z_{1}, z_{2}) = f_{1}(z_{1}) + f_{2}(z_{2}) = \alpha\|z_{1}\|_{2,1} + \beta\|z_{2}\|_{2,1}

        and the operator

        .. math:: K = \begin{bmatrix}
                    \nabla & -\mathbb{I}\\
                    \mathbb{O} & \mathcal{E}
                  \end{bmatrix}
        
        Therefore, 

        .. math:: f(K \begin{bmatrix}
                       u \\
                       w 
                     \end{bmatrix}) = f_{1}(\nabla u - w) + f_{2}(\mathcal{E}w)
        

        In that case, the proximal operator of TGV does not have an exact solution and we use an iterative 
        algorithm to solve:


        .. math:: \mathrm{prox}_{\tau \mathrm{TGV}_{\alpha,\beta}}(b) := \underset{u}{\mathrm{argmin}} \frac{1}{2\tau}\|u - b\|^{2} + \mathrm{TGV_{\alpha, \beta}}(u) \Leftrightarrow

        .. math:: \underset{u,w}{\mathrm{argmin}} \frac{1}{2\tau}\|u - b\|^{2} +  \alpha \|\nabla u - w\|_{2,1} + \beta\|\mathcal{E}w\|_{2,1} 

        
        The algorithm used for the proximal operator of TGV is the Primal Dual Hybrid Algorithm, see :class:`.PDHG`.


        Parameters
        ----------
        max_iteration : :obj:`int`, default = 100
            Maximum number of iterations for the PDHG algorithm.
        correlation : :obj:`str`, default = `Space`
            Correlation between `Space` and/or `SpaceChannels` for the :class:`.GradientOperator`.
        backend :  :obj:`str`, default = `c`      
            Backend to compute the :class:`.GradientOperator`  
        split : :obj:`boolean`, default = False
            Splits the Gradient into spatial gradient and spectral or temporal gradient for multichannel data.
        verbose : :obj:`int`, default = 0
            Prints primal, dual and primal-dual objectives of the :class:`PDHG` algorithm.
        update_objective_interval : :obj:`int`, default = :code:`max_iteration`
            Updates the objectives of the :class:`PDHG` algorithm.


        Examples
        --------

        .. math:: \mathrm{prox}_{\tau\mathrm{TGV}_{\alpha, \beta}}(b) = \underset{u}{\mathrm{argmin}} \frac{1}{2\tau}\|u - b\|^{2} + \mathrm{TGV}_{\alpha, \beta}(u) 


        >>> alpha1 = 1.0
        >>> alpha0 = 2.0
        >>> TGV = TotalGeneralisedVariation(alpha1 = alpha1, alpha0 = alpha0)
        >>> sol = TGV.proximal(b, tau = 1.0)


        """   
        
            
    def __init__(self,
                 reg_parameter=1.0,
                 alpha1 = 1.0,
                 alpha0 = 2.0,
                 max_iteration = 100, 
                 update_objective_interval = 50,
                 correlation = "Space",
                 backend = "c",
                 split = False,
                 verbose = 0,                  
                 warmstart=False):
        
        super(TotalGeneralisedVariation, self).__init__(L = None)

        # regularisation parameters for TGV
        self.alpha1 = alpha1
        self.alpha0 = alpha0
        self.reg_parameter = reg_parameter
                
        # Iterations for PDHG_TGV
        self.iterations = max_iteration
        
        # correlation space or spacechannels
        self.correlation = correlation

        # backend for the gradient
        self.backend = backend        
        
        # splitting Gradient
        self.split = split
                        
        self.verbose = verbose
        self.update_objective_interval = update_objective_interval

        # warm-start
        self.warmstart  = warmstart
        if self.warmstart:
            self.hasstarted = False        


    def __call__(self, x):
        
        if not hasattr(self, 'pdhg'):   
            return 0.0        
        else:              
            # Compute alpha1 * || Du - w || + alpha0 * ||Ew||, 
            # where (u,w) are solutions coming from the proximal method below.
            tmp = self.f(self.pdhg.operator.direct(self.pdhg.solution))
            return tmp

    def _setup_solver(self, x, tau=1.0):

        # tau_sqrt = np.sqrt(tau)
        
        if not hasattr(self, 'domain'):
            
            # sirf compatibility
            try:
                self.domain = x.geometry
            except:
                self.domain = x            
                        
        if not hasattr(self, 'operator'):
            
            self.Gradient = GradientOperator(self.domain, correlation = self.correlation, backend = self.backend)  
            self.SymGradient = SymmetrisedGradientOperator(self.Gradient.range, correlation = self.correlation, backend = self.backend)  
            self.ZeroOperator = ZeroOperator(self.domain, self.SymGradient.range)
            self.IdentityOperator = - IdentityOperator(self.Gradient.range)

            #    BlockOperator = [ Gradient      - Identity  ]
            #                    [ ZeroOperator   SymGradient] 
            self.operator = BlockOperator(self.Gradient, self.IdentityOperator, 
                                          self.ZeroOperator,self.SymGradient,shape=(2,2)) 

        if not all(hasattr(self, attr) for attr in ["g"]):
            self.g1 = 0.5*L2NormSquared(b = x)
            self.g2 = ZeroFunction()               
            self.g = BlockFunction(self.g1, self.g2)


        if not all(hasattr(self, attr) for attr in ["f"]):
            # parameters to set up PDHG algorithm
            self.f1 = tau*self.alpha1 * MixedL21Norm()
            self.f2 = tau*self.alpha0 * MixedL21Norm()
            self.f = BlockFunction(self.f1, self.f2)  
         
        if self.warmstart:     
            if self.hasstarted:
                tmp_initial = self.pdhg.solution
            else:
                tmp_initial = None
                self.hasstarted = True 
        else:
            tmp_initial = None

        # setup PDHG                           
        self.pdhg = PDHG(initial = tmp_initial, f = self.f, g=self.g, operator = self.operator,
                update_objective_interval = self.update_objective_interval,
                max_iteration = self.iterations) 
        self.pdhg.run(verbose=self.verbose) 

                                     
    def proximal(self, x, tau = 1.0, out = None):
        
        self._setup_solver(x, tau)  
                    
        if out is None:
            return self.pdhg.solution[0]
        else:
            out.fill(self.pdhg.solution[0])            

    def convex_conjugate(self,x):  
        
        return 0.0
    
    

