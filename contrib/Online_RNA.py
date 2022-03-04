# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi
#   (Collaborative Computational Project in Tomographic Imaging), with
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from cil.optimisation.algorithms import Algorithm
import numpy

class Online_RNA(Algorithm):
    r"""
        Online regularised non-linear acceleration

        'Online Regularized Nonlinear Acceleration'
        Damien Scieur · Edouard Oyallon · Alexandre d'Aspremont · Francis Bach
        
        https://arxiv.org/pdf/1805.09639.pdf

        .. math:: x_N = g(y_{N-1}) \;  y_N = RNA(X,Y,\lambda,\beta)

        Parameters
        ----------
        algo : Algorithm
                Algorithm that you'd like accelerating, i.e. ISTA/FISTA/PDHG... The "g" in: x_{i+1} = g(x_i)
        RNA_reg_param : positive :obj:`float`, default = None
            Regularisation parameter for RNA
        N : positive :obj:`int`, default = 5
            Number of iterations saved in the memory.
        beta : positive/negative :obj:`float`, default = -1
            Mixing parameter for Anderson-type acceleration (beta=-1 for no mixing)
        kwargs: Keyword arguments
            Arguments from the base class :class:`.Algorithm`.
    """

    def  __init__(self, algo=None, RNA_reg_param=None, N=5, beta=-1, **kwargs):

        super(Online_RNA, self).__init__(**kwargs)

        self.set_up(algo=algo, RNA_reg_param=RNA_reg_param, N=N, beta=beta)


    def set_up(self, algo, RNA_reg_param, N, beta):
        self.algo = algo
        self.RNA_reg_param = RNA_reg_param
        self.N = N
        self.beta = beta
        self.configured = True
        self.Y = self.algo.x_old.array.ravel()[None].T
        self.algo.__next__()
        self.X =  self.algo.x_old.array.ravel()[None].T
        self.max_iteration = 100000000

    def RNA(self):
        Res = self.X-self.Y
        e = numpy.ones((Res.shape[1], 1))
        if self.iteration <= 1:
            Res_norm = 1
        else:
            Res_norm = numpy.linalg.norm(Res,2)**2
        c_la = numpy.linalg.lstsq(Res.T@Res + self.RNA_reg_param * Res_norm * numpy.eye(Res.shape[1]), e, rcond = None)[0]
        c_la /= numpy.sum(c_la)
        return (self.Y - self.beta * Res) @ c_la

    def update(self):
        if self.iteration == 0:
            x_extra = self.algo.x_old.array.ravel()[None].T
        else:
            x_extra = self.RNA()
            
            self.algo.x_old.fill(numpy.reshape(x_extra,self.algo.x.shape))

            self.x = self.algo.x_old

            if self.algo.g.lower is not None and self.algo.g.upper is not None:
                self.x = self.algo.g.proximal(self.x,1)

        self.algo.__next__()
    
        if self.iteration <= 1:
            self.Y = x_extra.copy()
            self.X = self.algo.x_old.copy().array.ravel()[None].T
        elif self.iteration <= (self.N):
            self.Y = numpy.hstack((self.Y, x_extra))
            self.X = numpy.hstack((self.X, self.algo.x_old.copy().array.ravel()[None].T))
        else:
            self.Y = numpy.hstack((self.Y[:, 1:], x_extra))
            self.X = numpy.hstack((self.X[:, 1:], self.algo.x_old.copy().array.ravel()[None].T))
            
    def update_objective(self):
        self.loss.append(self.algo.loss[-1])