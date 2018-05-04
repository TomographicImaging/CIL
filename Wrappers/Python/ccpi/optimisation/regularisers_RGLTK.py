# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018 Jakob Jorgensen, Daniil Kazantsev and Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from ccpi.filters.regularisers import ROF_TV, FGP_TV, SB_TV
from ccpi.filters.cpu_regularisers import TV_ENERGY
import numpy as np


class ROF_TV_regulariser(BaseFunction):
    def __init__(self,lambdaReg,iterationsTV,tolerance,time_marchstep,device):
        # set parameters
        self.lambdaReg = lambdaReg
        self.iterationsTV = iterationsTV
        self.time_marchstep = time_marchstep
        self.device = device # string for 'cpu' or 'gpu'
    def fun(self,x):
        # evaluate objective function of TV gradient        
        EnergyValTV = TV_ENERGY(np.asarray(x.as_array(), dtype=np.float32), np.asarray(x.as_array(), dtype=np.float32), self.lambdaReg, 2)
        return EnergyValTV
    def prox(self,x,Lipshitz):
        pars = {'algorithm' : ROF_TV, \
               'input' : np.asarray(x.as_array(), dtype=np.float32),\
                'regularization_parameter':self.lambdaReg*Lipshitz, \
                'number_of_iterations' :self.iterationsTV ,\
                'time_marching_parameter':self.time_marchstep}
        
        out = ROF_TV(pars['input'], 
              pars['regularization_parameter'],
              pars['number_of_iterations'],
              pars['time_marching_parameter'], self.device)
        return VolumeData(out, dimension_labels=x.dimension_labels)

class FGP_TV_regulariser(BaseFunction):
    def __init__(self,lambdaReg,iterationsTV,tolerance,methodTV,nonnegativity,printing,device):
        # set parameters
        self.lambdaReg = lambdaReg
        self.iterationsTV = iterationsTV
        self.tolerance = tolerance
        self.methodTV = methodTV
        self.nonnegativity = nonnegativity
        self.printing = printing
        self.device = device # string for 'cpu' or 'gpu'
    def fun(self,x):
        # evaluate objective function of TV gradient        
        EnergyValTV = TV_ENERGY(np.asarray(x.as_array(), dtype=np.float32), np.asarray(x.as_array(), dtype=np.float32), self.lambdaReg, 2)
        return EnergyValTV
    def prox(self,x,Lipshitz):
        pars = {'algorithm' : FGP_TV, \
               'input' : np.asarray(x.as_array(), dtype=np.float32),\
                'regularization_parameter':self.lambdaReg*Lipshitz, \
                'number_of_iterations' :self.iterationsTV ,\
                'tolerance_constant':self.tolerance,\
                'methodTV': self.methodTV ,\
                'nonneg': self.nonnegativity ,\
                'printingOut': self.printing}
        
        out = FGP_TV(pars['input'], 
              pars['regularization_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['methodTV'],
              pars['nonneg'],
              pars['printingOut'], self.device)
        return VolumeData(out, dimension_labels=x.dimension_labels)


class SB_TV_regulariser(BaseFunction):
    def __init__(self,lambdaReg,iterationsTV,tolerance,methodTV,printing,device):
        # set parameters
        self.lambdaReg = lambdaReg
        self.iterationsTV = iterationsTV
        self.tolerance = tolerance
        self.methodTV = methodTV
        self.printing = printing
        self.device = device # string for 'cpu' or 'gpu'
    def fun(self,x):
        # evaluate objective function of TV gradient        
        EnergyValTV = TV_ENERGY(np.asarray(x.as_array(), dtype=np.float32), np.asarray(x.as_array(), dtype=np.float32), self.lambdaReg, 2)
        return EnergyValTV
    def prox(self,x,Lipshitz):
        pars = {'algorithm' : SB_TV, \
               'input' : np.asarray(x.as_array(), dtype=np.float32),\
                'regularization_parameter':self.lambdaReg*Lipshitz, \
                'number_of_iterations' :self.iterationsTV ,\
                'tolerance_constant':self.tolerance,\
                'methodTV': self.methodTV ,\
                'printingOut': self.printing}
        
        out = SB_TV(pars['input'], 
              pars['regularization_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['methodTV'],
              pars['printingOut'], self.device)
        return VolumeData(out, dimension_labels=x.dimension_labels)
