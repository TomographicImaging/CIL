# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018 Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import os    
from ccpi.framework import DataSetProcessor, DataSetProcessor23D , DataSet
from ccpi.filters.cpu_regularizers_boost import SplitBregman_TV , FGP_TV ,\
                                                LLT_model, PatchBased_Regul ,\
                                                TGV_PD
#from ccpi.filters.cpu_regularizers_cython import some

try:
    from ccpi.filters import gpu_regularizers as gpu
    class PatchBasedRegGPU(DataSetProcessor23D):
        '''Regularizers DataSetProcessor for PatchBasedReg
        
        
        '''
        
        def __init__(self):
            attributes = {'regularization_parameter':None, 
                          'searching_window_ratio': None, 
                          'similarity_window_ratio': None, 
                          'PB_filtering_parameter': None
                      }
            super(PatchBasedRegGPU, self).__init__(**attributes)
                
            
        def process(self):
            '''Executes the processor
                
            '''
            dsi = self.getInput()
            out = gpu.NML (dsi.as_array(), 
                             self.searching_window_ratio,
                             self.similarity_window_ratio,
                             self.regularization_parameter,
                             self.PB_filtering_parameter)  
            y = DataSet( out , False )
            return y
    class Diff4thHajiaboli(DataSetProcessor23D):
        '''Regularizers DataSetProcessor for PatchBasedReg
        
        
        '''
        
        def __init__(self):
            attributes = {'regularization_parameter':None, 
                          'searching_window_ratio': None, 
                          'similarity_window_ratio': None, 
                          'PB_filtering_parameter': None
                      }
            super(Diff4thHajiaboli, self).__init__(self, **attributes)
                
            
        def process(self):
            '''Executes the processor
                
            '''
            dsi = self.getInput()
            out = gpu.Diff4thHajiaboli (dsi.as_array(), 
                                        self.regularization_parameter, 
                                        self.number_of_iterations,
                                        self.edge_preserving_parameter)  
            y = DataSet( out , False )
            return y

except ImportError as ie:
    print (ie)
    


class SBTV(DataSetProcessor23D):
    '''Regularizers DataSetProcessor
    '''
    
    def __init__(self):
        attributes = {'regularization_parameter':None, 
                  'number_of_iterations': 35, 
                  'tolerance_constant': 0.0001, 
                  'TV_penalty':0
                  }
        super(SBTV , self).__init__(**attributes)
            
            
    def process(self):
        '''Executes the processor
        
        '''
    
        dsi = self.getInput()
        out = SplitBregman_TV (dsi.as_array(), 
                               self.regularization_parameter,
                               self.number_of_iterations,
                               self.tolerance_constant,
                               self.TV_penalty)  
        y = DataSet( out[0] , False )
        return y
    
class FGPTV(DataSetProcessor23D):
    '''Regularizers DataSetProcessor
    '''
    
    def __init__(self):
        attributes = {'regularization_parameter':None, 
                  'number_of_iterations': 35, 
                  'tolerance_constant': 0.0001, 
                  'TV_penalty':0
                  }
        super(FGPTV, self).__init__(**attributes)
            
        
    def process(self):
        '''Executes the processor
            
        '''
        dsi = self.getInput()
        out = FGP_TV (dsi.as_array(), 
                      self.regularization_parameter,
                      self.number_of_iterations,
                      self.tolerance_constant,
                      self.TV_penalty)  
        y = DataSet( out[0] , False )
        return y
    
class LLT(DataSetProcessor23D):
    '''Regularizers DataSetProcessor for LLT_model
    
    
    '''
    
    def __init__(self):
        attributes = {'regularization_parameter':None, 
                      'time_step': 0.0001, 
                      'number_of_iterations': 35, 
                      'tolerance_constant': 0, 
                      'restrictive_Z_smoothing': None 
                  }
        super(LLT, self).__init__(**attributes)
            
        
    def process(self):
        '''Executes the processor
            
        '''
        dsi = self.getInput()
        out = LLT_model (dsi.as_array(), 
                         self.regularization_parameter,
                         self.time_step,
                         self.number_of_iterations,
                         self.tolerance_constant,
                         self.restrictive_Z_smoothing)  
        y = DataSet( out[0] , False )
        return y
    
class PatchBasedReg(DataSetProcessor23D):
    '''Regularizers DataSetProcessor for PatchBasedReg
    
    
    '''
    
    def __init__(self):
        attributes = {'regularization_parameter':None, 
                      'searching_window_ratio': None, 
                      'similarity_window_ratio': None, 
                      'PB_filtering_parameter': None
                  }
        super(PatchBasedReg, self).__init__(**attributes)
            
        
    def process(self):
        '''Executes the processor
            
        '''
        dsi = self.getInput()
        out = PatchBased_Regul (dsi.as_array(), 
                         self.regularization_parameter,
                         self.searching_window_ratio,
                         self.similarity_window_ratio,
                         self.PB_filtering_parameter)  
        y = DataSet( out[0] , False )
        return y
    
class TGVPD(DataSetProcessor23D):
    '''Regularizers DataSetProcessor for PatchBasedReg
    
    
    '''
    
    def __init__(self,**kwargs):
        attributes = {'regularization_parameter':None, 
                      'first_order_term': None, 
                      'second_order_term': None, 
                      'number_of_iterations': None
                  }
        for key, value in kwargs.items():
            if key in attributes.keys():
                attributes[key] = value
                
        super(TGVPD, self).__init__(**attributes)
            
        
    def process(self):
        '''Executes the processor
            
        '''
        dsi = self.getInput()
        if dsi.number_of_dimensions == 2:
            out = TGV_PD(dsi.as_array(), 
                         self.regularization_parameter,
                         self.first_order_term, 
                         self.second_order_term , 
                         self.number_of_iterations)
            y = DataSet( out[0] , False )
        elif len(np.shape(input)) == 3:
            #assuming it's 3D
            # run independent calls on each slice
            out3d = dsi.as_array().copy()
            for i in range(np.shape(dsi.as_array())[0]):
                out = TGV_PD(dsi.as_array()[i], 
                                     self.regularization_parameter,
                                     self.first_order_term, 
                                     self.second_order_term , 
                                     self.number_of_iterations)
                # copy the result in the 3D image
                out3d[i] = out[0].copy()
            y = DataSet (out3d , False)
        return y
    
## self contained test
if __name__ == '__main__':
    filename = os.path.join(".." , ".." , ".." , ".." , 
                            "CCPi-FISTA_Reconstruction", "data" ,
                            "lena_gray_512.tif")
    Im = plt.imread(filename)                     
    Im = np.asarray(Im, dtype='float32')

    Im = Im/255

    perc = 0.075
    u0 = Im + np.random.normal(loc = Im ,
                                  scale = perc * Im , 
                                  size = np.shape(Im))
    # map the u0 u0->u0>0
    f = np.frompyfunc(lambda x: 0 if x < 0 else x, 1,1)
    u0 = f(u0).astype('float32')

    lena = DataSet(u0, False, ['X','Y'])
    
    ## plot 
    fig = plt.figure()
    
    a=fig.add_subplot(2,3,1)
    a.set_title('noise')
    imgplot = plt.imshow(u0#,cmap="gray"
                         )
    
    reg_output = []
    ##############################################################################
    # Call regularizer
    
  
    reg3 = SBTV()
    reg3.number_of_iterations = 40
    reg3.tolerance_constant = 0.0001
    reg3.regularization_parameter = 15
    reg3.TV_penalty = 0
    reg3.setInput(lena)
    dataprocessoroutput = reg3.getOutput()
    
    
    # plot
    a=fig.add_subplot(2,3,2)
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    a.text(0.05, 0.95, 'SBTV', transform=a.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    imgplot = plt.imshow(dataprocessoroutput.as_array(),\
                         #cmap="gray"
                         )
    ##########################################################################
    
    reg4 = FGPTV()
    reg4.number_of_iterations = 200
    reg4.tolerance_constant = 1e-4
    reg4.regularization_parameter = 0.05
    reg4.TV_penalty = 0
    reg4.setInput(lena)
    dataprocessoroutput2 = reg4.getOutput()
   
    # plot
    a=fig.add_subplot(2,3,3)    
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    a.text(0.05, 0.95, 'FGPTV', transform=a.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    imgplot = plt.imshow(dataprocessoroutput2.as_array(),\
                         #cmap="gray"
                         )
    
    ###########################################################################
    reg6 = LLT()
    reg6.regularization_parameter = 5
    reg6.time_step = 0.00035
    reg6.number_of_iterations = 350
    reg6.tolerance_constant = 0.0001
    reg6.restrictive_Z_smoothing = 0
    reg6.setInput(lena)
    llt = reg6.getOutput()
    # plot
    a=fig.add_subplot(2,3,4)    
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    a.text(0.05, 0.95, 'LLT', transform=a.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    imgplot = plt.imshow(llt.as_array(),\
                         #cmap="gray"
                         )
    ###########################################################################
    
    reg7 = PatchBasedReg()
    reg7.regularization_parameter = 0.05
    reg7.searching_window_ratio = 3
    reg7.similarity_window_ratio = 1
    reg7.PB_filtering_parameter = 0.06
    reg7.setInput(lena)
    pbr = reg7.getOutput()
    # plot
    a=fig.add_subplot(2,3,5)    
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    a.text(0.05, 0.95, 'PatchBasedReg', transform=a.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    imgplot = plt.imshow(pbr.as_array(),\
                         #cmap="gray"
                         )
    ###########################################################################
    
    reg5 = TGVPD()
    reg5.regularization_parameter = 0.07
    reg5.first_order_term = 1.3
    reg5.second_order_term = 1
    reg5.number_of_iterations = 550
    reg5.setInput(lena)
    tgvpd = reg5.getOutput()
    # plot
    a=fig.add_subplot(2,3,6)    
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    a.text(0.05, 0.95, 'TGVPD', transform=a.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    imgplot = plt.imshow(tgvpd.as_array(),\
                         #cmap="gray"
                         )
    if False:
        #reg4.input = None
        reg5 = FGPTV()
        reg5.number_of_iterations = 350
        reg5.tolerance_constant = 0.01
        reg5.regularization_parameter = 40
        reg5.TV_penalty = 0
        reg5.setInputProcessor(reg3)
        chain = reg5.process()
       
        a=fig.add_subplot(2,3,6)
        
        
        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # place a text box in upper left in axes coords
        a.text(0.05, 0.95, 'SBTV + FGPTV', transform=a.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
        imgplot = plt.imshow(chain.as_array(),\
                             #cmap="gray"
                             )
    plt.show()