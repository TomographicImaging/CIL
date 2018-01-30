# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 16:19:03 2018

@author: ofn77899
"""

import matplotlib.pyplot as plt
import numpy as np
import os    
from enum import Enum
import timeit
from ccpi.filters.cpu_regularizers_boost import SplitBregman_TV , FGP_TV ,\
                                                 LLT_model, PatchBased_Regul ,\
                                                 TGV_PD
from ccpi.framework import DataSetProcessor, DataSet

class SplitBregmanTVRegularizer(DataSetProcessor):
    '''Regularizers DataSetProcessor
    '''
    
    
    
    def __init__(self, input , regularization_parameter , number_of_iterations  = 35 ,\
                 tolerance_constant = 0.0001 , TV_penalty= 0, **wargs):
        kwargs = {'regularization_parameter':regularization_parameter, 
                  'number_of_iterations':number_of_iterations, 
                  'tolerance_constant':tolerance_constant, 
                  'TV_penalty':TV_penalty, 
                  'input' : input,
                  'output': None
                  }
        for key, value in wargs.items():
            kwargs[key] = value
        DataSetProcessor.__init__(self, **kwargs)
        
        
        
    def apply(self):
        pars = self.getParameterMap(['input' , 'regularization_parameter' ,
                                  'number_of_iterations', 'tolerance_constant' ,
                                  'TV_penalty' ])
    
        out = SplitBregman_TV (pars['input'].as_array(), pars['regularization_parameter'],
                              pars['number_of_iterations'],
                              pars['tolerance_constant'],
                              pars['TV_penalty'])  
        print (type(out))
        y = DataSet( out[0] , False )
        #self.setParameter(output_dataset=y)
        return y
    
class FGPTVRegularizer(DataSetProcessor):
    '''Regularizers DataSetProcessor
    '''
    
    
    
    def __init__(self, input , regularization_parameter , number_of_iterations  = 35 ,\
                 tolerance_constant = 0.0001 , TV_penalty= 0, **wargs):
        kwargs = {'regularization_parameter':regularization_parameter, 
                  'number_of_iterations':number_of_iterations, 
                  'tolerance_constant':tolerance_constant, 
                  'TV_penalty':TV_penalty, 
                  'input' : input,
                  'output': None
                  }
        for key, value in wargs.items():
            kwargs[key] = value
        DataSetProcessor.__init__(self, **kwargs)
        
        
        
    def apply(self):
        
        pars = self.getParameterMap(['input' , 'regularization_parameter' ,
                                  'number_of_iterations', 'tolerance_constant' ,
                                  'TV_penalty' ])
        
        if issubclass(type(pars['input']) , DataSetProcessor):
            pars['input'] = pars['input'].getOutput()
        
        out = FGP_TV (pars['input'].as_array(), 
              pars['regularization_parameter'],
              pars['number_of_iterations'],
              pars['tolerance_constant'], 
              pars['TV_penalty'])  
        y = DataSet( out[0] , False )
        #self.setParameter(output_dataset=y)
        return y
    
    def chain(self, other):
        if issubclass(type(other) , DataSetProcessor):
            self.setParameter(input = other.getOutput()[0])
       

if __name__ == '__main__':
    filename = os.path.join(".." , ".." , ".." , ".." , 
                            "CCPi-FISTA_Reconstruction", "data" ,
                            "lena_gray_512.tif")
    Im = plt.imread(filename)                     
    Im = np.asarray(Im, dtype='float32')
    
    perc = 0.15
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
    
    ####################### SplitBregman_TV #####################################
    # u = SplitBregman_TV(single(u0), 10, 30, 1e-04);
    
    start_time = timeit.default_timer()
    pars = {'algorithm' : SplitBregman_TV , \
            'input' : lena,
            'regularization_parameter':40 , \
    'number_of_iterations' :350 ,\
    'tolerance_constant':0.01 , \
    'TV_penalty': 0
    }
    
    
    reg = SplitBregmanTVRegularizer(pars['input'],
                                             pars['regularization_parameter'],
                              pars['number_of_iterations'],
                              pars['tolerance_constant'],
                              pars['TV_penalty'], 
                              hold_input=False, hold_output=True)
    splitbregman = reg.getOutput()
    
    #txtstr = printParametersToString(pars) 
    #txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
    #print (txtstr)
        
    
    a=fig.add_subplot(2,3,2)
    
    
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    a.text(0.05, 0.95, 'SplitBregman', transform=a.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    imgplot = plt.imshow(splitbregman.as_array(),\
                         #cmap="gray"
                         )
    pars = {'algorithm' : FGP_TV , \
        'input' : lena,
        'regularization_parameter':5e-5, \
        'number_of_iterations' :10 ,\
        'tolerance_constant':0.001,\
        'TV_penalty': 0
}
    reg2 = FGPTVRegularizer(pars['input'],
                              pars['regularization_parameter'],
                              pars['number_of_iterations'],
                              pars['tolerance_constant'],
                              pars['TV_penalty'], 
                              hold_input=False, hold_output=True)
    fgp = reg2.getOutput()
    
    #txtstr = printParametersToString(pars) 
    #txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
    #print (txtstr)
        
    
    a=fig.add_subplot(2,3,3)
    
    
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    a.text(0.05, 0.95, 'FGPTV', transform=a.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    imgplot = plt.imshow(fgp.as_array(),\
                         #cmap="gray"
                         )
    
    reg3 = FGPTVRegularizer(reg,
                           pars['regularization_parameter'],
                              pars['number_of_iterations'],
                              pars['tolerance_constant'],
                              pars['TV_penalty'], 
                              hold_input=False, hold_output=True)
    chain = reg3.getOutput()
    
    #txtstr = printParametersToString(pars) 
    #txtstr += "%s = %.3fs" % ('elapsed time',timeit.default_timer() - start_time)
    #print (txtstr)
        
    
    a=fig.add_subplot(2,3,4)
    
    
    # these are matplotlib.patch.Patch properties
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    a.text(0.05, 0.95, 'chain', transform=a.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    imgplot = plt.imshow(chain.as_array(),\
                         #cmap="gray"
                         )
    plt.show()