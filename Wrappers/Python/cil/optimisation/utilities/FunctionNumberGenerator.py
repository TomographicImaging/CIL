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

import numpy as np

def FunctionNumberGenerator(num_functions, sampling_method = "random"):

    r""" FunctionNumberGenerator

        The FunctionNumberGenerator selects randomly a number from a list of numbers/functions of length :code:`num_functions`.
        
        Parameters:
        -----------
        num_functions : :obj: int 
            The total number of functions used in our Stochastic estimators e.g.,
        sampling_method : :obj:`string`, Default = :code:`random`
            Selection process for each function in the list. It can be :code:`random`, :code:`random_permutation`,  :code:`fixed_permutation`.               
               -  :code:`random`: Every function is selected randomly with replacement.
               -  :code:`random_permutation`: Every function is selected randomly without replacement. After selecting all the functions in the list, i.e., after one epoch, the list is randomly permuted.
               -  :code:`fixed_permuation`: Every function is selected randomly without replacement and the list of function is permuted only once.

    Example
    -------

    >>> fng = FunctionNumberGenerator(10)
    >>> print(next(fng))

    >>> number_of_functions = 10
    >>> fng = FunctionNumberGenerator(number_of_functions, sampling_method="fixed_permutation")
    >>> epochs = 2
    >>> generated_numbers=[print(next(fng), end=' ') for _ in range(epochs*number_of_functions)]
                            
    """    

    if not isinstance(num_functions, int):
        raise ValueError(" Integer is required for `num_functions`, {} is passed. ".format(num_functions)  )

    # Accepted sampling_methods
    default_sampling_methods = ["random", "random_permutation","fixed_permutation"]

    if sampling_method not in default_sampling_methods:
        raise NotImplementedError("Only {} are implemented at the moment.".format(default_sampling_methods)) 

    replacement=False
    if sampling_method=="random":
        replacement=True
    else:
        if sampling_method=="random_permutation":
            shuffle="random"
        else:
            shuffle="single"

    function_num = -1
    index = 0    

    if replacement is False:                    
        # create a list of functions without replacement, first permutation
        list_of_functions = np.random.choice(range(num_functions),num_functions, replace=False)            
                          
    while(True):                
                
        if replacement is False:
            
            # list of functions already permuted
            function_num = list_of_functions[index]
            index+=1                
                            
            if index == num_functions:
                index=0
                
                # For random shuffle, at the end of each epoch, we permute the list again                    
                if shuffle=="random":                    
                    list_of_functions = np.random.choice(range(num_functions),num_functions, replace=False)                                                                                          
        else:
            
            # at each iteration (not epoch) function is randomly selected
            function_num = np.random.randint(0, num_functions)
        
        yield function_num        


