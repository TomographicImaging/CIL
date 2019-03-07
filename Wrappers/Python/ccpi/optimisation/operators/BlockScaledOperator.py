# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 12:36:40 2019

@author: ofn77899
"""
import numpy
from ccpi.optimisation.operators import BlockOperator


       
class BlockScaledOperator(BlockOperator):
    def __init__(self, *args, **kwargs):
        super(BlockScaledOperator, self).__init__(*args, **kwargs)
        scalar = kwargs.get('scalar',1)
        if isinstance (scalar, list) or isinstance(scalar, tuple) or \
            isinstance(scalar, numpy.ndarray):
            if len(scalars) != len(self.operators):
                raise ValueError('dimensions of scalars and operators do not match')
        else:
            scalar = [scalar for _ in self.operators]
        self.operators = [v * op for op in self.operators] 