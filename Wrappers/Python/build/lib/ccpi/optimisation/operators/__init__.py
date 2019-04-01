# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:56:27 2019

@author: ofn77899
"""

from .Operator import Operator
from .LinearOperator import LinearOperator
from .ScaledOperator import ScaledOperator
from .BlockOperator import BlockOperator
from .BlockScaledOperator import BlockScaledOperator


from .FiniteDifferenceOperator import FiniteDiff
from .GradientOperator import Gradient
from .SymmetrizedGradientOperator import SymmetrizedGradient
from .IdentityOperator import Identity
from .ZeroOperator import ZeroOp
