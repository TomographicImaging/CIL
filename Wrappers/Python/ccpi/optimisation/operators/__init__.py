# -*- coding: utf-8 -*-
#  CCP in Tomographic Imaging (CCPi) Core Imaging Library (CIL).

#   Copyright 2017 UKRI-STFC
#   Copyright 2017 University of Manchester

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from .Operator import Operator, LinearOperator, ScaledOperator, SumOperator,\
 CompositionOperator, Composition2Operator
#from .LinearOperator import LinearOperator
#from .ScaledOperator import ScaledOperator
from .BlockOperator import BlockOperator
# from .BlockScaledOperator import BlockScaledOperator

from .SparseFiniteDiff import SparseFiniteDiff
from .ShrinkageOperator import ShrinkageOperator

from .FiniteDifferenceOperator import FiniteDiff
from .GradientOperator import Gradient
from .SymmetrizedGradientOperator import SymmetrizedGradient
from .IdentityOperator import Identity
from .ZeroOperator import ZeroOperator
from .LinearOperatorMatrix import LinearOperatorMatrix

