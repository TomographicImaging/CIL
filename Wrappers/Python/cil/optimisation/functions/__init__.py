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

from .Function import Function
from .Function import SumFunction
from .Function import ScaledFunction
from .Function import SumScalarFunction
from .Function import ConstantFunction
from .Function import ZeroFunction
from .Function import TranslateFunction
from .L1Norm import L1Norm
from .L2NormSquared import L2NormSquared
from .L2NormSquared import WeightedL2NormSquared
from .LeastSquares import LeastSquares
from .BlockFunction import BlockFunction
from .OperatorCompositionFunction import OperatorCompositionFunction
from .MixedL21Norm import MixedL21Norm, SmoothMixedL21Norm
from .IndicatorBox import IndicatorBox
from .KullbackLeibler import KullbackLeibler
from .Rosenbrock import Rosenbrock
from .TotalVariation import TotalVariation
from .TotalGeneralisedVariation import TotalGeneralisedVariation
