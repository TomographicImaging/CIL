#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

from .Function import Function
from .Function import SumFunction
from .Function import ScaledFunction
from .Function import SumScalarFunction
from .Function import ConstantFunction
from .Function import ZeroFunction
from .Function import TranslateFunction
from .L1Norm import L1Norm, MixedL11Norm, soft_shrinkage
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
from .L1Sparsity import L1Sparsity
from .ApproximateGradientSumFunction import ApproximateGradientSumFunction
from .SGFunction import SGFunction
from .SAGFunction import SAGFunction, SAGAFunction

