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
from .Function import Function
from .ZeroFunction import ZeroFunction
from .L1Norm import L1Norm
from .L2NormSquared import L2NormSquared
from .ScaledFunction import ScaledFunction
from .BlockFunction import BlockFunction
from .FunctionOperatorComposition import FunctionOperatorComposition
from .MixedL21Norm import MixedL21Norm
from .IndicatorBox import IndicatorBox
from .KullbackLeibler import KullbackLeibler
from .Norm2Sq import Norm2Sq, StochasticNorm2Sq

