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


from .operators import Operator
from .CompositeOperator import CompositeOperator
from .CompositeDataContainer import CompositeDataContainer
from .FiniteDifferenceOperator import FiniteDiff
from .GradientOperator import Gradient
from .SymmetrizedGradientOperator import SymmetrizedGradient
from .IdentityOperator import Identity
from .ZeroOperator import ZeroOp
from .AstraProjectorSimpleOperator import AstraProjectorSimple



#import operators.Operator

#from .operators import Operator
#from .CompositeOperator import CompositeOperator
#from .CompositeDataContainer import CompositeDataContainer


