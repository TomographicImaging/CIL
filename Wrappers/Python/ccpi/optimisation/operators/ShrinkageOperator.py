#========================================================================
# Copyright 2019 Science Technology Facilities Council
# Copyright 2019 University of Manchester
#
# This work is part of the Core Imaging Library developed by Science Technology
# Facilities Council and University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#=========================================================================


class ShrinkageOperator():
    
    '''
    
        Proximal Operator for f(x) = \|\| x \|\|_{1}
        
            prox_{\tau * f}(x) = x.sign() * \max( |x| - \tau, 0 )
    
    
    '''
    
    def __init__(self):
        pass

    def __call__(self, x, tau, out=None):
        
        return x.sign() * (x.abs() - tau).maximum(0) 
   