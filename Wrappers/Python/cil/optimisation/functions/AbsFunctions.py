r'''Decorators and classes for methods and functions which act on the 
absolute value of the complex-valued input, 
math::  G(z) = H(abs(z))

This is accomplished calculating a bounded proximal operator and making
a change of phase,
math::  prox_G(z) = prox^+_H(r) \circ Phi,
where
math:: z = r \circ Phi,     r = abs(z), Phi = exp(i angl(z)),
and \circ is element-wise product.  prox^+ is the proximal map of H 
in which the minimisation carried out over the positive orthant.
For further details see https://doi.org/10.48550/arXiv.2410.22161


This work has been supported by the Royal Academy of Engineering and the 
Office of the Chief Science Adviser for National Security under the UK 
Intelligence Community Postdoctoral Research Fellowship programme.

Francis M Watson, University of Manchester 2024
'''

import numpy as np
from cil.optimisation.functions import Function, TotalVariation
from cil.framework import DataContainer
from typing import Optional



def take_abs_input(func):
    '''decorator for function to act on abs of input of a method'''
    def _take_abs_decorator(self, x: DataContainer, *args, **kwargs):
        try: 
            self._domain = None
        except:
            pass
        rgeo = x.geometry.copy()
        rgeo.dtype = np.float64
        r = rgeo.allocate()
        r.array = np.abs(x.array).astype(np.float64)
        fval = func(r, *args, **kwargs) # func(self, r, *args, **kwargs) for the abstract class implementation
        return fval
    return _take_abs_decorator

def abs_and_project(func):
    '''decorator for function to act on abs of input, 
    with return being projected to the angle of the input.
    Requires function return to have the same shape as input,
    such as prox.'''
    def _abs_project_decorator(self, x: DataContainer, *args, **kwargs):
        try: 
            self._domain = None
        except:
            pass
        rgeo = x.geometry.copy()
        rgeo.dtype = np.float64
        r = rgeo.allocate()
        r.array=np.abs(x.array).astype(np.float64)
        Phi = np.exp(1j*np.angle(x.array))
        out=kwargs.get('out',None)
        if out is not None:
            del kwargs['out']
        fvals = func(r, *args, **kwargs) # func(self, r, *args, **kwargs) for the abstract class implementation
        
        # Douglas-Rachford splitting to find solution in positive orthant
        if np.any(fvals.array<0):
            print('AbsFunctions: projection to +ve orthant triggered')
            cts = 0
            y = r.copy()
            while np.any(fvals.array<0):
                tmp = fvals.array - 0.5*y.array + 0.5*r.array
                tmp[tmp<0] = 0.
                y.array += tmp - fvals.array
                fvals = func(y,*args,**kwargs)
                cts +=1
                if cts>10:
                    fvals.array[fvals.array<0] = 0.
                    break
      
        if out is not None:
            out.array = fvals.array.astype(np.complex128)*Phi
        else:
            out = x.geometry.allocate()
            out.array = fvals.array.astype(np.complex128)*Phi
            return out
    return _abs_project_decorator


class FunctionOfAbs(Function):
    '''Encapsulates an existing function which is passed vector-abs(x)
    Proximal map and __call__ are modified accordingly'''
    
    def __init__(self, function: Function, assume_lower_semi:bool=False):
        self._function = function
        self._lower_semi = assume_lower_semi
        super().__init__(L=function._L)

    def __call__(self,x):
        call_abs = take_abs_input(self._function.__call__)
        return call_abs(self._function,x)
    
    def proximal(self, x, tau=1, out=None):
        prox_abs = abs_and_project(self._function.proximal)
        return prox_abs(self._function, x, tau=tau, out=out)
   
    
    def convex_conjugate(self, x):
        r'''If g= self._function is lower semi-continuous, convex, non-decreasing 
        finite at the origin, then f^*(z*) = g^+(|z*|), where the monotone conugate g^+ is
            g^+(z*) =sup {(z, z*) - g(z) : z >= O}
        see Convex Analysis, R. Tyrrell Rocakfellar, pp110-111
        The monotone conjugate will therefore be less than or equal to the convex conjugate, 
        since it is taken over a smaller set.  It is not available directly, but may coincide with
        the convex conjugate, which is therefore the best estimate we have.  This is only valid for
        real x. In other cases, a general convex conjugate is not available or defined.      
        '''
        
        if self._lower_semi:
            conv_abs = take_abs_input(self._function.convex_conjugate)
            return conv_abs(self._function,x)
        else:
            return 0.0
    
    # the superclass implementation of proximal_conjugate is the correct one

