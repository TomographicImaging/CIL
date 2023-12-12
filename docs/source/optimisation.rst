..     -*- coding: utf-8 -*-
      Copyright 2019 United Kingdom Research and Innovation
      Copyright 2019 The University of Manchester
    
      Licensed under the Apache License, Version 2.0 (the "License");
      you may not use this file except in compliance with the License.
      You may obtain a copy of the License at
    
          http://www.apache.org/licenses/LICENSE-2.0
    
      Unless required by applicable law or agreed to in writing, software
      distributed under the License is distributed on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
      See the License for the specific language governing permissions and
      limitations under the License.
    
     Authors:
     CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

Optimisation framework
**********************
This package allows rapid prototyping of optimisation-based reconstruction problems, i.e. defining and solving different optimization problems to enforce different properties on the reconstructed image.

Firstly, it provides an object-oriented framework for defining mathematical operators and functions as well a collection of useful example operators and functions. Both smooth and non-smooth functions can be used.

Further, it provides a number of high-level generic implementations of optimisation algorithms to solve genericlly formulated optimisation problems constructed from operator and function objects.

The fundamental components are:

+ :code:`Operator`: A class specifying a (currently linear) operator.
+ :code:`Function`: A class specifying mathematical functions such as a least squares data fidelity.
+ :code:`Algorithm`: Implementation of an iterative optimisation algorithm to solve a particular generic optimisation problem. Algorithms are iterable Python object which can be run in a for loop. Can be stopped and warm restarted.



Algorithms
==========

A number of generic algorithm implementations are provided including 
Gradient Descent (GD), Conjugate Gradient Least Squares (CGLS), 
Simultaneous Iterative Reconstruction Technique (SIRT), Primal Dual Hybrid 
Gradient (PDHG), Iterative Shrinkage Thresholding Algorithm (ISTA),
and Fast Iterative Shrinkage Thresholding Algorithm (FISTA).

An algorithm is designed for a particular generic optimisation problem accepts and number of 
instances of :code:`Function` derived classes and/or :code:`Operator` derived classes as input to 
define a specific instance of the generic optimisation problem to be solved.
They are iterable objects which can be run in a for loop. 
The user can provide a stopping criterion different than the default max_iteration.

New algorithms can be easily created by extending the :code:`Algorithm` class. 
The user is required to implement only 4 methods: set_up, __init__, update and update_objective.

+ :code:`set_up` and :code:`__init__` are used to configure the algorithm
+ :code:`update` is the actual iteration updating the solution
+ :code:`update_objective` defines how the objective is calculated.

For example, the implementation of the update of the Gradient Descent 
algorithm to minimise a Function will only be:

.. code-block :: python

    def update(self):
        self.x += -self.rate * self.objective_function.gradient(self.x)
    def update_objective(self):
        self.loss.append(self.objective_function(self.x))

The :code:`Algorithm` provides the infrastructure to continue iteration, to access the values of the 
objective function in subsequent iterations, the time for each iteration, and to provide a nice 
print to screen of the status of the optimisation.

Base class
----------
.. autoclass:: cil.optimisation.algorithms.Algorithm
   :members:
   :private-members:
   :special-members:

GD
--
.. autoclass:: cil.optimisation.algorithms.GD
   :members:
   :inherited-members: run, update_objective_interval, max_iteration

CGLS
----
.. autoclass:: cil.optimisation.algorithms.CGLS
   :members: 
   :inherited-members: run, update_objective_interval, max_iteration

SIRT
----
.. autoclass:: cil.optimisation.algorithms.SIRT
   :members: update, update_objective
   :inherited-members: run, update_objective_interval, max_iteration

ISTA
----
.. autoclass:: cil.optimisation.algorithms.ISTA
   :members:
   :special-members:
   :inherited-members: run, update_objective_interval, max_iteration
   
FISTA
-----
.. autoclass:: cil.optimisation.algorithms.FISTA
   :members:
   :special-members:
   :inherited-members: run, update_objective_interval, max_iteration

PDHG
----
.. autoclass:: cil.optimisation.algorithms.PDHG
   :members: update, set_step_sizes, update_step_sizes, update_objective
   :member-order: bysource
   :inherited-members: run, update_objective_interval, max_iteration

LADMM
-----
.. autoclass:: cil.optimisation.algorithms.LADMM
   :members:
   :inherited-members: run, update_objective_interval, max_iteration

SPDHG
-----
.. autoclass:: cil.optimisation.algorithms.SPDHG
   :members:
   :inherited-members: run, update_objective_interval, max_iteration



Operators
=========
The two most important methods are :code:`direct` and :code:`adjoint` 
methods that describe the result of applying the operator, and its 
adjoint respectively, onto a compatible :code:`DataContainer` input. 
The output is another :code:`DataContainer` object or subclass 
hereof. An important special case is to represent the tomographic 
forward and backprojection operations.


Operator base classes
---------------------

All operators extend the :code:`Operator` class. A special class is the :code:`LinearOperator`
which represents an operator for which the :code:`adjoint` operation is defined.
A :code:`ScaledOperator` represents the multiplication of any operator with a scalar.

.. autoclass:: cil.optimisation.operators.Operator
   :members:
   :special-members:

.. autoclass:: cil.optimisation.operators.LinearOperator
   :members:
   :special-members:

.. autoclass:: cil.optimisation.operators.ScaledOperator
   :members:
   :special-members:

.. autoclass:: cil.optimisation.operators.CompositionOperator
   :members:
   :special-members:

.. autoclass:: cil.optimisation.operators.DiagonalOperator
   :members:
   :special-members:

.. autoclass:: cil.optimisation.operators.ChannelwiseOperator
   :members:
   :special-members:

Trivial operators
-----------------

Trivial operators are the following.

.. autoclass:: cil.optimisation.operators.IdentityOperator
   :members:
   :special-members:

.. autoclass:: cil.optimisation.operators.ZeroOperator
   :members:
   :special-members:

.. autoclass:: cil.optimisation.operators.MatrixOperator
   :members:
   :special-members:

.. autoclass:: cil.optimisation.operators.MaskOperator
   :members:
   :special-members:


GradientOperator 
-----------------

.. autoclass:: cil.optimisation.operators.GradientOperator
   :members:


.. autoclass:: cil.optimisation.operators.FiniteDifferenceOperator
   :members:

.. autoclass:: cil.optimisation.operators.SparseFiniteDifferenceOperator
   :members:

.. autoclass:: cil.optimisation.operators.SymmetrisedGradientOperator
   :members:







Functions
=========

A :code:`Function` represents a mathematical function of one or more inputs 
and is intended to accept :code:`DataContainers` as input as well as any 
additional parameters. 

Fixed parameters can be passed in during the creation of the function object. 
The methods of the function reflect the properties of it, for example, if the function
represented is differentiable the function should contain a method :code:`gradient` 
which should return the gradient of the function evaluated at an input point. 
If the function is not differentiable but allows a simple proximal operator, 
the method :code:`proximal` should return the proximal operator evaluated at an
input point. The function value is evaluated by calling the function itself, 
e.g. :code:`f(x)` for a :code:`Function f` and input point :code:`x`.


Base classes
------------

.. autoclass:: cil.optimisation.functions.Function
   :members:
   :special-members:

.. autoclass:: cil.optimisation.functions.SumFunction
   :members:
   :special-members: 

.. autoclass:: cil.optimisation.functions.ScaledFunction
   :members:
   :special-members:

.. autoclass:: cil.optimisation.functions.SumScalarFunction
   :members:
   :special-members: 

.. autoclass:: cil.optimisation.functions.TranslateFunction
   :members:
   :special-members: 
   
Simple functions
----------------
.. autoclass:: cil.optimisation.functions.ConstantFunction
   :members:
   :special-members: 

.. autoclass:: cil.optimisation.functions.ZeroFunction
   :members:
   :special-members: 

.. autoclass:: cil.optimisation.functions.Rosenbrock
   :members:
   :special-members: 

Composition of operator and a function
--------------------------------------

This class allows the user to write a function which does the following:

.. math::

  F ( x ) = G ( Ax )

where :math:`A` is an operator. For instance the least squares function l2norm_ :code:`Norm2Sq` can
be expressed as 

.. math::

  F(x) = || Ax - b ||^2_2

.. code::python

  F1 = Norm2Sq(A, b)
  # or equivalently
  F2 = OperatorCompositionFunction(L2NormSquared(b=b), A)


.. autoclass:: cil.optimisation.functions.OperatorCompositionFunction
  :members:
  :special-members: 

Indicator box
-------------

.. autoclass:: cil.optimisation.functions.IndicatorBox
   :members:
   :special-members:


KullbackLeibler 
---------------

.. autoclass:: cil.optimisation.functions.KullbackLeibler
   :members:
   :special-members:

L1 Norm
-------

.. autoclass:: cil.optimisation.functions.L1Norm
   :members:
   :special-members:

L2 Norm Squared
-----------------------
.. l2norm:

.. autoclass:: cil.optimisation.functions.L2NormSquared
   :members:
   :special-members:

.. autoclass:: cil.optimisation.functions.WeightedL2NormSquared
   :members:
   :special-members:
   

Least Squares
-------------

.. autoclass:: cil.optimisation.functions.LeastSquares
   :members:
   :special-members:

Mixed L21 norm
--------------

.. autoclass:: cil.optimisation.functions.MixedL21Norm
   :members:
   :special-members:
   
Smooth Mixed L21 norm
---------------------

.. autoclass:: cil.optimisation.functions.SmoothMixedL21Norm
   :members:
   :special-members:   

Mixed L11 norm
---------------------

.. autoclass:: cil.optimisation.functions.MixedL11Norm
   :members:
   :special-members:      

Total variation
---------------

.. autoclass:: cil.optimisation.functions.TotalVariation
   :members:
   :special-members: 

Block Framework
***************

To be able to express more advanced optimisation problems we developed the
`Block Framework`_, which provides a generic strategy to treat variational 
problems in the following form:

.. math::
    \min \text{Regulariser} + \text{Fidelity} 

The block framework consists of:

+ `BlockDataContainer`_
+ `BlockFunction`_
+ `BlockOperator`_




The block framework allows writing more advanced `optimisation problems`_. Consider the typical 
`Tikhonov regularisation <https://en.wikipedia.org/wiki/Tikhonov_regularization>`_:

.. math:: 

  \underset{u}{\mathrm{argmin}}\begin{Vmatrix}A u - b \end{Vmatrix}^2_2 + \alpha^2\|Lu\|^2_2

where,

* :math:`A` is the projection operator
* :math:`b` is the acquired data
* :math:`u` is the unknown image to be solved for
* :math:`\alpha` is the regularisation parameter
* :math:`L` is a regularisation operator

The first term measures the fidelity of the solution to the data. The second term measures the 
fidelity to the prior knowledge we have imposed on the system, operator :math:`L`.  

This can be re-written equivalently in the block matrix form:

.. math::
  \underset{u}{\mathrm{argmin}}\begin{Vmatrix}\binom{A}{\alpha L} u - \binom{b}{0}\end{Vmatrix}^2_2

With the definitions:

* :math:`\tilde{A} = \binom{A}{\alpha L}`
* :math:`\tilde{b} = \binom{b}{0}`

this can now be recognised as a least squares problem which can be solved by any algorithm in the :code:`cil.optimisation`
which can solve least squares problem, e.g. CGLS.

.. math:: 

  \underset{u}{\mathrm{argmin}}\begin{Vmatrix}\tilde{A} u - \tilde{b}\end{Vmatrix}^2_2

To be able to express our optimisation problems in the matrix form above, we developed the so-called, 
Block Framework comprising 4 main actors: :code:`BlockGeometry`, :code:`BlockDataContainer`, 
:code:`BlockFunction` and :code:`BlockOperator`.



BlockDataContainer
==================

`BlockDataContainer`_ holds `DataContainer`_ as column vector. It is possible to 
do basic algebra between `BlockDataContainer`_ s and with numbers, list or numpy arrays. 

.. math:: 

  x = [x_{1}, x_{2} ]\in (X_{1}\times X_{2})

  y = [y_{1}, y_{2}, y_{3} ]\in(Y_{1}\times Y_{2} \times Y_{3})


.. autoclass:: cil.framework.BlockDataContainer
   :members:
   :special-members:


Block Function  
==============

`BlockFunction`_ acts on `BlockDataContainer`_ as a separable sum function:
    
      .. math:: 

          f = [f_1,...,f_n] \newline

          f([x_1,...,x_n]) = f_1(x_1) +  .... + f_n(x_n)


.. math::
  
  Y = \begin{bmatrix}
  y_{1}\\
  y_{2}\\
  y_{3}\\
  \end{bmatrix}, \quad  F  = [ f_{1}, f_{2}, f_{3} ]

  F(Y) : = f_{1}(y_{1}) + f_{2}(y_{2}) + f_{3}(y_{3})


.. autoclass:: cil.optimisation.functions.BlockFunction
   :members:
   :special-members:

Block Operator
==============

`BlockOperator`_ represent a block matrix with operators

.. math:: 
  K = \begin{bmatrix}
      A_{1} & A_{2} \\
      A_{3} & A_{4} \\
      A_{5} & A_{6}
 \end{bmatrix}_{(3,2)} *  \quad \underbrace{\begin{bmatrix}
 x_{1} \\
 x_{2}
 \end{bmatrix}_{(2,1)}}_{\textbf{x}} =  \begin{bmatrix}
 A_{1}x_{1}  + A_{2}x_{2}\\
 A_{3}x_{1}  + A_{4}x_{2}\\
 A_{5}x_{1}  + A_{6}x_{2}\\
 \end{bmatrix}_{(3,1)} =  \begin{bmatrix}
 y_{1}\\
 y_{2}\\
 y_{3}
 \end{bmatrix}_{(3,1)} = \textbf{y}

Column: Share the same domains :math:`X_{1}, X_{2}`

Rows: Share the same ranges :math:`Y_{1}, Y_{2}, Y_{3}`

.. math::
 K : (X_{1}\times X_{2}) \rightarrow (Y_{1}\times Y_{2} \times Y_{3})

:math:`A_{1}, A_{3}, A_{5}`: share the same domain :math:`X_{1}` and 
:math:`A_{2}, A_{4}, A_{6}`: share the same domain :math:`X_{2}`

.. math::

 A_{1}: X_{1} \rightarrow Y_{1} \\
 A_{3}: X_{1} \rightarrow Y_{2} \\
 A_{5}: X_{1} \rightarrow Y_{3} \\
 A_{2}: X_{2} \rightarrow Y_{1} \\ 
 A_{4}: X_{2} \rightarrow Y_{2} \\
 A_{6}: X_{2} \rightarrow Y_{3}

For instance with these ingredients one may write the following objective 
function,

.. math::
   \alpha ||\nabla u||_{2,1} + ||u - g||_2^2

where :math:`g` represent the measured values, :math:`u` the solution
:math:`\nabla` is the gradient operator, :math:`|| ~~ ||_{2,1}` is a norm for 
the output of the gradient operator and :math:`|| x-g ||^2_2` is 
least squares fidelity function as

.. math::
   K = \begin{bmatrix}
           \nabla \\
           \mathbb{1}
         \end{bmatrix}

 F(x) = \Big[ \alpha \lVert ~x~ \rVert_{2,1} ~~ , ~~ || x - g||_2^2 \Big]
 
 w = [ u ]

Then we have rewritten the problem as

.. math::
  F(Kw) =   \alpha \left\lVert \nabla u \right\rVert_{2,1} + ||u-g||^2_2

Which in Python would be like

.. code-block:: python

   op1 = GradientOperator(ig, correlation=GradientOperator.CORRELATION_SPACE)
   op2 = IdentityOperator(ig, ag)

   # Create BlockOperator
   K = BlockOperator(op1, op2, shape=(2,1) )

   # Create functions      
   F = BlockFunction(alpha * MixedL21Norm(), 0.5 * L2NormSquared(b=noisy_data))


.. autoclass:: cil.optimisation.operators.BlockOperator
   :members:
   :special-members: 


:ref:`Return Home <mastertoc>`

.. _BlockDataContainer: framework.html#cil.framework.BlockDataContainer
.. _BlockFunction: optimisation.html#cil.optimisation.functions.BlockFunction
.. _BlockOperator: optimisation.html#cil.optimisation.operators.BlockOperators


References
----------

.. bibliography::    
