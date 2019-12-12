Optimisation framework
**********************
This package allows rapid prototyping of optimisation-based reconstruction problems, i.e. defining and solving different optimization problems to enforce different properties on the reconstructed image.

Firstly, it provides an object-oriented framework for defining mathematical operators and functions as well a collection of useful example operators and functions. Both smooth and non-smooth functions can be used.

Further, it provides a number of high-level generic implementations of optimisation algorithms to solve genericlly formulated optimisation problems constructed from operator and function objects.

The fundamental components are:

+ :code:`Operator`: A class specifying a (currently linear) operator
+ :code:`Function`: A class specifying mathematical functions such as a least squares data fidelity.
+ :code:`Algorithm`: Implementation of an iterative optimisation algorithm to solve a particular generic optimisation problem. Algorithms are iterable Python object which can be run in a for loop. Can be stopped and warm restarted.

Algorithm
=========

A number of generic algorithm implementations are provided including 
Gradient Descent (GD), Conjugate Gradient Least Squares (CGLS), 
Simultaneous Iterative Reconstruction Technique (SIRT), Primal Dual Hybrid 
Gradient (PDHG) and Fast Iterative Shrinkage Thresholding Algorithm (FISTA).

An algorithm is designed for a 
particular generic optimisation problem accepts and number of 
:code:`Function`s and/or :code:`Operator`s as input to define a specific instance of 
the generic optimisation problem to be solved.
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

.. autoclass:: ccpi.optimisation.algorithms.Algorithm
   :members:
   :private-members:
   :special-members:
.. autoclass:: ccpi.optimisation.algorithms.GradientDescent
   :members:
.. autoclass:: ccpi.optimisation.algorithms.CGLS
   :members:
.. autoclass:: ccpi.optimisation.algorithms.FISTA
   :members:
   :special-members:
.. autoclass:: ccpi.optimisation.algorithms.PDHG
   :members:
.. autoclass:: ccpi.optimisation.algorithms.SIRT
   :members:

Operator
========
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

.. autoclass:: ccpi.optimisation.operators.Operator
   :members:
   :special-members:
.. autoclass:: ccpi.optimisation.operators.LinearOperator
   :members:
   :special-members:
.. autoclass:: ccpi.optimisation.operators.ScaledOperator
   :members:
   :special-members:

Trivial operators
-----------------

Trivial operators are the following.

.. autoclass:: ccpi.optimisation.operators.Identity
   :members:
   :special-members:

.. autoclass:: ccpi.optimisation.operators.ZeroOperator
   :members:
   :special-members:

.. autoclass:: ccpi.optimisation.operators.LinearOperatorMatrix
   :members:
   :special-members:


Gradient 
-----------------

In the following the required classes for the implementation of the :code:`Gradient` operator.

.. autoclass:: ccpi.optimisation.operators.Gradient
   :members:
   :special-members:

.. autoclass:: ccpi.optimisation.operators.FiniteDiff
   :members:
   :special-members:

.. autoclass:: ccpi.optimisation.operators.SparseFiniteDiff
   :members:
   :special-members:

.. autoclass:: ccpi.optimisation.operators.SymmetrizedGradient
   :members:
   :special-members:


Shrinkage operator
------------------

.. autoclass:: ccpi.optimisation.operators.ShrinkageOperator
   :members:
   :special-members:




Function
========

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

.. autoclass:: ccpi.optimisation.functions.Function
   :members:
   :special-members:

.. autoclass:: ccpi.optimisation.functions.ScaledFunction
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
  F2 = FunctionOperatorComposition(L2NormSquared(b=b), A)


.. autoclass:: ccpi.optimisation.functions.FunctionOperatorComposition
   :members:
   :special-members:

Indicator box
-------------

.. autoclass:: ccpi.optimisation.functions.IndicatorBox
   :members:
   :special-members:


KullbackLeibler 
---------------

.. autoclass:: ccpi.optimisation.functions.KullbackLeibler
   :members:
   :special-members:

L1 Norm
-------

.. autoclass:: ccpi.optimisation.functions.L1Norm
   :members:
   :special-members:

Squared L2 norm
---------------
.. l2norm:

.. autoclass:: ccpi.optimisation.functions.L2NormSquared
   :members:
   :special-members:

And a least squares function:

.. autoclass:: ccpi.optimisation.functions.Norm2Sq
   :members:
   :special-members:

Mixed L21 norm
--------------

.. autoclass:: ccpi.optimisation.functions.MixedL21Norm
   :members:
   :special-members:

.. autoclass:: ccpi.optimisation.functions.ZeroFunction
   :members:
   :special-members:


Block Framework
***************

The block framework allows to write 

Block Operator
==============

A Block matrix with operators 

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
  \end{bmatrix}_{(3,1)} = \textbf{y}$$

Column: Share the same domains :math:`X_{1}, X_{2}`
Rows: Share the same ranges :math:`Y_{1}, Y_{2}, Y_{3}`.

.. math::
  
  K : (X_{1}\times X_{2}) \rightarrow (Y_{1}\times Y_{2} \times Y_{3})

  A_{1}, A_{3}, A_{5}: \text{share the same domain }  X_{1}

  A_{2}, A_{4}, A_{6}: \text{share the same domain }  X_{2}

  A_{1}: X_{1} \rightarrow Y_{1}, \quad A_{3}: X_{1} \rightarrow Y_{2}, \quad  A_{5}: X_{1} \rightarrow Y_{3}
  
  A_{2}: X_{2} \rightarrow Y_{1}, \quad A_{4}: X_{2} \rightarrow Y_{2}, \quad  A_{6}: X_{2} \rightarrow Y_{3}
 


.. autoclass:: ccpi.optimisation.operators.BlockOperator
   :members:
   :special-members:
.. autoclass:: ccpi.optimisation.operators.BlockScaledOperator
   :members:
   :special-members:



Block Function  
---------------
A Block vector of functions, Size of vector coincides with the rows of :math:`K`:

.. math::
  
  Kx  = \begin{bmatrix}
  y_{1}\\
  y_{2}\\
  y_{3}\\
  \end{bmatrix}, \quad  f  = [ f_{1}, f_{2}, f_{3} ]

  f(Kx) : = f_{1}(y_{1}) + f_{2}(y_{2}) + f_{3}(y_{3})

.. autoclass:: ccpi.optimisation.operators.BlockFunction
   :members:
   :special-members:


Block DataContainer 
----------------------

.. math:: 

  x = [x_{1}, x_{2} ]\in (X_{1}\times X_{2})

  y = [y_{1}, y_{2}, y_{3} ]\in(Y_{1}\times Y_{2} \times Y_{3})


.. autoclass:: ccpi.optimisation.operators.BlockDataContainer
   :members:
   :special-members:

:ref:`Return Home <mastertoc>`
