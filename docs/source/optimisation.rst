Optimisation framework
**********************
This package allows rapid prototyping of optimisation-based reconstruction problems, i.e. defining and solving different optimization problems to enforce different properties on the reconstructed image.

Firstly, it provides an object-oriented framework for defining mathematical operators and functions as well a collection of useful example operators and functions. Both smooth and non-smooth functions can be used.

Further, it provides a number of high-level generic implementations of optimisation algorithms to solve genericlly formulated optimisation problems constructed from operator and function objects.

The fundamental components are:

+ Operator: A class specifying a (currently linear) operator
+ Function: A class specifying mathematical functions such as a least squares data fidelity.
+ Algorithm: Implementation of an iterative optimisation algorithm to solve a particular generic optimisation problem. Algorithms are iterable Python object which can be run in a for loop. Can be stopped and warm restarted.

Algorithm
=========

A number of generic algorithm implementations are provided including 
Gradient Descent (GD), Conjugate Gradient Least Squares (CGLS), 
Simultaneous Iterative Reconstruction Technique (SIRT), Primal Dual Hybrid 
Gradient (PDHG) and Fast Iterative Shrinkage Thresholding Algorithm (FISTA).

An algorithm is designed for a 
particular generic optimisation problem accepts and number of 
Functions and/or Operators as input to define a specific instance of 
the generic optimisation problem to be solved.
They are iterable objects which can be run in a for loop. 
The user can provide a stopping criterion different than the default max_iteration.

New algorithms can be easily created by extending the Algorithm class. The user is required to implement only 4 methods: set_up, __init__, update and update_objective.

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

The :code:`Algorithm` provides the infrastructure to continue iteration, to access the values of the objective function in subsequent iterations, the time for each iteration.

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

Operator
========
The two most important methods are :code:`direct` and :code:`adjoint` 
methods that describe the result of applying the operator, and its 
adjoint respectively, onto a compatible :code:`DataContainer` input. 
The output is another :code:`DataContainer` object or subclass 
hereof. An important special case is to represent the tomographic 
forward and backprojection operations.

.. autoclass:: ccpi.optimisation.operators.Operator
   :members:
   :special-members:
.. autoclass:: ccpi.optimisation.operators.LinearOperator
   :members:
   :special-members:
.. autoclass:: ccpi.optimisation.operators.ScaledOperator
   :members:
   :special-members:
.. autoclass:: ccpi.optimisation.operators.GradientOperator
   :members:
   :special-members:
.. autoclass:: ccpi.optimisation.operators.Identity
   :members:
   :special-members:
.. autoclass:: ccpi.optimisation.operators.LinearOperatorMatrix
   :members:
   :special-members:
.. autoclass:: ccpi.optimisation.operators.ShrinkageOperator
   :members:
   :special-members:
.. autoclass:: ccpi.optimisation.operators.SparseFiniteDiff
   :members:
   :special-members:
.. autoclass:: ccpi.optimisation.operators.SymmetrizedGradientOperator
   :members:
   :special-members:
.. autoclass:: ccpi.optimisation.operators.ZeroOperator
   :members:
   :special-members:
.. autoclass:: ccpi.optimisation.operators.BlockOperator
   :members:
   :special-members:
.. autoclass:: ccpi.optimisation.operators.BlockScaledOperator
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


.. autoclass:: ccpi.optimisation.functions.Function
   :members:
   :special-members:
.. autoclass:: ccpi.optimisation.functions.FunctionOperatorComposition
   :members:
   :special-members:
.. autoclass:: ccpi.optimisation.functions.IndicatorBox
   :members:
   :special-members:
.. autoclass:: ccpi.optimisation.functions.KullbackLeibler
   :members:
   :special-members:
.. autoclass:: ccpi.optimisation.functions.L1Norm
   :members:
   :special-members:
.. autoclass:: ccpi.optimisation.functions.L2NormSquared
   :members:
   :special-members:
.. autoclass:: ccpi.optimisation.functions.MixedL21Norm
   :members:
   :special-members:
.. autoclass:: ccpi.optimisation.functions.Norm2Sq
   :members:
   :special-members:
.. autoclass:: ccpi.optimisation.functions.ScaledFunction
   :members:
   :special-members:
.. autoclass:: ccpi.optimisation.functions.ZeroFunction
   :members:
   :special-members:


:ref:`Return Home <mastertoc>`
