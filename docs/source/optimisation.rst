Optimisation framework
**********************
This package allows rapid prototyping of optimisation-based reconstruction problems, i.e. defining and solving different optimization problems to enforce different properties on the reconstructed image.

Firstly, it provides an object-oriented framework for defining mathematical operators and functions as well a collection of useful example operators and functions. Both smooth and non-smooth functions can be used.

Further, it provides a number of high-level generic implementations of optimisation algorithms to solve genericlly formulated optimisation problems constructed from operator and function objects.

The fundamental components are:

+ Operator: A class specifying a (currently linear) operator
+ Function: A class specifying mathematical functions such as a least squares data fidelity.
+ Algorithm: Implementation of an iterative optimisation algorithm to solve a particular generic optimisation problem. Algorithms are iterable Python object which can be run in a for loop. Can be stopped and warm restarted.

To be able to express complex optimisation problems we developed the
block framework, which provides a generic strategy to treat variational 
problems in the following form:

.. math::
    \min \text{Regulariser} + \text{Fidelity} 

The block framework consists of:

+ BlockDataContainer
+ BlockFunction
+ BlockOperator

`BlockDataContainer`_ holds `DataContainer` as column vector. It is possible to 
do basic algebra between ``BlockDataContainer`` s and with numbers, list or numpy arrays. 

`BlockFunction`_ acts on ``BlockDataContainer`` as a separable sum function:
    
      .. math:: 

          f = [f_1,...,f_n] \newline

          f([x_1,...,x_n]) = f_1(x_1) +  .... + f_n(x_n)

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

   op1 = Gradient(ig, correlation=Gradient.CORRELATION_SPACE)
   op2 = Identity(ig, ag)

   # Create BlockOperator
   K = BlockOperator(op1, op2, shape=(2,1) )

   # Create functions      
   F = BlockFunction(alpha * MixedL21Norm(), 0.5 * L2NormSquared(b=noisy_data))


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
.. autoclass:: ccpi.optimisation.functions.BlockFunction
   :members:
   :special-members:


:ref:`Return Home <mastertoc>`

.. _BlockDataContainer: framework.html#ccpi.framework.BlockDataContainer
.. _BlockFunction: optimisation.html#ccpi.optimisation.functions.BlockFunction
.. _BlockOperator: optimisation.html#ccpi.optimisation.operators.BlockOperators
