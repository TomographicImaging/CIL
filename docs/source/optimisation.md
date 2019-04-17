Optimisation framework
======================

This package allows rapid prototyping of optimisation-based
reconstruction problems, i.e. defining and solving different
optimization problems to enforce different properties on the
reconstructed image.

Firstly, it provides an object-oriented framework for defining
mathematical operators and functions as well a collection of useful
example operators and functions. Both smooth and non-smooth functions
can be used.

Further, it provides a number of high-level generic implementations of
optimisation algorithms to solve genericlly formulated optimisation
problems constructed from operator and function objects.

The fundamental components are:

-   Operator: A class specifying a (currently linear) operator
-   Function: A class specifying mathematical functions such as a least
    squares data fidelity.
-   Algorithm: Implementation of an iterative optimisation algorithm to
    solve a particular generic optimisation problem. Algorithms are
    iterable Python object which can be run in a for loop. Can be
    stopped and warm restarted.

Algorithm
---------

A number of generic algorithm implementations are provided including
Gradient Descent CGLS and FISTA. An algorithm is designed for a
particular generic optimisation problem accepts and number of Functions
and/or Operators as input to define a specific instance of the generic
optimisation problem to be solved. They are iterable objects which can
be run in a for loop. The user can provide a stopping criterion
different than the default max\_iteration.

New algorithms can be easily created by extending the Algorithm class.
The user is required to implement only 4 methods: set\_up, \_\_init\_\_,
update and update\_objective.

-   `set_up` and `__init__` are used to configure the algorithm
-   `update` is the actual iteration updating the solution
-   `update_objective` defines how the objective is calculated.

For example, the implementation of the update of the Gradient Descent
algorithm to minimise a Function will only be:

The `Algorithm` provides the infrastructure to continue iteration, to
access the values of the objective function in subsequent iterations,
the time for each iteration.

::: {.autoclass members="" private-members="" special-members=""}
ccpi.optimisation.algorithms.Algorithm
:::

::: {.autoclass members=""}
ccpi.optimisation.algorithms.GradientDescent
:::

::: {.autoclass members=""}
ccpi.optimisation.algorithms.CGLS
:::

::: {.autoclass members=""}
ccpi.optimisation.algorithms.FISTA
:::

Operator
--------

The two most important methods are `direct` and `adjoint` methods that
describe the result of applying the operator, and its adjoint
respectively, onto a compatible `DataContainer` input. The output is
another `DataContainer` object or subclass hereof. An important special
case is to represent the tomographic forward and backprojection
operations.

::: {.autoclass members=""}
ccpi.optimisation.operators.Operator
:::

::: {.autoclass members=""}
ccpi.optimisation.operators.LinearOperator
:::

::: {.autoclass members=""}
ccpi.optimisation.operators.ScaledOperator
:::

Function
--------

A `Function` represents a mathematical function of one or more inputs
and is intended to accept `DataContainers` as input as well as any
additional parameters.

Fixed parameters can be passed in during the creation of the function
object. The methods of the function reflect the properties of it, for
example, if the function represented is differentiable the function
should contain a method `gradient` which should return the gradient of
the function evaluated at an input point. If the function is not
differentiable but allows a simple proximal operator, the method
`proximal` should return the proximal operator evaluated at an input
point. The function value is evaluated by calling the function itself,
e.g. `f(x)` for a `Function f` and input point `x`.

::: {.autoclass members=""}
ccpi.optimisation.functions.Function
:::

`Return Home <mastertoc>`{.interpreted-text role="ref"}
