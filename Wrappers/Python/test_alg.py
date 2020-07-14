from ccpi.optimisation.algorithms import *
from ccpi.framework import *
from ccpi.optimisation.operators import *
from ccpi.optimisation.functions import *

from ccpi.framework import TestData
from ccpi.utilities.display import plotter2D

ig = ImageGeometry(1,2,3)


x_init = ig.allocate()

loader = TestData()
gt = loader.load(TestData.CAMERA)
ig = gt.geometry

x_init = gt * 0.

# b = x_init.copy()
# fill with random numbers
# b.fill(numpy.random.random(x_init.shape))
# b = ig.allocate('random')
identity = Identity(ig)

norm2sq = Norm2Sq(identity, gt)
rate = 0.3
#rate = norm2sq.L / 2.1
print ("rate ", norm2sq.L / 2.1)

alg = GradientDescent(x_init=x_init, 
                        objective_function=norm2sq, 
                        rate=rate, max_iteration=100, atol=1e-9, rtol=1e-6)
#alg.max_iteration = 20
alg.run()
plotter2D([gt,alg.get_output()])

numpy.testing.assert_array_almost_equal(alg.x.as_array(), gt.as_array(), decimal=6)


ig = ImageGeometry(12,13,14)
x_init = ig.allocate()
# b = x_init.copy()
# fill with random numbers
# b.fill(numpy.random.random(x_init.shape))
b = ig.allocate('random')
identity = Identity(ig)

norm2sq = Norm2Sq(identity, b)
rate = norm2sq.L / 3.

alg = GradientDescent(x_init=x_init, 
                        objective_function=norm2sq, 
                        rate=rate, atol=1e-9, rtol=1e-6)
alg.max_iteration = 20
alg.run()

print ((b-alg.get_output()).squared_norm())
numpy.testing.assert_array_almost_equal(alg.x.as_array(), b.as_array(), decimal=6)