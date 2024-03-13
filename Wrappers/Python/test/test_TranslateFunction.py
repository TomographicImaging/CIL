#  Copyright 2020 United Kingdom Research and Innovation
#  Copyright 2020 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

import unittest
from utils import initialise_tests
from cil.optimisation.functions import L1Norm, \
                                        L2NormSquared, \
                                        TranslateFunction, \
                                        MixedL21Norm
from cil.optimisation.operators import GradientOperator
from cil.framework import ImageGeometry, DataContainer
import numpy as np
import logging
from testclass import CCPiTestClass as TestFunction

initialise_tests()

''' Here we test SumFunction class for different function

L2Norm, L1Norm, KullbackLeibler, ZeroFunction, ConstantFunction, Scalar

for call method
for gradient method



'''



class TestTranslaateFunction(TestFunction):


    def test_TranslateFunction(self):

        # Test TranslationFunction

        ig = ImageGeometry(4,4)
        tmp = ig.allocate('random', seed=10)
        b = ig.allocate('random', seed=10)
        scalar = 0.4
        tau = 0.05

        list1 = [ L2NormSquared(), scalar * L2NormSquared(), scalar * L2NormSquared(b=b),
                 L1Norm(), scalar * L1Norm(), scalar * L1Norm(b=b)]

        list1_shift = [ L2NormSquared().centered_at(ig.allocate()), scalar * L2NormSquared().centered_at(ig.allocate()), scalar * L2NormSquared().centered_at(b),
                        L1Norm().centered_at(ig.allocate()), scalar * L1Norm().centered_at(ig.allocate()), scalar * L1Norm().centered_at(b)]

        out_gradient1 = ig.allocate()
        out_gradient2 = ig.allocate()

        out_proximal1 = ig.allocate()
        out_proximal2 = ig.allocate()

        out_proximal_conj1 = ig.allocate()
        out_proximal_conj2 = ig.allocate()

        for func, func_shift in zip(list1, list1_shift):

            # check call
            res1 = func(tmp)
            res2 = func_shift(tmp)
            self.assertNumpyArrayAlmostEqual(res1, res2)

            try:
                # check gradient
                res1_gradient = func.gradient(tmp)
                res2_gradient = func_shift.gradient(tmp)
                self.assertNumpyArrayAlmostEqual(res1_gradient.as_array(), res2_gradient.as_array())

                # check gradient out
                func.gradient(tmp, out = out_gradient1)
                func_shift.gradient(tmp, out = out_gradient2)
                self.assertNumpyArrayAlmostEqual(out_gradient1.as_array(), out_gradient2.as_array())

            except NotImplementedError:
                logging.info('Function is not differentiable')

            # check proximal
            func.proximal(tmp, tau, out = out_proximal1)
            func_shift.proximal(tmp, tau, out = out_proximal2)
            self.assertNumpyArrayAlmostEqual(out_proximal1.as_array(), out_proximal2.as_array())

            # check proximal conjugate
            func.proximal_conjugate(tmp, tau, out = out_proximal_conj1)
            func_shift.proximal_conjugate(tmp, tau, out = out_proximal_conj2)
            self.assertNumpyArrayAlmostEqual(out_proximal_conj1.as_array(), out_proximal_conj1.as_array())


    def test_TranslateFunction_MixedL21Norm(self):

        print("Test for TranslateFunction for MixedL21Norm")

        ig = ImageGeometry(4,4)

        Grad = GradientOperator(ig)
        b = Grad.range_geometry().allocate('random', seed=10)

        alpha = 0.4
        f1 = alpha * MixedL21Norm()
        fun = TranslateFunction(f1, b)

        tmp_x = Grad.range_geometry().allocate('random', seed=10)

        res1 = fun(tmp_x)
        res2 = f1(tmp_x - b)
        self.assertAlmostEqual(res1, res2)
        print("Check call...OK")

        res1 = f1.convex_conjugate(tmp_x) - b.dot(tmp_x)
        res2 = fun.convex_conjugate(tmp_x)
        self.assertAlmostEqual(res1, res2)
        print("Check convex conjugate...OK (maybe inf=inf)")

        tau = 0.4
        res1 = fun.proximal(tmp_x, tau)
        res2 = f1.proximal(tmp_x - b, tau) + b

        self.assertNumpyArrayAlmostEqual(res1.get_item(0).as_array(), res2.get_item(0).as_array() )
        self.assertNumpyArrayAlmostEqual(res1.get_item(1).as_array(), res2.get_item(1).as_array() )
        print("Check prox...OK ")



if __name__ == '__main__':

    t = TestFunction()
    t.test_TranslateFunction()
    t.test_TranslateFunction_MixedL21Norm()
