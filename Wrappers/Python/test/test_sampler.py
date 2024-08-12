#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
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
import os
import sys
from testclass import CCPiTestClass
import numpy as np
from cil.optimisation.utilities import Sampler, SamplerRandom
initialise_tests()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class TestSamplers(CCPiTestClass):

    def example_function(self, iteration_number):
        return ((iteration_number+5) % 50)

    def test_init_Sampler(self):
        default_sampler = Sampler(50, self.example_function)
        self.assertEqual(default_sampler._num_indices, 50)
        self.assertEqual(default_sampler.num_indices, 50)
        self.assertListEqual(default_sampler._prob_weights, [1/50]*50)
        self.assertListEqual(default_sampler.prob_weights, [1/50]*50)
        self.assertEqual(default_sampler._iteration_number, 0)
        self.assertEqual(default_sampler.current_iter_number, 0)
        self.assertEqual(default_sampler._type, None)

        other_sampler = Sampler(55, self.example_function, sampling_type='banana',
                                prob_weights=list(range(55)/np.sum(range(55))))
        self.assertEqual(other_sampler._num_indices, 55)
        self.assertEqual(other_sampler.num_indices, 55)
        self.assertListEqual(other_sampler._prob_weights,
                             list(range(55)/np.sum(range(55))))
        self.assertListEqual(other_sampler.prob_weights,
                             list(range(55)/np.sum(range(55))))
        self.assertEqual(other_sampler._iteration_number, 0)
        self.assertEqual(other_sampler.current_iter_number, 0)
        self.assertEqual(other_sampler._type, 'banana')

        # Check probabilities sum to one and are positive
        with self.assertRaises(ValueError):
            Sampler.from_function(
                10, self.example_function, prob_weights=[1/11]*10)
        with self.assertRaises(ValueError):
            Sampler.from_function(10, self.example_function,
                                  prob_weights=[-1]+[2]+[0]*8)

        # Check function is callable
        with self.assertRaises(ValueError):
            Sampler.from_function(10, function='banana')

        # check num_subset is an integer
        with self.assertRaises(ValueError):
            Sampler.from_function(10.5, self.example_function)

    def test_init_RandomSampler(self):
        default_sampler = SamplerRandom(10)
        self.assertEqual(default_sampler._num_indices, 10)
        self.assertEqual(default_sampler.num_indices, 10)
        self.assertListEqual(default_sampler._prob_weights, [1/10]*10)
        self.assertListEqual(default_sampler.prob_weights, [1/10]*10)
        self.assertEqual(default_sampler._replace, True)
        self.assertEqual(default_sampler.replace, True)
        self.assertEqual(default_sampler._iteration_number, 0)
        self.assertEqual(default_sampler.current_iter_number, 0)
        self.assertEqual(default_sampler._type, 'random_with_replacement')
        self.assertEqual(default_sampler._sampling_list, None)

        other_sampler = SamplerRandom(
            11, seed=3, sampling_type='banana', replace=False, prob=list(range(11)/np.sum(range(11))))
        self.assertEqual(other_sampler._num_indices, 11)
        self.assertEqual(other_sampler.num_indices, 11)
        self.assertEqual(other_sampler._seed, 3)
        self.assertEqual(other_sampler.seed, 3)
        self.assertEqual(other_sampler._replace, False)
        self.assertEqual(other_sampler.replace, False)
        self.assertListEqual(other_sampler._prob_weights,
                             list(range(11)/np.sum(range(11))))
        self.assertListEqual(other_sampler.prob_weights,
                             list(range(11)/np.sum(range(11))))
        self.assertEqual(other_sampler._iteration_number, 0)
        self.assertEqual(other_sampler.current_iter_number, 0)
        self.assertEqual(other_sampler._type, 'banana')
        self.assertEqual(other_sampler._sampling_list, None)

    def test_from_function(self):

        sampler = Sampler.from_function(50, self.example_function)
        order = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        self.assertNumpyArrayEqual(sampler.get_samples(20), np.array(
            order)[:20])

        N = 25
        for i in range(N):
            self.assertEqual(next(sampler), order[i])

        self.assertEqual(sampler._iteration_number, N)
        self.assertEqual(sampler.current_iter_number, N)

        self.assertEqual(sampler.get_samples(
            550)[519], self.example_function(519))

        sampler = Sampler.from_function(50, self.example_function)
        self.assertListEqual(sampler.prob_weights, [1/50] * 50)
        self.assertEqual(sampler.num_indices, 50)
        self.assertEqual(sampler._type, 'from_function')

        sampler = Sampler.from_function(40, self.example_function, [1]+[0]*39)
        self.assertListEqual(sampler.prob_weights,  [1]+[0]*39)
        self.assertEqual(sampler.num_indices, 40)
        self.assertEqual(sampler._type, 'from_function')
    def test_sequential_iterator_and_get_samples(self):

        sampler = Sampler.sequential(10)
        self.assertEqual(sampler.num_indices, 10)
        self.assertEqual(sampler._type, 'sequential')
        self.assertListEqual(sampler.prob_weights, [1/10]*10)

        sampler = Sampler.sequential(10)
        self.assertNumpyArrayEqual(sampler.get_samples(20), np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

        for i in range(337):
            self.assertEqual(next(sampler), i % 10)

        self.assertNumpyArrayEqual(sampler.get_samples(20), np.array(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

    def test_random_without_replacement_iterator_and_get_samples(self):

        sampler = Sampler.random_without_replacement(8, seed=1)
        self.assertEqual(sampler.num_indices, 8)
        self.assertEqual(sampler._type, 'random_without_replacement')
        self.assertEqual(sampler._prob_weights,  [1/8]*8)
        self.assertEqual(sampler._seed, 1)
        self.assertListEqual(sampler.prob_weights, sampler._prob_weights)

        sampler = Sampler.random_without_replacement(7)
        self.assertEqual(sampler.num_indices, 7)
        self.assertEqual(sampler._type, 'random_without_replacement')
        self.assertEqual(sampler._prob_weights, [1/7]*7)
        self.assertListEqual(sampler.prob_weights, sampler._prob_weights)

        sampler = Sampler.random_without_replacement(7, seed=1)
        order = [2, 5, 0, 1, 4, 3, 6, 1, 6, 0, 4, 2, 3, 5, 5, 6, 2, 4, 0, 1, 3, 0,
                 2, 6, 3]
        self.assertNumpyArrayEqual(
            sampler.get_samples(25), np.array(order[:25]))

        for i in range(25):
            self.assertEqual(next(sampler), order[i])

        self.assertNumpyArrayEqual(
            sampler.get_samples(25), np.array(order[:25]))

    def test_herman_meyer_iterator_and_get_samples(self):

        sampler = Sampler.herman_meyer(12)
        self.assertEqual(sampler.num_indices, 12)
        self.assertEqual(sampler._type, 'herman_meyer')
        self.assertListEqual(sampler.prob_weights, [1/12] * 12)
        out = [sampler.next() for _ in range(12)]
        self.assertListEqual(out, [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11])

        sampler = Sampler.herman_meyer(12)
        order = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5,
                 11, 0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
        self.assertNumpyArrayEqual(
            sampler.get_samples(14), np.array(order[:14]))

        for i in range(25):
            self.assertEqual(sampler.next(), order[i % 12])

        self.assertNumpyArrayEqual(
            sampler.get_samples(14), np.array(order[:14]))

    def test_random_with_replacement_iterator_and_get_samples(self):

        sampler = Sampler.random_with_replacement(5)
        self.assertEqual(sampler.num_indices, 5)
        self.assertEqual(sampler._type, 'random_with_replacement')
        self.assertListEqual(sampler._prob_weights, [1/5] * 5)
        self.assertListEqual(sampler.prob_weights, [1/5] * 5)

        sampler = Sampler.random_with_replacement(4, [0.7, 0.1, 0.1, 0.1])
        self.assertEqual(sampler.num_indices, 4)
        self.assertEqual(sampler._type, 'random_with_replacement')
        self.assertListEqual(sampler.prob_weights, [0.7, 0.1, 0.1, 0.1])

        sampler = Sampler.random_with_replacement(5, seed=5)
        order = [1, 4, 1, 4, 2, 3, 3, 2, 1, 0, 0, 3,
                 2, 0, 4, 1, 2, 1, 3, 2, 2, 1, 1, 1, 1]
        self.assertNumpyArrayEqual(
            sampler.get_samples(14), np.array(order[:14]))

        for i in range(25):
            self.assertEqual(next(sampler), order[i])

        self.assertNumpyArrayEqual(
            sampler.get_samples(14), np.array(order[:14]))

        sampler = Sampler.random_with_replacement(
            4, [0.7, 0.1, 0.1, 0.1], seed=5)
        order = [0, 2, 0, 3, 0, 0, 1, 0, 0, 0, 0, 1,
                 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.assertNumpyArrayEqual(
            sampler.get_samples(14), np.array(order[:14]))

        for i in range(25):
            self.assertEqual(sampler.next(), order[i])

        self.assertNumpyArrayEqual(
            sampler.get_samples(14), np.array(order[:14]))

    def test_staggered_iterator_and_get_samples(self):

        sampler = Sampler.staggered(21, 4)
        self.assertEqual(sampler.num_indices, 21)
        self.assertEqual(sampler._type, 'staggered')
        out = [sampler.next() for _ in range(21)]
        self.assertListEqual(
            out, [0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19])
        self.assertListEqual(sampler.prob_weights, [1/21] * 21)

        with self.assertRaises(ValueError):
            Sampler.staggered(22, 25)

        sampler = Sampler.staggered(21, 4)
        order = [0, 4, 8, 12, 16, 20, 1, 5, 9, 13,
                 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19]
        self.assertNumpyArrayEqual(
            sampler.get_samples(10), np.array(order[:10]))

        for i in range(25):
            self.assertEqual(next(sampler), order[i % 21])

        self.assertNumpyArrayEqual(
            sampler.get_samples(10), np.array(order[:10]))
