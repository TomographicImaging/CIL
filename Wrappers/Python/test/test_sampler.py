# -*- coding: utf-8 -*-
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
from cil.optimisation.utilities import Sampler
initialise_tests()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class TestSamplers(CCPiTestClass):

    def example_function(self, iteration_number):
        if iteration_number < 500:
            np.random.seed(iteration_number)
            return (np.random.choice(49, 1)[0])
        else:
            np.random.seed(iteration_number)
            return (np.random.choice(50, 1)[0])

    def test_init(self):

        sampler = Sampler.sequential(10)
        self.assertEqual(sampler.max_index_number, 10)
        self.assertEqual(sampler._type, 'sequential')
        self.assertListEqual(sampler._order, list(range(10)))
        self.assertEqual(sampler._last_index, 9)
        self.assertListEqual(sampler.prob_weights, [1/10]*10)

        sampler = Sampler.random_without_replacement(7)
        self.assertEqual(sampler.max_index_number, 7)
        self.assertEqual(sampler._type, 'random_without_replacement')
        self.assertEqual(sampler._prob, [1/7]*7)
        self.assertListEqual(sampler.prob_weights, sampler._prob)

        sampler = Sampler.random_without_replacement(8, seed=1)
        self.assertEqual(sampler.max_index_number, 8)
        self.assertEqual(sampler._type, 'random_without_replacement')
        self.assertEqual(sampler._prob,  [1/8]*8)
        self.assertEqual(sampler._seed, 1)
        self.assertListEqual(sampler.prob_weights, sampler._prob)

        sampler = Sampler.herman_meyer(12)
        self.assertEqual(sampler.max_index_number, 12)
        self.assertEqual(sampler._type, 'herman_meyer')
        self.assertEqual(sampler._last_index, 11)
        self.assertListEqual(
            sampler._order, [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11])
        self.assertListEqual(sampler.prob_weights, [1/12] * 12)

        sampler = Sampler.random_with_replacement(5)
        self.assertEqual(sampler.max_index_number, 5)
        self.assertEqual(sampler._type, 'random_with_replacement')
        self.assertListEqual(sampler._prob, [1/5] * 5)
        self.assertListEqual(sampler.prob_weights, [1/5] * 5)

        sampler = Sampler.random_with_replacement(4, [0.7, 0.1, 0.1, 0.1])
        self.assertEqual(sampler.max_index_number, 4)
        self.assertEqual(sampler._type, 'random_with_replacement')
        self.assertListEqual(sampler._prob, [0.7, 0.1, 0.1, 0.1])
        self.assertListEqual(sampler.prob_weights, [0.7, 0.1, 0.1, 0.1])

        sampler = Sampler.staggered(21, 4)
        self.assertEqual(sampler.max_index_number, 21)
        self.assertEqual(sampler._type, 'staggered')
        self.assertListEqual(sampler._order, [
                             0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19])
        self.assertEqual(sampler._last_index, 20)
        self.assertListEqual(sampler.prob_weights, [1/21] * 21)

        with self.assertRaises(ValueError):
            Sampler.staggered(22, 25)

        
        #Check probabilities sum to one and are positive
        with self.assertRaises(ValueError): 
            Sampler.from_function(10, self.example_function, prob_weights=[1/11]*10)
        with self.assertRaises(ValueError): 
            Sampler.from_function(10, self.example_function, prob_weights=[-1]+[2]+[0]*8)
        

        sampler = Sampler.from_function(50, self.example_function)
        self.assertListEqual(sampler.prob_weights, [1/50] * 50)
        self.assertEqual(sampler.max_index_number, 50)
        self.assertEqual(sampler._type, 'from_function')
        
        sampler = Sampler.from_function(40, self.example_function, [1]+[0]*39)
        self.assertListEqual(sampler.prob_weights,  [1]+[0]*39)
        self.assertEqual(sampler.max_index_number, 40)
        self.assertEqual(sampler._type, 'from_function')
        
        #check probabilities sum to 1 and are positive
        with self.assertRaises(ValueError):
            Sampler.from_function(40, self.example_function, [0.9]+[0]*39)
        with self.assertRaises(ValueError):
            Sampler.from_function(40, self.example_function, [-1]+[2]+[0]*38)

    def test_from_function(self):

        sampler = Sampler.from_function(50, self.example_function)
        order = [44, 37, 40, 42, 46, 35, 10, 47, 3, 28, 9, 25,
                 11, 18, 43, 8, 41, 47, 42, 29, 35, 9, 4, 19, 34]
        for i in range(25):
            self.assertEqual(next(sampler), order[i])
            if i % 5 == 0:  # Check both that get samples works and doesn't interrupt the sampler
                self.assertNumpyArrayEqual(sampler.get_samples(), np.array(
                    order)[:20])

    def test_sequential_iterator_and_get_samples(self):
        # Test the squential sampler
        sampler = Sampler.sequential(10)
        for i in range(25):
            self.assertEqual(next(sampler), i % 10)
            if i % 5 == 0:  # Check both that get samples works and doesn't interrupt the sampler
                self.assertNumpyArrayEqual(sampler.get_samples(), np.array(
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

        sampler = Sampler.sequential(10)
        for i in range(25):
            # Repeat the test for .next()
            self.assertEqual(sampler.next(), i % 10)
            if i % 5 == 0:
                self.assertNumpyArrayEqual(sampler.get_samples(), np.array(
                    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

    def test_random_without_replacement_iterator_and_get_samples(self):
        # Test the random without replacement sampler
        sampler = Sampler.random_without_replacement(7, seed=1)
        order = [2, 5, 0, 2, 1, 0, 1, 2, 2, 3, 2, 4,
                 1, 6, 0, 4, 2, 3, 0, 1, 5, 6, 2, 4, 6]
        for i in range(25):
            self.assertEqual(next(sampler), order[i])
            if i % 4 == 0:  # Check both that get samples works and doesn't interrupt the sampler
                self.assertNumpyArrayEqual(
                    sampler.get_samples(6), np.array(order[:6]))

    def test_herman_meyer_iterator_and_get_samples(self):
        # Test the Herman Meyer sampler
        sampler = Sampler.herman_meyer(12)
        order = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5,
                 11, 0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
        for i in range(25):
            self.assertEqual(sampler.next(), order[i % 12])
            if i % 5 == 0:  # Check both that get samples works and doesn't interrupt the sampler
                self.assertNumpyArrayEqual(
                    sampler.get_samples(14), np.array(order[:14]))

    def test_random_with_replacement_iterator_and_get_samples(self):
        # Test the Random with replacement sampler
        sampler = Sampler.random_with_replacement(5, seed=5)
        order = [1, 4, 1, 4, 2, 3, 3, 2, 1, 0, 0, 3,
                 2, 0, 4, 1, 2, 1, 3, 2, 2, 1, 1, 1, 1]
        for i in range(25):
            self.assertEqual(next(sampler), order[i])
            if i % 5 == 0:  # Check both that get samples works and doesn't interrupt the sampler
                self.assertNumpyArrayEqual(
                    sampler.get_samples(14), np.array(order[:14]))

        sampler = Sampler.random_with_replacement(
            4, [0.7, 0.1, 0.1, 0.1], seed=5)
        order = [0, 2, 0, 3, 0, 0, 1, 0, 0, 0, 0, 1,
                 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(25):
            self.assertEqual(sampler.next(), order[i])
            if i % 5 == 0:  # Check both that get samples works and doesn't interrupt the sampler
                self.assertNumpyArrayEqual(
                    sampler.get_samples(14), np.array(order[:14]))

    def test_staggered_iterator_and_get_samples(self):
        # Test the staggered sampler
        sampler = Sampler.staggered(21, 4)
        order = [0, 4, 8, 12, 16, 20, 1, 5, 9, 13,
                 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19]
        for i in range(25):
            self.assertEqual(next(sampler), order[i % 21])
            if i % 5 == 0:  # Check both that get samples works and doesn't interrupt the sampler
                self.assertNumpyArrayEqual(
                    sampler.get_samples(10), np.array(order[:10]))


