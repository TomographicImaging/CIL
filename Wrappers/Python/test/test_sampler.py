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
from cil.framework import Sampler
initialise_tests()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class TestSamplers(CCPiTestClass):
    def test_init(self):

        sampler = Sampler.sequential(10)
        self.assertEqual(sampler.num_subsets, 10)
        self.assertEqual(sampler.type, 'sequential')
        self.assertListEqual(sampler.order, list(range(10)))
        self.assertListEqual(sampler.initial_order, list(range(10)))
        self.assertEqual(sampler.shuffle, False)
        self.assertEqual(sampler.prob, None)
        self.assertEqual(sampler.last_subset, 9)

        sampler = Sampler.randomWithoutReplacement(7, shuffle=True)
        self.assertEqual(sampler.num_subsets, 7)
        self.assertEqual(sampler.type, 'random_without_replacement')
        self.assertListEqual(sampler.order, list(range(7)))
        self.assertListEqual(sampler.initial_order, list(range(7)))
        self.assertEqual(sampler.shuffle, True)
        self.assertEqual(sampler.prob, None)
        self.assertEqual(sampler.last_subset, 6)

        sampler = Sampler.randomWithoutReplacement(8, shuffle=False, seed=1)
        self.assertEqual(sampler.num_subsets, 8)
        self.assertEqual(sampler.type, 'random_without_replacement')
        self.assertEqual(sampler.shuffle, False)
        self.assertEqual(sampler.prob, None)
        self.assertEqual(sampler.last_subset, 7)
        self.assertEqual(sampler.seed, 1)

        sampler = Sampler.hermanMeyer(12)
        self.assertEqual(sampler.num_subsets, 12)
        self.assertEqual(sampler.type, 'herman_meyer')
        self.assertEqual(sampler.shuffle, False)
        self.assertEqual(sampler.prob, None)
        self.assertEqual(sampler.last_subset, 11)
        self.assertListEqual(
            sampler.order, [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11])
        self.assertListEqual(sampler.initial_order, [
                             0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11])

        sampler = Sampler.randomWithReplacement(5)
        self.assertEqual(sampler.num_subsets, 5)
        self.assertEqual(sampler.type, 'random_with_replacement')
        self.assertEqual(sampler.order, None)
        self.assertEqual(sampler.initial_order, None)
        self.assertEqual(sampler.shuffle, False)
        self.assertListEqual(sampler.prob, [1/5] * 5)
        self.assertEqual(sampler.last_subset, 4)

        sampler = Sampler.randomWithReplacement(4, [0.7, 0.1, 0.1, 0.1])
        self.assertEqual(sampler.num_subsets, 4)
        self.assertEqual(sampler.type, 'random_with_replacement')
        self.assertEqual(sampler.order, None)
        self.assertEqual(sampler.initial_order, None)
        self.assertEqual(sampler.shuffle, False)
        self.assertListEqual(sampler.prob, [0.7, 0.1, 0.1, 0.1])
        self.assertEqual(sampler.last_subset, 3)

        sampler = Sampler.staggered(21, 4)
        self.assertEqual(sampler.num_subsets, 21)
        self.assertEqual(sampler.type, 'staggered')
        self.assertListEqual(sampler.order, [
                             0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19])
        self.assertListEqual(sampler.initial_order, [
                             0, 4, 8, 12, 16, 20, 1, 5, 9, 13, 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19])
        self.assertEqual(sampler.shuffle, False)
        self.assertEqual(sampler.prob, None)
        self.assertEqual(sampler.last_subset, 20)

        try:
            Sampler.staggered(22, 25)
        except ValueError:
            self.assertTrue(True)

        sampler = Sampler.customOrder([1, 4, 6, 7, 8, 9, 11])
        self.assertEqual(sampler.num_subsets, 7)
        self.assertEqual(sampler.type, 'custom_order')
        self.assertListEqual(sampler.order, [1, 4, 6, 7, 8, 9, 11])
        self.assertListEqual(sampler.initial_order, [1, 4, 6, 7, 8, 9, 11])
        self.assertEqual(sampler.shuffle, False)
        self.assertEqual(sampler.prob, None)
        self.assertEqual(sampler.last_subset, 6)


        
    def test_sequential_iterator_and_get_samples(self):
        
        #Test the squential sampler 
        sampler = Sampler.sequential(10)
        for i in range(25):
            self.assertEqual(next(sampler), i % 10)
            if i%5==0: # Check both that get samples works and doesn't interrupt the sampler
                self.assertNumpyArrayEqual(sampler.get_samples(), np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))
       
        sampler = Sampler.sequential(10)
        for i in range(25):
            self.assertEqual(sampler.next(), i % 10) # Repeat the test for .next()
            if i%5==0:
                self.assertNumpyArrayEqual(sampler.get_samples(), np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

    def test_random_without_replacement_iterator_and_get_samples(self):        
        #Test the random without replacement sampler 
        sampler = Sampler.randomWithoutReplacement(7, shuffle=True, seed=1)
        order = [6, 2, 1, 0, 4, 3, 5, 1, 0, 4, 2, 5,
                 6, 3, 3, 2, 1, 4, 0, 5, 6, 2, 6, 3, 4]
        for i in range(25):
            self.assertEqual(next(sampler), order[i])
            if i%4==0:# Check both that get samples works and doesn't interrupt the sampler
                self.assertNumpyArrayEqual(sampler.get_samples(6), np.array(order[:6]))
        
        #Repeat the test for shuffle=False
        sampler = Sampler.randomWithoutReplacement(8, shuffle=False, seed=1)
        order = [7, 2, 1, 6, 0, 4, 3, 5]
        for i in range(25):
            self.assertEqual(sampler.next(), order[i % 8])
            if i%5==0:# Check both that get samples works and doesn't interrupt the sampler
                self.assertNumpyArrayEqual(sampler.get_samples(5), np.array(order[:5]))

    def test_herman_meyer_iterator_and_get_samples(self): 
        #Test the Herman Meyer sampler
        sampler = Sampler.hermanMeyer(12)
        order = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11, 0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
        for i in range(25):
            self.assertEqual(sampler.next(), order[i % 12])
            if i%5==0:# Check both that get samples works and doesn't interrupt the sampler
                self.assertNumpyArrayEqual(sampler.get_samples(14), np.array(order[:14]))

    def test_random_with_replacement_iterator_and_get_samples(self): 
        #Test the Random with replacement sampler
        sampler = Sampler.randomWithReplacement(5, seed=5)
        order=[1, 4, 1, 4, 2, 3, 3, 2, 1, 0, 0, 3, 2, 0, 4, 1, 2, 1, 3, 2, 2, 1, 1, 1, 1]
        for i in range(25):
            self.assertEqual(next(sampler), order[i])
            if i%5==0:# Check both that get samples works and doesn't interrupt the sampler
                self.assertNumpyArrayEqual(sampler.get_samples(14), np.array(order[:14]))

        sampler = Sampler.randomWithReplacement(
            4, [0.7, 0.1, 0.1, 0.1], seed=5)
        order = [0, 2, 0, 3, 0, 0, 1, 0, 0, 0, 0, 1,
                 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(25):
            self.assertEqual(sampler.next(), order[i])
            if i%5==0:# Check both that get samples works and doesn't interrupt the sampler
                self.assertNumpyArrayEqual(sampler.get_samples(14), np.array(order[:14]))

    def test_staggered_iterator_and_get_samples(self): 
        #Test the staggered sampler
        sampler = Sampler.staggered(21, 4)
        order = [0, 4, 8, 12, 16, 20, 1, 5, 9, 13,
                 17, 2, 6, 10, 14, 18, 3, 7, 11, 15, 19]
        for i in range(25):
            self.assertEqual(next(sampler), order[i % 21])
            if i%5==0:# Check both that get samples works and doesn't interrupt the sampler
                self.assertNumpyArrayEqual(sampler.get_samples(10), np.array(order[:10]))

    def test_custom_order_iterator_and_get_samples(self): 
        #Test the custom order sampler
        sampler = Sampler.customOrder([1, 4, 6, 7, 8, 9, 11])
        order = [1, 4, 6, 7, 8, 9, 11,1, 4, 6, 7, 8, 9, 11,1, 4, 6, 7, 8, 9, 11,1, 4, 6, 7, 8, 9, 11]
        for i in range(25):
            self.assertEqual(sampler.next(), order[i % 7])
            if i%5==0:# Check both that get samples works and doesn't interrupt the sampler
                self.assertNumpyArrayEqual(sampler.get_samples(10), np.array(order[:10]))