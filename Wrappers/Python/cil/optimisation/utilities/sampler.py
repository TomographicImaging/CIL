# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi
#   (Collaborative Computational Project in Tomographic Imaging), with
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.


import numpy as np
import math
import time
from functools import partial


class Sampler():
    """     
    The user is recommended to not instantiate this class  directly but instead use one of the static methods that will return instances of different samplers.
    
    This class wraps a function that takes an in integer iteration number and returns an integer from {0, 1, …, S-1} with S=num_indices to produce a Sampler which returns a new index for each call of `next`.

    
    Custom samplers can be created by subclassing this sampler class. 

    

    Parameters
    ----------

    function: A function that takes an in integer iteration number and returns an integer from {0, 1, …, S-1} with S=num_indices. 

    num_indices: int
        One above the largest integer that could be drawn by the sampler. The sampler will select from a list of indices {0, 1, …, S-1} with S=num_indices. 

    sampling_type:str default is None 
        The sampling type used. Choose from  "sequential", "staggered", "herman_meyer" and "from_function". 

    prob_weights: list of floats of length num_indices that sum to 1.  Default is [1/num_indices]*num_indices 
        Consider that the sampler is called a large number of times this argument holds the expected number of times each index would be called,  normalised to 1. 

    Returns
    -------
    A Sampler that can be called with Sampler.next() or next(Sampler)

    Example
    -------
    >>> sampler = Sampler.random_with_replacement(5)
    >>> print(sampler.get_samples())
    [3 4 0 0 2 3 3 2 2 1 1 4 4 3 0 2 4 4 2 4]


    Example
    -------    
    >>> sampler = Sampler.staggered(21,4)
    >>> print(sampler.get_samples(5))
    [ 0  4  8 12 16]

    Example
    -------
    >>> sampler = Sampler.sequential(10)
    >>> print(sampler.get_samples(5))
    [0 1 2 3 4]

    Example
    -------
    >>> sampler = Sampler.herman_meyer(12)
    >>> print(sampler.get_samples(16))
    [ 0  6  3  9  1  7  4 10  2  8  5 11  0  6  3  9]



    Example
    -------
    This example creates a sampler that samples in order from a custom list. The num_indices  is 13, although note that the index 12 is never called by the sampler. The number of indices must be at least one greater than any of the elements in the custom_list. 
    The probability weights are calculated and passed to the sampler as they are not uniform. 

    >>> custom_list = [1,1,1,0,0,11,5,9,8,3]
    >>> num_indices = 13
    >>> 
    >>> def test_function(iteration_number, custom_list=custom_list):
    >>>    return(custom_list[iteration_number%len(custom_list)])
    >>> 
    >>> #calculate prob weights 
    >>> temp_list = []
    >>> for i in range(num_indices):
    >>>     temp_list.append(custom_list.count(i))
    >>> total = sum(temp_list)
    >>> prob_weights = [x/total for x in temp_list]
    >>> 
    >>> sampler = Sampler.from_function(num_indices=num_indices, function=test_function, prob_weights=prob_weights)
    >>> print(list(sampler.get_samples(25)))
    [1, 1, 1, 0, 0, 11, 5, 9, 8, 3, 1, 1, 1, 0, 0, 11, 5, 9, 8, 3, 1, 1, 1, 0, 0]
    >>> print(sampler)
    Sampler that wraps a function that takes an iteration number and selects from a list of indices {0, 1, …, S-1}, where S is the number of indices. 
    Type : from_function 
    Current iteration number : 11 
    number of indices : 13 
    Probability weights : [0.2, 0.3, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 0.0] 



    Note
    -----
    The optimal choice of sampler depends on the data and the number of calls to the sampler. 

    For random sampling with replacement, there is the possibility, with a small number of calls to the sampler that some indices will not have been selected. For the case of uniform probabilities, the default, the number of
    iterations required such that the probability that all indices have been selected at least once is greater than :math:`p` grows as :math:`nlog(n/p)` where `n` is `num_indices`. 
    For example, to be 99% certain that you have seen all indices, for `n=20` you should take at least 152 samples, `n=50` at least 426 samples. To be more likely than not, for `n=20` you should take 78 samples and `n=50` you should take 228 samples. 
    In general, we note that for a large number of samples (e.g. `>20*num_indices`), the density of the outputted samples looks more and more uniform. For a small number of samples (e.g. `<5*num_indices`) the user may wish to consider
    another sampling method e.g. random without replacement, which, when calling `num_indices` samples is guaranteed to draw each index exactly once.  
        """

    def __init__(self, num_indices, function,  sampling_type=None, prob_weights=None):

        self._type = sampling_type
        self._num_indices = num_indices
        self.function = function

        if prob_weights is None:
            prob_weights = [1/num_indices]*num_indices
        else:
            if abs(sum(prob_weights)-1) > 1e-6:
                raise ValueError('The provided prob_weights must sum to one')

            if any(np.array(prob_weights) < 0):
                raise ValueError(
                    'The provided prob_weights must be greater than or equal to zero')

        self._prob_weights = prob_weights
        self._iteration_number = 0

    @property
    def prob_weights(self):
        return self._prob_weights

    @property
    def num_indices(self):
        return self._num_indices

    @property
    def current_iter_number(self):
        return self._iteration_number

    def next(self):
        """ 
        Returns a sample from the list of indices `{0, 1, …, S-1}, where S is the number of indices and increments the sampler.
        """

        out = self.function(self._iteration_number)

        self._iteration_number += 1
        return out

    def __next__(self):
        return self.next()

    def get_samples(self,  num_samples=20):
        """
        Returns the first `num_samples` produced by the sampler as a numpy array.

        Parameters
        ----------
        num_samples: int, default = 20
            The number of samples to return. 
        """
        save_last_index = self._iteration_number
        self._iteration_number = 0

        output = [self.next() for _ in range(num_samples)]

        self._iteration_number = save_last_index

        return np.array(output)

    def __str__(self):
        repres = "Sampler that selects from a list of indices {0, 1, …, S-1}, where S is the number of indices. \n"
        repres += "Type : {} \n".format(self._type)
        repres += "Current iteration number : {} \n".format(
            self._iteration_number)
        repres += "Number of indices : {} \n".format(self._num_indices)
        repres += "Probability weights : {} \n".format(self._prob_weights)
        return repres

    @staticmethod
    def sequential(num_indices):
        """
        Instantiates a sampler that outputs sequentially. 

        Parameters
        ----------
        num_indices: int
            One above the largest integer that could be drawn by the sampler. The sampler will select from a list of indices {0, 1, …, S-1} with S=num_indices. 
        
        Returns
        -------
        A Sampler  that can be called with Sampler.next()  or next(Sampler) and outputs sequentially

        Example
        -------

        >>> sampler=Sampler.sequential(10)
        >>> print(sampler.get_samples(5))
        [0 1 2 3 4]
        """
        def function(x):
            return x % num_indices

        sampler = Sampler(function=function, num_indices=num_indices, sampling_type='sequential',  prob_weights=[
            1/num_indices]*num_indices)
        return sampler

    @staticmethod
    def _staggered_function(num_indices, stride, iter_number):
        """Function that takes in an iteration number and outputs an index number based on the staggered ordering.  """
        iter_number_mod = iter_number % num_indices
        floor = num_indices // stride
        mod = num_indices % stride

        if iter_number_mod < (floor + 1)*mod:
            row_number = iter_number_mod // (floor + 1)
            column_number = (iter_number_mod % (floor + 1))
        else:
            row_number = mod + (iter_number_mod - (floor+1)*mod) // floor
            column_number = (iter_number_mod - (floor+1)*mod) % floor

        return row_number + stride*column_number

    @staticmethod
    def staggered(num_indices, stride):
        """
        Instantiates a sampler which outputs in a staggered order. 

        Parameters
        ----------
        num_indices: int
            One above the largest integer that could be drawn by the sampler. The sampler will select from a list of indices {0, 1, …, S-1} with S=num_indices. 

        stride: int
            The sampler will output in the order {0, a, 2a, 3a, ...., 1, 1+a, 1+2a, 1+3a,...., 2, 2+a, 2+2a, 2+3a,...} where a=stride. 
            The stride should be less than the num_indices

        Returns
        -------
        A Sampler  that can be called with Sampler.next()  or next(Sampler) and outputs in a staggered ordering

        Example
        -------
        >>> sampler=Sampler.staggered(21,4)
        >>> print(sampler.get_samples(5))
        [ 0  4  8 12 16]
        """

        if stride >= num_indices:
            raise (ValueError('The stride should be less than the number of indices'))

        sampler = Sampler(function=partial(Sampler._staggered_function, num_indices, stride), num_indices=num_indices, sampling_type='staggered', prob_weights=[
            1/num_indices]*num_indices)

        return sampler

    @staticmethod
    def random_with_replacement(num_indices, prob=None, seed=None):
        """
        Instantiates a sampler which outputs from a list of indices {0, 1, …, S-1}, with S = num_indices, with given probability and with replacement. 

        Parameters
        ----------
        num_indices: int
            One above the largest integer that could be drawn by the sampler. The sampler will select from a list of indices {0, 1, …, S-1} with S = num_indices. 

        prob: list of floats of length num_indices that sum to 1. default is None
            This is the probability for each index to be called by next. If None, then the indices will be sampled uniformly. 

        seed:int, default is None
            Random seed for the random number generator.  If set to None, the seed will be set using the current time.

        Returns
        -------
        A Sampler  that can be called with Sampler.next()  or next(Sampler) that samples randomly with replacement 


        Example
        -------
        >>> sampler = Sampler.random_with_replacement(5)
        >>> print(sampler.get_samples(10))
        [3 4 0 0 2 3 3 2 2 1]

        >>> sampler = Sampler.random_with_replacement(4, [0.7,0.1,0.1,0.1])
        >>> print(sampler.get_samples(10))
        [0 1 3 0 0 3 0 0 0 0]
        """

        if prob == None:
            prob = [1/num_indices] * num_indices

        sampler = SamplerRandom(
            num_indices=num_indices, sampling_type='random_with_replacement', replace=True, prob=prob, seed=seed)
        return sampler

    @staticmethod
    def random_without_replacement(num_indices, seed=None):
        """
        Instantiates a sampler which outputs from a list of indices {0, 1, …, S-1}, with S=num_indices, uniformly randomly without replacement.

        Parameters
        ----------
        num_indices: int
            One above the largest integer that could be drawn by the sampler. The sampler will select from a list of indices {0, 1, …, S-1} with S=num_indices. 

        seed:int, default=None
            Random seed for the  random number generator.  If set to None, the seed will be set using the current time. 

        Returns
        -------
        A Sampler  that can be called with Sampler.next()  or next(Sampler) that samples randomly without replacement 
        Example
        -------
        >>> sampler=Sampler.randomWithoutReplacement(7, seed=1)
        >>> print(sampler.get_samples(16))
        [6 2 1 0 4 3 5 1 0 4 2 5 6 3 3 2]

        """

        sampler = SamplerRandom(
            num_indices=num_indices, sampling_type='random_without_replacement', replace=False,  seed=seed)
        return sampler

    @staticmethod
    def from_function(num_indices, function, prob_weights=None):
        """
        Instantiates a sampler that wraps a function that takes an iteration number and selects from a list of indices {0, 1, …, S-1}.

        Parameters
        ----------
        num_indices: int
            One above the largest integer that could be drawn by the sampler. The sampler will select from a list of indices {0, 1, …, S-1} with S=num_indices. 

        function: A function that takes an in integer iteration number and returns an integer from {0, 1, …, S-1} with S=num_indices. 

        prob_weights: list of floats of length num_indices that sum to 1. Default is [1/num_indices]*num_indices 
            Consider that the sampler is called a large number of times this argument holds the expected number of times each index would be called,  normalised to 1. 


        Note
        -----
        If your function involves a random number generator, then it may be easier to subclass the SamplerRandom class instead. 

        Returns
        -------
        A Sampler that wraps a function and can be called with Sampler.next()  or next(Sampler) 


        Example
        -------
        This example creates a sampler that samples in order from a custom list. The num_indices  is 13, although note that the index 12 is never called by the sampler. The number of indices must be at least one greater than any of the elements in the custom_list. 
        The probability weights are calculated and passed to the sampler as they are not uniform. 

        >>> custom_list = [1,1,1,0,0,11,5,9,8,3]
        >>> num_indices = 13
        >>> 
        >>> def test_function(iteration_number, custom_list=custom_list):
        >>>    return(custom_list[iteration_number%len(custom_list)])
        >>> 
        >>> #calculate prob weights 
        >>> temp_list = []
        >>> for i in range(num_indices):
        >>>     temp_list.append(custom_list.count(i))
        >>> total = sum(temp_list)
        >>> prob_weights = [x/total for x in temp_list]
        >>> 
        >>> sampler = Sampler.from_function(num_indices=num_indices, function=test_function, prob_weights=prob_weights)
        >>> print(list(sampler.get_samples(25)))
        [1, 1, 1, 0, 0, 11, 5, 9, 8, 3, 1, 1, 1, 0, 0, 11, 5, 9, 8, 3, 1, 1, 1, 0, 0]
        >>> print(sampler)
        Sampler that wraps a function that takes an iteration number and selects from a list of indices {0, 1, …, S-1}, where S is the number of indices. 
        Type : from_function 
        Current iteration number : 11 
        number of indices : 13 
        Probability weights : [0.2, 0.3, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 0.0] 

        """

        if prob_weights is None:
            prob_weights = [1/num_indices]*num_indices

        sampler = Sampler(
            num_indices=num_indices, sampling_type='from_function', function=function, prob_weights=prob_weights)
        return sampler

    @staticmethod
    def _prime_factorisation(n):
        """
        Parameters
        ----------

        n: int
            The number to be factorised.

        Returns
        -------
        
        factors: list of ints
            The prime factors of n.

        """
        factors = []

        while n % 2 == 0:
            n //= 2
            factors.append(2)

        i = 3
        while i*i <= n:
            while n % i == 0:
                n //= i
                factors.append(i)
            i += 2

        if n > 1:
            factors.append(n)

        return factors

    @staticmethod
    def _herman_meyer_function(num_indices,  addition_arr, repeat_length_arr, iteration_number):
        """
        Parameters
        ----------
        num_indices: int
            The number of indices to be sampled from.

        addition_arr: list of ints
            The product of all factors at indices greater than the current factor.

        repeat_length_arr: list of ints
            The product of all factors at indices less than the current factor.

        iteration_number: int
            The current iteration number.

        Returns
        -------
        index: int
            The index to be sampled from.

        """

        index = 0
        for n in range(len(addition_arr)):
            addition = addition_arr[n]
            repeat_length = repeat_length_arr[n]

            length = num_indices // (addition*repeat_length)
            arr = np.arange(length) * addition

            ind = math.floor(iteration_number/repeat_length) % length
            index += arr[ind]

        return index

    @staticmethod
    def herman_meyer(num_indices):
        """
        Instantiates a sampler which outputs in a Herman Meyer order.

        Parameters
        ----------
        num_indices: int
            One above the largest integer that could be drawn by the sampler. The sampler will select from a list of indices {0, 1, …, S-1} with S=num_indices. For Herman-Meyer sampling this number should not be prime. 

        Reference
        ----------
        Herman GT, Meyer LB. Algebraic reconstruction techniques can be made computationally efficient. IEEE Trans Med Imaging.  doi: 10.1109/42.241889.

        Returns
        -------
        A Sampler  that can be called with Sampler.next()  or next(Sampler) and outputs in a Herman Meyer ordering 

        Example
        -------
        >>> sampler=Sampler.herman_meyer(12)
        >>> print(sampler.get_samples(16))
        [ 0  6  3  9  1  7  4 10  2  8  5 11  0  6  3  9]
        """

        factors = Sampler._prime_factorisation(num_indices)

        n_factors = len(factors)
        if n_factors == 1:
            raise ValueError(
                'Herman Meyer sampling defaults to sequential ordering if the number of indices is prime. Please use an alternative sampling method or change the number of indices. ')

        addition_arr = np.empty(n_factors, dtype=np.int64)
        repeat_length_arr = np.empty(n_factors, dtype=np.int64)

        repeat_length = 1
        addition = num_indices
        for i in range(n_factors):
            addition //= factors[i]
            addition_arr[i] = addition

            repeat_length_arr[i] = repeat_length
            repeat_length *= factors[i]

        hmf_call = partial(Sampler._herman_meyer_function,
                           num_indices, addition_arr, repeat_length_arr)

        # define the sampler
        sampler = Sampler(function=hmf_call,
                          num_indices=num_indices, sampling_type='herman_meyer', prob_weights=[1/num_indices]*num_indices)

        return sampler


class SamplerRandom(Sampler):
    """     
    The user is recommended to not instantiate this class  directly but instead use one of the static methods  in the parent Sampler class that will return instances of different samplers.
    
    This class produces Samplers that output random samples with and without replacement from the set {0, 1, …, S-1} where S=num_indices.

    Custom samplers can be created by subclassing this sampler class. 

    Parameters
    ----------

    num_indices: int
        One above the largest integer that could be drawn by the sampler. The sampler will select from a list of indices {0, 1, …, S-1} with S=num_indices. 

    sampling_type:str default = 'random_with_replacement"
        The sampling type used. Choose from "random_with_replacement" and "random_without_replacement"

    prob_weights: list of floats of length num_indices that sum to 1.  Default is [1/num_indices]*num_indices 
        Consider that the sampler is called a large number of times this argument holds the expected number of times each index would be called,  normalised to 1. 

    replace: bool, default is True 
        If True, sample with replace, otherwise sample without replacement

    seed:int, default = None
        Random seed for the methods that use a numpy random number generator.  If set to None, the seed will be set using the current time.

    Returns
    -------
    A Sampler that can be called with Sampler.next() or next(Sampler)

    Example
    -------
    >>> sampler = Sampler.random_with_replacement(5)
    >>> print(sampler.get_samples())
    [3 4 0 0 2 3 3 2 2 1 1 4 4 3 0 2 4 4 2 4]

    Example
    -------
    >>> sampler=Sampler.randomWithoutReplacement(7, seed=1)
    >>> print(sampler.get_samples(16))
    [6 2 1 0 4 3 5 1 0 4 2 5 6 3 3 2]

    """


    def __init__(self, num_indices,  seed=None, replace=True, prob=None,  sampling_type='random_with_replacement'):

        if seed is not None:
            self._seed = seed
        else:
            self._seed = int(time.time())
        self._generator = np.random.RandomState(self._seed)
        self._sampling_list = None
        self._replace = replace

        super(SamplerRandom, self).__init__(num_indices, self.function,
                                            sampling_type=sampling_type, prob_weights=prob)

    @property
    def seed(self):
        return self._seed

    @property
    def replace(self):
        return self._replace

    def function(self, iteration_number):
        """ For each iteration number this function samples from a randomly generated list in order. Every num_indices the list is re-created. """
        location = iteration_number % self._num_indices
        if location == 0:
            self._sampling_list = self._generator.choice(
                self._num_indices, self._num_indices, p=self._prob_weights, replace=self._replace)
        out = self._sampling_list[location]
        return out

    def get_samples(self,  num_samples=20):
        """
        Returns the first `num_samples` produced by the sampler as a numpy array.

        Parameters
        ----------
        num_samples: int, default = 20
            The number of samples to return. 
        """
        save_last_index = self._iteration_number
        self._iteration_number = 0

        save_generator = self._generator
        self._generator = np.random.RandomState(self._seed)
        save_sampling_list = self._sampling_list

        output = [self.next() for _ in range(num_samples)]

        self._iteration_number = save_last_index

        self._generator = save_generator
        self._sampling_list = save_sampling_list

        return np.array(output)

    def __str__(self):  
        repres=super().__str__()
        repres += "Seed : {} \n".format(self._seed)
        return repres


