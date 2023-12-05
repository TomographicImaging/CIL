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


class SamplerFromFunction():
    """     
        A class that wraps a function that takes an iteration number and selects from a list of indices {0, 1, …, S-1}.
        The function next() outputs a single next index from the list {0,1,…,S-1}.To be run again and again, depending on how many iterations.

        It is recommended to use the static methods to configure your Sampler object rather than initialising this class directly: the user should call this through Sampler.from_function(max_index_number, function, prob_weights) from cil.optimisation.utilities.sampler.

        Parameters
        ----------
        
        function: A function that takes an in integer iteration number and returns an integer from {0, 1, …, S-1} with S=max_index_number. 
        
        max_index_number: int
            The sampler will select from a list of indices {0, 1, …, S-1} with S=max_index_number. 

        sampling_type:str
            The sampling type used. This is set to the default "from_function".

        prob_weights: list of floats of length max_index_number that sum to 1.  Default is [1/max_index_number]*max_index_number 
            Consider that the sampler is called a large number of times this argument holds the expected number of times each index would be called,  normalised to 1. 
        
        Example
        -------
        >>> def test_function(iteration_number):
        >>>     if iteration_number<500:
        >>>         np.random.seed(iteration_number)
        >>>         return(np.random.choice(49,1)[0])
        >>>     else:
        >>>         np.random.seed(iteration_number)
        >>>         return(np.random.choice(50,1)[0])
        >>>
        >>> Sampler.from_function(max_index_number, function, prob_weights=None)
        >>> for _ in range(11):
        >>>     print(sampler.next())
        >>> print(list(sampler.get_samples(25)))
        44
        37
        40
        42
        46
        35
        10
        47
        3
        28
        9
        [44, 37, 40, 42, 46, 35, 10, 47, 3, 28, 9, 25, 11, 18, 43, 8, 41, 47, 42, 29, 35, 9, 4, 19, 34]


        Example
        -------
        >>> custom_list=[1,1,1,0,0,11,5,9,8,3]
        >>> max_index_number=13
        >>> 
        >>> def test_function(iteration_number, custom_list=custom_list):
            return(custom_list[iteration_number%len(custom_list)])
        >>> 
        >>> #calculate prob weights 
        >>> temp_list = []
        >>> for i in range(max_index_number):
        >>>     temp_list.append(custom_list.count(i))
        >>> total = sum(temp_list)
        >>> prob_weights = [x/total for x in temp_list]
        >>> 
        >>> sampler=Sampler.from_function(max_index_number=max_index_number, function=test_function, prob_weights=prob_weights)
        >>> for _ in range(11):
        >>>     print(sampler.next())
        >>> print(list(sampler.get_samples(25)))
        >>> print(sampler)
        1
        1
        1
        0
        0
        11
        5
        9
        8
        3
        1
        [1, 1, 1, 0, 0, 11, 5, 9, 8, 3, 1, 1, 1, 0, 0, 11, 5, 9, 8, 3, 1, 1, 1, 0, 0]
        Sampler that wraps a function that takes an iteration number and selects from a list of indices {0, 1, …, S-1}, where S is the max index number. 
        Type : from_function 
        Current iteration number : 11 
        Max index number : 13 
        Probability weights : [0.2, 0.3, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 0.0] 
        
        
        Note
        -----
        If your function involves a random number generator, then the seed should also depend on the iteration number, see the first example in the documentation, otherwise
        the `get_samples()` function may not accurately return the correct samples and may interrupt the next sample returned. 
        """
        
    def __init__(self, function, max_index_number,  sampling_type='from_function', prob_weights=None):
        
        self._type = sampling_type
        self._max_index_number = max_index_number
        self.function = function

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
    def max_index_number(self):
        return self._max_index_number 
    
    
    def next(self):
        """ 
        Returns and increments the sampler 
        """
        out = self.function(self._iteration_number)
        self._iteration_number += 1
        return out

    def __next__(self):
        return self.next()

    def get_samples(self,  num_samples=20):
        """
        Returns the first `num_samples` produced by the sampler as a numpy array.

        num_samples: int, default=20
            The number of samples to return. 
        """
        save_last_index = self._iteration_number
        self._iteration_number = 0
        output = [self.next() for _ in range(num_samples)]
        self._iteration_number = save_last_index
        return np.array(output)

    def __str__(self):
        repres = "Sampler that wraps a function that takes an iteration number and selects from a list of indices {0, 1, …, S-1}, where S is the max index number. \n"
        repres += "Type : {} \n".format(self._type)
        repres += "Current iteration number : {} \n".format(self._iteration_number)
        repres += "Max index number : {} \n".format(self._max_index_number)
        repres += "Probability weights : {} \n".format(self._prob_weights)
        return repres
    
class SamplerFromOrder():

    def __init__(self,  order, max_index_number, sampling_type,  prob_weights=None):
        """
       This sampler will sample from a list `order` that is passed. 

        It is recommended to use the static methods to configure your Sampler object rather than initialising this class directly: the user should call this through cil.optimisation.utilities.sampler and choose the desired static method from the Sampler class.  

        Parameters
        ----------
        order: list of indices
            The list of indices the method selects from using next. 
            
        max_index_number: int
            The elements in `order` should be chosen from  {0, 1, …, S-1} with S=max_index_number. 

        sampling_type:str
            The sampling type used. Choose from "sequential", "herman_meyer", and "staggered"

        prob_weights: list of floats of length max_index_number that sum to 1. 
            Consider that the sampler is called a large number of times this argument holds the expected number of times each index would be called,  normalised to 1. 

        Example
        -------
       
        
        >>> sampler=Sampler.staggered(21,4)
        >>> print(sampler.get_samples(5))
        >>> for _ in range(15):
        >>>    print(sampler.next())
        >>> print(sampler.get_samples(5))

        [ 0  4  8 12 16]
        0
        4
        8
        12
        16
        20
        1
        5
        9
        13
        17
        2
        6
        10
        14
        [ 0  4  8 12 16]
        """


        if prob_weights is None:
            prob_weights= max_index_number*[1/max_index_number]
            
        if abs(sum(prob_weights)-1) > 1e-6:
            raise ValueError('The provided prob_weights must sum to one')

        if any(np.array(prob_weights) < 0):
            raise ValueError(
                'The provided prob_weights must be greater than or equal to zero')
            
            
        self._prob_weights = prob_weights
        self._type = sampling_type
        self._max_index_number = max_index_number
        self._order = order
        self._last_index = len(order)-1

   
    @property
    def prob_weights(self):
        return self._prob_weights
    
    @property
    def max_index_number(self):
        return self._max_index_number 
    
    
    def next(self):
        """Returns and increments the sampler """

        self._last_index = (self._last_index+1) % len(self._order)
        return self._order[self._last_index]

    def __next__(self):
        return self.next()

    def get_samples(self,  num_samples=20):
        """
        Returns the first `num_samples` as a numpy array.

        num_samples: int, default=20
            The number of samples to return. 
        """
        save_last_index = self._last_index
        self._last_index = len(self._order)-1
        output = [self.next() for _ in range(num_samples)]
        self._last_index = save_last_index
        return np.array(output)

    def __str__(self):
        repres = "Sampler that outputs in order from a list of integers taken from {0, 1, …, S-1}, where S is the max index number.  \n"
        repres += "Type : {} \n".format(self._type)
        repres += "Order : {}  \n".format(self._order)
        repres += "Max index number : {} \n".format(self._max_index_number)
        repres += "Current iteration number (modulo the max index number) : {} \n".format(self._last_index)
        repres += "Probability weights : {} \n".format(self._prob_weights)
        return repres

class SamplerRandom():
    r"""
    A class to select from a list of indices {0, 1, …, S-1} using numpy.random.choice with and without replacement. 
    The function next() outputs a single next index from the list {0,1,…,S-1} .  

    It is recommended to use the static methods to configure your Sampler object rather than initialising this class directly: the user should call this through cil.optimisation.utilities.sampler and choose the desired static method from the Sampler class.  

    Parameters
    ----------
    max_index_number: int
        The sampler will select from a list of indices {0, 1, …, S-1} with S=max_index_number. 

    sampling_type:str
         The sampling type used. Choose from "random_with_replacement" and "random_without_replacement"

    replace= bool
        If True, sample with replace, otherwise sample without replacement

    prob: list of floats of length max_index_number that sum to 1. 
        For random sampling with replacement, this is the probability for each index to be called by next. 

    seed:int, default=None
        Random seed for the methods that use a numpy random number generator.  If set to None, the seed will be set using the current time.  

    prob_weights: list of floats of length max_index_number that sum to 1. 
        Consider that the sampler is called a large number of times this argument holds the expected number of times each index would be called,  normalised to 1. 
    
    Example
    -------
    >>> sampler=Sampler.random_with_replacement(5)
    >>> print(sampler.get_samples(10))
    [3 4 0 0 2 3 3 2 2 1]

    >>> sampler=Sampler.random_with_replacement(4, [0.7,0.1,0.1,0.1])
    >>> print(sampler.get_samples(10))
    [0 1 3 0 0 3 0 0 0 0]
        
    >>> sampler=Sampler.randomWithoutReplacement(7, seed=1)
    >>> print(sampler.get_samples(16))
    [6 2 1 0 4 3 5 1 0 4 2 5 6 3 3 2]
    """

    def __init__(self, max_index_number, replace,  sampling_type,  prob=None, seed=None):
        
        self._replace = replace
        self._prob = prob

        if prob is None:
            self._prob = [1/max_index_number]*max_index_number

        if replace:
            self._prob_weights = self._prob
        else:
            self._prob_weights = [1/max_index_number]*max_index_number

        if abs(sum(self._prob_weights)-1) > 1e-6:
            raise ValueError('The provided prob_weights must sum to one')

        if any(np.array(self._prob_weights) < 0):
            raise ValueError(
                'The provided prob_weights must be greater than or equal to zero')

        self._type = sampling_type
        self._max_index_number = max_index_number

        if seed is not None:
            self._seed = seed
        else:
            self._seed = int(time.time())

        self._generator = np.random.RandomState(self._seed)
        
    @property
    def prob_weights(self):
        return self._prob_weights
    
    @property
    def max_index_number(self):
        return self._max_index_number

    def next(self):
        """ Returns and increments the sampler """

        return int(self._generator.choice(self._max_index_number, 1, p=self._prob, replace=self._replace))

    def __next__(self):
        return self.next()

    def get_samples(self,  num_samples=20):
        """
        Returns the first `num_samples` as a numpy array.

        num_samples: int, default=20
            The number of samples to return. 

        """
        save_generator = self._generator
        self._generator = np.random.RandomState(self._seed)
        output = [self.next() for _ in range(num_samples)]
        self._generator = save_generator
        return np.array(output)


    def __str__(self):
            repres = "Sampler that wraps numpy.random.choice to sample from {0, 1, …, S-1}, where S is the max index number."
            repres += "Type : {} \n".format(self._type)
            repres += "max index number : {} \n".format(self._max_index_number)
            repres += "Probability weights : {} \n".format(self._prob_weights)
            return repres

class Sampler():

    r"""
    This class follows the factory design pattern. It is not instantiated but has 6 static methods that will return instances of 6 different samplers, which require a variety of parameters.
    
    Each factory method will instantiate a  class to select from the list of indices `{0, 1, …, S-1}, where S is the max index number.`.
    
    Common in each instantiated  class, the function `next()` outputs a single next index from the list {0,1,…,S-1}. Each class also has a `get_samples(n)` function which will output the first `n` samples. 

    Example
    -------
    >>> sampler=Sampler.sequential(10)
    >>> print(sampler.get_samples(5))
    >>> for _ in range(11):
            print(sampler.next())
    [0 1 2 3 4]
    0
    1
    2
    3
    4
    5
    6
    7
    8
    9
    0

    Example
    -------
    >>> sampler=Sampler.random_with_replacement(5)
    >>> for _ in range(12):
    >>>     print(next(sampler))
    >>> print(sampler.get_samples())
    3
    4
    0
    0
    2
    3
    3
    2
    2
    1
    1
    4
    [3 4 0 0 2 3 3 2 2 1 1 4 4 3 0 2 4 4 2 4]

    Note
    -----
    The optimal choice of sampler depends on the data and the number of calls to the sampler. 

    For random sampling with replacement, there is the possibility, with a small number of calls to the sampler that some indices will not have been selected. For the case of uniform probabilities, the default, the number of
    iterations required such that the probability that all indices have been selected at least once is greater than :math:`p` grows as :math:`nlog(n/p)` where `n` is `max_index_number`. 
    For example, to be 99% certain that you have seen all indices, for `n=20` you should take at least 152 samples, `n=50` at least 426 samples. To be more likely than not, for `n=20` you should take 78 samples and `n=50` you should take 228 samples. 
    In general, we note that for a large number of samples (e.g. `>20*max_index_number`), the density of the outputted samples looks more and more uniform. For a small number of samples (e.g. `<5*max_index_number`) the user may wish to consider
    another sampling method e.g. random without replacement, which, when calling `max_index_number` samples is guaranteed to draw each index exactly once.  

    """

    @staticmethod
    def sequential(max_index_number):
        """
        Instantiates a sampler that outputs sequentially. 

        Parameters
        ----------
        max_index_number: int
            The sampler will select from a list of indices {0, 1, …, S-1} with S=max_index_number. 

        Example
        -------

        >>> sampler=Sampler.sequential(10)
        >>> print(sampler.get_samples(5))
        >>> for _ in range(11):
                print(sampler.next())
        [0 1 2 3 4]
        0
        1
        2
        3
        4
        5
        6
        7
        8
        9
        0
        """
        order = list(range(max_index_number))
        sampler = SamplerFromOrder(max_index_number=max_index_number, sampling_type='sequential', order=order, prob_weights=[
                                   1/max_index_number]*max_index_number)
        return sampler

   

    @staticmethod
    def herman_meyer(max_index_number):
        """
        Instantiates a sampler which outputs in a Herman Meyer order.
        
        Parameters
        ----------
        max_index_number: int
            The sampler will select from a list of indices {0, 1, …, S-1} with S=max_index_number. For Herman-Meyer sampling this number should not be prime. 

        Reference
        ----------
        Herman GT, Meyer LB. Algebraic reconstruction techniques can be made computationally efficient. IEEE Trans Med Imaging.  doi: 10.1109/42.241889.

        Example
        -------
        >>> sampler=Sampler.herman_meyer(12)
        >>> print(sampler.get_samples(16))
        [ 0  6  3  9  1  7  4 10  2  8  5 11  0  6  3  9]

        """
        def _herman_meyer_order(n):
            # Assuming that the indices are in geometrical order
            n_variable = n
            i = 2
            factors = []

            while i * i <= n_variable:
                if n_variable % i:
                    i += 1
                else:
                    n_variable //= i
                    factors.append(i)

            if n_variable > 1:
                factors.append(n_variable)

            n_factors = len(factors)

            if n_factors == 0:
                raise ValueError(
                    'Herman Meyer sampling defaults to sequential ordering if the max index number is prime. Please use an alternative sampling method or change the max index number. ')

            order = [0 for _ in range(n)]
            value = 0

            for factor_n in range(n_factors):
                n_rep_value = 0

                if factor_n == 0:
                    n_change_value = 1
                else:
                    n_change_value = math.prod(factors[:factor_n])

                for element in range(n):
                    mapping = value
                    n_rep_value += 1

                    if n_rep_value >= n_change_value:
                        value = value + 1
                        n_rep_value = 0

                    if value == factors[factor_n]:
                        value = 0

                    order[element] = order[element] + \
                        math.prod(factors[factor_n+1:]) * mapping
            return order

        order = _herman_meyer_order(max_index_number)
        sampler = SamplerFromOrder(
            max_index_number=max_index_number, sampling_type='herman_meyer', order=order, prob_weights=[1/max_index_number]*max_index_number)
        return sampler

    @staticmethod
    def staggered(max_index_number, offset):
        """
        Instantiates a sampler which outputs in a staggered order. 
        
        Parameters
        ----------
        max_index_number: int
            The sampler will select from a list of indices {0, 1, …, S-1} with S=max_index_number. 

        offset: int
            The sampler will output in the order {0, a, 2a, 3a, ...., 1, 1+a, 1+2a, 1+3a,...., 2, 2+a, 2+2a, 2+3a,...} where a=offset. 
            The offset should be less than the max_index_number

        Example
        -------
        >>> sampler=Sampler.staggered(21,4)
        >>> print(sampler.get_samples(5))
        >>> for _ in range(15):
        >>>    print(sampler.next())
        >>> print(sampler.get_samples(5))
        [ 0  4  8 12 16]
        0
        4
        8
        12
        16
        20
        1
        5
        9
        13
        17
        2
        6
        10
        14
        [ 0  4  8 12 16]
        """

        if offset >= max_index_number:
            raise (ValueError('The offset should be less than the max index number'))

        indices = list(range(max_index_number))
        order = []
        [order.extend(indices[i::offset]) for i in range(offset)]
        sampler = SamplerFromOrder(max_index_number=max_index_number, sampling_type='staggered', order=order, prob_weights=[
                                   1/max_index_number]*max_index_number)
        return sampler

    @staticmethod
    def random_with_replacement(max_index_number, prob=None, seed=None):
        """
        Instantiates a sampler which outputs from a list of indices {0, 1, …, S-1}, with S=max_index_number, with given probability and with replacement. 

        Parameters
        ----------
        max_index_number: int
            The sampler will select from a list of indices {0, 1, …, S-1} with S=max_index_number. 

        prob: list of floats of length max_index_number that sum to 1. default=None
            This is the probability for each index to be called by next. If None, then the indices will be sampled uniformly. 

        seed:int, default=None
            Random seed for the random number generator.  If set to None, the seed will be set using the current time.


        Example
        -------
        >>> sampler=Sampler.random_with_replacement(5)
        >>> print(sampler.get_samples(10))
        [3 4 0 0 2 3 3 2 2 1]

        
        >>> sampler=Sampler.random_with_replacement(4, [0.7,0.1,0.1,0.1])
        >>> print(sampler.get_samples(10))
        [0 1 3 0 0 3 0 0 0 0]
        """

        if prob == None:
            prob = [1/max_index_number] * max_index_number

        sampler = SamplerRandom(
            max_index_number=max_index_number, sampling_type='random_with_replacement', replace=True, prob=prob, seed=seed)
        return sampler

    @staticmethod
    def random_without_replacement(max_index_number, seed=None, prob=None):
        """
        Instantiates a sampler which outputs from a list of indices {0, 1, …, S-1}, with S=max_index_number, uniformly randomly without replacement.

        Parameters
        ----------
        max_index_number: int
            The sampler will select from a list of indices {0, 1, …, S-1} with S=max_index_number. 

        seed:int, default=None
            Random seed for the  random number generator.  If set to None, the seed will be set using the current time. 

        Example
        -------
        >>> sampler=Sampler.randomWithoutReplacement(7, seed=1)
        >>> print(sampler.get_samples(16))
        [6 2 1 0 4 3 5 1 0 4 2 5 6 3 3 2]

        """

        sampler = SamplerRandom(
            max_index_number=max_index_number, sampling_type='random_without_replacement', replace=False,  seed=seed, prob=prob)
        return sampler

    @staticmethod
    def from_function(max_index_number, function, prob_weights=None):
        """
        Instantiates a sampler that wraps a function that takes an iteration number and selects from a list of indices {0, 1, …, S-1}.

        Parameters
        ----------
        max_index_number: int
            The sampler will select from a list of indices {0, 1, …, S-1} with S=max_index_number. 

        function: A function that takes an in integer iteration number and returns an integer from {0, 1, …, S-1} with S=max_index_number. 

        prob_weights: list of floats of length max_index_number that sum to 1. Default is [1/max_index_number]*max_index_number 
            Consider that the sampler is called a large number of times this argument holds the expected number of times each index would be called,  normalised to 1. 


        Note
        -----
        If your function involves a random number generator, then the seed should also depend on the iteration number, see the example in the documentation, otherwise
        the `get_samples()` function may not accurately return the correct samples and may interrupt the next sample returned. 

        Example
        -------
        >>> def test_function(iteration_number):
        >>>     if iteration_number<500:
        >>>         np.random.seed(iteration_number)
        >>>         return(np.random.choice(49,1)[0])
        >>>     else:
        >>>         np.random.seed(iteration_number)
        >>>         return(np.random.choice(50,1)[0])


        >>> sampler=Sampler.from_function(50, test_function)
        >>> for _ in range(11):
        >>>     print(sampler.next())
        >>> print(list(sampler.get_samples(25)))
        44
        37
        40
        42
        46
        35
        10
        47
        3
        28
        9
        [44, 37, 40, 42, 46, 35, 10, 47, 3, 28, 9, 25, 11, 18, 43, 8, 41, 47, 42, 29, 35, 9, 4, 19, 34]
        
        Example
        -------
        >>> custom_list=[1,1,1,0,0,11,5,9,8,3]
        >>> max_index_number=13
        >>> 
        >>> def test_function(iteration_number, custom_list=custom_list):
            return(custom_list[iteration_number%len(custom_list)])
        >>> 
        >>> #calculate prob weights 
        >>> temp_list = []
        >>> for i in range(max_index_number):
        >>>     temp_list.append(custom_list.count(i))
        >>> total = sum(temp_list)
        >>> prob_weights = [x/total for x in temp_list]
        >>> 
        >>> sampler=Sampler.from_function(max_index_number=max_index_number, function=test_function, prob_weights=prob_weights)
        >>> for _ in range(11):
        >>>     print(sampler.next())
        >>> print(list(sampler.get_samples(25)))
        >>> print(sampler)
        1
        1
        1
        0
        0
        11
        5
        9
        8
        3
        1
        [1, 1, 1, 0, 0, 11, 5, 9, 8, 3, 1, 1, 1, 0, 0, 11, 5, 9, 8, 3, 1, 1, 1, 0, 0]
        Sampler that wraps a function that takes an iteration number and selects from a list of indices {0, 1, …, S-1}, where S is the max index number. 
        Type : from_function 
        Current iteration number : 11 
        Max index number : 13 
        Probability weights : [0.2, 0.3, 0.0, 0.1, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 0.0] 
        
        
        Note
        -----
        If your function involves a random number generator, then the seed should also depend on the iteration number, see the first example in the documentation, otherwise
        the `get_samples()` function may not accurately return the correct samples and may interrupt the next sample returned. 
        """
        
        if prob_weights is None:
            prob_weights = [1/max_index_number]*max_index_number

        sampler = SamplerFromFunction(
            max_index_number=max_index_number, sampling_type='from_function', function=function, prob_weights=prob_weights)
        return sampler
