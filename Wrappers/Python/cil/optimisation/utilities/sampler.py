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
from functools import partial
import time
import numbers

class Sampler():
    # TODO: Work out how to make the examples testable
    """
    Initialises a sampler that returns and then increments indices from a sequence defined by a function.

    Static methods to easily configure several samplers are provided, such as sequential, staggered, Herman-Mayer, random with and without replacement.

    Custom deterministic samplers can be created by using the `from_function` static method or by subclassing this sampler class.

    Parameters
    ----------

    function : Callable[[int], int]
        A function that takes an integer iteration number and returns an integer between 0 and num_indices.

    num_indices: int
        The sampler will select from a range of indices 0 to num_indices.

    sampling_type:str, optional,  default = None
        The sampling type used. This is recorded for reference and printed when `print` is called.

    prob_weights: list of floats of length num_indices that sum to 1.  Default is [1 / num_indices] * num_indices
        Consider that the sampler is incremented a large number of times this argument holds the expected number of times each index would be outputted,  normalised to 1.

    Returns
    -------
    Sampler
        An instance of the Sampler class representing the desired configuration.

    Example
    -------
    >>> sampler = Sampler.random_with_replacement(5)
    >>> print(sampler.get_samples(20))
    [3 4 0 0 2 3 3 2 2 1 1 4 4 3 0 2 4 4 2 4]
    >>> print(next(sampler))
    3
    >>> print(sampler.next())
    4


    >>> sampler = Sampler.staggered(num_indices=21, stride=4)
    >>> print(next(sampler))
    0
    >>> print(sampler.next())
    4
    >>> print(sampler.get_samples(5))
    [ 0  4  8 12 16]

    Example
    -------
    >>> sampler = Sampler.sequential(10)
    >>> print(sampler.get_samples(5))
    >>> print(next(sampler))
    0
    [0 1 2 3 4]
    >>> print(sampler.next())
    1

    Example
    -------
    >>> sampler = Sampler.herman_meyer(12)
    >>> print(sampler.get_samples(16))
    [ 0  6  3  9  1  7  4 10  2  8  5 11  0  6  3  9]



    Example
    --------
    This example creates a sampler that outputs sequential indices, starting from 1.

    >>> num_indices=10
    >>>
    >>> def my_sampling_function(iteration_number):
    >>>     return (iteration_number+1)%10
    >>>
    >>> sampler = Sampler.from_function(num_indices=num_indices, function=my_sampling_function)
    >>> print(list(sampler.get_samples(25)))
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5]



    Note
    -----
    The optimal choice of sampler depends on the data and the number of calls to the sampler.  Note that a low number of calls to a random sampler won't give an even distribution.
    For a small number of samples (e.g. `<5*num_indices`) the user may wish to consider another sampling method e.g. random without replacement, which, when calling `num_indices` samples is guaranteed to draw each index exactly once.
        """

    def __init__(self, num_indices, function,  sampling_type=None, prob_weights=None):

        self._type = sampling_type

        if isinstance (num_indices, numbers.Integral):
            self._num_indices = num_indices
        else:
            raise ValueError('`num_indices` should be an integer. ')

        if callable(function):
            self._function = function
        else:
            raise ValueError('`function` should be an callable function. ')

        if prob_weights is None:
            prob_weights = [1 / num_indices] * num_indices
        else:
            if abs(sum(prob_weights) - 1) > 1e-6:
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
        Returns a sample from the list of indices `{0, 1, …, N-1}, where N is the number of indices and increments the sampler.
        """

        out = self._function(self._iteration_number)

        self._iteration_number += 1
        return out

    def __next__(self):
        return self.next()

    def get_samples(self,  num_samples):
        """
        Generates a list of the first num_samples output by the sampler. Calling this does not increment the sampler index or affect the behaviour of the sampler .

        Parameters
        ----------
        num_samples: int
            The number of samples to return.

        Returns
        --------
        List
            The first `num_samples" output by the sampler.
        """
        save_last_index = self._iteration_number
        self._iteration_number = 0

        output = [self.next() for _ in range(num_samples)]

        self._iteration_number = save_last_index

        return np.array(output)
    
    
    def get_previous_samples(self):
        """
        Generates a list of the samples outputted by the sampler since it was initialised. Calling this does not increment the sampler index or affect the behaviour of the sampler .

        Returns
        --------
        List
            A list of the samples outputted by the sampler since it was initialised
        """

        return get_samples(self._iteration_number)
    
    def get_current_sample(self):
        """
        Returns the current sample of the sampler without incrementing the sampler index.

        Returns
        --------
        int
            The current sample of the sampler.
        """
        return self._function(self._iteration_number)

    def __str__(self):
        repres = "Sampler that selects from a list of indices {0, 1, …, N-1}, where N is the number of indices. \n"
        repres += "Type : {} \n".format(self._type)
        repres += "Current iteration number : {} \n".format(
            self._iteration_number)
        repres += "Number of indices : {} \n".format(self._num_indices)
        repres += "Probability weights : {} \n".format(self._prob_weights)
        return repres

    @staticmethod
    def sequential(num_indices):
        """
        Instantiates a sampler that outputs sequential indices.

        Parameters
        ----------
        num_indices: int
            The sampler will select from a range of indices 0 to num_indices.

        Returns
        -------
        Sampler
            An instance of the Sampler class that will generate indices sequentially.

        Example
        -------
        >>> sampler = Sampler.sequential(10)
        >>> print(sampler.get_samples(5))
        >>> print(next(sampler))
        0
        [0 1 2 3 4]
        >>> print(sampler.next())
        1
        """
        def function(x):
            return x % num_indices

        sampler = Sampler(function=function, num_indices=num_indices,
                          sampling_type='sequential'
                          )
        return sampler

    @staticmethod
    def _staggered_function(num_indices, stride, iter_number):
        """Function that takes in an iteration number and outputs an index number based on the staggered ordering.

        Parameters
        ----------
        num_indices: int
            The sampler will select from a range of indices 0 to num_indices.

        stride: int
            The stride between returned indices. The stride should be less than the num_indices.

        iter_number: int
            The current iteration number of the sampler.

        Returns
        -------
        int
            The index to be outputted by the sampler corresponding to the `iter_number`

        """
        if not isinstance (num_indices, numbers.Integral):
            raise ValueError('`num_indices` should be an integer. ')

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
            The sampler will select from a range of indices 0 to num_indices.

        stride: int
            The stride between returned indices. The stride should be less than the num_indices.

        Returns
        -------
        Sampler
            An instance of the Sampler class that will generate indices in a staggered pattern.


        Example
        -------
        >>> sampler = Sampler.staggered(num_indices=21, stride=4)
        >>> print(next(sampler))
        0
        >>> print(sampler.next())
        4
        >>> print(sampler.get_samples(5))
        [ 0  4  8 12 16]
        Example
        -------
        >>> sampler = Sampler.staggered(num_indices=17, stride=8)
        >>> print(next(sampler))
        0
        >>> print(sampler.next())
        8
        >>> print(sampler.get_samples(10))
        [ 0  8  16 1 9 2 10 3 11 4]


        """

        if stride >= num_indices:
            raise (ValueError('The stride should be less than the number of indices'))

        sampler = Sampler(function=partial(Sampler._staggered_function, num_indices, stride),
                          num_indices=num_indices, sampling_type='staggered'
                          )

        return sampler

    @staticmethod
    def random_with_replacement(num_indices, prob=None, seed=None):
        """
        Instantiates a sampler which outputs an index between 0 - num_indices with a given probability.

        Parameters
        ----------
        num_indices: int
            The sampler will select from a range of indices 0 to num_indices

        prob: list of floats, optional
            The probability for each index to be selected by the 'next' operation. If not provided, the indices will be sampled uniformly. The list should have a length equal to num_indices, and the values should sum to 1

        seed:int, optional
            Used to initialise the random number generator where repeatability is required.

        Returns
        -------
        `RandomSampler`
            An instance of the `RandomSampler` class that will generate indices randomly with replacement

        Example
        -------
        >>> sampler = Sampler.random_with_replacement(5)
        >>> print(sampler.get_samples(10))
        [3 4 0 0 2 3 3 2 2 1]
        >>> print(next(sampler))
        3
        >>> print(sampler.next())
        4

        >>> sampler = Sampler.random_with_replacement(num_indices=4, prob=[0.7,0.1,0.1,0.1])
        >>> print(sampler.get_samples(10))
        [0 1 3 0 0 3 0 0 0 0]
        """

        sampler = SamplerRandom(
            num_indices=num_indices,
            sampling_type='random_with_replacement',
            prob=prob,
            replace=True,
            seed=seed
        )
        return sampler

    @staticmethod
    def random_without_replacement(num_indices, seed=None):
        """
        Instantiates a sampler which outputs an index between 0 - num_indices. Once sampled the index will not be sampled again until all indices have been returned.

        Parameters
        ----------
        num_indices: int
            The sampler will select from a range of indices 0 to num_indices.

        seed: int, optional
            Used to initialise the random number generator where repeatability is required.

        Returns
        -------
        `RandomSampler`
            An instance of the `RandomSampler` class that will generate indices randomly without replacement

        Example
        -------
        >>> sampler=Sampler.randomWithoutReplacement(num_indices=7, seed=1)
        >>> print(sampler.get_samples(16))
        [6 2 1 0 4 3 5 1 0 4 2 5 6 3 3 2]

        """

        sampler = SamplerRandom(
            num_indices=num_indices,
            sampling_type='random_without_replacement',
            replace=False,
            seed=seed
        )
        return sampler

    @staticmethod
    def from_function(num_indices, function, prob_weights=None):
        """
        Instantiate a sampler that wraps a function for index selection.

        Parameters
        ----------
        num_indices: int
            The sampler will select from a range of indices 0 to num_indices.

    function : callable
        A deterministic function that takes an integer as an argument, representing the iteration number, and returns an integer between 0 and num_indices. The function signature should be function(iteration_number: int) -> int

        prob_weights: list of floats of length num_indices that sum to 1. Default is [1 / num_indices] * num_indices
            Consider that the sampler is incremented a large number of times this argument holds the expected number of times each index would be outputted,  normalised to 1.

        Returns
        -------
        Sampler
            An instance of the Sampler class which samples from a function.


        Example
        --------
        This example creates a sampler that always outputs 2.  The probability weights are passed to the sampler as they are not uniform.

        >>> num_indices=3
        >>>
        >>> def my_sampling_function(iteration_number):
        >>>     return 2
        >>>
        >>> sampler = Sampler.from_function(num_indices=num_indices, function=my_sampling_function, prob_weights=[0, 0, 1])
        >>> print(list(sampler.get_samples(12)))
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]


        Example
        --------
        This example creates a sampler that outputs sequential indices, starting from 1.  The probability weights are not passed to the sampler as they are uniform.

        >>> num_indices=10
        >>>
        >>> def my_sampling_function(iteration_number):
        >>>     return (iteration_number+1)%10
        >>>
        >>> sampler = Sampler.from_function(num_indices=num_indices, function=my_sampling_function)
        >>> print(list(sampler.get_samples(25)))
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5]


        Example
        -------
        This example creates a sampler that samples in order from a custom list. The num_indices  is 6, although note that the index 5 is never output by the sampler. The number of indices must be at least one greater than any of the elements in the custom_list.
        The probability weights are passed to the sampler as they are not uniform.

        >>> custom_list = [0,0,0,0,0,0,3,2,1,4]
        >>> num_indices = 6
        >>>
        >>> def my_sampling_function(iteration_number, custom_list=custom_list]):
        >>>    return(custom_list[iteration_number%len(custom_list)])
        >>>
        >>> sampler = Sampler.from_function(num_indices=num_indices, function=my_sampling_function, prob_weights=[0.6, 0.1, 0.1, 0.1, 0.1, 0.0])
        >>> print(list(sampler.get_samples(25)))
        [0, 0, 0, 0, 0, 0, 3, 2, 1, 4, 0, 0, 0, 0, 0, 0, 3, 2, 1, 4, 0, 0, 0, 0, 0]
        >>> print(sampler)
        Sampler that wraps a function that takes an iteration number and selects from a list of indices {0, 1, …, N-1}, where N is the number of indices.
        Type : from_function
        Current iteration number : 0
        number of indices : 6
        Probability weights : [0.6, 0.1, 0.1, 0.1, 0.1, 0.0]

        """

        sampler = Sampler(
            num_indices=num_indices,
            sampling_type='from_function',
            function=function,
            prob_weights=prob_weights
        )
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
        list of ints
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
        int
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
        r"""Instantiates a sampler which outputs in a Herman Meyer order.

        Parameters
        ----------
        num_indices: int
            The sampler will select from a range of indices 0 to num_indices. For Herman-Meyer sampling this number should not be prime.
        
        Returns
        -------
        Sampler
            An instance of the Sampler class which outputs in a Herman Meyer order.
        
        
        
            
        Reference
        ----------
        With thanks to Imraj Singh and Zeljko Kereta for their help with the initial implementation of the Herman Meyer sampling. Their implementation was used in:

        Singh I, et al. Deep Image Prior PET Reconstruction using a SIRF-Based Objective - IEEE MIC, NSS & RTSD 2022. https://discovery.ucl.ac.uk/id/eprint/10176077/1/MIC_Conference_Record.pdf

        The sampling method was introduced in:

        Herman GT, Meyer LB. Algebraic reconstruction techniques can be made computationally efficient. IEEE Trans Med Imaging.  doi: 10.1109/42.241889.

        Example
        -------
        >>> sampler=Sampler.herman_meyer(12)
        >>> print(sampler.get_samples(16))
        [ 0  6  3  9  1  7  4 10  2  8  5 11  0  6  3  9]
        """

        if not isinstance (num_indices, numbers.Integral):
            raise ValueError('`num_indices` should be an integer. ')

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
                          num_indices=num_indices,
                          sampling_type='herman_meyer',
                          prob_weights=[1 / num_indices] * num_indices
                          )

        return sampler


class SamplerRandom(Sampler):
    """
    The user is recommended to not instantiate this class  directly but instead use one of the static methods  in the parent Sampler class that will return instances of different samplers.

    This class produces Samplers that output random samples with and without replacement from the set {0, 1, …, N-1} where N=num_indices.

    Custom random samplers can be created by subclassing this sampler class.

    Parameters
    ----------

    num_indices: int
        The sampler will select from a range of indices 0 to num_indices.

    sampling_type:str, optional,  default = 'random_with_replacement"
        The sampling type used. This is recorded for reference and printed when `print` is called.

    prob_weights: list of floats of length num_indices that sum to 1.  Default is [1 / num_indices] * num_indices
        Consider that the sampler is incremented a large number of times this argument holds the expected number of times each index would be outputted,  normalised to 1.

    replace: bool, default is True
        If True, sample with replace, otherwise sample without replacement

    seed:int, optional
        Used to initialise the random number generator where repeatability is required.

    Returns
    -------
    Sampler
        An instance of the Sampler class representing the desired configuration.

    Example
    -------
    >>> sampler = Sampler.random_with_replacement(5)
    >>> print(sampler.get_samples(20))
    [3 4 0 0 2 3 3 2 2 1 1 4 4 3 0 2 4 4 2 4]

    Example
    -------
    >>> sampler=Sampler.randomWithoutReplacement(num_indices=7, seed=1)
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

        super(SamplerRandom, self).__init__(num_indices, self._function,
                                            sampling_type=sampling_type, prob_weights=prob)

    @property
    def seed(self):
        return self._seed

    @property
    def replace(self):
        return self._replace

    def _function(self, iteration_number):
        """ For each iteration number this function samples from a randomly generated list in order. Every num_indices the list is re-created. """
        location = iteration_number % self._num_indices
        if location == 0:
            self._sampling_list = self._generator.choice(
                self._num_indices, self._num_indices, p=self._prob_weights, replace=self._replace)
        self._current_sample = self._sampling_list[location]
        return self._current_sample

    def get_samples(self,  num_samples):
        """
        Generates a list of the first num_samples output by the sampler. Calling this does not increment the sampler index or affect the behaviour of the sampler .

        Parameters
        ----------
        num_samples: int
            The number of samples to return.
        Returns
        -------
        list
            The first `num_samples` produced by the sampler
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
    
    def get_current_sample(self):
        """
        Returns the current sample of the sampler without incrementing the sampler index.

        Returns
        --------
        int
            The current sample of the sampler.
        """

        return self._current_sample

    def __str__(self):
        repres = super().__str__()
        repres += "Seed : {} \n".format(self._seed)
        return repres
