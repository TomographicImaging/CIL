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

from cil.utilities.dataexample import TestData

def gaussian(image, seed=None, clip=True, **kwargs):
    '''Gaussian-distributed additive noise.

        seed : int, optional
            If provided, this will set the random seed before generating noise,
            for valid pseudo-random comparisons.
        clip : bool, optional
            If True (default), the output will be clipped after noise applied.
            This is needed to maintain the proper image data range. If False, clipping
            is not applied, and the output may extend beyond the range [-1, 1].
        mean : float, optional
            Mean of random distribution.
            Default : 0.
        var : float, optional
            Variance of random distribution.
            Note: variance = (standard deviation) ** 2. Default : 0.01

        '''
    return TestData.random_noise(image, mode='gaussian', seed=seed, clip=clip, **kwargs)

def poisson(image, seed=None, clip=True, **kwargs):
    '''Poisson-distributed noise generated from the data.

        seed : int, optional
            If provided, this will set the random seed before generating noise,
            for valid pseudo-random comparisons.
        clip : bool, optional
            If True (default), the output will be clipped after noise applied.
            This is needed to maintain the proper image data range. If False, clipping
            is not applied, and the output may extend beyond the range [-1, 1].

        '''
    return TestData.random_noise(image, mode='poisson', seed=seed, clip=clip, **kwargs)

def salt(image, seed=None, **kwargs):
    '''Replaces random pixels with 1.

        seed : int, optional
            If provided, this will set the random seed before generating noise,
            for valid pseudo-random comparisons.
        amount : float, optional
            Proportion of image pixels to replace with noise on range [0, 1].
            Used in 'salt', 'pepper', and 'salt & pepper'. Default : 0.05
        '''
    return TestData.random_noise(image, mode='salt', seed=seed, clip=True, **kwargs)

def pepper(image, seed=None, **kwargs):
    '''Replaces random pixels with 0 (for unsigned images) or -1 (for signed images).

        seed : int, optional
            If provided, this will set the random seed before generating noise,
            for valid pseudo-random comparisons.
        amount : float, optional
            Proportion of image pixels to replace with noise on range [0, 1].
            Used in 'salt', 'pepper', and 'salt & pepper'. Default : 0.05
        '''
    return TestData.random_noise(image, mode='pepper', seed=seed, clip=True, **kwargs)

def saltnpepper(image, seed=None, **kwargs):
    '''Replaces random pixels with either 1 or `low_val`

    `low_val` is 0 for unsigned images or -1 for signed images.

        seed : int, optional
            If provided, this will set the random seed before generating noise,
            for valid pseudo-random comparisons.
        amount : float, optional
            Proportion of image pixels to replace with noise on range [0, 1].
            Used in 'salt', 'pepper', and 'salt & pepper'. Default : 0.05
        salt_vs_pepper : float, optional
            Proportion of salt vs. pepper noise for 's&p' on range [0, 1].
            Higher values represent more salt. Default : 0.5 (equal amounts)'''
    return TestData.random_noise(image, mode='s&p', seed=seed, clip=True, **kwargs)

def speckle(image, seed=None, clip=True, **kwargs):
    '''Multiplicative noise

    using out = image + n*image, where
                        n is uniform noise with specified mean & variance.

    seed : int, optional
            If provided, this will set the random seed before generating noise,
            for valid pseudo-random comparisons.
        clip : bool, optional
            If True (default), the output will be clipped after noise applied.
            This is needed to maintain the proper image data range. If False, clipping
            is not applied, and the output may extend beyond the range [-1, 1].
        mean : float, optional
            Mean of random distribution.
            Default : 0.
        var : float, optional
            Variance of random distribution.
            Note: variance = (standard deviation) ** 2. Default : 0.01
        '''
    return TestData.random_noise(image, mode='speckle', seed=seed, clip=clip, **kwargs)

def localvar(image, seed=None, clip=True, **kwargs):
    '''Gaussian-distributed additive noise, with specified
                        local variance at each point of `image`.

        seed : int, optional
            If provided, this will set the random seed before generating noise,
            for valid pseudo-random comparisons.
        clip : bool, optional
            If True (default), the output will be clipped after noise applied. This is
            needed to maintain the proper image data range. If False, clipping
            is not applied, and the output may extend beyond the range [-1, 1].
        mean : float, optional
            Mean of random distribution. Used in 'gaussian' and 'speckle'.
            Default : 0.
        var : float, optional
            Note: variance = (standard deviation) ** 2. Default : 0.01
        local_vars : ndarray, optional
            Array of positive floats, same shape as `image`, defining the local
            variance at every image point. Used in 'localvar'.
        '''
    return TestData.random_noise(image, mode='localvar', seed=seed, clip=clip, **kwargs)
