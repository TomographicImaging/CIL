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

from cil.framework import ImageData, ImageGeometry, DataContainer
import numpy
import numpy as np
from PIL import Image
import os
import os.path
import sys
from cil.io import NEXUSDataReader

data_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__),
        '../data/')
)

# this is the default location after a conda install
data_dir = os.path.abspath(
    os.path.join(sys.prefix, 'share','cil')
)

class DATA(object):
    @classmethod
    def dfile(cls):
        return None
    @classmethod
    def get(cls, size=None, scale=(0,1), **kwargs):
        ddir = kwargs.get('data_dir', data_dir)
        loader = TestData(data_dir=ddir)
        return loader.load(cls.dfile(), size, scale, **kwargs)

class BOAT(DATA):
    @classmethod
    def dfile(cls):
        return TestData.BOAT
class CAMERA(DATA):
    @classmethod
    def dfile(cls):
        return TestData.CAMERA
class PEPPERS(DATA):
    @classmethod
    def dfile(cls):
        return TestData.PEPPERS
class RESOLUTION_CHART(DATA):
    @classmethod
    def dfile(cls):
        return TestData.RESOLUTION_CHART
class SIMPLE_PHANTOM_2D(DATA):
    @classmethod
    def dfile(cls):
        return TestData.SIMPLE_PHANTOM_2D
class SHAPES(DATA):
    @classmethod
    def dfile(cls):
        return TestData.SHAPES
class RAINBOW(DATA):
    @classmethod
    def dfile(cls):
        return TestData.RAINBOW
class SYNCHROTRON_PARALLEL_BEAM_DATA(DATA):
    @classmethod
    def get(cls, **kwargs):
        '''
        A DLS dataset

        Parameters
        ----------
        data_dir: str, optional
           The path to the data directory

        Returns
        -------
        AcquisitionData
            The DLS dataset
        '''

        ddir = kwargs.get('data_dir', data_dir)
        loader = NEXUSDataReader()
        loader.set_up(file_name=os.path.join(os.path.abspath(ddir), '24737_fd_normalised.nxs'))
        return loader.read()
class SIMULATED_PARALLEL_BEAM_DATA(DATA):
    @classmethod
    def get(cls, **kwargs):
        '''
        A simulated parallel-beam dataset generated from SIMULATED_SPHERE_VOLUME

        Parameters
        ----------
        data_dir: str, optional
           The path to the data directory

        Returns
        -------
        AcquisitionData
            The simulated spheres dataset
        '''

        ddir = kwargs.get('data_dir', data_dir)
        loader = NEXUSDataReader()
        loader.set_up(file_name=os.path.join(os.path.abspath(ddir), 'sim_parallel_beam.nxs'))
        return loader.read()
class SIMULATED_CONE_BEAM_DATA(DATA):
    @classmethod
    def get(cls, **kwargs):
        '''
        A cone-beam dataset generated from SIMULATED_SPHERE_VOLUME

        Parameters
        ----------
        data_dir: str, optional
           The path to the data directory

        Returns
        -------
        AcquisitionData
            The simulated spheres dataset
        '''

        ddir = kwargs.get('data_dir', data_dir)
        loader = NEXUSDataReader()
        loader.set_up(file_name=os.path.join(os.path.abspath(ddir), 'sim_cone_beam.nxs'))
        return loader.read()
class SIMULATED_SPHERE_VOLUME(DATA):
    @classmethod
    def get(cls, **kwargs):
        '''
        A simulated volume of spheres

        Parameters
        ----------
        data_dir: str, optional
           The path to the data directory

        Returns
        -------
        ImageData
            The simulated spheres volume
        '''

        ddir = kwargs.get('data_dir', data_dir)
        loader = NEXUSDataReader()
        loader.set_up(file_name=os.path.join(os.path.abspath(ddir), 'sim_volume.nxs'))
        return loader.read()

class TestData(object):
    '''Class to return test data

    provides 6 dataset:
    BOAT = 'boat.tiff'
    CAMERA = 'camera.png'
    PEPPERS = 'peppers.tiff'
    RESOLUTION_CHART = 'resolution_chart.tiff'
    SIMPLE_PHANTOM_2D = 'hotdog'
    SHAPES =  'shapes.png'
    RAINBOW = 'rainbow.png'
    '''
    BOAT = 'boat.tiff'
    CAMERA = 'camera.png'
    PEPPERS = 'peppers.tiff'
    RESOLUTION_CHART = 'resolution_chart.tiff'
    SIMPLE_PHANTOM_2D = 'hotdog'
    SHAPES =  'shapes.png'
    RAINBOW =  'rainbow.png'

    def __init__(self, **kwargs):
        self.data_dir = kwargs.get('data_dir', data_dir)

    def load(self, which, size=None, scale=(0,1), **kwargs):
        '''
        Return a test data of the requested image

        Parameters
        ----------
        which: str
           Image selector: BOAT, CAMERA, PEPPERS, RESOLUTION_CHART, SIMPLE_PHANTOM_2D, SHAPES, RAINBOW
        size: tuple, optional
            The size of the returned ImageData. If None default will be used for each image type
        scale: tuple, optional
            The scale of the data values

        Returns
        -------
        ImageData
            The simulated spheres volume
        '''
        if which not in [TestData.BOAT, TestData.CAMERA,
                         TestData.PEPPERS, TestData.RESOLUTION_CHART,
                         TestData.SIMPLE_PHANTOM_2D, TestData.SHAPES,
                         TestData.RAINBOW]:
            raise ValueError('Unknown TestData {}.'.format(which))
        if which == TestData.SIMPLE_PHANTOM_2D:
            if size is None:
                N = 512
                M = 512
            else:
                N = size[0]
                M = size[1]

            sdata = numpy.zeros((N, M))
            sdata[int(round(N/4)):int(round(3*N/4)), int(round(M/4)):int(round(3*M/4))] = 0.5
            sdata[int(round(N/8)):int(round(7*N/8)), int(round(3*M/8)):int(round(5*M/8))] = 1
            ig = ImageGeometry(voxel_num_x = M, voxel_num_y = N, dimension_labels=[ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X])
            data = ig.allocate()
            data.fill(sdata)

        elif which == TestData.SHAPES:

            with Image.open(os.path.join(self.data_dir, which)) as f:

                if size is None:
                    N = 200
                    M = 300
                else:
                    N = size[0]
                    M = size[1]

                ig = ImageGeometry(voxel_num_x = M, voxel_num_y = N, dimension_labels=[ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X])
                data = ig.allocate()
                tmp = numpy.array(f.convert('L').resize((M,N)))
                data.fill(tmp/numpy.max(tmp))

        else:
            with Image.open(os.path.join(self.data_dir, which)) as tmp:

                if size is None:
                    N = tmp.size[1]
                    M = tmp.size[0]
                else:
                    N = size[0]
                    M = size[1]

                bands = tmp.getbands()
                if len(bands) > 1:
                    if len(bands) == 4:
                        tmp = tmp.convert('RGB')
                        bands = tmp.getbands()

                    ig = ImageGeometry(voxel_num_x=M, voxel_num_y=N, channels=len(bands),
                    dimension_labels=[ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X,ImageGeometry.CHANNEL])
                    data = ig.allocate()
                    data.fill(numpy.array(tmp.resize((M,N))))
                    data.reorder([ImageGeometry.CHANNEL,ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X])
                    data.geometry.channel_labels = bands
                else:
                    ig = ImageGeometry(voxel_num_x = M, voxel_num_y = N, dimension_labels=[ImageGeometry.HORIZONTAL_Y, ImageGeometry.HORIZONTAL_X])
                    data = ig.allocate()
                    data.fill(numpy.array(tmp.resize((M,N))))


            if scale is not None:
                dmax = data.as_array().max()
                dmin = data.as_array().min()
                # scale 0,1
                data = (data -dmin) / (dmax - dmin)
                if scale != (0,1):
                    #data = (data-dmin)/(dmax-dmin) * (scale[1]-scale[0]) +scale[0])
                    data *= (scale[1]-scale[0])
                    data += scale[0]
        # print ("data.geometry", data.geometry)
        return data

    @staticmethod
    def random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs):
        '''Function to add noise to input image

        :param image: input dataset, DataContainer of numpy.ndarray
        :param mode: type of noise
        :param seed: seed for random number generator
        :param clip: should clip the data.
        See https://github.com/scikit-image/scikit-image/blob/master/skimage/util/noise.py

        '''
        if hasattr(image, 'as_array'):
            arr = TestData.scikit_random_noise(image.as_array(), mode=mode, seed=seed, clip=clip,
                  **kwargs)
            out = image.copy()
            out.fill(arr)
            return out
        elif issubclass(type(image), numpy.ndarray):
            return TestData.scikit_random_noise(image, mode=mode, seed=seed, clip=clip,
                   **kwargs)

    @staticmethod
    def scikit_random_noise(image, mode='gaussian', seed=None, clip=True, **kwargs):
        """
        Function to add random noise of various types to a floating-point image.
        Parameters
        ----------
        image : ndarray
            Input image data. Will be converted to float.
        mode : str, optional
            One of the following strings, selecting the type of noise to add:
            - 'gaussian'  Gaussian-distributed additive noise.
            - 'localvar'  Gaussian-distributed additive noise, with specified
                        local variance at each point of `image`.
            - 'poisson'   Poisson-distributed noise generated from the data.
            - 'salt'      Replaces random pixels with 1.
            - 'pepper'    Replaces random pixels with 0 (for unsigned images) or
                        -1 (for signed images).
            - 's&p'       Replaces random pixels with either 1 or `low_val`, where
                        `low_val` is 0 for unsigned images or -1 for signed
                        images.
            - 'speckle'   Multiplicative noise using out = image + n*image, where
                        n is uniform noise with specified mean & variance.
        seed : int, optional
            If provided, this will set the random seed before generating noise,
            for valid pseudo-random comparisons.
        clip : bool, optional
            If True (default), the output will be clipped after noise applied
            for modes `'speckle'`, `'poisson'`, and `'gaussian'`. This is
            needed to maintain the proper image data range. If False, clipping
            is not applied, and the output may extend beyond the range [-1, 1].
        mean : float, optional
            Mean of random distribution. Used in 'gaussian' and 'speckle'.
            Default : 0.
        var : float, optional
            Variance of random distribution. Used in 'gaussian' and 'speckle'.
            Note: variance = (standard deviation) ** 2. Default : 0.01
        local_vars : ndarray, optional
            Array of positive floats, same shape as `image`, defining the local
            variance at every image point. Used in 'localvar'.
        amount : float, optional
            Proportion of image pixels to replace with noise on range [0, 1].
            Used in 'salt', 'pepper', and 'salt & pepper'. Default : 0.05
        salt_vs_pepper : float, optional
            Proportion of salt vs. pepper noise for 's&p' on range [0, 1].
            Higher values represent more salt. Default : 0.5 (equal amounts)
        Returns
        -------
        out : ndarray
            Output floating-point image data on range [0, 1] or [-1, 1] if the
            input `image` was unsigned or signed, respectively.
        Notes
        -----
        Speckle, Poisson, Localvar, and Gaussian noise may generate noise outside
        the valid image range. The default is to clip (not alias) these values,
        but they may be preserved by setting `clip=False`. Note that in this case
        the output may contain values outside the ranges [0, 1] or [-1, 1].
        Use this option with care.
        Because of the prevalence of exclusively positive floating-point images in
        intermediate calculations, it is not possible to intuit if an input is
        signed based on dtype alone. Instead, negative values are explicitly
        searched for. Only if found does this function assume signed input.
        Unexpected results only occur in rare, poorly exposes cases (e.g. if all
        values are above 50 percent gray in a signed `image`). In this event,
        manually scaling the input to the positive domain will solve the problem.
        The Poisson distribution is only defined for positive integers. To apply
        this noise type, the number of unique values in the image is found and
        the next round power of two is used to scale up the floating-point result,
        after which it is scaled back down to the floating-point image range.
        To generate Poisson noise against a signed image, the signed image is
        temporarily converted to an unsigned image in the floating point domain,
        Poisson noise is generated, then it is returned to the original range.

        This function is adapted from scikit-image.
        https://github.com/scikit-image/scikit-image/blob/master/skimage/util/noise.py

        Copyright (C) 2019, the scikit-image team
        All rights reserved.

        Redistribution and use in source and binary forms, with or without
        modification, are permitted provided that the following conditions are
        met:

        1. Redistributions of source code must retain the above copyright
            notice, this list of conditions and the following disclaimer.
        2. Redistributions in binary form must reproduce the above copyright
            notice, this list of conditions and the following disclaimer in
            the documentation and/or other materials provided with the
            distribution.
        3. Neither the name of skimage nor the names of its contributors may be
            used to endorse or promote products derived from this software without
            specific prior written permission.

        THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
        IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
        WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
        DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
        INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
        (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
        SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
        HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
        STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
        IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
        POSSIBILITY OF SUCH DAMAGE.

        """
        mode = mode.lower()

        # Detect if a signed image was input
        if image.min() < 0:
            low_clip = -1.
        else:
            low_clip = 0.

        image = numpy.asarray(image, dtype=(np.float64))
        if seed is not None:
            np.random.seed(seed=seed)

        allowedtypes = {
            'gaussian': 'gaussian_values',
            'localvar': 'localvar_values',
            'poisson': 'poisson_values',
            'salt': 'sp_values',
            'pepper': 'sp_values',
            's&p': 's&p_values',
            'speckle': 'gaussian_values'}

        kwdefaults = {
            'mean': 0.,
            'var': 0.01,
            'amount': 0.05,
            'salt_vs_pepper': 0.5,
            'local_vars': np.zeros_like(image) + 0.01}

        allowedkwargs = {
            'gaussian_values': ['mean', 'var'],
            'localvar_values': ['local_vars'],
            'sp_values': ['amount'],
            's&p_values': ['amount', 'salt_vs_pepper'],
            'poisson_values': []}

        for key in kwargs:
            if key not in allowedkwargs[allowedtypes[mode]]:
                raise ValueError('%s keyword not in allowed keywords %s' %
                                (key, allowedkwargs[allowedtypes[mode]]))

        # Set kwarg defaults
        for kw in allowedkwargs[allowedtypes[mode]]:
            kwargs.setdefault(kw, kwdefaults[kw])

        if mode == 'gaussian':
            noise = np.random.normal(kwargs['mean'], kwargs['var'] ** 0.5,
                                    image.shape)
            out = image + noise

        elif mode == 'localvar':
            # Ensure local variance input is correct
            if (kwargs['local_vars'] <= 0).any():
                raise ValueError('All values of `local_vars` must be > 0.')

            # Safe shortcut usage broadcasts kwargs['local_vars'] as a ufunc
            out = image + np.random.normal(0, kwargs['local_vars'] ** 0.5)

        elif mode == 'poisson':
            # Determine unique values in image & calculate the next power of two
            vals = len(np.unique(image))
            vals = 2 ** np.ceil(np.log2(vals))

            # Ensure image is exclusively positive
            if low_clip == -1.:
                old_max = image.max()
                image = (image + 1.) / (old_max + 1.)

            # Generating noise for each unique value in image.
            out = np.random.poisson(image * vals) / float(vals)

            # Return image to original range if input was signed
            if low_clip == -1.:
                out = out * (old_max + 1.) - 1.

        elif mode == 'salt':
            # Re-call function with mode='s&p' and p=1 (all salt noise)
            out = TestData.random_noise(image, mode='s&p', seed=seed,
                            amount=kwargs['amount'], salt_vs_pepper=1.)

        elif mode == 'pepper':
            # Re-call function with mode='s&p' and p=1 (all pepper noise)
            out = TestData.random_noise(image, mode='s&p', seed=seed,
                            amount=kwargs['amount'], salt_vs_pepper=0.)

        elif mode == 's&p':
            out = image.copy()
            p = kwargs['amount']
            q = kwargs['salt_vs_pepper']
            flipped = np.random.choice([True, False], size=image.shape,
                                    p=[p, 1 - p])
            salted = np.random.choice([True, False], size=image.shape,
                                    p=[q, 1 - q])
            peppered = ~salted
            out[flipped & salted] = 1
            out[flipped & peppered] = low_clip

        elif mode == 'speckle':
            noise = np.random.normal(kwargs['mean'], kwargs['var'] ** 0.5,
                                    image.shape)
            out = image + image * noise

        # Clip back to original range, if necessary
        if clip:
            out = np.clip(out, low_clip, 1.0)

        return out
