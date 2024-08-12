#  Copyright 2018 United Kingdom Research and Innovation
#  Copyright 2018 The University of Manchester
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
import math

import numpy

from .block import BlockGeometry


class Partitioner(object):
    '''Interface for AcquisitionData to be able to partition itself in a number of batches.

    This class, by multiple inheritance with AcquisitionData, allows the user to partition the data,
    by using the method ``partition``.
    The partitioning will generate a ``BlockDataContainer`` with appropriate ``AcquisitionData``.

    '''
    # modes of partitioning
    SEQUENTIAL = 'sequential'
    STAGGERED = 'staggered'
    RANDOM_PERMUTATION = 'random_permutation'

    def _partition_indices(self, num_batches, indices, stagger=False):
        """Partition a list of indices into num_batches of indices.

        Parameters
        ----------
        num_batches : int
            The number of batches to partition the indices into.
        indices : list of int, int
            The indices to partition. If passed a list, this list will be partitioned in ``num_batches``
            partitions. If passed an int the indices will be generated automatically using ``range(indices)``.
        stagger : bool, default False
            If True, the indices will be staggered across the batches.

        Returns
        --------
        list of list of int
            A list of batches of indices.
        """

        # Partition the indices into batches.
        if isinstance(indices, int):
            indices = list(range(indices))

        num_indices = len(indices)
        # sanity check
        if num_indices < num_batches:
            raise ValueError(
                'The number of batches must be less than or equal to the number of indices.'
            )

        if stagger:
            batches = [indices[i::num_batches] for i in range(num_batches)]

        else:
            # we split the indices with floor(N/M)+1 indices in N%M groups
            # and floor(N/M) indices in the remaining M - N%M groups.

            # rename num_indices to N for brevity
            N = num_indices
            # rename num_batches to M for brevity
            M = num_batches
            batches = [
                indices[j:j + math.floor(N / M) + 1] for j in range(N % M)
            ]
            offset = N % M * (math.floor(N / M) + 1)
            for i in range(M - N % M):
                start = offset + i * math.floor(N / M)
                end = start + math.floor(N / M)
                batches.append(indices[start:end])

        return batches

    def _construct_BlockGeometry_from_indices(self, indices):
        '''Convert a list of boolean masks to a list of BlockGeometry.

        Parameters
        ----------
          indices : list of lists of indices

        Returns
        -------
            BlockGeometry
        '''
        ags = []
        for mask in indices:
            ag = self.geometry.copy()
            ag.config.angles.angle_data = numpy.take(self.geometry.angles, mask, axis=0)
            ags.append(ag)
        return BlockGeometry(*ags)

    def partition(self, num_batches, mode, seed=None):
        '''Partition the data into ``num_batches`` batches using the specified ``mode``.


        The modes are

        1. ``sequential`` - The data will be partitioned into ``num_batches`` batches of sequential indices.

        2. ``staggered`` - The data will be partitioned into ``num_batches`` batches of sequential indices, with stride equal to ``num_batches``.

        3. ``random_permutation`` - The data will be partitioned into ``num_batches`` batches of random indices.

        Parameters
        ----------
        num_batches : int
            The number of batches to partition the data into.
        mode : str
            The mode to use for partitioning. Must be one of ``sequential``, ``staggered`` or ``random_permutation``.
        seed : int, optional
            The seed to use for the random permutation. If not specified, the random number
            generator will not be seeded.


        Returns
        -------
        BlockDataContainer
            Block of `AcquisitionData` objects containing the data requested in each batch

        Example
        -------

        Partitioning a list of ints [0, 1, 2, 3, 4, 5, 6, 7, 8] into 4 batches will return:

        1. [[0, 1, 2], [3, 4], [5, 6], [7, 8]] with ``sequential``
        2. [[0, 4, 8], [1, 5], [2, 6], [3, 7]] with ``staggered``
        3. [[8, 2, 6], [7, 1], [0, 4], [3, 5]] with ``random_permutation`` and seed 1

        '''
        if mode == Partitioner.SEQUENTIAL:
            return self._partition_deterministic(num_batches, stagger=False)
        elif mode == Partitioner.STAGGERED:
            return self._partition_deterministic(num_batches, stagger=True)
        elif mode == Partitioner.RANDOM_PERMUTATION:
            return self._partition_random_permutation(num_batches, seed=seed)
        else:
            raise ValueError('Unknown partition mode {}'.format(mode))

    def _partition_deterministic(self, num_batches, stagger=False, indices=None):
        '''Partition the data into ``num_batches`` batches.

        Parameters
        ----------
        num_batches : int
            The number of batches to partition the data into.
        stagger : bool, optional
            If ``True``, the batches will be staggered. Default is ``False``.
        indices : list of int, optional
            The indices to partition. If not specified, the indices will be generated from the number of projections.
        '''
        if indices is None:
            indices = self.geometry.num_projections
        partition_indices = self._partition_indices(num_batches, indices, stagger)
        blk_geo = self._construct_BlockGeometry_from_indices(partition_indices)

        # copy data
        out = blk_geo.allocate(None)
        axis = self.dimension_labels.index('angle')

        for i in range(num_batches):
            out[i].fill(
                numpy.squeeze(
                    numpy.take(self.array, partition_indices[i], axis=axis)
                )
            )

        return out

    def _partition_random_permutation(self, num_batches, seed=None):
        '''Partition the data into ``num_batches`` batches using a random permutation.

        Parameters
        ----------
        num_batches : int
            The number of batches to partition the data into.
        seed : int, optional
            The seed to use for the random permutation. If not specified, the random number generator
            will not be seeded.

        '''
        if seed is not None:
            numpy.random.seed(seed)

        indices = numpy.arange(self.geometry.num_projections)
        numpy.random.shuffle(indices)

        indices = list(indices)

        return self._partition_deterministic(num_batches, stagger=False, indices=indices)
