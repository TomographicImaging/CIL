#  Copyright 2025 United Kingdom Research and Innovation
#  Copyright 2025 The University of Manchester
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
from utils import has_tigre, has_nvidia, initialise_tests

from cil.utilities.dataexample import (
    SIMULATED_PARALLEL_BEAM_DATA,
    SIMULATED_CONE_BEAM_DATA,
    SIMULATED_SPHERE_VOLUME
)
from cil.processors import TransmissionAbsorptionConverter
from cil.framework import ImageData

import numpy as np
from unittest_parametrize import parametrize
from unittest_parametrize import ParametrizedTestCase

initialise_tests()

import contextlib
import importlib.metadata
import json
import os
import tempfile
import warnings
from unittest.mock import patch

if has_tigre:
    from tigre.utilities.gpu import GpuIds
    from cil.plugins.tigre import ProjectionOperator, tigre_algo_wrapper
    from cil.plugins.tigre import Algorithms as tigre_algorithms
    from cil.plugins.tigre.Algorithms import (NEXT_TIGRE_RELEASE,
                                              NUMPY2_PROMOTION_ALGORITHMS,
                                              SINGLE_ANGLE_ALGORITHMS,
                                              _has_tigre_fista_float64_bug,
                                              _has_tigre_single_angle_bug,
                                              _tigre_older_than)

has_tigre_single_angle_bug = has_tigre and _has_tigre_single_angle_bug()
has_tigre_fista_float64_bug = has_tigre and _has_tigre_fista_float64_bug()


def gpu_count(gpuids=None):
    """Number of GPUs `gpuids` holds, or the number TIGRE would use by default if it is None."""
    if gpuids is None:
        if not (has_tigre and has_nvidia):
            return 0
        try:
            gpuids = GpuIds()
        except Exception:
            return 0

    return len(getattr(gpuids, 'devices', None) or [])


def multi_gpu_ids(devices=2):
    """A GpuIds listing `devices` entries, so the multi-GPU path can be tested on one GPU."""
    gpuids = GpuIds()
    gpuids.devices = (gpuids.devices * devices)[:devices]
    return gpuids


class TestTigreReconstructionAlgorithms(ParametrizedTestCase,  unittest.TestCase):

    
    @staticmethod
    def get_geometry_data(geometry_type):
        if geometry_type == "parallel_2d":
            data = SIMULATED_PARALLEL_BEAM_DATA.get().get_slice(vertical='centre')
            gt = SIMULATED_SPHERE_VOLUME.get().get_slice(vertical='centre')
        elif geometry_type == "parallel_3d":
            data = SIMULATED_PARALLEL_BEAM_DATA.get()
            gt = SIMULATED_SPHERE_VOLUME.get()
        elif geometry_type == "cone_2d":
            gt = SIMULATED_SPHERE_VOLUME.get().get_slice(vertical='centre')
            data = SIMULATED_CONE_BEAM_DATA.get().get_slice(vertical='centre')
        elif geometry_type == "cone_3d":
            gt = SIMULATED_SPHERE_VOLUME.get()
            data = SIMULATED_CONE_BEAM_DATA.get()
        else:
            raise ValueError(f"Unknown geometry type: {geometry_type}")
        
        absorption = TransmissionAbsorptionConverter()(data)
        ig = gt.geometry
        return ig, absorption, gt



    
    def run_algorithm(self, algorithm_name, geometry_type, expect_warning=False, **kwargs):
        ig, absorption, gt = self.get_geometry_data(geometry_type)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            algo = tigre_algo_wrapper(
                algorithm_name=algorithm_name,
                initial=None,
                image_geometry=ig,
                data=absorption,
                number_iterations=2,
                **kwargs
            )
            img, qual = algo.run()

        # `expect_warning` marks the TV algorithms on 2D data, which only warn when the
        # reconstruction is spread over more than one GPU
        warning_msgs = [str(warn.message) for warn in w]
        self.assertEqual(
            any("CERN/TIGRE#681" in msg for msg in warning_msgs),
            expect_warning and gpu_count(kwargs.get('gpuids')) > 1,
            f"Unexpected 2D TV denoising warning state for {algorithm_name} with {geometry_type}"
        )

        self.assertIsInstance(img, ImageData)
        self.assertEqual(img.shape, ig.shape)
        self.assertEqual(img.dtype, ig.dtype)
        self.assertEqual(img.geometry, ig)
        if qual is not None:
            self.assertTrue(isinstance(qual, (float, int, np.ndarray)))

    
    @parametrize(
        ("algorithm_name", "kwargs", "expect_warning", "geometry_type"),
        [
            ("sart", {}, False, "parallel_2d"),
            ("sirt", {}, False, "parallel_3d"),
            ("ossart", {}, False, "cone_2d"),
            ("lsmr", {}, False, "cone_3d"),
            ("cgls", {}, False, "parallel_2d"),
            ('hybrid_lsqr', {}, False, "cone_2d"),
             ("ista", {
                "hyper": lambda self, ig, ag: ProjectionOperator(ig, ag).norm()**2,
                "Quameasopts": ['RMSE'],
                "tvlambda": 0.01
            }, True, "cone_2d"),
            ("fista", {
                "hyper": lambda self, ig, ag: 2 * ProjectionOperator(ig, ag).norm()**2,
                "Quameasopts": ['RMSE'],
                "tvlambda": 0.001
            }, True, "cone_2d"),
            ("sart_tv", {"tvlambda": 50}, True, "parallel_2d"),
            ("ossart_tv", {"tvlambda": 0.005}, True, "parallel_2d"),
        ]
    )
    @unittest.skipUnless(has_tigre, "Requires TIGRE")
    @unittest.skipUnless(has_nvidia, "Requires NVIDIA GPU for TIGRE")
    def test_tigre_algorithms_with_geometries(self, algorithm_name, kwargs, expect_warning, geometry_type):
        if has_tigre_single_angle_bug and algorithm_name in SINGLE_ANGLE_ALGORITHMS:
            self.skipTest(
                f"TIGRE's {algorithm_name} projects one angle at a time and cannot run with this "
                "version of TIGRE, see _has_tigre_single_angle_bug")

        if has_tigre_fista_float64_bug and algorithm_name in NUMPY2_PROMOTION_ALGORITHMS:
            self.skipTest(
                f"TIGRE's {algorithm_name} upcasts its working volume to float64 under NumPy 2 "
                "and cannot run with this version of TIGRE, see _has_tigre_fista_float64_bug")

        ig, absorption, _ = self.get_geometry_data(geometry_type)


        gpuids = GpuIds()
        if expect_warning:
            gpuids.devices = [0]
        kwargs['gpuids'] = gpuids


        resolved_kwargs = {
            k: v(self, ig, absorption.geometry) if callable(v) else v
            for k, v in kwargs.items()
        }
        self.run_algorithm(algorithm_name, geometry_type, expect_warning=expect_warning, **resolved_kwargs)


class TestTigreAlgorithmBuffers(ParametrizedTestCase, unittest.TestCase):
    """
    Test that the algorithm buffers are handled correctly.
    user's `initial` is never touched and every `run` starts afresh from it.
    """

    @staticmethod
    def _setup(geometry_type, initial_value=None):
        """Build a cheap sirt wrapper, with a non-zero `initial` if `initial_value` is given."""
        ig, absorption, _ = TestTigreReconstructionAlgorithms.get_geometry_data(geometry_type)
        initial = None if initial_value is None else ig.allocate(initial_value)
        algo = tigre_algo_wrapper(
            algorithm_name='sirt',
            initial=initial,
            image_geometry=ig,
            data=absorption,
            number_iterations=2,
        )
        return ig, initial, algo

    @parametrize(("geometry_type",), [("parallel_2d",), ("parallel_3d",)])
    @unittest.skipUnless(has_tigre, "Requires TIGRE")
    @unittest.skipUnless(has_nvidia, "Requires NVIDIA GPU for TIGRE")
    def test_run_is_repeatable(self, geometry_type):
        _, _, algo = self._setup(geometry_type)
        first, _ = algo.run()
        second, _ = algo.run()
        np.testing.assert_allclose(first.as_array(), second.as_array(), atol=1e-8)

    @parametrize(("geometry_type",), [("parallel_2d",), ("parallel_3d",)])
    @unittest.skipUnless(has_tigre, "Requires TIGRE")
    @unittest.skipUnless(has_nvidia, "Requires NVIDIA GPU for TIGRE")
    def test_initial_is_not_modified(self, geometry_type):
        _, initial, algo = self._setup(geometry_type, initial_value=0.5)
        before = initial.as_array().copy()
        algo.run()
        np.testing.assert_array_equal(initial.as_array(), before)

    @parametrize(("geometry_type",), [("parallel_2d",), ("parallel_3d",)])
    @unittest.skipUnless(has_tigre, "Requires TIGRE")
    @unittest.skipUnless(has_nvidia, "Requires NVIDIA GPU for TIGRE")
    def test_out_matches_return(self, geometry_type):
        ig, _, algo = self._setup(geometry_type)
        expected, _ = algo.run()

        out = ig.allocate(0)
        returned, _ = algo.run(out=out)
        self.assertIs(returned, out)
        np.testing.assert_allclose(out.as_array(), expected.as_array(), atol=1e-8)

