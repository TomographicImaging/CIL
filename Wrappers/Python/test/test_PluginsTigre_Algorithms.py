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
from cil.processors import TransmissionAbsorptionConverter, Slicer
from cil.framework import ImageData

import numpy as np
from unittest_parametrize import parametrize
from unittest_parametrize import ParametrizedTestCase

initialise_tests()

import warnings
    
from testclass import CCPiTestClass

if has_tigre:
    from tigre.utilities.gpu import GpuIds
    from cil.plugins.tigre import ProjectionOperator, tigre_algo_wrapper

class TestTigreReconstructionAlgorithms(ParametrizedTestCase,  unittest.TestCase):

    
    def get_geometry_data(self, geometry_type):
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



    
    def run_algorithm(self, name, geometry_type, expect_warning=False, **kwargs):
        ig, absorption, gt = self.get_geometry_data(geometry_type)

        if expect_warning:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                algo = tigre_algo_wrapper(
                    name=name,
                    initial=None,
                    image_geometry=ig,
                    data=absorption,
                    niter=2,
                    **kwargs
                )
                img, qual = algo.run()
                warning_msgs = [str(warn.message) for warn in w]
                self.assertTrue(
                    any("incorrect results in the TV denoising step" in msg for msg in warning_msgs),
                    f"Expected warning not raised for {name} with {geometry_type}"
                )
        else:
            algo = tigre_algo_wrapper(
                name=name,
                initial=None,
                image_geometry=ig,
                data=absorption,
                niter=2,
                **kwargs
            )
            img, qual = algo.run()

        self.assertIsInstance(img, ImageData)
        self.assertEqual(img.shape, ig.shape)
        if qual is not None:
            self.assertTrue(isinstance(qual, (float, int, np.ndarray)))

    
    @parametrize(
        ("name", "kwargs", "expect_warning", "geometry_type"),
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
    def test_tigre_algorithms_with_geometries(self, name, kwargs, expect_warning, geometry_type):
        ig, absorption, _ = self.get_geometry_data(geometry_type)
        
        
        gpuids = GpuIds()
        if expect_warning:
            gpuids.devices = [0]
        kwargs['gpuids'] = gpuids


        resolved_kwargs = {
            k: v(self, ig, absorption.geometry) if callable(v) else v
            for k, v in kwargs.items()
        }
        self.run_algorithm(name, geometry_type, expect_warning=expect_warning, **resolved_kwargs)


