from .testclass import CCPiTestClass
import numpy
from ccpi.framework import TestData
import os



class TestTestData(CCPiTestClass):
    def test_random_noise(self):
        #loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
        #data_dir=os.path.join(os.path.dirname(__file__),'..', 'data')
        loader = TestData()

        camera = loader.load(TestData.CAMERA)

        noisy_camera = TestData.random_noise(camera, seed=1)
        norm = (camera-noisy_camera).norm()
        self.assertAlmostEqual(norm, 48.881268, places=4)
