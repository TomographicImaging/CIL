import numpy
from ccpi.framework import TestData
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from testclass import CCPiTestClass
import unittest

class TestTestData(CCPiTestClass):
    def test_random_noise(self):
        # loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
        # data_dir=os.path.join(os.path.dirname(__file__),'..', 'data')
        loader = TestData()
        camera = loader.load(TestData.CAMERA)
        noisy_camera = TestData.random_noise(camera, seed=1)
        norm = (camera - noisy_camera).norm()
        self.assertAlmostEqual(norm, 48.881268, places=4)

    def test_load_CAMERA(self):

        loader = TestData()
        res = False
        try:
            image = loader.load(TestData.CAMERA)
            if (image.shape[0] == 512) and (image.shape[1] == 512):
                res = True
            else:
                print("Image dimension mismatch")
        except FileNotFoundError:
            print("File not found")
        except:
            print("Failed to load file")

        self.assertTrue(res)


    def test_load_BOAT(self):
        loader = TestData()
        res = False
        try:
            image = loader.load(TestData.BOAT)
            if (image.shape[0] == 512) and (image.shape[1] == 512):
                res = True
            else:
                print("Image dimension mismatch")
        except FileNotFoundError:
            print("File not found")
        except:
            print("Failed to load file")

        self.assertTrue(res)

    def test_load_PEPPERS(self):
        loader = TestData()
        res = False
        try:
            image = loader.load(TestData.PEPPERS)
            if (image.shape[0] == 512) and (image.shape[1] == 512) and (image.shape[2] == 3):
                res = True
            else:
                print("Image dimension mismatch")
        except FileNotFoundError:
            print("File not found")
        except:
            print("Failed to load file")

        self.assertTrue(res)

    def test_load_RESOLUTION_CHART(self):
        loader = TestData()
        res = False
        try:
            image = loader.load(TestData.RESOLUTION_CHART)
            if (image.shape[0] == 512) and (image.shape[1] == 512):
                res = True
            else:
                print("Image dimension mismatch")
        except FileNotFoundError:
            print("File not found")
        except:
            print("Failed to load file")

        self.assertTrue(res)

    def test_load_SIMPLE_PHANTOM_2D(self):
        loader = TestData()
        res = False
        try:
            image = loader.load(TestData.SIMPLE_PHANTOM_2D)
            if (image.shape[0] == 512) and (image.shape[1] == 512):
                res = True
            else:
                print("Image dimension mismatch")
        except FileNotFoundError:
            print("File not found")
        except:
            print("Failed to load file")

        self.assertTrue(res)

    def test_load_SHAPES(self):
        loader = TestData()
        res = False
        try:
            image = loader.load(TestData.SHAPES)
            if (image.shape[0] == 200) and (image.shape[1] == 300):
                res = True
            else:
                print("Image dimension mismatch")
        except FileNotFoundError:
            print("File not found")
        except:
            print("Failed to load file")

        self.assertTrue(res)
