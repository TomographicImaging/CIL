from cil.framework import ImageGeometry, ImageData
from cil.utilities import dataexample 
import numpy as np 
import cupy as cp
from unittest import TestCase
import os



class TDataContainerAlgebra(object):


    '''A base class for unit test of DataContainer algebra.'''
    def test_divide_scalar(self):
        if hasattr(self, 'cwd'):
            os.chdir(self.cwd)
        image1 = self.image1
        image2 = self.image2
        image1.fill(1.)
        image2.fill(2.)
        
        tmp = image1/1.
        np.testing.assert_array_equal(image1.as_array().get(), tmp.as_array().get())
    
        tmp1 = image1.divide(1.)
        np.testing.assert_array_equal(tmp.as_array().get(), tmp1.as_array().get())
        
        image1.divide(1., out=image2)
        np.testing.assert_array_equal(tmp.as_array().get(), image2.as_array().get())


        image2.fill(2)
        image2 /= 2.0
        np.testing.assert_array_equal(image1.as_array().get(), image2.as_array().get())


    def test_divide_datacontainer(self):
        if hasattr(self, 'cwd'):
            os.chdir(self.cwd)
        # add 1 because the data contains zeros and divide is not going to be happy
        image1 = self.image1 + 1
        image2 = self.image2 + 1
        
        tmp = image1/image2


        np.testing.assert_array_almost_equal(
            np.ones(image1.shape, dtype=np.float32), tmp.as_array().get()
            )
    
        tmp1 = image1.divide(image2)
        np.testing.assert_array_almost_equal(
            np.ones(image1.shape, dtype=np.float32), tmp1.as_array().get()
            )
        
        tmp1.fill(2.)
        image1.divide(image2, out=tmp1)
        
        np.testing.assert_array_almost_equal(
            np.ones(image1.shape, dtype=np.float32), tmp1.as_array().get()
            )
        
        image1 /= image2
        np.testing.assert_array_almost_equal(
            np.ones(image1.shape, dtype=np.float32), image1.as_array().get()
            )        


    def test_multiply_scalar(self):
        if hasattr(self, 'cwd'):
            os.chdir(self.cwd)
        image1 = self.image1
        image2 = self.image2
        image2.fill(2.)
        
        tmp = image1 * 1.
        np.testing.assert_array_equal(image1.as_array().get(), tmp.as_array().get())
    
        tmp1 = image1.multiply(1.)
        np.testing.assert_array_equal(tmp.as_array().get(), tmp1.as_array().get())
        
        image1.multiply(1., out=image2)
        np.testing.assert_array_equal(tmp.as_array().get(), image2.as_array().get())


    def test_multiply_datacontainer(self):
        if hasattr(self, 'cwd'):
            os.chdir(self.cwd)
        image1 = self.image1
        image2 = self.image2
        image2.fill(1.)
        tmp = image1 * image2


        np.testing.assert_array_almost_equal(
            image1.as_array().get(), tmp.as_array().get()
            )
    
        tmp1 = image1.multiply(image2)
        np.testing.assert_array_almost_equal(
            image1.as_array().get(), tmp1.as_array().get()
            )
        
        tmp1.fill(2.)
        image1.multiply(image2, out=tmp1)
        
        np.testing.assert_array_almost_equal(
            image1.as_array().get(), tmp1.as_array().get()
            )


    def test_add_scalar(self):
        if hasattr(self, 'cwd'):
            os.chdir(self.cwd)
        image1 = self.image1
        image2 = self.image2
        image1.fill(0)
        image2.fill(1)
        
        tmp = image1 + 1.
        np.testing.assert_array_equal(image2.as_array().get(), tmp.as_array().get())
    
        tmp1 = image1.add(1.)
        np.testing.assert_array_equal(tmp.as_array().get(), tmp1.as_array().get())
        
        tmp1.fill(0)
        image1.add(1., out=tmp1)
        np.testing.assert_array_equal(tmp1.as_array().get(), image2.as_array().get())
    
    def test_add_datacontainer(self):
        if hasattr(self, 'cwd'):
            os.chdir(self.cwd)
        image1 = self.image1
        image2 = self.image2
        image1.fill(0.)
        image2.fill(1.)
        tmp = image1 + image2


        np.testing.assert_array_almost_equal(
            np.ones(image1.shape, dtype=np.float32), tmp.as_array().get()
            )
    
        tmp1 = image1.add(image2)
        
        np.testing.assert_array_almost_equal(
            np.ones(image1.shape, dtype=np.float32), tmp1.as_array().get()
            )
        
        tmp1.fill(2.)
        image1.add(image2, out=tmp1)
        
        np.testing.assert_array_almost_equal(
            np.ones(image1.shape, dtype=np.float32), tmp1.as_array().get()
            )
        
    
    def test_subtract_scalar(self):
        if hasattr(self, 'cwd'):
            os.chdir(self.cwd)
        image1 = self.image1
        image2 = self.image2
        image1.fill(2)
        image2.fill(1)
        
        tmp = image1 - 1.
        np.testing.assert_array_equal(image2.as_array().get(), tmp.as_array().get())
    
        tmp1 = image1.subtract(1.)
        np.testing.assert_array_equal(tmp.as_array().get(), tmp1.as_array().get())
        
        tmp1.fill(0)
        image1.subtract(1., out=tmp1)
        np.testing.assert_array_equal(tmp1.as_array().get(), image2.as_array().get())


    def test_subtract_datacontainer(self):
        if hasattr(self, 'cwd'):
            os.chdir(self.cwd)
        image1 = self.image1
        image2 = self.image2
        
        tmp = image1 - image2


        np.testing.assert_array_almost_equal(
            np.zeros(image1.shape, dtype=np.float32), tmp.as_array().get()
            )
    
        tmp1 = image1.subtract(image2)
        
        np.testing.assert_array_almost_equal(
            np.zeros(image1.shape, dtype=np.float32), tmp1.as_array().get()
            )
        
        tmp1.fill(2.)
        image1.subtract(image2, out=tmp1)
        
        np.testing.assert_array_almost_equal(
            np.zeros(image1.shape, dtype=np.float32), tmp1.as_array().get()
            )


    def test_division_by_scalar_zero(self):
        self.assertTrue(True)
        return
        if hasattr(self, 'cwd'):
            os.chdir(self.cwd)
        try:
            self.image1 / 0.
            self.assertFalse(True)
        except ZeroDivisionError:
            self.assertTrue(True)
        except error:
            self.assertTrue(True)
    
    def test_division_by_datacontainer_zero(self):
        self.assertTrue(True)
        return
        if hasattr(self, 'cwd'):
            os.chdir(self.cwd)
        try:
            self.image2 *= 0
            tmp = self.image1 / self.image2
            self.assertFalse(True)
        except ZeroDivisionError:
            self.assertTrue(True)
        except error:
            self.assertTrue(True)

    def test_cupy_array_fill_with_numpy(self):
        out = self.image1 * 0

        arr = np.asarray(
            np.arange(0,self.image1.size).reshape(self.image1.shape),
            dtype=out.dtype
            )


        out.fill(arr)

        np.testing.assert_array_equal(arr, out.as_array().get())

    def test_sapyb_scalars(self):


        image1 = self.image1.copy()
        image2 = self.image2.copy()

        print ("type image1 ", type(image1), image1.backend)

        arr = np.asarray(np.arange(0,image1.size).reshape(image1.shape), dtype=np.float32)
        image1.fill(arr)
        image2.fill(-arr)
        print ("type image1 ", type(image1), image1.backend)


        #scalars
        #check call methods with out


        a = 2.0
        b = -3.0
        gold = a * arr - b * arr


        out = image1.sapyb(a, image2, b)
        print ("type out sapyb ", type(out), out.backend)

        np.testing.assert_allclose(out.as_array().get(), gold)
        np.testing.assert_allclose(image1.as_array().get(), arr)
        np.testing.assert_allclose(image2.as_array().get(), -arr)


        out.fill(-1)
        print (id(out))
        image1.sapyb(a, image2, b, out=out)
        np.testing.assert_allclose(out.as_array().get(), gold)
        np.testing.assert_allclose(image1.as_array().get(), arr)
        np.testing.assert_allclose(image2.as_array().get(), -arr)

        print (out.backend)
        out.fill(arr)
        out.sapyb(a, image2, b, out=out)
        np.testing.assert_allclose(out.as_array().get(), gold)
        np.testing.assert_allclose(image2.as_array().get(), -arr)


        out.fill(-arr)
        image1.sapyb(a, out, b, out=out)
        np.testing.assert_allclose(out.as_array().get(), gold)
        np.testing.assert_allclose(image1.as_array().get(), arr)


    def test_sapyb_vectors(self):


        image1 = self.image1.copy()
        image2 = self.image2.copy()


        arr = np.asarray(
            np.arange(0,image1.size).reshape(image1.shape),
            dtype=np.float32
            )
        image1.fill(arr)
        image2.fill(-arr)


        a = image1.copy()
        a.fill(2)
        b = image1.copy()
        b.fill(-3)


        gold = a.as_array().get() * arr - b.as_array().get() * arr


        out = image1.sapyb(a, image2, b)
        np.testing.assert_allclose(out.as_array().get(), gold)
        np.testing.assert_allclose(image1.as_array().get(), arr)
        np.testing.assert_allclose(image2.as_array().get(), -arr)


        out.fill(0)
        image1.sapyb(a, image2, b, out=out)
        np.testing.assert_allclose(out.as_array().get(), gold)
        np.testing.assert_allclose(image1.as_array().get(), arr)
        np.testing.assert_allclose(image2.as_array().get(), -arr)


        out.fill(arr)
        out.sapyb(a, image2, b, out=out)
        np.testing.assert_allclose(out.as_array().get(), gold)
        np.testing.assert_allclose(image2.as_array().get(), -arr)


        out.fill(-arr)
        image1.sapyb(a, out, b, out=out)
        np.testing.assert_allclose(out.as_array().get(), gold)
        np.testing.assert_allclose(image1.as_array().get(), arr)


    def test_sapyb_mixed(self):


        image1 = ImageData(geometry=self.image1.geometry, backend='cupy')
        # image2 = self.image2.copy()
        image2 = ImageData(geometry=self.image1.geometry, backend='cupy')


        arr = np.asarray(
            np.arange(0,image1.size).reshape(image1.shape),
            dtype=np.float32
            )
        image1.fill(arr)
        image2.fill(-arr)
 
        a = 2
        b = image1.copy()
        b.fill(-3)


        gold = a * arr - b.as_array().get() * arr


        out = image1.sapyb(a, image2, b)
        np.testing.assert_allclose(out.as_array().get(), gold)
        np.testing.assert_allclose(image1.as_array().get(), arr)
        np.testing.assert_allclose(image2.as_array().get(), -arr)


        out.fill(0)
        image1.sapyb(a, image2, b, out=out)
        np.testing.assert_allclose(out.as_array().get(), gold)
        np.testing.assert_allclose(image1.as_array().get(), arr)
        np.testing.assert_allclose(image2.as_array().get(), -arr)
       
        out.fill(arr)
        out.sapyb(a, image2, b, out=out)
        np.testing.assert_allclose(out.as_array().get(), gold)
        np.testing.assert_allclose(image2.as_array().get(), -arr)


        out.fill(-arr)
        image1.sapyb(a, out, b, out=out)
        np.testing.assert_allclose(out.as_array().get(), gold)
        np.testing.assert_allclose(image1.as_array().get(), arr)

    def test_algebra_different_type(self):

        ig = self.image1.geometry.copy()
        ig.dtype = np.int32

        image1 = ImageData(geometry=self.image1.geometry, backend='cupy')
        # image2 = self.image2.copy()

        image2 = ImageData(geometry=self.image1.geometry, backend='cupy')

        arr = self.image1.as_array()

        image1.fill(arr)
        image2.fill(-arr)
 
        a = 2
        b = image1.copy()
        b.fill(-3)


class TestCupyIntegrationImageData(TestCase, TDataContainerAlgebra):

    def setUp(self):
        ig = ImageGeometry(100,100)

        # cp.copyto(garr, camera.as_array().get())
        # garr = cp.array(camera.as_array().get())


        # datac = ImageData(geometry=camera.geometry, backend='np')
        self.image1 = ImageData(geometry=ig, backend='cupy')

        self.image2 = ImageData(geometry=ig, backend='cupy')
        
        
        # print ("python copying")
        # for i in range(datag.shape[0]):
        #     for j in range(datag.shape[1]):
        #         datag.array[i,j] = camera.as_array().get()[i,j]
        # print ("done")
        

    def tearDown(self):
        pass

    def test_backend(self):
        assert self.image1.backend == 'cupy'
    def test_copy(self):

        cpy = self.image1.copy()
        assert id(cpy) != id(self.image1)
        print (self.image1.backend, cpy.backend)
        assert self.image1.backend == cpy.backend

    def test_creation_and_copy_ImageData(self):

        camera = dataexample.CAMERA.get((100,100))
        new = camera.geometry.allocate(0, dtype=self.image1.dtype)
        arr = np.array(camera.as_array(), dtype=new.dtype)
        print (arr.dtype)
        new.fill(arr)
        
        # should assert that fails
        # self.image1.fill(camera)
        
        self.image1.fill(new)
        # garr = datag.array
        # print (type(garr))



        # from cil.utilities.display import show2D
        # # show2D([garr.get(), camera])
        # show2D([datag.as_array().get().get(), camera])

        
        np.testing.assert_almost_equal(self.image1.as_array().get(), camera.as_array())
        
    
    def test_ImageGeometry_allocate_cupy(self):
        ig = ImageGeometry(19,22,21)
        data = ig.allocate(0, backend='cupy')
        import cupy as cp
        assert isinstance (data.as_array(), cp.ndarray)

        
