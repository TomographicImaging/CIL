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

import unittest
import numpy
from cil.framework import ImageGeometry

from cil.optimisation.operators import GradientOperator
from cil.optimisation.operators import LinearOperator

class TestGradientOperator(unittest.TestCase):

    def __init__(self, *args, **kwargs):

        N, M, K = 10, 11, 12
        channels = 12

        self.voxel_size_x, self.voxel_size_y, self.voxel_size_z = 0.1, 0.3, 0.6

        self.ig_2D = ImageGeometry(voxel_num_x = M, voxel_num_y = N) 
        self.ig_2D_chan = ImageGeometry(voxel_num_x = M, voxel_num_y = N, channels = channels) 
        self.ig_3D = ImageGeometry(voxel_num_x = M, voxel_num_y = N, voxel_num_z= K) 
        self.ig_3D_chan = ImageGeometry(voxel_num_x = M, voxel_num_y = N, channels = channels, voxel_num_z= K) 

        self.ig_2D_voxel = ImageGeometry(voxel_num_x = M, voxel_num_y = N, voxel_size_x=self.voxel_size_x, voxel_size_y=self.voxel_size_y) 
        self.ig_2D_chan_voxel = ImageGeometry(voxel_num_x = M, voxel_num_y = N, channels = channels, voxel_size_x=self.voxel_size_x, voxel_size_y=self.voxel_size_y) 
        self.ig_3D_voxel = ImageGeometry(voxel_num_x = M, voxel_num_y = N, voxel_num_z= K, voxel_size_x=self.voxel_size_x, voxel_size_y=self.voxel_size_y, voxel_size_z=self.voxel_size_z) 
        self.ig_3D_chan_voxel = ImageGeometry(voxel_num_x = M, voxel_num_y = N, channels = channels, voxel_num_z= K, voxel_size_y=self.voxel_size_y, voxel_size_x=self.voxel_size_x, voxel_size_z=self.voxel_size_z)         

        self.list_geometries = [self.ig_2D, self.ig_2D_chan, self.ig_3D, self.ig_3D_chan,
                                self.ig_2D_voxel, self.ig_2D_chan_voxel, self.ig_3D_voxel, self.ig_3D_chan_voxel]
        self.bconditions = ['Neumann', 'Periodic']                                
        self.backend = ['numpy','c']
        self.correlation = ['Space','SpaceChannels']
        self.method = ['forward', 'backward', 'centered']

        super(TestGradientOperator, self).__init__(*args, **kwargs)

    def print_assertion_info(self, geom = None, bnd = None, backend = None, method = None, corr = None, split = None):

        print( " Test Failed ")
        print( " ImageGeometry {} \n".format(geom))
        print( " Bnd Cond {} ".format(bnd))                                
        print( " Backend {} ".format(backend))
        print( " Method {} ".format(method))
        print( " Correlation {} ".format(corr)) 
        if split is not None:
            print( " Split {} ".format(split))        
            

    def test_GradientOperator_linearity(self):

        for geom in self.list_geometries:            
            for bnd in self.bconditions:
                for backend in self.backend:
                    for corr in self.correlation:
                        for method in self.method:                       
                                                                                          
                            Grad = GradientOperator(geom, 
                                                    bnd_cond = bnd,
                                                    backend = backend, 
                                                    correlation = corr, method = method)
                            try:                                                    
                                self.assertTrue(LinearOperator.dot_test(Grad))
                            except AssertionError:    
                                self.print_assertion_info(geom,bnd,backend,corr,method)
                                raise
                                                            
                            

    def test_GradientOperator_norm(self):

        for i in self.list_geometries:            
            for j in self.bconditions:
                for k in self.backend:
                    for z in self.correlation:
                        for l in self.method:

                            if i.channels == 1:
                            
                                if i.length==2:                                                             
                                    norm = numpy.sqrt((2/i.voxel_size_y)**2 + (2/i.voxel_size_x)**2)
                                elif i.length==3:  
                                    norm = numpy.sqrt((2/i.voxel_size_z)**2 + (2/i.voxel_size_y)**2 + (2/i.voxel_size_x)**2)  

                            else:

                                if z == 'Space':     
                                    if i.length==3:
                                        norm = numpy.sqrt((2/i.voxel_size_y)**2 + (2/i.voxel_size_x)**2)
                                    else:
                                        norm = numpy.sqrt(4 + (2/i.voxel_size_z)**2 + (2/i.voxel_size_y)**2 + (2/i.voxel_size_x)**2)                                        

                                else:

                                    if i.length==3:
                                        norm = numpy.sqrt(4 + (2/i.voxel_size_y)**2 + (2/i.voxel_size_x)**2)
                                    else:
                                        norm = numpy.sqrt(4 + (2/i.voxel_size_z)**2 + (2/i.voxel_size_y)**2 + (2/i.voxel_size_x)**2)                                      

                                                
                            Grad = GradientOperator(i, 
                                                    bnd_cond = j,
                                                    backend = k, 
                                                    correlation = z, method=l)
                            try:                                                    
                                numpy.testing.assert_approx_equal(Grad.norm(), norm, significant = 1) 
                            except AssertionError:    
                                self.print_assertion_info(i,j,k,z,l)
                                raise

    def test_GradientOperator_in_place_vs_allocate_direct(self):

        for geom in self.list_geometries:            
            for bnd in self.bconditions:
                for backend in self.backend:
                    for corr in self.correlation:
                        for method in self.method:   
                            Grad = GradientOperator(geom, 
                                                    bnd_cond = bnd,
                                                    backend = backend, 
                                                    correlation = corr, method=method)
                            tmp_x = Grad.domain.allocate('random')
                            res_in_place = Grad.direct(tmp_x)
                            res_allocate = Grad.range.allocate()
                            Grad.direct(tmp_x, out = res_allocate)

                            for m in range(len(Grad.range.geometries)):

                                try:
                                    numpy.testing.assert_array_almost_equal(res_in_place[m].array, 
                                                                    res_allocate[m].array)                            
                                except:
                                    self.print_assertion_info(geom,bnd,backend,corr,method) 
                                    raise   

    def test_GradientOperator_in_place_vs_allocate_adjoint(self):

        for geom in self.list_geometries:            
            for bnd in self.bconditions:
                for backend in self.backend:
                    for corr in self.correlation:
                        for method in self.method:  
                            Grad = GradientOperator(geom, 
                                                    bnd_cond = bnd,
                                                    backend = backend, 
                                                    correlation = corr, method=method)
                            tmp_x = Grad.range.allocate('random')
                            res_in_place = Grad.adjoint(tmp_x)
                            res_allocate = Grad.domain.allocate()
                            Grad.adjoint(tmp_x, out = res_allocate)

                            try:
                                numpy.testing.assert_array_almost_equal(res_in_place.array, 
                                                                res_allocate.array)                            
                            except:
                                self.print_assertion_info(geom,bnd,backend,corr,method, split) 
                                raise  

    def test_GradientOperator_range_shape(self):

        Grad2D = GradientOperator(self.ig_2D)
        numpy.testing.assert_equal(Grad2D.range.shape, (2,1)) 

        Grad2D = GradientOperator(self.ig_2D_chan, correlation="Space")
        numpy.testing.assert_equal(Grad2D.range.shape, (2,1))             

        Grad2D_chan = GradientOperator(self.ig_2D_chan, correlation="SpaceChannels")
        numpy.testing.assert_equal(Grad2D_chan.range.shape, (3,1))    

        Grad3D = GradientOperator(self.ig_3D)
        numpy.testing.assert_equal(Grad3D.range.shape, (3,1))   

        Grad3D_chan = GradientOperator(self.ig_3D_chan, correlation="Space")
        numpy.testing.assert_equal(Grad3D_chan.range.shape, (3,1))  

        Grad3D_chan = GradientOperator(self.ig_3D_chan, correlation="SpaceChannels")
        numpy.testing.assert_equal(Grad3D_chan.range.shape, (4,1))  

    def test_GradientOperator_split_shape(self):

        for geom in [self.ig_2D, self.ig_2D_chan, self.ig_3D, self.ig_3D_chan]:

            for split in [True, False]:

                Grad = GradientOperator(geom, split = split, correlation="SpaceChannels")

                if geom == self.ig_2D:
                    shape = (2,1)
                elif geom == self.ig_3D:
                    shape = (3,1)
                elif geom == self.ig_2D_chan:
                    if split:
                        shape = (2,1)
                    else:
                        shape = (3,1)    
                elif geom == self.ig_3D_chan:  
                    if split:
                        shape = (2,1)
                    else:
                        shape = (4,1)                                           

                try:
                    numpy.testing.assert_equal(Grad.range.shape, shape)  
                except:
                    self.print_assertion_info(geom, None, None, None, split=split)
                    raise

    def test_GradientOperator_split_direct_adjoint(self):
        
        # Test split for direct and adjoint in 2D + channels geometry
        geom = self.ig_2D_chan
        Grad2D_split_false = GradientOperator(geom, split = False, correlation="SpaceChannels")
        Grad2D_split_true = GradientOperator(geom, split = True, correlation="SpaceChannels")

        tmp_x = geom.allocate('random')
        res1 = Grad2D_split_false.direct(tmp_x)
        res2 = Grad2D_split_true.direct(tmp_x)
        res1_adj = Grad2D_split_false.adjoint(res1)
        res2_adj = Grad2D_split_true.adjoint(res2)

        numpy.testing.assert_array_almost_equal(res1[0].array, res2[0].array)
        numpy.testing.assert_array_almost_equal(res1[1].array, res2[1][0].array)  
        numpy.testing.assert_array_almost_equal(res1[2].array, res2[1][1].array) 
        numpy.testing.assert_array_almost_equal(res1_adj.array, res2_adj.array)     

        # Test split for direct and adjoint in 3D + channels geometry
        geom = self.ig_3D_chan
        Grad3D_split_false = GradientOperator(geom, split = False, correlation="SpaceChannels")
        Grad3D_split_true = GradientOperator(geom, split = True, correlation="SpaceChannels")

        tmp_x = geom.allocate('random')
        res1 = Grad3D_split_false.direct(tmp_x)
        res2 = Grad3D_split_true.direct(tmp_x)
        res1_adj = Grad3D_split_false.adjoint(res1)
        res2_adj = Grad3D_split_true.adjoint(res2)

        numpy.testing.assert_array_almost_equal(res1[0].array, res2[0].array)
        numpy.testing.assert_array_almost_equal(res1[1].array, res2[1][0].array)  
        numpy.testing.assert_array_almost_equal(res1[2].array, res2[1][1].array) 
        numpy.testing.assert_array_almost_equal(res1[3].array, res2[1][2].array) 
        numpy.testing.assert_array_almost_equal(res1_adj.array, res2_adj.array)                  
            
    def test_Gradient_operator_numpy_vs_c(self):

        for geom in self.list_geometries:
            for bnd in self.bconditions:
                for corr in self.correlation:

                    Grad_c = GradientOperator(geom, bnd_cond = bnd, correlation= corr, backend = 'c')  
                    Grad_numpy =  GradientOperator(geom, bnd_cond = bnd, correlation= corr, backend = 'numpy')

                    tmp_x = geom.allocate('random')
                    res1_c = Grad_c.direct(tmp_x)
                    res1_np = Grad_numpy.direct(tmp_x)
                    
                    # Check direct of numpy vs c in place
                    for m in range(len(Grad_c.range.geometries)):
                        try:
                            numpy.testing.assert_array_almost_equal(res1_c[m].array, 
                                                            res1_np[m].array)                            
                        except:
                            print("Check direct of numpy vs c in place")
                            self.print_assertion_info(geom, bnd, None, None, corr) 
                            raise   

                    res1_c_out = Grad_c.range.allocate()                    
                    Grad_c.direct(tmp_x, out = res1_c_out)

                    res1_np_out = Grad_numpy.range.allocate()                    
                    Grad_numpy.direct(tmp_x, out = res1_np_out)     
                    
                    # Check direct of numpy vs c allocate
                    for m in range(len(Grad_c.range.geometries)):
                        try:
                            numpy.testing.assert_array_almost_equal(res1_c_out[m].array, 
                                                            res1_np_out[m].array)                            
                        except:
                            print("Check direct of numpy vs c allocate")
                            self.print_assertion_info(geom, bnd, None, None, corr) 
                            raise  

                    tmp_x = Grad_c.range.allocate('random')
                    res1_c = Grad_c.adjoint(tmp_x)
                    res1_np = Grad_numpy.adjoint(tmp_x)
 
                    # Check adjoint of numpy vs c in place
                    try:
                        numpy.testing.assert_array_almost_equal(res1_c.array, 
                                                        res1_np.array, decimal=5)                            
                    except:
                        print("Check adjoint of numpy vs c in place")
                        self.print_assertion_info(geom, bnd, None, None, corr) 
                        raise   

                    res1_c_out = Grad_c.domain.allocate()                    
                    Grad_c.adjoint(tmp_x, out = res1_c_out)

                    res1_np_out = Grad_numpy.domain.allocate()                    
                    Grad_numpy.adjoint(tmp_x, out = res1_np_out)     

                    # Check adjoint of numpy vs c allocate
                    try:
                        numpy.testing.assert_array_almost_equal(res1_c_out.array, 
                                                        res1_np_out.array, decimal=5)                            
                    except:
                        print("Check adjoint of numpy vs c allocate")
                        self.print_assertion_info(geom, bnd, None, None, corr) 
                        raise    

