# -*- coding: utf-8 -*-
#  CCP in Tomographic Imaging (CCPi) Core Imaging Library (CIL).

#   Copyright 2017 UKRI-STFC
#   Copyright 2017 University of Manchester

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
        

from ccpi.processors import RingRemover
from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry
from ccpi.astra.operators import AstraProjectorSimple

import tomophantom
from tomophantom import TomoP2D       
import os
import numpy as np
import unittest
from ccpi.io import NEXUSDataReader
try:
    import wget
    has_wget = True
except ImportError as ie:
    has_wget = False
#from ccpi.utilities.display import plotter2D
from packaging import version
skip_test = not has_wget
print ("Numpy version ", np.version.version)
if version.parse(np.version.version) < version.parse("1.14"):
    print ("Numpy version ", np.version.version)
    skip_test = True
class TestRingProcessor(unittest.TestCase):
    
    def setUp(self):
        self.list_of_files = []
        self.url = 'https://www.ccpi.ac.uk/sites/www.ccpi.ac.uk/files/'
        self.cwd = os.getcwd()
        np.random.seed(1)
    # @property
    # def list_of_files(self):
    #     return self._list_of_files
    # @list_of_files.setter
    # def list_of_files(self, value):
    #     print("adding", value)
    #     self._list_of_files.append(value)

    def tearDown(self):
        for f in self.list_of_files:
            the_file = os.path.join(self.cwd , f)
            if os.path.exists(the_file):
                os.remove(the_file)
                print ("removed" , the_file)

    @unittest.skipIf(skip_test, "Numpy <= 1.13")
    def test_2D_demo_ring(self):
        
        print("Start 2D ring removal in simulated sinogram")
        
        model = 1 # select a model number from the library
        N = 512 # set dimension of the phantom
        path = os.path.dirname(tomophantom.__file__)
        path_library2D = os.path.join(path, "Phantom2DLibrary.dat")
        
        phantom_2D = TomoP2D.Model(model, N, path_library2D)    
        # data = ImageData(phantom_2D)
        ig = ImageGeometry(voxel_num_x=N, voxel_num_y=N, voxel_size_x = 0.1, voxel_size_y = 0.1)
        data = ig.allocate(None)
        data.fill(phantom_2D)
        
        # Create acquisition data and geometry
        detectors = N
        angles = np.linspace(0, np.pi, 120)
        ag = AcquisitionGeometry('parallel','2D',angles, detectors, pixel_size_h = 0.1)
            
        Aop = AstraProjectorSimple(ig, ag, 'cpu')
        sin = ag.allocate()
        Aop.direct(data, out = sin)
        
        sin_stripe = 0*sin
    #        sin_stripe = sin
        tmp = sin.as_array()
        tmp[:,::20]=0
        sin_stripe.fill(tmp)
                
        ring_removal = RingRemover(4, "db25", 20, info = True)
        ring_removal.set_input(sin_stripe)
        ring_recon = ring_removal.get_output()
        
        # load ring processed result sinogram 2D 
        wget.download(self.url + 'result_sinogram_2D_ring_remover.nxs', 
          out = self.cwd)
        self.list_of_files.append('result_sinogram_2D_ring_remover.nxs')
        
        reader = NEXUSDataReader()
        reader.set_up(nexus_file = os.path.join(self.cwd, self.list_of_files[-1]))
        tmp = reader.load_data()      
        
        print("Check ring remover sinogram 2D")
        np.testing.assert_array_almost_equal(tmp.as_array(), ring_recon.as_array()) 
        # plotter2D([tmp, ring_recon], titles=['saved', 'recon'])
        # np.testing.assert_allclose(tmp.as_array(), ring_recon.as_array(), rtol=1e4)
        print("Test passed\n")        
                    
    @unittest.skipIf(skip_test, "has wget or Numpy < 1.14")
    def test_3D(self):
        
        print("Start 3D ring removal in real data")
        wget.download(self.url + 'Sinogram_Rock_sample_3D.nxs', 
          out = self.cwd)
        self.list_of_files.append('Sinogram_Rock_sample_3D.nxs')
        fname = os.path.join(self.cwd, self.list_of_files[-1])
        # Load data --> sirt after flat and after ring
        read_sinogram3D = NEXUSDataReader()
        read_sinogram3D.set_up(nexus_file = fname)
        
        sinogram3D = read_sinogram3D.load_data()
        
        print(sinogram3D.shape)
        
        ring_removal = RingRemover(4, "db25", 20, info = True)
        ring_removal.set_input(sinogram3D)
        ring_recon = ring_removal.get_output()
        
        # load ring processed result sinogram rock 3D 
        self.list_of_files.append('result_sinogram_rock_3D_ring_remover.nxs')
        wget.download(self.url + self.list_of_files[-1],
           out = self.cwd)
        reader = NEXUSDataReader()
        reader.set_up(nexus_file = os.path.join(self.cwd, self.list_of_files[-1]))
        tmp = reader.load_data()      
        
        print("Check ring remover sinogram 3D")
        np.testing.assert_array_almost_equal(tmp.as_array(), ring_recon.as_array())          
        print("Test passed\n")
        
    @unittest.skipIf(skip_test, "has wget or Numpy < 1.14")
    def test_2D_channels(self):
        
        print("Start 2D+channels ring removal in real data")
        self.list_of_files.append('Sinogram_Rock_sample_2D_channels.nxs')
        wget.download(self.url + self.list_of_files[-1],
           out = self.cwd)
        
        read_sinogram2D_chan = NEXUSDataReader()
        read_sinogram2D_chan.set_up(nexus_file = 
              os.path.join(self.cwd, self.list_of_files[-1]))
        
        sinogram2D_chan = read_sinogram2D_chan.load_data()
        
        print(sinogram2D_chan.shape)
        
        ring_removal = RingRemover(4, "db25", 20, info = True)
        ring_removal.set_input(sinogram2D_chan)
        ring_recon = ring_removal.get_output()
        
        # load ring processed result sinogram rock 3D 
        self.list_of_files.append('result_sinogram_rock_2D_channels_ring_remover.nxs')
        wget.download(self.url + self.list_of_files[-1],
           out = self.cwd)
        
        reader = NEXUSDataReader()
        reader.set_up(nexus_file = os.path.join(self.cwd, self.list_of_files[-1]))
        tmp = reader.load_data() 

        print("Check ring remover sinogram 2D_channels")
        np.testing.assert_array_almost_equal(tmp.as_array(), ring_recon.as_array()) 
        print("Test passed\n")        
            
    @unittest.skipIf(skip_test, "has wget or Numpy < 1.14")
    def test_3D_channels(self):
        
        print("Start 3D+channels ring removal in real data")
        self.list_of_files.append('Sinogram_Rock_sample_3D_channels.nxs')
        wget.download(self.url + self.list_of_files[-1],
           out = self.cwd)
        
        # Load data --> sirt after flat and after ring
        read_sinogram3D_chan = NEXUSDataReader()
        read_sinogram3D_chan.set_up(nexus_file = 
           os.path.join(self.cwd , self.list_of_files[-1]))
        sinogram3D_chan = read_sinogram3D_chan.load_data()
        
        print(sinogram3D_chan.shape)
        
        ring_removal = RingRemover(4, "db25", 20, info = True)
        ring_removal.set_input(sinogram3D_chan)
        ring_recon = ring_removal.get_output()
        
        # load ring processed result sinogram rock 3D channels
        self.list_of_files.append('result_sinogram_rock_3D_channels_ring_remover.nxs')
        wget.download(self.url + self.list_of_files[-1],
           out = self.cwd)
        
        reader = NEXUSDataReader()
        reader.set_up(nexus_file = 
           os.path.join(self.cwd, self.list_of_files[-1]))
        tmp = reader.load_data() 

        print("Check ring remover sinogram 3D_channels")
        np.testing.assert_array_almost_equal(tmp.as_array(), ring_recon.as_array()) 
        print("Test passed\n")        
                
                

if __name__ == '__main__':
    
    d = TestRingProcessor()
    d.test_2D_demo_ring()
    d.test_3D()
    d.test_2D_channels()
    d.test_3D_channels()
      
    

   
#test_2D_channels()
#test_3D()    
#test_3D_channels()    
 
              
    
    
    
    
