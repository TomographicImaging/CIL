#  Copyright 2022 United Kingdom Research and Innovation
#  Copyright 2022 The University of Manchester
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

import numpy as np
from cil.optimisation.operators import LinearOperator
from cil.utilities import dataexample
from cil.framework import AcquisitionGeometry
from cil.framework import DataOrder

class SimData(object):

    def _get_roi_3D(self):
        roi = [20,30,40]
        center_offset = [50,-20,5]

        self.ig_roi = self.ig.copy()
        self.ig_roi.voxel_num_x = roi[2]
        self.ig_roi.voxel_num_y = roi[1]
        self.ig_roi.voxel_num_z = roi[0]
        self.ig_roi.center_x = center_offset[2]*self.ig_roi.voxel_size_x
        self.ig_roi.center_y = center_offset[1]*self.ig_roi.voxel_size_y
        self.ig_roi.center_z = center_offset[0]*self.ig_roi.voxel_size_z

        index_roi = [None]*3
        for i in range(3):
            ind0 = center_offset[i] + (self.img_data.shape[i] - roi[i])//2
            ind1 = ind0 + roi[i]
            index_roi[i] = (ind0, ind1)

        self.gold_roi = self.img_data.array[index_roi[0][0]:index_roi[0][1],index_roi[1][0]:index_roi[1][1],index_roi[2][0]:index_roi[2][1]]


    def _get_roi_2D(self):
        roi = [30,40]
        center_offset = [-20,5]

        self.ig_roi = self.ig.copy()
        self.ig_roi.voxel_num_x = roi[1]
        self.ig_roi.voxel_num_y = roi[0]
        self.ig_roi.center_x = center_offset[1]*self.ig_roi.voxel_size_x
        self.ig_roi.center_y = center_offset[0]*self.ig_roi.voxel_size_y

        index_roi = [None]*2
        for i in range(2):
            ind0 = center_offset[i] + (self.img_data.shape[i] - roi[i])//2
            ind1 = ind0 + roi[i]
            index_roi[i] = (ind0, ind1)

        self.gold_roi = self.img_data.array[index_roi[0][0]:index_roi[0][1],index_roi[1][0]:index_roi[1][1]]


    def Cone3D(self):
        self.acq_data = dataexample.SIMULATED_CONE_BEAM_DATA.get()
        self.acq_data.reorder(self.backend)
        
        self.img_data = dataexample.SIMULATED_SPHERE_VOLUME.get()

        self.acq_data=np.log(self.acq_data)
        self.acq_data*=-1.0

        self.ig = self.img_data.geometry
        self.ag = self.acq_data.geometry

        self._get_roi_3D()


    def Parallel3D(self):
        self.acq_data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()
        self.acq_data.reorder(self.backend)
        
        self.img_data = dataexample.SIMULATED_SPHERE_VOLUME.get()

        self.acq_data=np.log(self.acq_data)
        self.acq_data*=-1.0

        self.ig = self.img_data.geometry
        self.ag = self.acq_data.geometry

        self._get_roi_3D()


    def Cone2D(self):

        self.acq_data = dataexample.SIMULATED_CONE_BEAM_DATA.get().get_slice(vertical='centre')
        self.acq_data.reorder(self.backend)
        
        self.img_data = dataexample.SIMULATED_SPHERE_VOLUME.get().get_slice(vertical='centre')

        self.acq_data=np.log(self.acq_data)
        self.acq_data*=-1.0

        self.ig = self.img_data.geometry
        self.ag = self.acq_data.geometry

        self._get_roi_2D()


    def Parallel2D(self):

        self.acq_data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get().get_slice(vertical='centre')
        self.acq_data.reorder(self.backend)
        
        self.img_data = dataexample.SIMULATED_SPHERE_VOLUME.get().get_slice(vertical='centre')

        self.acq_data=np.log(self.acq_data)
        self.acq_data*=-1.0

        self.ig = self.img_data.geometry
        self.ag = self.acq_data.geometry

        self._get_roi_2D()


class TestCommon_ProjectionOperator_TOY(object):
    '''
    Tests behaviour of the operators on a variety of geometries.
    '''
    def Cone3D(self):
        '''
            These are all single cone beam projection geometries. Pixels of  1, 2, 0.5, 0.5, Voxels of 1, 2, 0.5, 0.25
        '''

        self.test_geometries=[]
        ag_test_1 = AcquisitionGeometry.create_Cone3D(source_position=[0,-1000,0],detector_position=[0,0,0])\
                                            .set_panel([16,16],[1,1])\
                                            .set_angles([0])
        ag_test_1.set_labels(DataOrder.get_order_for_engine(self.backend, ag_test_1))

        
        ig_test_1 = ag_test_1.get_ImageGeometry()
        norm_1 = 4
        self.test_geometries.append((ag_test_1, ig_test_1, 4))


        ag_test_2 = AcquisitionGeometry.create_Cone3D(source_position=[0,-1000,0],detector_position=[0,0,0])\
                                            .set_panel([16,16],[2,2])\
                                            .set_angles([0])
        ag_test_2.set_labels(DataOrder.get_order_for_engine(self.backend, ag_test_2))

        ig_test_2 = ag_test_2.get_ImageGeometry()
        norm_2 = 8
        self.test_geometries.append((ag_test_2, ig_test_2, norm_2))


        ag_test_3 = AcquisitionGeometry.create_Cone3D(source_position=[0,-1000,0],detector_position=[0,0,0])\
                                            .set_panel([16,16],[0.5,0.5])\
                                            .set_angles([0])
        ag_test_3.set_labels(DataOrder.get_order_for_engine(self.backend, ag_test_3))
        ig_test_3 = ag_test_3.get_ImageGeometry()

        norm_3 = 2
        self.test_geometries.append((ag_test_3, ig_test_3, norm_3))


        ag_test_4 = AcquisitionGeometry.create_Cone3D(source_position=[0,-1000,0],detector_position=[0,1000,0])\
                                            .set_panel([16,16],[0.5,0.5])\
                                            .set_angles([0])
        ag_test_4.set_labels(DataOrder.get_order_for_engine(self.backend, ag_test_4))
        ig_test_4 = ag_test_4.get_ImageGeometry()

        norm_4 = 1
        self.test_geometries.append((ag_test_4, ig_test_4, norm_4))


    def Cone2D(self):
        '''
            These are all single cone beam projection geometries. Pixels of  1, 2, 0.5, 0.5, Voxels of 1, 2, 0.5, 0.25
        '''

        self.test_geometries=[]
        ag_test_1 = AcquisitionGeometry.create_Cone2D(source_position=[0,-1000],detector_position=[0,0])\
                                            .set_panel(16,1)\
                                            .set_angles([0])
        ag_test_1.set_labels(DataOrder.get_order_for_engine(self.backend, ag_test_1))

        
        ig_test_1 = ag_test_1.get_ImageGeometry()
        norm_1 = 4
        self.test_geometries.append((ag_test_1, ig_test_1, 4))


        ag_test_2 = AcquisitionGeometry.create_Cone2D(source_position=[0,-1000],detector_position=[0,0])\
                                            .set_panel(16,2)\
                                            .set_angles([0])
        ag_test_2.set_labels(DataOrder.get_order_for_engine(self.backend, ag_test_2))

        ig_test_2 = ag_test_2.get_ImageGeometry()
        norm_2 = 8
        self.test_geometries.append((ag_test_2, ig_test_2, norm_2))


        ag_test_3 = AcquisitionGeometry.create_Cone2D(source_position=[0,-1000],detector_position=[0,0])\
                                            .set_panel(16,0.5)\
                                            .set_angles([0])
        ag_test_3.set_labels(DataOrder.get_order_for_engine(self.backend, ag_test_3))
        ig_test_3 = ag_test_3.get_ImageGeometry()

        norm_3 = 2
        self.test_geometries.append((ag_test_3, ig_test_3, norm_3))


        ag_test_4 = AcquisitionGeometry.create_Cone2D(source_position=[0,-1000],detector_position=[0,1000])\
                                            .set_panel(16,0.5)\
                                            .set_angles([0])
        ag_test_4.set_labels(DataOrder.get_order_for_engine(self.backend, ag_test_4))
        ig_test_4 = ag_test_4.get_ImageGeometry()

        norm_4 = 1
        self.test_geometries.append((ag_test_4, ig_test_4, norm_4))


    def Parallel3D(self):
        '''
            These are all single parallel beam projection geometries. Pixels & voxels of  1, 2, 0.5
        '''

        self.test_geometries=[]
        ag_test_1 = AcquisitionGeometry.create_Parallel3D()\
                                            .set_panel([16,16],[1,1])\
                                            .set_angles([0])
        ag_test_1.set_labels(DataOrder.get_order_for_engine(self.backend, ag_test_1))

        
        ig_test_1 = ag_test_1.get_ImageGeometry()
        norm_1 = 4
        self.test_geometries.append((ag_test_1, ig_test_1, norm_1))

        ag_test_2 = AcquisitionGeometry.create_Parallel3D()\
                                            .set_panel([16,16],[2,2])\
                                            .set_angles([0])
        ag_test_2.set_labels(DataOrder.get_order_for_engine(self.backend, ag_test_2))


        ig_test_2 = ag_test_2.get_ImageGeometry()
        norm_2 = 8
        self.test_geometries.append((ag_test_2, ig_test_2, norm_2))


        ag_test_3 = AcquisitionGeometry.create_Parallel3D()\
                                            .set_panel([16,16],[0.5,0.5])\
                                            .set_angles([0])
        ag_test_3.set_labels(DataOrder.get_order_for_engine(self.backend, ag_test_3))


        ig_test_3 = ag_test_3.get_ImageGeometry()
        norm_3 = 2
        self.test_geometries.append((ag_test_3, ig_test_3, norm_3))


    def Parallel2D(self):
        '''
            These are all single parallel beam projection geometries. Pixels & voxels of  1, 2, 0.5
        '''

        self.test_geometries=[]
        ag_test_1 = AcquisitionGeometry.create_Parallel2D()\
                                            .set_panel(16,1)\
                                            .set_angles([0])

        ag_test_1.set_labels(DataOrder.get_order_for_engine(self.backend, ag_test_1))

        ig_test_1 = ag_test_1.get_ImageGeometry()
        norm_1 = 4
        self.test_geometries.append((ag_test_1, ig_test_1, norm_1))

        ag_test_2 = AcquisitionGeometry.create_Parallel2D()\
                                            .set_panel(16,2)\
                                            .set_angles([0])
        ag_test_2.set_labels(DataOrder.get_order_for_engine(self.backend, ag_test_2))


        ig_test_2 = ag_test_2.get_ImageGeometry()
        norm_2 = 8
        self.test_geometries.append((ag_test_2, ig_test_2, norm_2))


        ag_test_3 = AcquisitionGeometry.create_Parallel2D()\
                                            .set_panel(16,0.5)\
                                            .set_angles([0])
        ag_test_3.set_labels(DataOrder.get_order_for_engine(self.backend, ag_test_3))

        ig_test_3 = ag_test_3.get_ImageGeometry()
        norm_3 = 2
        self.test_geometries.append((ag_test_3, ig_test_3, norm_3))


    def test_norm(self):
        count = 1
        for ag, ig, norm in self.test_geometries:
            Op = self.ProjectionOperator(ig, ag,  **self.PO_args)
            n = Op.norm()

            diff =abs(n-norm)
            self.assertLess(diff, self.tolerance_norm, "Geometry: {0} Calculated norm {1}, expected {2}".format(count, n, norm))


    def test_linearity(self):
        count = 1
        for ag, ig, _ in self.test_geometries:
            Op = self.ProjectionOperator(ig, ag,  **self.PO_args)
            res = LinearOperator.dot_test(Op, tolerance=self.tolerance_linearity)
            self.assertTrue(res, "Geometry: {0} Dot-test out of tolerance {1}".format(count, self.tolerance_linearity))
            count +=1



class TestCommon_ProjectionOperator(object):

    '''
    These are all single projection geometries approximating parallel beam.
    '''

    def Cone3D(self):
        self.ag = AcquisitionGeometry.create_Cone3D(source_position=[0,-100000,0],detector_position=[0,0,0])\
                                            .set_panel([16,16],[1,1])\
                                            .set_angles([0])\
                                            .set_labels(['vertical','horizontal'])

    def Cone2D(self):
        self.ag = AcquisitionGeometry.create_Cone2D(source_position=[0,-100000],detector_position=[0,0])\
                                            .set_panel(16,1)\
                                            .set_angles([0])\
                                            .set_labels(['horizontal'])

    def Parallel3D(self):
        self.ag = AcquisitionGeometry.create_Parallel3D()\
                                            .set_panel([16,16],[1,1])\
                                            .set_angles([0])\
                                            .set_labels(['vertical','horizontal'])

    def Parallel2D(self):
        self.ag = AcquisitionGeometry.create_Parallel2D()\
                                            .set_panel(16,1)\
                                            .set_angles([0])\
                                            .set_labels(['horizontal'])

    def test_forward_projector(self):

        #create checker-board image
        res = np.zeros((16,16,16))
        ones = np.ones((4,16,4))
        for k in range(4):
            for i in range(4):
                if (i + k)% 2 == 0: 
                    res[k*4:(k+1)*4,:,i*4:(i+1)*4] = ones

        #create checker-board forward projection for parallel rays
        checker = np.zeros((16,16))
        ones = np.ones((4,4))
        for j in range(4):
            for i in range(4):
                if (i + j)% 2 == 0: 
                    checker[j*4:(j+1)*4,i*4:(i+1)*4] = ones * 16

        if self.ag.dimension == '2D':
            checker = checker[0]
            res = res[0]

        ig = self.ag.get_ImageGeometry()
        volume = ig.allocate(0)

        volume.fill(res)

        Op = self.ProjectionOperator(ig, self.ag, **self.PO_args)
        fp = Op.direct(volume)

        np.testing.assert_allclose(fp.array, checker, atol = self.tolerance_fp)


    def test_backward_projector(self):

        #create checker-board projection
        checker = np.zeros((16,16))
        ones = np.ones((4,4))
        for j in range(4):
            for i in range(4):
                if (i + j)% 2 == 0: 
                    checker[j*4:(j+1)*4,i*4:(i+1)*4] = ones

        #create backprojection of checker-board
        res = np.zeros((16,16,16))
        ones = np.ones((4,16,4))
        for k in range(4):
            for i in range(4):
                if (i + k)% 2 == 0: 
                    res[k*4:(k+1)*4,:,i*4:(i+1)*4] = ones

        if self.ag.dimension == '2D':
            checker = checker[0]
            res = res[0]

        ig = self.ag.get_ImageGeometry()
        data = self.ag.allocate(0)

        data.fill(checker)

        Op = self.ProjectionOperator(ig, self.ag, **self.PO_args)
        bp = Op.adjoint(data)

        if self.ag.geom_type == 'cone':
            #as cone beam res is not perfect grid
            np.testing.assert_allclose(bp.array, res, atol=1e-3)
        else:
            np.testing.assert_equal(bp.array, res)


class TestCommon_ProjectionOperator_SIM(SimData):
    '''
    Tests forward and backward operators function with and without 'out'
    '''
    def test_forward_projector(self):
        Op = self.ProjectionOperator(self.ig, self.ag, **self.PO_args)
        fp = Op.direct(self.img_data)
        np.testing.assert_allclose(fp.as_array(), self.acq_data.as_array(),atol=self.tolerance_fp)        

        fp2 = fp.copy()
        fp2.fill(0)
        Op.direct(self.img_data,out=fp2)
        np.testing.assert_allclose(fp.as_array(), fp2.as_array(),1e-8)    


    def test_backward_projectors_functionality(self):
        #this checks mechanics but not value
        Op = self.ProjectionOperator(self.ig, self.ag, **self.PO_args)
        bp = Op.adjoint(self.acq_data)

        bp2 = bp.copy()
        bp2.fill(0)
        Op.adjoint(self.acq_data,out=bp2)
        np.testing.assert_allclose(bp.as_array(), bp2.as_array(), 1e-8)    


    def test_input_arguments(self):

        #default image_geometry, named parameter acquisition_geometry
        Op = self.ProjectionOperator(acquisition_geometry=self.ag, **self.PO_args)
        fp = Op.direct(self.img_data)
        np.testing.assert_allclose(fp.as_array(), self.acq_data.as_array(),atol=self.tolerance_fp)        


class TestCommon_FBP_SIM(SimData):
    '''
    FBP tests on simulated data
    '''
    def test_FBP(self):

        self.FBP = self.FBP(self.ig, self.ag, **self.FBP_args)
        reco = self.FBP(self.acq_data)
        np.testing.assert_allclose(reco.as_array(), self.img_data.as_array(),atol=self.tolerance_fbp)    

        reco2 = reco.copy()
        reco2.fill(0)
        self.FBP(self.acq_data,out=reco2)
        np.testing.assert_allclose(reco.as_array(), reco2.as_array(),atol=1e-8)   


    def test_FBP_roi(self):
        self.FBP = self.FBP(self.ig_roi, self.ag, **self.FBP_args)
        reco = self.FBP(self.acq_data)
        np.testing.assert_allclose(reco.as_array(), self.gold_roi, atol=self.tolerance_fbp_roi) 

    
    def test_input_arguments(self):

        #default image_geometry, named parameter acquisition_geometry
        self.FBP = self.FBP(acquisition_geometry = self.ag, **self.FBP_args)
        reco = self.FBP(self.acq_data)
        np.testing.assert_allclose(reco.as_array(), self.img_data.as_array(),atol=self.tolerance_fbp)    

class TestCommon_ProjectionOperatorBlockOperator(object):
    # def setUp(self):
    #     data = dataexample.SIMULATED_PARALLEL_BEAM_DATA.get()
    #     self.data = data.get_slice(vertical='centre')
    #     K = ProjectionOperator(image_geometry=ig, acquisition_geometry=data.geometry)
    #     A = ProjectionOperator(image_geometry=ig, acquisition_geometry=self.data.geometry)
    #     self.projectionOperator = (A, K)
    def partition_test(self):
        
        A, K = self.projectionOperator

        u = A.adjoint(self.data)
        v = K.adjoint(self.datasplit)

        # the images are not entirely the same as the BlockOperator's requires to 
        # add all the data of the adjoint operator, which may result in a slightly
        # different image
        np.testing.assert_allclose(u.as_array(), v.as_array(), rtol=1.2e-6, atol=1.6e-4)

        # test if using data output

        v.fill(0)
        K.adjoint(self.datasplit, out=v)
        np.testing.assert_allclose(u.as_array(), v.as_array(), rtol=1.2e-6, atol=1.6e-4)



        x = A.direct(u)
        y = K.direct(v)

        # let's check that the data is the same
        def check_data_is_the_same(x,y):
            k = 0
            wrong = 0
            for el in y.containers:
                for j in range(el.shape[0]):
                    try:
                        np.testing.assert_allclose(el.as_array()[j], x.as_array()[k], atol=7e-2, rtol=1e-6)
                    except AssertionError as ae:
                        print(ae)
                        wrong += 1
                    # show2D([el.as_array()[j], x.as_array()[k]], cmap=['inferno', 'inferno'])
                    k += 1

            assert wrong == 0
        
            # reassemlbe the data
            out = x * 0
            k = 0
            for i, el in enumerate(y.containers):
                # print (i, el.shape)
                for j in range(el.shape[0]):
                    out.array[k] = el.as_array()[j]
                    k += 1

            # show2D([out, x, out-x], cmap=['inferno', 'inferno', 'seismic'], title=['out', 'x', 'diff'], \
            #     num_cols=3)
            np.testing.assert_allclose(out.as_array(), x.as_array(), atol=1e-2, rtol=1e-6)

        check_data_is_the_same(x,y)
        y.fill(0)
        K.direct(v, out=y)
        check_data_is_the_same(x,y)