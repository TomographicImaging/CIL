from ccpi.framework import AcquisitionGeometry, ImageGeometry
from ccpi.framework import DataContainer, AcquisitionData
import numpy
import tomophantom
from tomophantom import TomoP2D
from ccpi.utilities.display import plotter2D
import os

class AcquisitionGeometrySubsetGenerator(object):
    RANDOM='random'
    UNIFORM_SAMPLING='uniform'
    
    ### Changes in the Operator required to work as OS operator
    @staticmethod
    def generate_subset(ag, subset_id, number_of_subsets, method='random'):
        ags = ag.clone()
        angles = ags.angles
        if method == 'random':
            indices = AcquisitionGeometrySubsetGenerator.random_indices(angles, subset_id, number_of_subsets)
        else:
            raise ValueError('Can only do '.format('random'))
        ags.angles = ags.angles[indices]
        return ags , indices
    @staticmethod
    def random_indices(angles, subset_id, number_of_subsets):
        N = int(numpy.floor(float(len(angles))/float(number_of_subsets)))
        indices = numpy.asarray(range(len(angles)))
        numpy.random.shuffle(indices)
        indices = indices[:N]
        ret = numpy.asarray(numpy.zeros_like(angles), dtype=numpy.bool)
        for i,el in enumerate(indices):
            ret[el] = True
        return ret

    @staticmethod
    def generate_subsets(acquisition_geometry, number_of_subsets, method):
        subsets = [AcquisitionGeometrySubsetGenerator.random_indices(
                        acquisition_geometry.angles, subset_id, number_of_subsets) 
                                        for subset_id in range(number_of_subsets)]
        return subsets


def generate_subsets(self, number_of_subsets, method):
    self.number_of_subsets = number_of_subsets
    subsets = [AcquisitionGeometrySubsetGenerator.random_indices(
                    self.angles, subset_id, number_of_subsets) 
                                    for subset_id in range(number_of_subsets)]
    self.subsets = subsets[:]
    return subsets




numpy.random.seed(1)


model = 12 # select a model number from the library
N = 128 # set dimension of the phantom
device = 'gpu'
path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")
angles = numpy.linspace(0, numpy.pi, 180, dtype='float32')

phantom = TomoP2D.Model(model, N, path_library2D) 
sino = TomoP2D.ModelSino(model, N, N, angles * 180. / numpy.pi, path_library2D)
# plotter2D(sino)

# Define image geometry.
ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N, 
                   voxel_size_x = 0.1,
                   voxel_size_y = 0.1)
im_data = ig.allocate()
im_data.fill(phantom)

detectors = N
ag = AcquisitionGeometry('parallel','2D', angles, detectors,
                        pixel_size_h = 0.1)

print ("Number of subsets", ag.number_of_subsets)
print ("Subset_id", ag.subset_id)
data = ag.allocate()


data2 = ag.allocate()

print (id (data.as_array()), id(data2.as_array()))

data.fill(sino)
data2.fill(sino)

print (id (data.as_array()), id(data2.as_array()))

# plotter2D(data)

#subsets = AcquisitionGeometrySubsetGenerator.generate_subsets(ag, 18, 'random')

data.geometry.generate_subsets(10, 'random')
data.geometry.subset_id = 0

print (data.geometry.subsets[3])

# for el in ag.subsets:
#     print (el)

# plotter2D([data.as_array()[ag.subsets[0]], 
#            data.as_array()[ag.subsets[1]],
#            data.as_array()[ag.subsets[2]],
#            data.as_array()[ag.subsets[3]]])


sub0 = data.as_array()
data.geometry.subset_id = 1
sub1 = data.as_array()

numpy.random.seed(1)
data2.geometry.generate_subsets(10, 'random')
data2.geometry.subset_id = 8
print (data2.geometry.subsets[3])

s1 = data2.geometry.subsets[3]
s2 = data.geometry.subsets[0]

try:
    numpy.testing.assert_array_equal(s1,s2)
except AssertionError as ae:
    pass

print ("data.as_array().shape", data.as_array().shape)
print ("data2.as_array().shape", data2.as_array().shape)

#plotter2D([sub0 - sub1,
#           data2.as_array() -  data.as_array()])
print ("ids of data and data2", id (data.as_array()), id(data2.as_array()))
#out = data2.as_array() / data.as_array()


plotter2D([data- data2])

data2.geometry.subset_id = 0
plotter2D([data - data2])
assert id(data.as_array()) != id (data2.as_array())
assert id(data.geometry) != id (data2.geometry)
numpy.testing.assert_array_equal(data.as_array(),data2.as_array())
#print (out.as_array().min(), out.as_array().max())