import tomophantom
from tomophantom import TomoP3D
from ccpi.optimisation.algorithms import CGLS, RCGLS
from ccpi.framework import BlockDataContainer, ImageGeometry, AcquisitionGeometry
from ccpi.optimisation.operators import BlockOperator, Gradient
from ccpi.astra.operators import AstraProjector3DSimple
from ccpi.utilities.display import plotter2D
import os
import numpy as np
import matplotlib.pyplot as plt

# Set up image geometry
num_vx = 128
num_vy = 128
num_vz = 128

ig = ImageGeometry(voxel_num_x = num_vx, voxel_num_y = num_vy, voxel_num_z = num_vz)

print(ig)

# Load Shepp-Logan phantom 
model = 13
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "Phantom3DLibrary.dat")

#tomophantom takes angular input in degrees
phantom_3D = TomoP3D.Model(model, (num_vx, num_vy, num_vz), path_library3D)
#set up acquisition geometry
number_pixels_x = 128
number_projections = 180
angles = np.linspace(0, np.pi, number_projections, dtype=np.float32)
ag = AcquisitionGeometry(geom_type='parallel', angles=angles, 
                         pixel_num_h=number_pixels_x, 
                         pixel_num_v=num_vz, 
                         dimension_labels=['vertical', 'angle', 'horizontal'])
print(ag)


phantom_sino = TomoP3D.ModelSino(model, num_vx, number_pixels_x, num_vz, angles*180./np.pi, path_library3D)
# Rescale the tomophantom data, set the max absorbtion to 25%
set_ratio_absorption = 0.25
new_max_value = -np.log(set_ratio_absorption)
sino_max = np.amax(phantom_sino)
scale = new_max_value/sino_max

# Allocate the image data container and copy the dataset in.
# This is only used as a reference to the ground truth
model = ig.allocate(0)
model.fill(phantom_3D*scale)

# Allocate the acquisition data container and copy the sinogram in
sinogram = ag.allocate(0)
sinogram.fill(phantom_sino*scale)

#plotter2D([model.subset(vertical=64), sinogram.subset(vertical=64)])
background_counts = 1000 #lower counts will increase the noise

counts = float(background_counts) * np.exp(-sinogram.as_array())
noisy_counts = np.random.poisson(counts)
sino_out = -np.log(noisy_counts/background_counts)

sinogram_noisy = ag.allocate()
sinogram_noisy.fill(sino_out)


device = "gpu"
# operator = AstraProjector3DSimple(ig, ag)

L = Gradient(ig)




def find_almost_equal_decimal(a,b,dtype,decimal=7,maxrecur=100):
    print ("testing ")
    print("a {}\nb {}". format(a, b))
    #decimal = 7
    #max = 100
    while maxrecur > 0:
        try:
            rerun = False
            #print ("direct", direct[0][0], direct[1][0])
            np.testing.assert_almost_equal(a,b, decimal=decimal)
        except AssertionError as ae:
            # print (ae)
            rerun = True
        if rerun:
            maxrecur -= 1
            decimal -= 1
        else:
            break
    return decimal

# test direct and norm
direct = []
a = L.direct(model)
direct.append([ a.squared_norm(), a ])
direct.append(L.direct_L21norm(model))
decimal = find_almost_equal_decimal(direct[0][0], direct[1][0],np.float32, decimal=7)
print ("direct decimal", decimal)

# test adjoint and norm
adjoint = []

a = L.adjoint(direct[0][1])
adjoint.append( [ a.squared_norm(), a ] )
adjoint.append(L.adjoint_L21norm(direct[0][1]))

a = L.adjoint(direct[1][1])
adjoint.append( [a.squared_norm(), a] )
adjoint.append(L.adjoint_L21norm(direct[1][1]))

decimal1 = find_almost_equal_decimal(adjoint[0][0], adjoint[1][0],np.float32, decimal=7)
decimal2 = find_almost_equal_decimal(adjoint[2][0], adjoint[3][0],np.float32, decimal=7)
print ("adjoint decimal", decimal1, decimal2)

# test scaled operator

alpha = 1/13.2
L1 = alpha * L
# test direct and norm

direct = []
a = L1.direct(model)
direct.append([ a.squared_norm(), a ])
direct.append(list(L.direct_L21norm(model)))
direct[1][0] *= alpha * alpha
direct[1][1] *= alpha
decimal = find_almost_equal_decimal(direct[0][0], direct[1][0],np.float32, decimal=7)
print ("scaled direct decimal", decimal)
print (type(direct[1][1]))
decimal = find_almost_equal_decimal(direct[1][1].squared_norm(), direct[1][0],np.float32, decimal=7)
print ("scaled direct decimal with L21norm", decimal)


# test adjoint and norm
adjoint = []

a = L1.adjoint(direct[0][1])
adjoint.append( [ a.squared_norm(), a ] )
adjoint.append(list(L.adjoint_L21norm(direct[0][1])))
adjoint[1][0] *= alpha*alpha
adjoint[1][1] *= alpha
decimal1 = find_almost_equal_decimal(adjoint[0][0], adjoint[1][0],np.float32, decimal=7)
print ("scaled adjoint decimal1 using scaled direct", decimal1)


a = L1.adjoint(direct[1][1])
adjoint.append( [a.squared_norm(), a] )

adjoint.append(list(L.adjoint_L21norm(direct[1][1])))

adjoint[3][0] *= alpha*alpha
adjoint[3][1] *= alpha

decimal2 = find_almost_equal_decimal(adjoint[2][0], adjoint[3][0],np.float32, decimal=7)
print ("scaled adjoint decimal2 using L21 scaled", decimal2)
