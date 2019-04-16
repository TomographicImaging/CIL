import numpy
import tomophantom
from tomophantom import TomoP3D
from tomophantom.supp.artifacts import ArtifactsClass as Artifact
import os

import matplotlib.pyplot as plt

from ccpi.optimisation.algorithms import CGLS
from ccpi.plugins.ops import CCPiProjectorSimple
from ccpi.optimisation.ops import PowerMethodNonsquare
from ccpi.optimisation.ops import TomoIdentity
from ccpi.optimisation.funcs import Norm2sq, Norm1
from ccpi.framework import ImageGeometry, AcquisitionGeometry, ImageData, AcquisitionData
from ccpi.optimisation.algorithms import GradientDescent
from ccpi.framework import BlockDataContainer
from ccpi.optimisation.operators import BlockOperator


model = 13 # select a model number from tomophantom library
N_size = 64 # Define phantom dimensions using a scalar value (cubic phantom)
path = os.path.dirname(tomophantom.__file__)
path_library3D = os.path.join(path, "Phantom3DLibrary.dat")

#This will generate a N_size x N_size x N_size phantom (3D)
phantom_tm = TomoP3D.Model(model, N_size, path_library3D)

# detector column count (horizontal)
detector_horiz = int(numpy.sqrt(2)*N_size)
# detector row count (vertical) (no reason for it to be > N)
detector_vert = N_size
# number of projection angles
angles_num = int(0.5*numpy.pi*N_size)
# angles are expressed in degrees
angles = numpy.linspace(0.0, 179.9, angles_num, dtype='float32')


acquisition_data_array = TomoP3D.ModelSino(model, N_size,
                                                 detector_horiz, detector_vert,
                                                 angles,
                                                 path_library3D)

tomophantom_acquisition_axes_order = ['vertical', 'angle', 'horizontal']

artifacts = Artifact(acquisition_data_array)


tp_acq_data = AcquisitionData(artifacts.noise(0.2, 'Gaussian'),
                              dimension_labels=tomophantom_acquisition_axes_order)
#print ("size", acquisition_data.shape)
print ("horiz", detector_horiz)
print ("vert", detector_vert)
print ("angles", angles_num)

tp_acq_geometry = AcquisitionGeometry(geom_type='parallel', dimension='3D',
                                      angles=angles,
                                      pixel_num_h=detector_horiz,
                                      pixel_num_v=detector_vert,
                                      channels=1,
                                      )

acq_data = tp_acq_geometry.allocate()
#print (tp_acq_geometry)
print ("AcquisitionData", acq_data.shape)
print ("TomoPhantom", tp_acq_data.shape, tp_acq_data.dimension_labels)

default_acquisition_axes_order = ['angle', 'vertical', 'horizontal']

acq_data2 = tp_acq_data.subset(dimensions=default_acquisition_axes_order)
print ("AcquisitionData", acq_data2.shape, acq_data2.dimension_labels)
print ("AcquisitionData {} TomoPhantom {}".format(id(acq_data2.as_array()),
                                                     id(acquisition_data_array)))

fig = plt.figure()
plt.subplot(1,2,1)
plt.imshow(acquisition_data_array[20])
plt.title('Sinogram')
plt.subplot(1,2,2)
plt.imshow(tp_acq_data.as_array()[20])
plt.title('Sinogram + noise')
plt.show()

# Set up Operator object combining the ImageGeometry and AcquisitionGeometry
# wrapping calls to CCPi projector.

ig = ImageGeometry(voxel_num_x=detector_horiz,
                   voxel_num_y=detector_horiz,
                   voxel_num_z=detector_vert)
A = CCPiProjectorSimple(ig, tp_acq_geometry)
# Forward and backprojection are available as methods direct and adjoint. Here
# generate test data b and some noise

#b = A.direct(Phantom)
b = acq_data2

#z = A.adjoint(b)


# Using the test data b, different reconstruction methods can now be set up as
# demonstrated in the rest of this file. In general all methods need an initial
# guess and some algorithm options to be set. Note that 100 iterations for
# some of the methods is a very low number and 1000 or 10000 iterations may be
# needed if one wants to obtain a converged solution.
x_init = ImageData(geometry=ig,
                   dimension_labels=['horizontal_x','horizontal_y','vertical'])
X_init = BlockDataContainer(x_init)
B = BlockDataContainer(b,
                       ImageData(geometry=ig, dimension_labels=['horizontal_x','horizontal_y','vertical']))

# setup a tomo identity
Ibig = 4e1 * TomoIdentity(geometry=ig)
Ismall = 1e-3 * TomoIdentity(geometry=ig)
Iok = 7.6e0 * TomoIdentity(geometry=ig)

# composite operator
Kbig = BlockOperator(A, Ibig, shape=(2,1))
Ksmall = BlockOperator(A, Ismall, shape=(2,1))
Kok = BlockOperator(A, Iok, shape=(2,1))

#out = K.direct(X_init)
#x0 = x_init.copy()
#x0.fill(numpy.random.randn(*x0.shape))
#lipschitz = PowerMethodNonsquare(A, 5, x0)
#print("lipschitz", lipschitz)

#%%

simplef = Norm2sq(A, b, memopt=False)
#simplef.L = lipschitz[0]/3000.
simplef.L = 0.00003

f = Norm2sq(Kbig,B)
f.L = 0.00003

fsmall = Norm2sq(Ksmall,B)
fsmall.L = 0.00003

fok = Norm2sq(Kok,B)
fok.L = 0.00003

print("setup gradient descent")
gd = GradientDescent( x_init=x_init, objective_function=simplef,
                      rate=simplef.L)
gd.max_iteration = 5
simplef2 = Norm2sq(A, b, memopt=True)
#simplef.L = lipschitz[0]/3000.
simplef2.L = 0.00003
print("setup gradient descent")
gd2 = GradientDescent( x_init=x_init, objective_function=simplef2,
                      rate=simplef2.L)
gd2.max_iteration = 5

Kbig.direct(X_init)
Kbig.adjoint(B)
print("setup CGLS")
cg = CGLS()
cg.set_up(X_init, Kbig, B )
cg.max_iteration = 10

print("setup CGLS")
cgsmall = CGLS()
cgsmall.set_up(X_init, Ksmall, B )
cgsmall.max_iteration = 10


print("setup CGLS")
cgs = CGLS()
cgs.set_up(x_init, A, b )
cgs.max_iteration = 10

print("setup CGLS")
cgok = CGLS()
cgok.set_up(X_init, Kok, B )
cgok.max_iteration = 10
# #
#out.__isub__(B)
#out2 = K.adjoint(out)

#(2.0*self.c)*self.A.adjoint( self.A.direct(x) - self.b )


for _ in gd:
    print ("GradientDescent iteration {} {}".format(gd.iteration, gd.get_last_loss()))
#gd2.run(5,verbose=True)
print("CGLS block lambda big")
cg.run(10, lambda it,val: print ("CGLS big iteration {} objective {}".format(it,val)))

print("CGLS standard")
cgs.run(10, lambda it,val: print ("CGLS standard iteration {} objective {}".format(it,val)))

print("CGLS block lambda small")
cgsmall.run(10, lambda it,val: print ("CGLS small iteration {} objective {}".format(it,val)))
print("CGLS block lambdaok")
cgok.run(10, verbose=True)
# #    for _ in cg:
#    print ("iteration {} {}".format(cg.iteration, cg.get_current_loss()))
# #
# #    fig = plt.figure()
# #    plt.imshow(cg.get_output().get_item(0,0).subset(vertical=0).as_array())
# #    plt.title('Composite CGLS')
# #    plt.show()
# #
# #    for _ in cgs:
#    print ("iteration {} {}".format(cgs.iteration, cgs.get_current_loss()))
# #
Phantom = ImageData(phantom_tm)

theslice=40

fig = plt.figure()
plt.subplot(2,3,1)
plt.imshow(numpy.flip(Phantom.subset(vertical=theslice).as_array(),axis=0), cmap='gray')
plt.clim(0,0.7)
plt.title('Simulated Phantom')
plt.subplot(2,3,2)
plt.imshow(gd.get_output().subset(vertical=theslice).as_array(), cmap='gray')
plt.clim(0,0.7)
plt.title('Simple Gradient Descent')
plt.subplot(2,3,3)
plt.imshow(cgs.get_output().subset(vertical=theslice).as_array(), cmap='gray')
plt.clim(0,0.7)
plt.title('Simple CGLS')
plt.subplot(2,3,5)
plt.imshow(cg.get_output().get_item(0).subset(vertical=theslice).as_array(), cmap='gray')
plt.clim(0,0.7)
plt.title('Composite CGLS\nbig lambda')
plt.subplot(2,3,6)
plt.imshow(cgsmall.get_output().get_item(0).subset(vertical=theslice).as_array(), cmap='gray')
plt.clim(0,0.7)
plt.title('Composite CGLS\nsmall lambda')
plt.subplot(2,3,4)
plt.imshow(cgok.get_output().get_item(0).subset(vertical=theslice).as_array(), cmap='gray')
plt.clim(0,0.7)
plt.title('Composite CGLS\nok lambda')
plt.show()


#Ibig = 7e1 * TomoIdentity(geometry=ig)
#Kbig = BlockOperator(A, Ibig, shape=(2,1))
#cg2 = CGLS(x_init=X_init, operator=Kbig, data=B)
#cg2.max_iteration = 10
#cg2.run(10, verbose=True)
