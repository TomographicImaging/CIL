# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 14:26:21 2018

@author: ofn77899
"""

from ccpi.framework import ImageData , AcquisitionData, ImageGeometry, AcquisitionGeometry
from ccpi.optimisation.algs import FISTA, FBPD, CGLS
from ccpi.optimisation.funcs import Norm2sq, Norm1
from ccpi.reconstruction.ccpiops import CCPiProjectorSimple
from ccpi.reconstruction.parallelbeam import alg as pbalg
from ccpi.reconstruction.processors import CCPiForwardProjector, CCPiBackwardProjector , \
Normalizer , CenterOfRotationFinder , AcquisitionDataPadder

from ccpi.io.reader import NexusReader

import numpy
import matplotlib.pyplot as plt

import os
import pickle


def avg_img(image):
    shape = list(numpy.shape(image))
    l = shape.pop(0)
    avg = numpy.zeros(shape)
    for i in range(l):
        avg += image[i] / l
    return avg
    

reader = NexusReader(os.path.join(".." ,".." ,".." , "data" , "24737_fd.nxs" ))

dims = reader.get_projection_dimensions()
print (dims)

flat = avg_img(reader.load_flat())
dark = avg_img(reader.load_dark())

norm = Normalizer(flat_field=flat, dark_field=dark)

norm.set_input(reader.get_acquisition_data())

cor = CenterOfRotationFinder()
cor.set_input(norm.get_output())
center_of_rotation = cor.get_output()
voxel_per_pixel = 1

padder = AcquisitionDataPadder(center_of_rotation=center_of_rotation)
padder.set_input(norm.get_output())
padded_data = padder.get_output()

pg = padded_data.geometry
geoms = pbalg.pb_setup_geometry_from_acquisition(padded_data.as_array(),
                                                pg.angles,
                                                center_of_rotation,
                                                voxel_per_pixel )
vg = ImageGeometry(voxel_num_x=geoms['output_volume_x'],
                   voxel_num_y=geoms['output_volume_y'], 
                   voxel_num_z=geoms['output_volume_z'])
#data = numpy.reshape(reader.getAcquisitionData())
print ("define projector")
Cop = CCPiProjectorSimple(vg, pg)
# Create least squares object instance with projector and data.
print ("Create least squares object instance with projector and data.")
f = Norm2sq(Cop,padded_data,c=0.5)
print ("Initial guess")
# Initial guess
x_init = ImageData(geometry=vg, dimension_labels=['horizontal_x','horizontal_y','vertical'])
#invL = 0.5
#g = f.grad(x_init)
#print (g)
#u = x_init - invL*f.grad(x_init)
        
#%%
print ("run FISTA")
# Run FISTA for least squares without regularization
opt = {'tol': 1e-4, 'iter': 10}
x_fista0, it0, timing0, criter0 = FISTA(x_init, f, None, opt=opt)
pickle.dump(x_fista0, open("fista0.pkl", "wb"))


plt.imshow(x_fista0.subset(horizontal_x=80).array)
plt.title('FISTA0')
#plt.show()

# Now least squares plus 1-norm regularization
lam = 0.1
g0 = Norm1(lam)

# Run FISTA for least squares plus 1-norm function.
x_fista1, it1, timing1, criter1 = FISTA(x_init, f, g0,opt=opt)
pickle.dump(x_fista1, open("fista1.pkl", "wb"))

plt.imshow(x_fista0.subset(horizontal_x=80).array)
plt.title('FISTA1')
#plt.show()

plt.semilogy(criter1)
#plt.show()

# Run FBPD=Forward Backward Primal Dual method on least squares plus 1-norm
x_fbpd1, it_fbpd1, timing_fbpd1, criter_fbpd1 = FBPD(x_init,None,f,g0,opt=opt)
pickle.dump(x_fbpd1, open("fbpd1.pkl", "wb"))

plt.imshow(x_fbpd1.subset(horizontal_x=80).array)
plt.title('FBPD1')
#plt.show()

plt.semilogy(criter_fbpd1)
#plt.show()

# Now FBPD for least squares plus TV
#lamtv = 1
#gtv = TV2D(lamtv)

#x_fbpdtv, it_fbpdtv, timing_fbpdtv, criter_fbpdtv = FBPD(x_init,None,f,gtv,opt=opt)

#plt.imshow(x_fbpdtv.subset(vertical=0).array)
#plt.show()

#plt.semilogy(criter_fbpdtv)
#plt.show()  


# Run CGLS, which should agree with the FISTA0
x_CGLS, it_CGLS, timing_CGLS, criter_CGLS = CGLS(x_init, Cop, padded_data, opt=opt)
pickle.dump(x_CGLS, open("cgls.pkl", "wb"))
plt.imshow(x_CGLS.subset(horizontal_x=80).array)
plt.title('CGLS')
plt.title('CGLS recon, compare FISTA0')
#plt.show()

plt.semilogy(criter_CGLS)
plt.title('CGLS criterion')
#plt.show()


cols = 4
rows = 1
current = 1
fig = plt.figure()
# projections row

current = current 
a=fig.add_subplot(rows,cols,current)
a.set_title('FISTA0')
imgplot = plt.imshow(x_fista0.subset(horizontal_x=80).as_array())

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FISTA1')
imgplot = plt.imshow(x_fista1.subset(horizontal_x=80).as_array())

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FBPD1')
imgplot = plt.imshow(x_fbpd1.subset(horizontal_x=80).as_array())

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('CGLS')
imgplot = plt.imshow(x_CGLS.subset(horizontal_x=80).as_array())

plt.show()


#%%
fig = plt.figure()
# projections row
b=fig.add_subplot(1,1,1)
b.set_title('criteria')
imgplot = plt.loglog(criter0 , label='FISTA0')
imgplot = plt.loglog(criter1 , label='FISTA1')
imgplot = plt.loglog(criter_fbpd1, label='FBPD1')
imgplot = plt.loglog(criter_CGLS, label='CGLS')
#imgplot = plt.loglog(criter_fbpdtv, label='FBPD TV')
b.legend(loc='right')
plt.show()