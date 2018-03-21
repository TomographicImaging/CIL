# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 14:26:21 2018

@author: ofn77899
"""

from ccpi.framework import ImageData , AcquisitionData, ImageGeometry, AcquisitionGeometry
from ccpi.reconstruction.algs import FISTA, FBPD, CGLS
from ccpi.reconstruction.funcs import Norm2sq, Norm1
from ccpi.reconstruction.ops import CCPiProjectorSimple
from ccpi.reconstruction.parallelbeam import alg as pbalg
from ccpi.processors import CCPiForwardProjector, CCPiBackwardProjector , \
Normalizer , CenterOfRotationFinder , AcquisitionDataPadder

from ccpi.io.reader import NexusReader

import numpy
import matplotlib.pyplot as plt

import os

def add_dimension(data, fill_with, axis, start=True):
    delta = data.shape[data.get_dimension_axis(axis)] - fill_with.shape[fill_with.get_dimension_axis(axis)] 
    command = 'data.array['
    i = 0
    for k,v in data.dimension_labels.items():
        if axis == v:
            if start:
                command = command + str(delta) + ":"
            else:
                l = data.get_dimension_size(axis) - delta
                command = command + "0:" + str(l)
        else:
            command = command + ":"
            
        if i < data.number_of_dimensions -1:
            command = command + ','
        i += 1
    command = command + "] = fill_with.array[:]" 
    #print (command)
    exec(command)
    #return command


def avg_img(image):
    shape = list(numpy.shape(image))
    l = shape.pop(0)
    avg = numpy.zeros(shape)
    for i in range(l):
        avg += image[i] / l
    return avg
    
def setupCCPiGeometries(voxel_num_x, voxel_num_y, voxel_num_z, angles, counter):
    vg = ImageGeometry(voxel_num_x=voxel_num_x,voxel_num_y=voxel_num_y, voxel_num_z=voxel_num_z)
    Phantom_ccpi = ImageData(geometry=vg,
                        dimension_labels=['horizontal_x','horizontal_y','vertical'])
    #.subset(['horizontal_x','horizontal_y','vertical'])
    # ask the ccpi code what dimensions it would like
        
    voxel_per_pixel = 1
    geoms = pbalg.pb_setup_geometry_from_image(Phantom_ccpi.as_array(),
                                                angles,
                                                voxel_per_pixel )
    
    pg = AcquisitionGeometry('parallel',
                              '3D',
                              angles,
                              geoms['n_h'], 1,
                              geoms['n_v'], 1 #2D in 3D is a slice 1 pixel thick
                              )
    
    center_of_rotation = Phantom_ccpi.get_dimension_size('horizontal_x') / 2
    ad = AcquisitionData(geometry=pg,dimension_labels=['angle','vertical','horizontal'])
    geoms_i = pbalg.pb_setup_geometry_from_acquisition(ad.as_array(),
                                                angles,
                                                center_of_rotation,
                                                voxel_per_pixel )

    #print (counter)
    counter+=1
    #print (geoms , geoms_i)
    if counter < 4:
        if (not ( geoms_i == geoms )):
            print ("not equal and {0}".format(counter))
            X = max(geoms['output_volume_x'], geoms_i['output_volume_x'])
            Y = max(geoms['output_volume_y'], geoms_i['output_volume_y'])
            Z = max(geoms['output_volume_z'], geoms_i['output_volume_z'])
            return setupCCPiGeometries(X,Y,Z,angles, counter)
        else:
            print ("return geoms {0}".format(geoms))
            return geoms
    else:
        print ("return geoms_i {0}".format(geoms_i))
        return geoms_i

reader = NexusReader(os.path.join(".." ,".." ,".." , "data" , "24737_fd.nxs" ))

dims = reader.get_projection_dimensions()
print (dims)

flat = avg_img(reader.load_flat())
dark = avg_img(reader.load_dark())

data = reader.getAcquisitionData()
norm = Normalizer(flat_field=flat, dark_field=dark)

norm.set_input(reader.getAcquisitionData())

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

plt.imshow(x_fista0.subset(horizontal_x=80).array)
plt.title('FISTA0')
plt.show()

# Now least squares plus 1-norm regularization
lam = 0.1
g0 = Norm1(lam)

# Run FISTA for least squares plus 1-norm function.
x_fista1, it1, timing1, criter1 = FISTA(x_init, f, g0,opt=opt)

plt.imshow(x_fista0.subset(horizontal_x=80).array)
plt.title('FISTA1')
plt.show()

plt.semilogy(criter1)
plt.show()

# Run FBPD=Forward Backward Primal Dual method on least squares plus 1-norm
x_fbpd1, it_fbpd1, timing_fbpd1, criter_fbpd1 = FBPD(x_init,None,f,g0,opt=opt)

plt.imshow(x_fbpd1.subset(horizontal_x=80).array)
plt.title('FBPD1')
plt.show()

plt.semilogy(criter_fbpd1)
plt.show()

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

plt.imshow(x_CGLS.subset(horizontal_x=80).array)
plt.title('CGLS')
plt.title('CGLS recon, compare FISTA0')
plt.show()

plt.semilogy(criter_CGLS)
plt.title('CGLS criterion')
plt.show()