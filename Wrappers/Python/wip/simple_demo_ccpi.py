#import sys
#sys.path.append("..")

from ccpi.framework import ImageData , ImageGeometry, AcquisitionGeometry
from ccpi.reconstruction.algs import FISTA, FBPD, CGLS
from ccpi.reconstruction.funcs import Norm2sq, Norm1 , TV2D
from ccpi.astra.astra_ops import AstraProjectorSimple
from ccpi.reconstruction.ops import CCPiProjectorSimple
from ccpi.reconstruction.parallelbeam import alg as pbalg
from ccpi.processors import CCPiForwardProjector, CCPiBackwardProjector 

import numpy as np
import matplotlib.pyplot as plt

test_case = 1   # 1=parallel2D, 2=cone2D, 3=parallel3D

# Set up phantom
N = 128
vert = 8
# Set up measurement geometry
angles_num = 20; # angles number
det_w = 1.0
det_num = N
SourceOrig = 200
OrigDetec = 0

if test_case==1:
    angles = np.linspace(0,np.pi,angles_num,endpoint=False)
elif test_case==2:
    angles = np.linspace(0,2*np.pi,angles_num,endpoint=False)
elif test_case == 3:
    angles = np.linspace(0,np.pi,angles_num,endpoint=False)
else:
    NotImplemented


def setupCCPiGeometries(voxel_num_x, voxel_num_y, voxel_num_z, counter):
    vg = ImageGeometry(voxel_num_x=voxel_num_x,voxel_num_y=voxel_num_y, voxel_num_z=voxel_num_z)
    Phantom = ImageData(geometry=vg)#.subset(['horizontal_x','horizontal_y','vertical'])
    # ask the ccpi code what dimensions it would like
    Phantom_ccpi = Phantom.subset(['horizontal_x','horizontal_y','vertical'])
        
    voxel_per_pixel = 1
    geoms = pbalg.pb_setup_geometry_from_image(Phantom_ccpi.as_array(),
                                                angles,
                                                voxel_per_pixel )
    
    pg = AcquisitionGeometry('parallel',
                              '3D',
                              angles,
                              geoms['n_h'],det_w,
                              geoms['n_v'], det_w #2D in 3D is a slice 1 pixel thick
                              )
    
    center_of_rotation = Phantom_ccpi.get_dimension_size('horizontal_x') / 2
    geoms_i = pbalg.pb_setup_geometry_from_image(Phantom_ccpi.as_array(),
                                                angles,
                                                voxel_per_pixel )

    counter+=1
    if (not ( geoms_i == geoms )) and counter < 4:
        setupCCPiGeometries(geoms_i['output_volume_x'],geoms_i['output_volume_y'],
                            geoms_i['output_volume_z'],)
    else:
        return geoms_i

geoms = setupCCPiGeometries(N,N,vert,0)
vg = ImageGeometry(voxel_num_x=geoms['output_volume_x'],
                   voxel_num_y=geoms['output_volume_y'], 
                   voxel_num_z=geoms['output_volume_z'])
Phantom = ImageData(geometry=vg)#.subset(['horizontal_x','horizontal_y','vertical'])
    

#x = Phantom.as_array()
i = 0
while i < geoms['n_v']:
    #x = Phantom.subset(vertical=i, dimensions=['horizontal_x','horizontal_y']).array
    x = Phantom.subset(vertical=i).array
    x[round(N/4):round(3*N/4),round(N/4):round(3*N/4)] = 1.0
    x[round(N/8):round(7*N/8),round(3*N/8):round(5*N/8)] = 2.0
    Phantom.fill(x, vertical=i)
    i += 1

plt.imshow(Phantom.subset(vertical=0).as_array())
plt.show()



# Parallelbeam geometry test
if test_case==1:
    #Phantom_ccpi = Phantom.subset(['horizontal_x','horizontal_y','vertical'])
    #Phantom_ccpi.geometry = vg.clone()
    center_of_rotation = Phantom.get_dimension_size('horizontal_x') / 2
        
    pg = AcquisitionGeometry('parallel',
                          '3D',
                          angles,
                          geoms['n_h'],det_w,
                          geoms['n_v'], det_w #2D in 3D is a slice 1 pixel thick
                          )
elif test_case==2:
    raise NotImplemented('cone beam projector not yet available')
    pg = AcquisitionGeometry('cone',
                          '2D',
                          angles,
                          det_num,
                          det_w,
                          vert, det_w, #2D in 3D is a slice 1 pixel thick 
                          dist_source_center=SourceOrig, 
                          dist_center_detector=OrigDetec)

# ASTRA operator using volume and sinogram geometries
#Aop = AstraProjectorSimple(vg, pg, 'cpu')
Cop = CCPiProjectorSimple(vg, pg)

# Try forward and backprojection
b = Cop.direct(Phantom)
out2 = Cop.adjoint(b)

plt.imshow(b.subset( vertical=4).array)
plt.show()

plt.imshow(out2.subset( vertical=4).array)
plt.show()

# Create least squares object instance with projector and data.
f = Norm2sq(Cop,b,c=0.5)

# Initial guess
x_init = ImageData(geometry=vg)
#%%
# Run FISTA for least squares without regularization
x_fista0, it0, timing0, criter0 = FISTA(x_init, f, None)

plt.imshow(x_fista0.array[0])
plt.title('FISTA0')
plt.show()

# Now least squares plus 1-norm regularization
lam = 0.1
g0 = Norm1(lam)

# Run FISTA for least squares plus 1-norm function.
x_fista1, it1, timing1, criter1 = FISTA(x_init, f, g0)

plt.imshow(x_fista1.array[0])
plt.title('FISTA1')
plt.show()

plt.semilogy(criter1)
plt.show()

# Run FBPD=Forward Backward Primal Dual method on least squares plus 1-norm
opt = {'tol': 1e-4, 'iter': 10000}
x_fbpd1, it_fbpd1, timing_fbpd1, criter_fbpd1 = FBPD(x_init,None,f,g0,opt=opt)

plt.imshow(x_fbpd1.array[0])
plt.title('FBPD1')
plt.show()

plt.semilogy(criter_fbpd1)
plt.show()

# Now FBPD for least squares plus TV
lamtv = 1
gtv = TV2D(lamtv)

x_fbpdtv, it_fbpdtv, timing_fbpdtv, criter_fbpdtv = FBPD(x_init,None,f,gtv,opt=opt)

plt.imshow(x_fbpdtv.array[0])
plt.show()

plt.semilogy(criter_fbpdtv)
plt.show()


# Run CGLS, which should agree with the FISTA0
x_CGLS, it_CGLS, timing_CGLS, criter_CGLS = CGLS(Cop, b, 1000, x_init)

plt.imshow(x_CGLS.array)
plt.title('CGLS')
plt.title('CGLS recon, compare FISTA0')
plt.show()

plt.semilogy(criter_CGLS)
plt.title('CGLS criterion')
plt.show()


#%%

clims = (-0.5,2.5)
cols = 3
rows = 2
current = 1
fig = plt.figure()
# projections row
a=fig.add_subplot(rows,cols,current)
a.set_title('phantom {0}'.format(np.shape(Phantom.as_array())))

imgplot = plt.imshow(Phantom.as_array(),vmin=clims[0],vmax=clims[1])

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FISTA0')
imgplot = plt.imshow(x_fista0.as_array(),vmin=clims[0],vmax=clims[1])

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FISTA1')
imgplot = plt.imshow(x_fista1.as_array(),vmin=clims[0],vmax=clims[1])

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FBPD1')
imgplot = plt.imshow(x_fbpd1.as_array(),vmin=clims[0],vmax=clims[1])

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('CGLS')
imgplot = plt.imshow(x_CGLS.as_array(),vmin=clims[0],vmax=clims[1])

current = current + 1
a=fig.add_subplot(rows,cols,current)
a.set_title('FBPD TV')
imgplot = plt.imshow(x_fbpdtv.as_array(),vmin=clims[0],vmax=clims[1])

fig = plt.figure()
# projections row
b=fig.add_subplot(1,1,1)
b.set_title('criteria')
imgplot = plt.loglog(criter0 , label='FISTA0')
imgplot = plt.loglog(criter1 , label='FISTA1')
imgplot = plt.loglog(criter_fbpd1, label='FBPD1')
imgplot = plt.loglog(criter_CGLS, label='CGLS')
imgplot = plt.loglog(criter_fbpdtv, label='FBPD TV')
b.legend(loc='right')
plt.show()
#%%