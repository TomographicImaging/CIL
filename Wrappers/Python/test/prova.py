from cil.framework import ImageGeometry, ImageData
from cil.utilities import dataexample 
import numpy as np 
import cupy as cp


ig = ImageGeometry(100,100)

camera = dataexample.CAMERA.get((100,100))
# cp.copyto(garr, camera.as_array())
garr = cp.array(camera.as_array())


datac = ImageData(geometry=camera.geometry, backend='numpy')
datag = ImageData(geometry=camera.geometry, backend='cupy')


# print ("python copying")
# for i in range(datag.shape[0]):
#     for j in range(datag.shape[1]):
#         datag.array[i,j] = camera.as_array()[i,j]
# print ("done")
datag.fill(camera)
# garr = datag.array
# print (type(garr))



from cil.utilities.display import show2D
# show2D([garr.get(), camera])
show2D([datag.as_array().get(), camera])

