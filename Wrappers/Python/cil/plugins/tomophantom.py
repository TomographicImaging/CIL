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

from cil.framework import ImageGeometry, AcquisitionGeometry
from cil.framework import ImageData, AcquisitionData
import tomophantom
from tomophantom import TomoP2D, TomoP3D
import os

path = os.path.dirname(tomophantom.__file__)
path_library2D = os.path.join(path, "Phantom2DLibrary.dat")
path_library3D = os.path.join(path, "Phantom3DLibrary.dat")

# # Define image geometry.
# ig = ImageGeometry(voxel_num_x = N, voxel_num_y = N, 
#                    voxel_size_x = 0.1,
#                    voxel_size_y = 0.1)
# im_data = ig.allocate()
# im_data.fill(phantom)

# show(im_data, title = 'TomoPhantom', cmap = 'inferno')
def name_to_model_number(model, dims=2):
    if model == 'shepp-logan':
        if dims == 2:
            return 1
        else:
            return 13
    else:
        return model

def get_ImageData(model, geometry):
    '''Returns an ImageData relative to geometry with the model model from tomophantom'''
    ig = geometry.copy()
    num_dims = len(ig.dimension_labels)
    
    if ImageGeometry.CHANNEL in ig.dimension_labels:
        if num_dims == 4:
            # 3D+time for tomophantom
            pass
        elif num_dims == 3:
            # 2D+time for tomophantom
            pass
        else:
            raise ValueError('Wrong ImageGeometry')
    else:
        if num_dims == 3:
            # 3D
            if not ((ig.shape[0] == ig.shape[1]) and (ig.shape[1] == ig.shape[2])) :
                raise ValueError('Can only handle cubic ImageData, got shape'.format(ig.shape))
            N = ig.shape[0]
            num_model = name_to_model_number(model)
            phantom_arr = TomoP3D.Model(num_model, N, path_library3D)
        elif num_dims == 2:
            # 2D
            if ig.shape[0] != ig.shape[1]:
                raise ValueError('Can only handle square ImageData, got shape'.format(ig.shape))
            N = ig.shape[0]
            num_model = name_to_model_number(model)
            phantom_arr = TomoP2D.Model(num_model, N, path_library2D)
        else:
            raise ValueError('Wrong ImageGeometry')

        im_data = ImageData(phantom_arr, geometry=ig.copy(), suppress_warning=True)
        return im_data
    pass


def get_AcquisitionData(model, geometry):
    '''Returns an AcquisitionData relative to geometry with the model model from tomophantom
    
    Notice this is only for circular paralell beam scans.
    '''
    pass

if __name__ == '__main__':
    # ig = ImageGeometry(512,512,512)

    # phantom = tomophantom.get_ImageData(model=12, ig)
    # shepp_logan = tomophantom.get_ImageData(model='shepp-logan', ig)

    # # only for simple parallel beam
    # ag = AcquisitionGeometry.createParallel2D()
    # # finish set up geometry
    # ag.set_panel((80,80))
    # ag.set_angles(angles, angle_unit='degree')

    # ad = tomophantom.get_AcquisitionData(model='shepp-logan', ag)
    pass