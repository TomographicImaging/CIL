from ccpi.framework import *
from ccpi.io import NEXUSDataReader
import os
from PIL import Image

class TIFFWriter(object):
    
    def __init__(self,
                 **kwargs):
        
        self.data_container = kwargs.get('data_container', None)
        self.file_name = kwargs.get('file_name', None)
        
        if ((self.data_container is not None) and (self.file_name is not None)):
            self.set_up(data_container = self.data_container,
                        file_name = self.file_name)
        
    def set_up(self,
               data_container = None,
               file_name = None):
        
        self.data_container = data_container
        self.file_name = file_name
        
        if not ((isinstance(self.data_container, ImageData)) or 
                (isinstance(self.data_container, AcquisitionData))):
            raise Exception('Writer supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData')

    def write_file(self):
        '''alias of write'''
        return self.write()
    
    def write(self):
        ndim = len(self.data_container.shape)
        if ndim == 2:
            # save single slice
            with open(self.file_name, 'wb') as f:
                Image.fromarray(self.data_container.as_array()).save(f, 'tiff')
        elif ndim == 3:
            for sliceno in range(self.data_container.shape[0]):
                print ("Saving {}/{}".format(sliceno, self.data_container.shape[0]))
                # save single slice
                # pattern = self.file_name.split('.')
                dimension = self.data_container.dimension_labels[0]
                fname = "{}_idx_{:04}.tiff".format(self.file_name.strip(".tiff"), sliceno)
                with open(fname, 'wb') as f:
                    Image.fromarray(self.data_container.as_array()[sliceno]).save(f, 'tiff')
        elif ndim == 4:
            for sliceno1 in range(self.data_container.shape[0]):
                # save single slice
                # pattern = self.file_name.split('.')
                dimension = [ self.data_container.dimension_labels[0] ]
                for sliceno2 in range(self.data_container.shape[1]):
                    idx = self.data_container.shape[0] * sliceno2 + sliceno1 
                    fname = "{}_{}_{}_idx_{:04}.tiff".format(self.file_name.strip(".tiff"), 
                        self.data_container.shape[0], self.data_container.shape[1], idx)
                    with open(fname, 'wb') as f:
                        Image.fromarray(self.data_container.as_array()[sliceno1][sliceno2]).save(f, 'tiff')
        else:
            raise ValueError('Cannot handle more than 4 dimensions')

working_directory = os.path.abspath('/mnt/data/CCPi/Dataset/EM/Cryo_Sample_Data_1/')
os.chdir(working_directory)


reader = NEXUSDataReader()
i = 50
fname = "./Block_CGLS_scale_1_gamma1_it_{:02}.nxs".format(i)
reader.set_up(nexus_file=fname)

writer = TIFFWriter(data_container=reader.load_data().subset(dimensions=['horizontal_y', 'horizontal_x', 'vertical']), file_name="cil/Block_CGLS_scale_1_gamma1_it_50.tiff")
writer.write()