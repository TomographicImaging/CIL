import os


class BaseReader(object):
    def __init__(self, **kwargs):
        '''
        Constructor:
        
        Input:
            
            file_name:      full path to file

            slice_roi: dictionary with roi to load 
                {'angle': (start, end, step), 
                 'horizontal': (start, end, step), 
                 'vertical': (start, end, step)}
                or 
                {'axis_0': (start, end, step), 
                 'axis_1': (start, end, step), 
                 'axis_2': (start, end, step)}
                in the case of a TIFF stack.
                'step' defines standard numpy slicing.
                -1 is a shortcut to load all elements along an axis.
                Start and end can be specified as None which is equivalent 
                to start = 0 and end = load everything to the end, respectively.
                Start and end also can be negative.

            binning_roi: dictionary with roi to load 
                {'angle': (start, end, step), 
                 'horizontal': (start, end, step), 
                 'vertical': (start, end, step)}
                or 
                {'axis_0': (start, end, step), 
                 'axis_1': (start, end, step), 
                 'axis_2': (start, end, step)}
                in the case of a TIFF stack.
                'step' number of pixels is binned together, values of resulting binned
                pixels are calculated as average. 
                -1 is a shortcut to load all elements along an axis.
                Start and end can be specified as None which is equivalent 
                to start = 0 and end = load everything to the end, respectively.
                Start and end also can be negative.
        '''

    def set_up(self, file_name=None, slice_roi=None, binning_roi=None, **kwargs):
        
        self.file_name = os.path.abspath(file_name)
            
        if self.file_name == None:
            raise Exception('Path to nexus file is required.')
        
        # check if file exists
        if not(os.path.isfile(self.file_name)):
            raise Exception('File\n {}\n does not exist.'.format(self.file_name))

        self.set_slice(slice_roi)
        self.set_binning(binning_roi)  
        

    def set_slice(self, roi):
        '''
        roi: dictionary with roi to load 
            {'angle': (start, end, step), 
                'horizontal': (start, end, step), 
                'vertical': (start, end, step)}
            or 
            {'axis_0': (start, end, step), 
                'axis_1': (start, end, step), 
                'axis_2': (start, end, step)}
            in the case of a TIFF stack.
            'step' defines standard numpy slicing.
            -1 is a shortcut to load all elements along an axis.
            Start and end can be specified as None which is equivalent 
            to start = 0 and end = load everything to the end, respectively.
            Start and end also can be negative.
        '''
        self.slice_roi = roi
        # in subclasses would have check for labels
        

    def set_binning(self, roi):
        '''
        roi: dictionary with roi to load 
            {'angle': (start, end, step), 
                'horizontal': (start, end, step), 
                'vertical': (start, end, step)}
            or 
            {'axis_0': (start, end, step), 
                'axis_1': (start, end, step), 
                'axis_2': (start, end, step)}
            in the case of a TIFF stack.
            'step' number of pixels is binned together, values of resulting binned
            pixels are calculated as average. 
        '''
        self.binning_roi = roi
        # in subclasses would have check for labels
        

    def read(self):
        '''
        Returns either an ImageData or Acquisition Data containing the uncompressed data as numpy.float32
        '''
        raise NotImplementedError("read method is not implemented for BaseReader.")


    # TODO:
    # should we have read_as, read_as_original, etc. here?
    # possibly rename slice_roi, binning_roi -> slicing and binning or bin and slice ?