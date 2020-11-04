from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ccpi.processors.CofR_xcorr import CofR_xcorr

class CentreOfRotationCorrector(object):

    @staticmethod
    def xcorr(slice_index='centre', projection_index=0, ang_tol=0.1):
        """
        Performs a single line cross-correlation with projections at 180 degees seperation.
        Only suitable for parallel-beam geometry
        """
        proccessor = CofR_xcorr(slice_index, projection_index, ang_tol)
        return proccessor
