# This script demonstrates how to load IMAT fits data
# into the CIL optimisation framework and run reconstruction methods.
#
# Demo to reconstruct energy-discretized channels of IMAT data

# needs dxchange: conda install -c conda-forge dxchange
# needs astropy: conda install astropy


# All third-party imports.
import numpy as np
import matplotlib.pyplot as plt
from dxchange.reader import read_fits
from astropy.io import fits

# All own imports.
from ccpi.framework import AcquisitionData, AcquisitionGeometry, ImageGeometry, ImageData, DataContainer
from ccpi.astra.ops import AstraProjectorSimple, AstraProjector3DSimple
from ccpi.optimisation.algs import CGLS, FISTA
from ccpi.optimisation.funcs import Norm2sq, Norm1
from ccpi.plugins.regularisers import FGP_TV

# set main parameters here
n = 512 
totalAngles = 250 # total number of projection angles
# spectral discretization parameter
num_average = 145 # channel discretization frequency (total number of averaged channels)
numChannels = 2970 # 2970
totChannels = round(numChannels/num_average) # the resulting number of channels
Projections_stack = np.zeros((num_average,n,n),dtype='uint16')
ProjAngleChannels = np.zeros((totalAngles,totChannels,n,n),dtype='float32')

#########################################################################
print ("Loading the data...")
MainPath = '/media/algol/HD-LXU3/DATA_DANIIL/' # path to data
pathname0 = '{!s}{!s}'.format(MainPath,'PSI_DATA/DATA/Sample/')
counterFileName = 4675
# A main loop over all available angles 
for ll in range(0,totalAngles,1):
    pathnameData = '{!s}{!s}{!s}/'.format(pathname0,'angle',str(ll))
    filenameCurr = '{!s}{!s}{!s}'.format('IMAT0000',str(counterFileName),'_Tomo_test_000_')
    counterT = 0
    # loop over reduced channels (discretized)
    for i in range(0,totChannels,1):
        sumCount = 0
        # loop over actual channels to obtain averaged one
        for j in range(0,num_average,1):
            if counterT < 10:
                outfile = '{!s}{!s}{!s}{!s}.fits'.format(pathnameData,filenameCurr,'0000',str(counterT))
            if ((counterT  >= 10) & (counterT < 100)):
                outfile = '{!s}{!s}{!s}{!s}.fits'.format(pathnameData,filenameCurr,'000',str(counterT))
            if ((counterT  >= 100) & (counterT < 1000)):
                outfile = '{!s}{!s}{!s}{!s}.fits'.format(pathnameData,filenameCurr,'00',str(counterT))
            if ((counterT  >= 1000) & (counterT < 10000)):
                outfile = '{!s}{!s}{!s}{!s}.fits'.format(pathnameData,filenameCurr,'0',str(counterT))
            try:
                Projections_stack[j,:,:] = read_fits(outfile)
            except:
                print("Fits is corrupted, skipping no.", counterT)
                sumCount -= 1
            counterT += 1
            sumCount += 1
        AverageProj=np.sum(Projections_stack,axis=0)/sumCount # averaged projection over "num_average" channels
        ProjAngleChannels[ll,i,:,:] = AverageProj
    print("Angle is processed", ll)
    counterFileName += 1
#########################################################################

flat1 = read_fits('{!s}{!s}{!s}'.format(MainPath,'PSI_DATA/DATA/','OpenBeam_aft1/IMAT00004932_Tomo_test_000_SummedImg.fits'))
nonzero = flat1 > 0
# Apply flat field and take negative log
for ll in range(0,totalAngles,1):
    for i in range(0,totChannels,1):
        ProjAngleChannels[ll,i,nonzero] = ProjAngleChannels[ll,i,nonzero]/flat1[nonzero]

eqzero = ProjAngleChannels == 0
ProjAngleChannels[eqzero] = 1
ProjAngleChannels_NormLog = -np.log(ProjAngleChannels) # normalised and neg-log data

# extact sinogram over energy channels
selectedVertical_slice = 256
sino_all_channels = ProjAngleChannels_NormLog[:,:,:,selectedVertical_slice]
# Set angles to use
angles = np.linspace(-np.pi,np.pi,totalAngles,endpoint=False)

# set the geometry
ig = ImageGeometry(n,n)
ag = AcquisitionGeometry('parallel',
                             '2D',
                             angles,
                             n,1)
Aop = AstraProjectorSimple(ig, ag, 'gpu')


# loop to reconstruct energy channels
REC_chan = np.zeros((totChannels, n, n), 'float32')
for i in range(0,totChannels,1):
    sino_channel = sino_all_channels[:,i,:] # extract a sinogram for i-th channel

    print ("Initial guess")
    x_init = ImageData(geometry=ig)
    
    # Create least squares object instance with projector and data.
    print ("Create least squares object instance with projector and data.")
    f = Norm2sq(Aop,DataContainer(sino_channel),c=0.5)
    
    print ("Run FISTA-TV for least squares")
    lamtv = 10
    opt = {'tol': 1e-4, 'iter': 200}
    g_fgp = FGP_TV(lambdaReg = lamtv,
                 iterationsTV=50,
                 tolerance=1e-6,
                 methodTV=0,
                 nonnegativity=0,
                 printing=0,
                 device='gpu')
    
    x_fista_fgp, it1, timing1, criter_fgp = FISTA(x_init, f, g_fgp, opt)
    REC_chan[i,:,:] = x_fista_fgp.array
    """
    plt.figure()
    plt.subplot(121)
    plt.imshow(x_fista_fgp.array, vmin=0, vmax=0.05)
    plt.title('FISTA FGP TV')
    plt.subplot(122)
    plt.semilogy(criter_fgp)
    plt.show()
    """
    """
    print ("Run CGLS for least squares")
    opt = {'tol': 1e-4, 'iter': 20}
    x_init = ImageData(geometry=ig)
    x_CGLS, it_CGLS, timing_CGLS, criter_CGLS = CGLS(x_init, Aop, DataContainer(sino_channel), opt=opt)
    
    plt.figure()
    plt.imshow(x_CGLS.array,vmin=0, vmax=0.05)
    plt.title('CGLS')
    plt.show()
    """
# Saving images into fits using astrapy if required
add_val = np.min(REC_chan[:])
REC_chan += abs(add_val)
REC_chan = REC_chan/np.max(REC_chan[:])*65535
counter = 0
filename = 'FISTA_TV_imat_slice'
for i in range(totChannels):
    outfile = '{!s}_{!s}_{!s}.fits'.format(filename,str(selectedVertical_slice),str(counter))
    hdu = fits.PrimaryHDU(np.uint16(REC_chan[i,:,:]))
    hdu.writeto(outfile, overwrite=True)
    counter += 1