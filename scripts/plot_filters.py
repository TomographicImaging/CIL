import numpy as np
from scipy.fft import fftfreq
import matplotlib.pyplot as plt
from cil.recon import FBP

fft_order = 8
filter_length = 2**fft_order
print(filter_length)

# in cycles/pixel
freq = fftfreq(filter_length)
#print(freq)
# in pi radians
freq*=2

ind_sorted = np.argsort(freq)
#print(ind_sorted)
ramp = abs(freq)

# must be less than 1.0 
nyquist_limit = 1.0
cut_off_frequency = nyquist_limit

ramp[ramp > nyquist_limit] = 0
ramp[ramp > cut_off_frequency] = 0

# plt.plot(freq[ind_sorted],ramp[ind_sorted], label='ramp')

filter_array1_ram_lak = ramp.copy()
filter_array1_ram_lak[filter_array1_ram_lak > cut_off_frequency] = 0
plt.plot(freq[ind_sorted], filter_array1_ram_lak[ind_sorted], label='ram-lak')

filter_array1_shep = ramp * np.sinc(freq/2)
plt.plot(freq[ind_sorted], filter_array1_shep[ind_sorted], label='shepp-logan')

filter_array1_cos = ramp * np.cos(freq*np.pi/2)
plt.plot(freq[ind_sorted], filter_array1_cos[ind_sorted], label='cosine')

filter_array1_hamming = ramp * (0.54 + 0.46 * np.cos(freq*np.pi))
plt.plot(freq[ind_sorted], filter_array1_hamming[ind_sorted], label='hamming')

filter_array1_hann = ramp * (0.5 + 0.5 * np.cos(freq*np.pi))
plt.plot(freq[ind_sorted], filter_array1_hann[ind_sorted], label='hann')

filters = [ramp, filter_array1_shep, filter_array1_cos, filter_array1_hamming, filter_array1_hann]
filter_name = ['ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann']

plt.ylim(0,nyquist_limit)
plt.xlim(-nyquist_limit,nyquist_limit)
plt.xlabel('Frequency (rads/pixel)')
plt.ylabel('Magnitude')


theta = np.linspace(-1, 1, 9, True)

plt.xticks(theta, ['-π', '-3π/4', '-π/2', '-π/4', '0', 'π/4', 'π/2', '3π/4', 'π'])

plt.legend()
plt.show()

instance = FBP.GenericFilteredBackProjection(input, image_geometry=None, filter='ram-lak', backend='tigre')
instance.get_filter_array()
