#%%
import numpy as np
from scipy.fft import fftfreq
import matplotlib.pyplot as plt

from skimage.transform.radon_transform import _get_fourier_filter



#%%
fft_order = 8

filter_length = 2**fft_order

#in cycles/pixel
freq = fftfreq(filter_length)

ind_sorted = np.argsort(freq)
ramp = abs(freq)

# must be less than 0.5
nyquist_limit = 0.5
cut_off_frequency = nyquist_limit

ramp[ramp > nyquist_limit] = 0
ramp[ramp > cut_off_frequency] = 0

plt.plot(freq[ind_sorted],ramp[ind_sorted], label='ramp')


filter_array1_ram_lak = ramp.copy()
filter_array1_ram_lak[filter_array1_ram_lak > cut_off_frequency] = 0
plt.plot(freq[ind_sorted], filter_array1_ram_lak[ind_sorted], label='ram-lak')

filter_array1_shep = ramp * np.sinc(freq)
plt.plot(freq[ind_sorted], filter_array1_shep[ind_sorted], label='shepp-logan')

filter_array1_cos = ramp * np.cos(freq*np.pi)
plt.plot(freq[ind_sorted], filter_array1_cos[ind_sorted], label='cosine')

filter_array1_hamming = ramp * (0.54 + 0.46 * np.cos(2*freq*np.pi))
plt.plot(freq[ind_sorted], filter_array1_hamming[ind_sorted], label='hamming')

filter_array1_hann = ramp * (0.5 + 0.5 * np.cos(2*freq*np.pi))
plt.plot(freq[ind_sorted], filter_array1_hann[ind_sorted], label='hann')

filters = [ramp, filter_array1_shep, filter_array1_cos, filter_array1_hamming, filter_array1_hann]
filter_name = ['ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann']

plt.ylim(0,nyquist_limit)
plt.xlim(-nyquist_limit,nyquist_limit)
plt.xlabel('Frequency (cycles/pixel)')
plt.ylabel('Magnitude')

plt.legend()



#%% skimage
import matplotlib.pyplot as plt

filters = ['ramp', 'shepp-logan', 'cosine', 'hamming', 'hann']

freq = fftfreq(2**8,0.5)
ind_sorted = np.argsort(freq)

for ix, f in enumerate(filters):
    response = _get_fourier_filter(2**fft_order, f)
    plt.plot(freq[ind_sorted],response[ind_sorted], label=f)
plt.legend()



#%%

#units!

fft_order = 8
filter_length = 2**fft_order

# in cycles/pixel
freq = fftfreq(filter_length)
# in pi radians
freq*=2

ind_sorted = np.argsort(freq)
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
# %%
