#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 23:09:29 2018

@author: evangelos
"""

import numpy as np
from scipy import signal
sig = np.repeat([0., 1., 0.], 100)
win = signal.hann(50)
filtered = signal.convolve(sig, win, mode='same') / sum(win)

#%%

import matplotlib.pyplot as plt
fig, (ax_orig, ax_win, ax_filt) = plt.subplots(3, 1, sharex=True)
ax_orig.plot(sig)
ax_orig.set_title('Original pulse')
ax_orig.margins(0, 0.1)
ax_win.plot(win)
ax_win.set_title('Filter impulse response')
ax_win.margins(0, 0.1)
ax_filt.plot(filtered)
ax_filt.set_title('Filtered signal')
ax_filt.margins(0, 0.1)
fig.tight_layout()
fig.show()

#%%

from cvxpy import *

N = 10
x1_denoise = Variable((N,N))

from scipy import signal
from scipy import misc
#ascent = misc.ascent()
#scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],[-10+0j, 0+ 0j, +10 +0j],[ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy
#grad = Variable(signal.convolve2d(ascent, scharr, boundary='symm', mode='same'))


#import matplotlib.pyplot as plt
#fig, (ax_orig, ax_mag, ax_ang) = plt.subplots(3, 1, figsize=(6, 15))
#ax_orig.imshow(ascent, cmap='gray')
#ax_orig.set_title('Original')
#ax_orig.set_axis_off()
#ax_mag.imshow(np.absolute(grad), cmap='gray')
#ax_mag.set_title('Gradient magnitude')
#ax_mag.set_axis_off()
#ax_ang.imshow(np.angle(grad), cmap='hsv') # hsv is cyclic, like angles
#ax_ang.set_title('Gradient orientation')
#ax_ang.set_axis_off()
#fig.show()

plt.imshow(np.abs(grad))
plt.show()