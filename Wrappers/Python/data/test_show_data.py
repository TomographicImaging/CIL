#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 20:43:48 2019

@author: evangelos
"""

from ccpi.data import camera, boat, peppers
import matplotlib.pyplot as plt


d = camera(size=(256,256))

plt.imshow(d.as_array())
plt.colorbar()
plt.show()

d1 = boat(size=(256,256))

plt.imshow(d1.as_array())
plt.colorbar()
plt.show()


d2 = peppers(size=(256,256))

plt.imshow(d2.as_array())
plt.colorbar()
plt.show()