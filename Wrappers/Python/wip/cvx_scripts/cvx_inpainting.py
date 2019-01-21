#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 18:41:05 2019

@author: evangelos
"""
import numpy as np
from skimage.util import random_noise
from skimage import data, transform, color
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from cvx_functions import *
from cvxpy import *

cat = rgb2gray(data.chelsea())
original = transform.resize(cat,[128,128], mode = 'reflect', anti_aliasing=True)

N, M = original.shape

# Create domains
domain = 'text' # area, pixels, text

if domain == 'area':
    known = np.ones((N,M))
    known[0:73,88:120] = 0
elif domain == 'pixels':
    known = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            if np.random.random() > 0.3:
                known[i, j] = 1
elif domain == 'text':
    image = Image.fromarray(original)
    fnt = ImageFont.truetype('/Library/Fonts/Arial.ttf', 20)
    d = ImageDraw.Draw(image)
    d.text((20,3), "I am a dog ", font=fnt, fill = 0)
    d.text((20,25), "I am a dog ", font=fnt, fill = 0)
    d.text((20,45), "I am a dog ", font=fnt, fill = 0)
    d.text((20,65), "I am a dog ", font=fnt, fill = 0)
    d.text((20,85), "I am a dog ", font=fnt, fill = 0)
    d.text((20,105), "I am a dog ", font=fnt, fill = 0)
    corrupted = np.asarray(image, dtype = 'float64')
    known = np.ones((N,M))
    known[corrupted==0]=0
   
                              
#%%
# Create problem, choose regulariser and parameters
reg = 'TGV'

if reg == 'TV':
    u = Variable((N, M))
    alpha = 0.005
    regulariser = alpha * tv(u)
elif reg == 'TV2':
    u = Variable((N, M))
    alpha = 0.0005
    regulariser = alpha * tv2(u)
elif reg == 'ICTV':
    u = Variable((N, M))
    v = Variable((N, M))
    alpha0 = 0.04
    alpha1 = 0.08
    regulariser = ictv(u, v, alpha0, alpha1)  
elif reg == 'TGV':
    u = Variable((N, M))
    w1 = Variable((N, M))
    w2 = Variable((N, M))
    alpha0 = 0.01
    alpha1 = 0.001
    regulariser = tgv(u,w1,w2,alpha0, alpha1)  
    
fidelity = 0.5 * sum_squares(multiply(known, u) - multiply(known,original)) 

obj = Minimize(regulariser + fidelity)
prob = Problem(obj)
prob.solve(verbose = True, solver = MOSEK)    

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
# Display the in-painted image.
ax[1,0].imshow(u.value, cmap='gray');
ax[1,0].set_title("In-Painted Image")
ax[1,0].axis('off')

ax[0,0].imshow(original*known, cmap='gray');
ax[0,0].set_title("Corrupted")
ax[0,0].axis('off');

ax[0,1].imshow(original, cmap='gray');
ax[0,1].set_title("Original")
ax[0,1].axis('off')

ax[1,1].imshow(np.abs(original - u.value), cmap='gray');
ax[1,1].set_title("Difference")
ax[1,1].axis('off');


