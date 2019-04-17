#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 10:30:46 2019

@author: evangelos
"""

import odl


# --- Set up problem definition --- #


# Define function space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300])

# Create phantom
data = odl.phantom.shepp_logan(space, modified=True)
data = odl.phantom.salt_pepper_noise(data)

# Create gradient operator
grad = odl.Gradient(space)


# --- Set up the inverse problem --- #

# Create data discrepancy by translating the l1 norm
l1_norm = odl.solvers.L1Norm(space)
data_discrepancy = l1_norm.translated(data)

# l2-squared norm of gradient
regularizer = 0.05 * odl.solvers.L2NormSquared(grad.range) * grad

# --- Select solver parameters and solve using proximal gradient --- #

# Select step-size that guarantees convergence.
gamma = 0.01

# Optionally pass callback to the solver to display intermediate results
callback = (odl.solvers.CallbackPrintIteration() &
            odl.solvers.CallbackShow())

# Run the algorithm (ISTA)
x = space.zero()
odl.solvers.proximal_gradient(
    x, f=data_discrepancy, g=regularizer, niter=200, gamma=gamma,
    callback=callback)
