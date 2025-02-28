#  Copyright 2021 United Kingdom Research and Innovation
#  Copyright 2021 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

import os
import subprocess
import sys
import importlib

import logging
import cil.version
import pprint
import unittest

# initialise_tests should be imported and called in each test file to change default behaviour
# it must be in each file as each may get run as an independent processes

def startTestRun(self):
    """Called once before any tests are executed.
    """
    #set logging
    logging.basicConfig(level=logging.WARNING)

    print("\n----------------------------------------------------------------------")
    print("TEST SYSTEM CONFIGURATION")
    print("CIL version: ", cil.version.version)
    pprint.pprint(system_state)
    print("----------------------------------------------------------------------\n")
    sys.stdout.flush()

def initialise_tests():
    """
    Updates the unittests.TestResults.StartTestRun method with our version.
    """
    setattr(unittest.TestResult, 'startTestRun', startTestRun)


# system state dictionary to be populated
system_state = {}

#nvidia
try:
    subprocess.check_output('nvidia-smi')
    has_nvidia = True
except:
    if os.environ.get("TESTS_FORCE_GPU", ""):
        raise ImportError
    has_nvidia = False
system_state['has_nvidia']=has_nvidia

#astra
module_info = importlib.util.find_spec("astra")
if module_info is None:
    if os.environ.get("TESTS_FORCE_GPU", ""):
        raise ImportError
    has_astra = False
else:
    has_astra = True
system_state['has_astra']=has_astra

#tigre
module_info = importlib.util.find_spec("tigre")
if module_info is None:
    if os.environ.get("TESTS_FORCE_GPU", ""):
        raise ImportError
    has_tigre = False
else:
    has_tigre = True
system_state['has_tigre']=has_tigre

#cvxpy
module_info = importlib.util.find_spec("cvxpy")
if module_info is None:
    has_cvxpy = False
else:
    has_cvxpy = True
system_state['has_cvxpy']=has_cvxpy


#ipp
from cil.framework import cilacc
has_ipp = hasattr(cilacc, 'filter_projections_avh')
system_state['has_ipp']=has_ipp


#ccpi-regularisation toolkit
module_info = importlib.util.find_spec("ccpi")
if module_info != None:
    module_info = importlib.util.find_spec("ccpi.filters.regularisers")

if module_info is None:
    has_ccpi_regularisation = False
else:
    has_ccpi_regularisation = True

system_state['has_ccpi_regularisation']= has_ccpi_regularisation


#tomophantom
module_info = importlib.util.find_spec("tomophantom")
if module_info is None:
    has_tomophantom = False
else:
    has_tomophantom = True
system_state['has_tomophantom']= has_tomophantom


#has_numba
module_info = importlib.util.find_spec("numba")
if module_info is None:
    has_numba = False
else:
    has_numba = True
system_state['has_numba']= has_numba

# has_matplotlib
module_info = importlib.util.find_spec("matplotlib")
if module_info is None:
    has_matplotlib = False
else:
    has_matplotlib = True
system_state['has_matplotlib'] = has_matplotlib


# to disable prints from 3rd part libraries and tests
def disable_print():
    sys.stdout = open(os.devnull, 'w')

def enable_prints():
    sys.stdout = sys.__stdout__
