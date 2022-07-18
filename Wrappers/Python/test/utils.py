# -*- coding: utf-8 -*-
#  Copyright 2018 - 2022 United Kingdom Research and Innovation
#  Copyright 2018 - 2022 The University of Manchester
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

import os
import subprocess
import sys
import importlib

import logging
import cil.version

# startTestRun should be imported and used in each test file to change default behaviour
# it must be in each file as each may get run as an independent process
# use:
# setattr(unittest.TestResult, 'startTestRun', utils.startTestRun)

def startTestRun(self):
    """Called once before any tests are executed.
    """
    #set logging
    logging.basicConfig(level=logging.WARNING)


    print("\n----------------------------------------------------------------------")
    print("TEST SYSTEM CONFIGURATION")
    print("CIL version: ", cil.version.version)
    print(system_state)
    print("----------------------------------------------------------------------\n", flush=True)

# system state dictionary to be populated
system_state = {}

#nvidia
try:
    subprocess.check_output('nvidia-smi') == 0
    has_nvidia = True
except subprocess.CalledProcessError:
    has_nvidia = False
system_state['has_nvidia']=has_nvidia

#astra
module_info = importlib.util.find_spec("astra")
if module_info is None:
    has_astra = False
else:
    has_astra = True
system_state['has_astra']=has_astra

#astra_gpu
if  has_astra and has_nvidia:
    has_gpu_astra = True

#tigre
module_info = importlib.util.find_spec("tigre")
if module_info is None:
    has_tigre = False
else:
    has_tigre = True
system_state['has_tigre']=has_tigre

#tigre_gpu
if  has_tigre and has_nvidia:
    has_gpu_tigre = True

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


# to disable prints from 3rd part libraries and tests
def disable_print():
    sys.stdout = open(os.devnull, 'w')

def enable_prints():
    sys.stdout = sys.__stdout__