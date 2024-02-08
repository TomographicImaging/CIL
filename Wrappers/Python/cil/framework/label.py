# -*- coding: utf-8 -*-
#  Copyright 2024 United Kingdom Research and Innovation
#  Copyright 2024 The University of Manchester
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
# Joshua DM Hellier (University of Manchester)
# Nicholas Whyatt (UKRI-STFC)

from typing import TypedDict


class ImageLabels(TypedDict):
    RANDOM: str
    RANDOM_INT: str
    CHANNEL: str
    VERTICAL: str
    HORIZONTAL_X: str
    HORIZONTAL_Y: str


class AcquisitionLabels(TypedDict):
    RANDOM: str
    RANDOM_INT: str
    ANGLE_UNIT: str
    DEGREE: str
    RADIAN: str
    CHANNEL: str
    ANGLE: str
    VERTICAL: str
    HORIZONTAL: str
    PARALLEL: str
    CONE: str
    DIM2: str
    DIM3: str


image_labels: ImageLabels = {"RANDOM": "random",
                             "RANDOM_INT": "random_int",
                             "CHANNEL": "channel",
                             "VERTICAL": "vertical",
                             "HORIZONTAL_X": "horizontal_x",
                             "HORIZONTAL_Y": "horizontal_y"}

acquisition_labels: AcquisitionLabels = {"RANDOM": "random",
                                         "RANDOM_INT": "random_int",
                                         "ANGLE_UNIT": "angle_unit",
                                         "DEGREE": "degree",
                                         "RADIAN": "radian",
                                         "CHANNEL": "channel",
                                         "ANGLE": "angle",
                                         "VERTICAL": "vertical",
                                         "HORIZONTAL": "horizontal",
                                         "PARALLEL": "parallel",
                                         "CONE": "cone",
                                         "DIM2": "2D",
                                         "DIM3": "3D"}
