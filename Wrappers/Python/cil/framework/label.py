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

from typing import TypedDict, List

from .base import BaseAcquisitionGeometry


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


class DataOrder(TypedDict):
    ENGINES: List[str]
    ASTRA_IG_LABELS: List[str]
    TIGRE_IG_LABELS: List[str]
    ASTRA_AG_LABELS: List[str]
    TIGRE_AG_LABELS: List[str]
    CIL_IG_LABELS: List[str]
    CIL_AG_LABELS: List[str]
    TOMOPHANTOM_IG_LABELS: List[str]

    @staticmethod  # type: ignore[misc]
    def get_order_for_engine(engine, geometry):
        if engine == 'astra':
            if isinstance(geometry, BaseAcquisitionGeometry):
                dim_order = data_order["ASTRA_AG_LABELS"]
            else:
                dim_order = data_order["ASTRA_IG_LABELS"]
        elif engine == 'tigre':
            if isinstance(geometry, BaseAcquisitionGeometry):
                dim_order = data_order["TIGRE_AG_LABELS"]
            else:
                dim_order = data_order["TIGRE_IG_LABELS"]
        elif engine == 'cil':
            if isinstance(geometry, BaseAcquisitionGeometry):
                dim_order = data_order["CIL_AG_LABELS"]
            else:
                dim_order = data_order["CIL_IG_LABELS"]
        else:
            raise ValueError("Unknown engine expected one of {0} got {1}".format(data_order["ENGINES"], engine))

        dimensions = []
        for label in dim_order:
            if label in geometry.dimension_labels:
                dimensions.append(label)

        return dimensions

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

data_order: DataOrder = \
    {"ENGINES": ["astra", "tigre", "cil"],
     "ASTRA_IG_LABELS": [image_labels["CHANNEL"], image_labels["VERTICAL"], image_labels["HORIZONTAL_Y"], image_labels["HORIZONTAL_X"]],
     "TIGRE_IG_LABELS": [image_labels["CHANNEL"], image_labels["VERTICAL"], image_labels["HORIZONTAL_Y"], image_labels["HORIZONTAL_X"]],
     "ASTRA_AG_LABELS": [acquisition_labels["CHANNEL"], acquisition_labels["VERTICAL"], acquisition_labels["ANGLE"], acquisition_labels["HORIZONTAL"]],
     "TIGRE_AG_LABELS": [acquisition_labels["CHANNEL"], acquisition_labels["ANGLE"], acquisition_labels["VERTICAL"], acquisition_labels["HORIZONTAL"]],
     "CIL_IG_LABELS": [image_labels["CHANNEL"], image_labels["VERTICAL"], image_labels["HORIZONTAL_Y"], image_labels["HORIZONTAL_X"]],
     "CIL_AG_LABELS": [acquisition_labels["CHANNEL"], acquisition_labels["ANGLE"], acquisition_labels["VERTICAL"], acquisition_labels["HORIZONTAL"]],
     "TOMOPHANTOM_IG_LABELS": [image_labels["CHANNEL"], image_labels["VERTICAL"], image_labels["HORIZONTAL_Y"], image_labels["HORIZONTAL_X"]]
    }

get_order_for_engine = DataOrder.get_order_for_engine  # type: ignore[attr-defined]


def check_order_for_engine(engine, geometry):
    order_requested = get_order_for_engine(engine, geometry)

    if order_requested == list(geometry.dimension_labels):
        return True
    else:
        raise ValueError(
            "Expected dimension_label order {0}, got {1}.\nTry using `data.reorder('{2}')` to permute for {2}"
            .format(order_requested, list(geometry.dimension_labels), engine))
