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

from enum import Enum, EnumMeta

class _LabelsBase(Enum):

    @classmethod
    def validate(cls, label):
        try:
            for member in cls:
                if member == label:
                    return True 
        except:
            pass

        raise ValueError(f"Expected one of {[e.value for e in cls]} from {cls.__name__}, got {label}")

    @classmethod
    def get_enum_member(cls, label):
        try:
            for member in cls:
                if member == label:
                    return member 
        except:
            pass
        
        raise ValueError(f"Expected one of {[e.value for e in cls]} from {cls.__name__}, got {label}")
        
    @classmethod
    def get_enum_value(cls, label):
        return cls.get_enum_member(label).value
        
    def __eq__(self, other):
        if self.value == other:
            return True
        return False
        
    def __contains__(cls, item):
        for member in cls:
            if member.value == item:
                return True
        return False
        
# Needed for python < 3.12
EnumMeta.__contains__ = _LabelsBase.__contains__

class Backends(_LabelsBase):
    ASTRA = "astra"
    TIGRE = "tigre"
    CIL = "cil"

class ImageDimensionLabels(_LabelsBase):
    CHANNEL = "channel"
    VERTICAL = "vertical"
    HORIZONTAL_X = "horizontal_x"
    HORIZONTAL_Y = "horizontal_y"

    @classmethod
    def get_default_order_for_engine(cls, engine):
        engine_orders = {
            Backends.ASTRA.value: [cls.CHANNEL.value, cls.VERTICAL.value, cls.HORIZONTAL_Y.value, cls.HORIZONTAL_X.value],
            Backends.TIGRE.value: [cls.CHANNEL.value, cls.VERTICAL.value, cls.HORIZONTAL_Y.value, cls.HORIZONTAL_X.value],
            Backends.CIL.value: [cls.CHANNEL.value, cls.VERTICAL.value, cls.HORIZONTAL_Y.value, cls.HORIZONTAL_X.value]
        }
        Backends.validate(engine)
        engine = Backends.get_enum_value(engine)

        return engine_orders[engine]

    @classmethod
    def get_order_for_engine(cls, engine, geometry):

        dim_order = cls.get_default_order_for_engine(engine)
        dimensions = [label for label in dim_order if label in geometry.dimension_labels ]

        return dimensions

    @classmethod
    def check_order_for_engine(cls, engine, geometry):
        order_requested = cls.get_order_for_engine(engine, geometry)

        if order_requested == list(geometry.dimension_labels):
            return True
        else:
            raise ValueError(
                "Expected dimension_label order {0}, got {1}.\nTry using `data.reorder('{2}')` to permute for {2}"
                .format(order_requested, list(geometry.dimension_labels), engine))
        
class AcquisitionDimensionLabels(_LabelsBase):
    CHANNEL = "channel"
    ANGLE = "angle"
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"

    @classmethod
    def get_default_order_for_engine(cls, engine):
        engine_orders = {
            Backends.ASTRA.value: [cls.CHANNEL.value, cls.VERTICAL.value, cls.ANGLE.value, cls.HORIZONTAL.value],
            Backends.TIGRE.value: [cls.CHANNEL.value, cls.ANGLE.value, cls.VERTICAL.value, cls.HORIZONTAL.value],
            Backends.CIL.value: [cls.CHANNEL.value, cls.ANGLE.value, cls.VERTICAL.value, cls.HORIZONTAL.value]
        }
        Backends.validate(engine)
        engine = Backends.get_enum_value(engine)

        return engine_orders[engine]

    @classmethod
    def get_order_for_engine(cls, engine, geometry):

        dim_order = cls.get_default_order_for_engine(engine)
        dimensions = [label for label in dim_order if label in geometry.dimension_labels ]

        return dimensions

    @classmethod
    def check_order_for_engine(cls, engine, geometry):
        order_requested = cls.get_order_for_engine(engine, geometry)

        if order_requested == list(geometry.dimension_labels):
            return True
        else:
            raise ValueError(
                "Expected dimension_label order {0}, got {1}.\nTry using `data.reorder('{2}')` to permute for {2}"
                .format(order_requested, list(geometry.dimension_labels), engine))
    
class FillTypes(_LabelsBase):
    RANDOM = "random"
    RANDOM_INT = "random_int"

class UnitsAngles(_LabelsBase):
    DEGREE = "degree"
    RADIAN = "radian"

class AcquisitionTypes(_LabelsBase):
    PARALLEL = "parallel"
    CONE = "cone"

class AcquisitionDimensions(_LabelsBase):
    DIM2 = "2D"
    DIM3 = "3D"

