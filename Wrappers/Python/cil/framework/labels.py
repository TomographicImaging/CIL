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

from enum import Enum

class _LabelsBase(Enum):

    @classmethod
    def validate(cls, label):
        """
        Validate if the given label is present in the class or its values.
        Parameters:
        label (str): The label to validate.
        Returns:
        bool: True if the label is present in the class or its values
        Raises:
        ValueError: If the label is not present in the class or its values.
        """
        if isinstance(label, cls):
            return True
        elif label in [e.name for e in cls]:
            return True
        elif label in [e.value for e in cls]:
            return True
        else:
            raise ValueError(f"Expected one of {[e.value for e in cls]}, got {label}")

    @classmethod
    def member_from_value(cls, label):
        if isinstance(label, str):
            label = label.lower()

        for member in cls:
            if member.value == label:
                return member
        raise ValueError(f"{label} is not a valid {cls.__name__}")

    @classmethod
    def member_from_key(cls, label):
        for member in cls:
            if member.name == label:
                return member
        raise ValueError(f"{label} is not a valid {cls.__name__}")
    
    @classmethod
    def get_enum_member(cls, label):
        if isinstance(label, cls):
            return label
        elif label in [e.name for e in cls]:
            return cls.member_from_key(label)
        elif label in [e.value for e in cls]:
            return cls.member_from_value(label)
        else:
            raise ValueError(f"{label} is not a valid {cls.__name__}")
        
    @classmethod
    def get_enum_value(cls, label):
        return cls.get_enum_member(label).value
        
    def __eq__(self, other):
        if isinstance(other, str):
            return self.value == other or self.name == other
        return super().__eq__(other)

class Backends(_LabelsBase):
    ASTRA = "astra"
    TIGRE = "tigre"
    CIL = "cil"

class DimensionLabelsImage(_LabelsBase):
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
        
class DimensionLabelsAcquisition(_LabelsBase):
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

class AcquisitionType(_LabelsBase):
    PARALLEL = "parallel"
    CONE = "cone"

class AcquisitionDimension(_LabelsBase):
    DIM2 = "2D"
    DIM3 = "3D"

