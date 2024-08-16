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
    """
    Base class for labels enumeration. These changes are needed for python < 3.12.

    Methods:
    --------
    - __eq__(other): Checks if the enum or enum values are equal
    - __contains__(item): Checks if the enum contains the given value
    """

    def __eq__(self, other):
        if self.value == other:
            return True
        return False

    def __contains__(self, item):
        for member in self:
            if member.value == item:
                return True
        return False

# Needed for python < 3.12
EnumMeta.__contains__ = _LabelsBase.__contains__

class Backends(_LabelsBase):
    """
    Available backends for CIL.

    Attributes
    ----------
    ASTRA ('astra'): The ASTRA toolbox.
    TIGRE ('tigre'): The TIGRE toolbox.
    CIL ('cil'): Native CIL implementation.

    Examples
    --------
    FBP(data, backend=Backends.ASTRA)
    FBP(data, backend="astra")
    """
    ASTRA = "astra"
    TIGRE = "tigre"
    CIL = "cil"

class ImageDimensionLabels(_LabelsBase):
    """
    Available dimension labels for image data.
    
    Attributes
    ----------
    CHANNEL ('channel'): The channel dimension.
    VERTICAL ('vertical'): The vertical dimension.
    HORIZONTAL_X ('horizontal_x'): The horizontal dimension in x.
    HORIZONTAL_Y ('horizontal_y'): The horizontal dimension in y.

    Examples
    --------
    data.reorder([ImageDimensionLabels.HORIZONTAL_X, ImageDimensionLabels.VERTICAL])
    data.reorder(["horizontal_x", "vertical"])
    """
    CHANNEL = "channel"
    VERTICAL = "vertical"
    HORIZONTAL_X = "horizontal_x"
    HORIZONTAL_Y = "horizontal_y"

    @classmethod
    def get_order_for_engine(cls, engine, geometry=None):
        """
        Returns the order of dimensions for a specific engine and geometry.

        Parameters:
        ----------
        engine : str
            The engine name.
        geometry : ImageGeometry, optional
            The geometry object. If None, the default order is returned.

        Returns:
        --------
        list
            The order of dimensions for the given engine and geometry.
        """
        order = [cls.CHANNEL.value, cls.VERTICAL.value, \
                 cls.HORIZONTAL_Y.value, cls.HORIZONTAL_X.value]
        
        engine_orders = {
            Backends.ASTRA.value: order,
            Backends.TIGRE.value: order,
            Backends.CIL.value: order
        }

        dim_order = engine_orders[Backends(engine).value]

        if geometry is None:
            return dim_order
        else:
            return [label for label in dim_order if label in geometry.dimension_labels ]

    @classmethod
    def check_order_for_engine(cls, engine, geometry):
        """
        Checks if the order of dimensions is correct for a specific engine and geometry.

        Parameters:
        ----------
        engine : str
            The engine name.
        geometry : ImageGeometry
            The geometry object.

        Returns:
        --------
        bool
            True if the order of dimensions is correct.

        Raises:
        -------
        ValueError
            If the order of dimensions is incorrect.
        """
        order_requested = cls.get_order_for_engine(engine, geometry)

        if order_requested == list(geometry.dimension_labels):
            return True
        else:
            raise ValueError(
                f"Expected dimension_label order {order_requested}, \
                    got {list(geometry.dimension_labels)}.\n\
                    Try using `data.reorder('{engine}')` to permute for {engine}")

class AcquisitionDimensionLabels(_LabelsBase):
    """
    Available dimension labels for acquisition data.
    
    Attributes
    ----------
    CHANNEL ('channel'): The channel dimension.
    ANGLE ('angle'): The angle dimension.
    VERTICAL ('vertical'): The vertical dimension.
    HORIZONTAL ('horizontal'): The horizontal dimension.
    
    Examples
    --------
    data.reorder([AcquisitionDimensionLabels.CHANNEL,
                  AcquisitionDimensionLabels.ANGLE,
                  AcquisitionDimensionLabels.HORIZONTAL])
    data.reorder(["channel", "angle", "horizontal"])
    """

    CHANNEL = "channel"
    ANGLE = "angle"
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


    @classmethod
    def get_order_for_engine(cls, engine, geometry=None):
        """
        Returns the order of dimensions for a specific engine and geometry.

        Parameters:
        ----------
        engine : str
            The engine name.
        geometry : AcquisitionGeometry, optional
            The geometry object. If None, the default order is returned.

        Returns:
        --------
        list
            The order of dimensions for the given engine and geometry.
        """
        engine_orders = {
            Backends.ASTRA.value: [cls.CHANNEL.value, cls.VERTICAL.value, \
                                   cls.ANGLE.value, cls.HORIZONTAL.value],
            Backends.TIGRE.value: [cls.CHANNEL.value, cls.ANGLE.value, \
                                   cls.VERTICAL.value, cls.HORIZONTAL.value],
            Backends.CIL.value: [cls.CHANNEL.value, cls.ANGLE.value, \
                                 cls.VERTICAL.value, cls.HORIZONTAL.value]
        }

        dim_order = engine_orders[Backends(engine).value]

        if geometry is None:
            return dim_order
        else:
            return [label for label in dim_order if label in geometry.dimension_labels ]


    @classmethod
    def check_order_for_engine(cls, engine, geometry):
        """
        Checks if the order of dimensions is correct for a specific engine and geometry.

        Parameters:
        ----------
        engine : str
            The engine name.
        geometry : AcquisitionGeometry

        Returns:
        --------
        bool
            True if the order of dimensions is correct

        Raises:
        -------
        ValueError
            If the order of dimensions is incorrect.
        """

        order_requested = cls.get_order_for_engine(engine, geometry)

        if order_requested == list(geometry.dimension_labels):
            return True
        else:
            raise ValueError(
                f"Expected dimension_label order {order_requested}, \
                got {list(geometry.dimension_labels)}.\n\
                Try using `data.reorder('{engine}')` to permute for {engine}")

class FillTypes(_LabelsBase):
    """
    Available fill types for image data.

    Attributes
    ----------
    RANDOM ('random'): Fill with random values.
    RANDOM_INT ('random_int'): Fill with random integers.

    Examples
    --------
    data.fill(FillTypes.random)
    data.fill("random")
    """

    RANDOM = "random"
    RANDOM_INT = "random_int"

class UnitsAngles(_LabelsBase):
    """
    Available units for angles.

    Attributes
    ----------
    DEGREE ('degree'): Degrees.
    RADIAN ('radian'): Radians.

    Examples
    --------
    data.geometry.set_unitangles(angle_data, angle_units=UnitsAngles.DEGREE)
    data.geometry.set_unit(angle_data, angle_units="degree")
    """

    DEGREE = "degree"
    RADIAN = "radian"

class AcquisitionTypes(_LabelsBase):
    """
    Available acquisition types.

    Attributes
    ----------
    PARALLEL ('parallel'): Parallel beam.
    CONE ('cone'): Cone beam.
    """

    PARALLEL = "parallel"
    CONE = "cone"

class AcquisitionDimensions(_LabelsBase):
    """
    Available acquisition dimensions.

    Attributes
    ----------
    DIM2 ('2D'): 2D acquisition.
    DIM3 ('3D'): 3D acquisition.
    """

    DIM2 = "2D"
    DIM3 = "3D"
