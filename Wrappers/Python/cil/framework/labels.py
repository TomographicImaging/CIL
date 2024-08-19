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
from enum import Enum, Flag as _Flag, auto, unique
try:
    from enum import EnumType
except ImportError: # Python<3.11
    from enum import EnumMeta as EnumType


class _StrEnumMeta(EnumType):
    """Python<3.12 requires this in a metaclass (rather than directly in StrEnum)"""
    def __contains__(self, item: str) -> bool:
        try:
            key = item.upper()
        except (AttributeError, TypeError):
            return False
        return key in self.__members__ or item in self.__members__.values()


@unique
class StrEnum(str, Enum, metaclass=_StrEnumMeta):
    """Case-insensitive StrEnum"""
    @classmethod
    def _missing_(cls, value: str):
        return cls.__members__.get(value.upper(), None)

    def __eq__(self, value: str) -> bool:
        try:
            value = self.__class__[value.upper()]
        except (KeyError, ValueError, AttributeError):
            pass
        return super().__eq__(value)

    def __hash__(self) -> int:
        """consistent hashing for dictionary keys"""
        return hash(self.value)

    # compatibility with Python>=3.11 `enum.StrEnum`
    __str__ = str.__str__
    __format__ = str.__format__

    @staticmethod
    def _generate_next_value_(name: str, start, count, last_values) -> str:
        return name.lower()


class Backends(StrEnum):
    """
    Available backends for CIL.

    Examples
    --------
    ```
    FBP(data, backend=Backends.ASTRA)
    FBP(data, backend="astra")
    ```
    """
    ASTRA = auto()
    TIGRE = auto()
    CIL = auto()


class ImageDimensionLabels(StrEnum):
    """
    Available dimension labels for image data.

    Examples
    --------
    ```
    data.reorder([ImageDimensionLabels.HORIZONTAL_X, ImageDimensionLabels.VERTICAL])
    data.reorder(["horizontal_x", "vertical"])
    ```
    """
    CHANNEL = auto()
    VERTICAL = auto()
    HORIZONTAL_X = auto()
    HORIZONTAL_Y = auto()

    @classmethod
    def get_order_for_engine(cls, engine: str, geometry=None) -> tuple:
        """
        Returns the order of dimensions for a specific engine and geometry.

        Parameters
        ----------
        geometry: ImageGeometry, optional
            If unspecified, the default order is returned.
        """
        order = cls.CHANNEL, cls.VERTICAL, cls.HORIZONTAL_Y, cls.HORIZONTAL_X
        engine_orders = {Backends.ASTRA: order, Backends.TIGRE: order, Backends.CIL: order}
        dim_order = engine_orders[Backends(engine)]

        if geometry is None:
            return dim_order
        return tuple(label for label in dim_order if label in geometry.dimension_labels)

    @classmethod
    def check_order_for_engine(cls, engine: str, geometry) -> bool:
        """
        Returns True iff the order of dimensions is correct for a specific engine and geometry.

        Parameters
        ----------
        geometry: ImageGeometry

        Raises
        ------
        ValueError if the order of dimensions is incorrect.
        """
        order_requested = cls.get_order_for_engine(engine, geometry)
        if order_requested == tuple(geometry.dimension_labels):
            return True
        raise ValueError(
            f"Expected dimension_label order {order_requested}"
            f" got {tuple(geometry.dimension_labels)}."
            f" Try using `data.reorder('{engine}')` to permute for {engine}")


class AcquisitionDimensionLabels(StrEnum):
    """
    Available dimension labels for acquisition data.

    Examples
    --------
    ```
    data.reorder([AcquisitionDimensionLabels.CHANNEL,
                  AcquisitionDimensionLabels.ANGLE,
                  AcquisitionDimensionLabels.HORIZONTAL])
    data.reorder(["channel", "angle", "horizontal"])
    ```
    """
    CHANNEL = auto()
    ANGLE = auto()
    VERTICAL = auto()
    HORIZONTAL = auto()

    @classmethod
    def get_order_for_engine(cls, engine: str, geometry=None) -> tuple:
        """
        Returns the order of dimensions for a specific engine and geometry.

        Parameters
        ----------
        geometry : AcquisitionGeometry, optional
            If unspecified, the default order is returned.
        """
        engine_orders = {
            Backends.ASTRA: (cls.CHANNEL, cls.VERTICAL, cls.ANGLE, cls.HORIZONTAL),
            Backends.TIGRE: (cls.CHANNEL, cls.ANGLE, cls.VERTICAL, cls.HORIZONTAL),
            Backends.CIL: (cls.CHANNEL, cls.ANGLE, cls.VERTICAL, cls.HORIZONTAL)}
        dim_order = engine_orders[Backends(engine)]

        if geometry is None:
            return dim_order
        return tuple(label for label in dim_order if label in geometry.dimension_labels)

    @classmethod
    def check_order_for_engine(cls, engine: str, geometry) -> bool:
        """
        Returns True iff the order of dimensions is correct for a specific engine and geometry.

        Parameters
        ----------
        geometry: AcquisitionGeometry

        Raises
        ------
        ValueError if the order of dimensions is incorrect.
        """
        order_requested = cls.get_order_for_engine(engine, geometry)
        if order_requested == tuple(geometry.dimension_labels):
            return True
        raise ValueError(
            f"Expected dimension_label order {order_requested},"
            f" got {tuple(geometry.dimension_labels)}."
            f" Try using `data.reorder('{engine}')` to permute for {engine}")


class FillTypes(StrEnum):
    """
    Available fill types for image data.

    Attributes
    ----------
    RANDOM: Fill with random values.
    RANDOM_INT: Fill with random integers.

    Examples
    --------
    ```
    data.fill(FillTypes.RANDOM)
    data.fill("random")
    ```
    """
    RANDOM = auto()
    RANDOM_INT = auto()


class UnitsAngles(StrEnum):
    """
    Available units for angles.

    Examples
    --------
    ```
    data.geometry.set_unitangles(angle_data, angle_units=UnitsAngles.DEGREE)
    data.geometry.set_unit(angle_data, angle_units="degree")
    ```
    """
    DEGREE = auto()
    RADIAN = auto()


class _FlagMeta(EnumType):
    """Python<3.12 requires this in a metaclass (rather than directly in Flag)"""
    def __contains__(self, item) -> bool:
        return item.upper() in self.__members__ if isinstance(item, str) else super().__contains__(item)


@unique
class Flag(_Flag, metaclass=_FlagMeta):
    """Case-insensitive Flag"""
    @classmethod
    def _missing_(cls, value):
        return cls.__members__.get(value.upper(), None) if isinstance(value, str) else super()._missing_(value)

    def __eq__(self, value: str) -> bool:
        return super().__eq__(self.__class__[value.upper()] if isinstance(value, str) else value)


class AcquisitionType(Flag):
    """
    Available acquisition types & dimensions.

    Attributes
    ----------
    PARALLEL: Parallel beam.
    CONE: Cone beam.
    DIM2: 2D acquisition.
    DIM3: 3D acquisition.
    """
    PARALLEL = auto()
    CONE = auto()
    DIM2 = auto()
    DIM3 = auto()

    @classmethod
    def _missing_(cls, value):
        """2D/3D aliases"""
        if isinstance(value, str):
            value = {'2D': 'DIM2', '3D': 'DIM3'}.get(value.upper(), value)
        return super()._missing_(value)

    def __str__(self) -> str:
        """2D/3D special handling"""
        return '2D' if self == self.DIM2 else '3D' if self == self.DIM3 else self.name
