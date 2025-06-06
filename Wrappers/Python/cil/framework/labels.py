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


class Backend(StrEnum):
    """
    Available backends for CIL.

    Examples
    --------
    ```
    FBP(data, backend=Backend.ASTRA)
    FBP(data, backend="astra")
    ```
    """
    ASTRA = auto()
    TIGRE = auto()
    CIL = auto()


class _DimensionBase:
    @classmethod
    def _default_order(cls, engine: str) -> tuple:
        raise NotImplementedError

    @classmethod
    def get_order_for_engine(cls, engine: str, geometry=None) -> tuple:
        """
        Returns the order of dimensions for a specific engine and geometry.

        Parameters
        ----------
        geometry: ImageGeometry | AcquisitionGeometry
            If unspecified, the default order is returned.
        """
        order = cls._default_order(engine)
        if geometry is None:
            return order
        return tuple(label for label in order if label in geometry.dimension_labels)

    @classmethod
    def check_order_for_engine(cls, engine: str, geometry) -> bool:
        """
        Returns True iff the order of dimensions is correct for a specific engine and geometry.

        Parameters
        ----------
        geometry: ImageGeometry | AcquisitionGeometry

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


class ImageDimension(_DimensionBase, StrEnum):
    """
    Available dimension labels for image data.

    Examples
    --------
    >>> data.reorder([ImageDimension.HORIZONTAL_X, ImageDimension.VERTICAL])
    >>> data.reorder(["horizontal_x", "vertical"])

    """
    CHANNEL = auto()
    VERTICAL = auto()
    HORIZONTAL_X = auto()
    HORIZONTAL_Y = auto()

    @classmethod
    def _default_order(cls, engine: str) -> tuple:
        engine = Backend(engine)
        orders = {
            Backend.ASTRA: (cls.CHANNEL, cls.VERTICAL, cls.HORIZONTAL_Y, cls.HORIZONTAL_X),
            Backend.TIGRE: (cls.CHANNEL, cls.VERTICAL, cls.HORIZONTAL_Y, cls.HORIZONTAL_X),
            Backend.CIL: (cls.CHANNEL, cls.VERTICAL, cls.HORIZONTAL_Y, cls.HORIZONTAL_X)}
        return orders[engine]


class AcquisitionDimension(_DimensionBase, StrEnum):
    """
    Available dimension labels for acquisition data.

    Examples
    --------
    >>> data.reorder([AcquisitionDimension.CHANNEL,
                  AcquisitionDimension.ANGLE,
                  AcquisitionDimension.HORIZONTAL])
    >>> data.reorder(["channel", "angle", "horizontal"])
    """
    CHANNEL = auto()
    ANGLE = auto()
    VERTICAL = auto()
    HORIZONTAL = auto()

    @classmethod
    def _default_order(cls, engine: str) -> tuple:
        engine = Backend(engine)
        orders = {
            Backend.ASTRA: (cls.CHANNEL, cls.VERTICAL, cls.ANGLE, cls.HORIZONTAL),
            Backend.TIGRE: (cls.CHANNEL, cls.ANGLE, cls.VERTICAL, cls.HORIZONTAL),
            Backend.CIL: (cls.CHANNEL, cls.ANGLE, cls.VERTICAL, cls.HORIZONTAL)}
        return orders[engine]


class FillType(StrEnum):
    """
    Available fill types for image data.

    Attributes
    ----------
    RANDOM:
        Fill with random values.
    RANDOM_INT:
        Fill with random integers.
    RANDOM_DEPRECATED:
        Fill with random values using numpy.random.random_sample method
    RANDOM_INT_DEPRECATED:
        Fill with random integers using numpy.random.randint method

    Examples
    --------
    >>> data.fill(FillType.RANDOM)
    >>> data.fill("random")
    """
    RANDOM = auto()
    RANDOM_INT = auto()
    RANDOM_DEPRECATED = auto()
    RANDOM_INT_DEPRECATED = auto()


class AngleUnit(StrEnum):
    """
    Available units for angles.

    Examples
    --------
    >>> data.geometry.set_angles(angle_data, angle_units=AngleUnit.DEGREE)
    >>> data.geometry.set_angles(angle_data, angle_units="degree")

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
        return super().__eq__(self.__class__(value.upper()) if isinstance(value, str) else value)


class AcquisitionType(Flag):
    """
    Available acquisition types & dimensions.

    WARNING: It's best to use strings rather than integers to initialise.
    >>> AcquisitionType(3) == AcquisitionType(2 | 1) == AcquisitionType.CONE|PARALLEL != AcquisitionType('3D')

    Attributes
    ----------
    PARALLEL:
        Parallel beam.
    CONE:
        Cone beam.
    DIM2:
        2D acquisition.
    DIM3:
        3D acquisition.
    """
    PARALLEL = auto()
    CONE = auto()
    DIM2 = auto()
    DIM3 = auto()

    def validate(self):
        """
        Check if the geometry and dimension types are allowed
        """
        assert len(self.dimension) < 2, f"{self} must be 2D xor 3D"
        assert len(self.geometry) < 2, f"{self} must be parallel xor cone beam"
        return self

    @property
    def dimension(self):
        """
        Returns the label for the dimension type
        """
        return self & (self.DIM2 | self.DIM3)

    @property
    def geometry(self):
        """
        Returns the label for the geometry type
        """
        return self & (self.PARALLEL | self.CONE)

    @classmethod
    def _missing_(cls, value):
        """2D/3D aliases"""
        if isinstance(value, str):
            value = {'2D': 'DIM2', '3D': 'DIM3'}.get(value.upper(), value)
        return super()._missing_(value)

    def __str__(self) -> str:
        """2D/3D special handling"""
        return '2D' if self == self.DIM2 else '3D' if self == self.DIM3 else (self.name or super().__str__())

    def __hash__(self) -> int:
        """consistent hashing for dictionary keys"""
        return hash(self.value)

    # compatibility with Python>=3.11 `enum.Flag`
    def __len__(self) -> int:
        return bin(self.value).count('1')
