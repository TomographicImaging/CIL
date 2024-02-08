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
