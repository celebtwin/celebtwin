"""Enums for ANN strategies.

Provide only the enums that are used in the ANN strategies, for the benefit of
type checking.

This module should only import standard library to support lazy loading.
"""

import functools
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .. import logic


class Model(str, Enum):
    FACENET = "Facenet"
    FACENET512 = "Facenet512"
    VGG_FACE = "VGG-Face"
    OPENFACE = "OpenFace"
    DEEPFACE = "DeepFace"
    DEEPID = "DeepID"
    DLIB = "Dlib"
    ARC_FACE = "ArcFace"
    S_FACE = "SFace"
    GHOST_FACE_NET = "GhostFaceNet"

    @functools.cached_property
    def embedding_size(self) -> int:
        return _embedding_size_of[self]

    @functools.cached_property
    def normalization(self) -> str:
        return _normalization_of[self]


_embedding_size_of = {
    'Facenet': 128,
    'Facenet512': 512,
    'VGG-Face': 4096,
    "OpenFace": 128,
    "DeepFace": 4096,
    "DeepID": 512,  # or 4096 depending on the version
    "Dlib": 128,
    "ArcFace": 512,
    "SFace": 512,
    "GhostFaceNet": 512,
}

_normalize = "Facenet2018"  # simply / 127.5 - 1
_normalization_of = {
    'Facenet': _normalize,
    'Facenet512': _normalize,
    'VGG-Face': 'VGGFace2',
    "OpenFace": _normalize,
    "DeepFace": _normalize,
    "DeepID": _normalize,
    "Dlib": _normalize,
    "ArcFace": "ArcFace",
    "SFace": _normalize,
    "GhostFaceNet": _normalize,
}


class Detector(str, Enum):
    OPENCV = "opencv"
    RETINA_FACE = "retinaface"
    MTCNN = "mtcnn"
    SSD = "ssd"
    DLIB = "dlib"
    MEDIAPIPE = "mediapipe"
    YOLOV8 = "yolov8"
    YOLOV11N = "yolov11n"
    YOLOV11S = "yolov11s"
    CENTERFACE = "centerface"
    BUILTIN = "builtin"
    SKIP = "skip"


class ANNBackend(str, Enum):
    ANNOY = "annoy"
    BRUTE_FORCE = "brute"
    HNSW = "hnsw"

    @property
    def strategy_class(self) -> type["logic.ann.ANNStrategy"]:
        return _strategy_class_of()[self]


@functools.cache
def _strategy_class_of() -> dict[ANNBackend, type["logic.ann.ANNStrategy"]]:
    from celebtwin.logic import ann
    return {
        ANNBackend.ANNOY: ann.AnnoyStrategy,
        ANNBackend.BRUTE_FORCE: ann.BruteForceStrategy,
        ANNBackend.HNSW: ann.HnswStrategy,
    }
