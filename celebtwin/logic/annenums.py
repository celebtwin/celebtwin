"""Enums for ANN strategies.

Provide only the enums that are used in the ANN strategies, for the benefit of
type checking.
"""

import functools
from enum import Enum

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

_normalization_of = {
    'Facenet': 'Facenet2018',
    'Facenet512': 'Facenet2018',
    'VGG-Face': 'VGGFace2',
    "OpenFace": 'Facenet2018',
    "DeepFace": 'Facenet2018',
    "DeepID": 'Facenet2018',
    "Dlib": 'Facenet2018',
    "ArcFace": "ArcFace",
    "SFace": "Facenet2018",
    "GhostFaceNet": "Facenet2018",
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
    SKIP = "skip"


class ANNBackend(str, Enum):
    ANNOY = "annoy"
    BRUTE_FORCE = "brute"
    HNSW = "hnsw"

    @functools.cached_property
    def strategy_class(self) -> 'type[celebtwin.logic.ann.ANNStrategy]':
        from celebtwin.logic import ann
        if self == ANNBackend.ANNOY:
            return ann.AnnoyStrategy
        elif self == ANNBackend.BRUTE_FORCE:
            return ann.BruteForceStrategy
        elif self == ANNBackend.HNSW:
            return ann.HnswStrategy


_ann_backend_of = {b.value: b for b in ANNBackend}
