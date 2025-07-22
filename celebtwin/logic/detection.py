"""Face detection and alignment."""

from dataclasses import dataclass
from pathlib import Path

import deepface.models.Detector  # type: ignore
import deepface.modules.modeling  # type: ignore
import numpy as np
import tensorflow as tf

from .annenums import Detector

# from deepface.modules.detection import project_facial_area

Image = str | Path | np.ndarray | tf.Tensor

@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True)
class Box:
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class Face:
    box: Box
    left_eye: Point
    right_eye: Point
    confidence: float


def detect_faces(detector: Detector, image: Image) -> list[Face]:
    if detector == Detector.BUILTIN:
        from ..logic import preproc_face
        faces = preproc_face.detect_faces(image)
    else:
        face_detector: deepface.models.detector.DeepFaceDetector = \
            deepface.modules.modeling.build_model(
                task="face_detector", model_name=detector.value)
        faces = [Face(
            box=Box(face.x, face.y, face.w, face.h),
            left_eye=Point(*face.left_eye),
            right_eye=Point(*face.right_eye),
            confidence=face.confidence,
        ) for face in face_detector.detect_faces(image)]
    faces.sort(key=lambda face: face.box.width * face.box.height, reverse=True)
    return faces
