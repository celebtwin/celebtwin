
import functools
from dataclasses import dataclass
from math import atan2, degrees
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN  # type: ignore


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
    left_eye: Point
    right_eye: Point
    box: Box


mtcnn_detector = MTCNN()


def detect_faces(image: str | Path | np.ndarray | tf.Tensor) -> list[Face]:
    if isinstance(image, (str, Path)):
        image_data = tf.io.read_file(str(image))
        image = tf.image.decode_image(image_data, channels=3)
    elif not isinstance(image, (np.ndarray, tf.Tensor)):
        raise ValueError("image must be a file path, numpy array or tf.Tensor")

    unwrap = functools.partial(map, lambda x: x.item())
    return [Face(
        left_eye=Point(*unwrap(detection["keypoints"]["left_eye"])),
        right_eye=Point(*unwrap(detection["keypoints"]["right_eye"])),
        box=Box(*unwrap(detection["box"]))
    ) for detection in mtcnn_detector.detect_faces(image)]


class NoFaceDetectedError(Exception):
    """Raised when face detection failed during image preprocessing."""

    def __init__(self, path: Path | None = None):
        if path is None:
            message = "❌ No face detected"
        else:
            message = f"❌ No face detected: {path}"
        super().__init__(message)


def preprocess_face_aligned(path: Path) -> np.ndarray:
    """Load an image, detect a face, crop and align the image.

    The image is cropped to include only the face. Resizing should be done by the caller.
    """
    # MTCNN requires a RGB image. Apply grayscale conversion later.
    image = tf.image.decode_image(
        tf.io.read_file(str(path)),
        channels=3, expand_animations=False)

    faces = detect_faces(image)
    if not faces:
        raise NoFaceDetectedError(path)
    face = faces[0]

    # Angle between the eyes and the horizontal axis.
    left_eye = face.left_eye
    right_eye = face.right_eye
    dx = right_eye.x - left_eye.x
    dy = right_eye.y - left_eye.y
    angle = degrees(atan2(dy, dx))

    eyes_center = Point((left_eye.x + right_eye.x) / 2,
                        (left_eye.y + right_eye.y) / 2)

    # Rotate the image to align the eyes horizontally.
    rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
    aligned_image = cv2.warpAffine(
        image.numpy(), rotation_matrix, (image.shape[1], image.shape[0])
    )

    # Detect face again after rotation to get the best cropping box.
    faces_aligned = detect_faces(aligned_image)
    if not faces_aligned:
        raise NoFaceDetectedError(path)

    # Crop the face from the aligned image
    box = faces_aligned[0].box
    cropped_image = aligned_image[
        box.top:box.top + box.height, box.left:box.left + box.width]
    return cropped_image
